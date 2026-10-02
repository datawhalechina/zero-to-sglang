# Chapter 8 RadixAttention and Prefix Caching

In the previous part, we introduced how paging can efficiently manage KV Cache memory allocation and reduce fragmentation. **This chapter focuses further on a practical problem in online LLM inference: can the KV Cache be shared and reused across different requests to reduce GPU memory usage even further?** The answer is yes. Prefix caching (Prefix Cache) makes this possible by directly reusing the computed KV states of requests that share a common prefix. RadixAttention goes one step further: it combines KV Cache physical-block mapping, efficient prefix matching, scheduler coordination, and LRU eviction through a Radix Tree, forming an automated, **token-level** sharing mechanism.

## 1 Learning Objectives

This chapter starts from the problems left by traditional KV Cache management, introduces Prefix Cache, and then explores how it supports efficient KV Cache management during LLM inference. After reading this chapter, you should be able to answer the following questions:

- What problems does traditionally managed KV Cache leave behind after a request finishes?
- Why can a prefix be reused directly, while an arbitrary segment cannot?
- Why is a path-compressed Radix Tree more suitable for managing shared prefixes than a token-by-token Trie?
- What are the complete procedures for longest-prefix matching, insertion, branching, and node splitting, and can they be derived from a simple example?
- What information is stored in a Radix Tree node? How are a logical token-id sequence and physical cache slots in the KV Cache Pool associated with each other?
- In the complete RadixAttention cache-management lifecycle, what roles do match, schedule, lock, evict, and free play, and how do these operations work together?

## 2 From KV Cache to Prefix Cache

In Chapter 4 of Part I, we introduced the lifecycle and memory usage of the KV Cache during inference with an LLM that uses Full Attention, as well as the problems faced by traditional KV Cache management. We begin with a brief review.

---

### 2.1 Limitations of Traditional KV Cache

The **core benefit of a traditional KV Cache** is that, during autoregressive generation, the Key/Value tensors computed for previously processed tokens at every layer are cached. During prefill, the complete request is processed and its KV states are written to the cache. During each subsequent decode step, only Q, K, and V for the new token need to be computed; the new K and V are appended to the cache, and the current Q attends to all previously cached KV states. Historical K and V therefore do not need to be recomputed.

In many earlier basic implementations, however, **the KV Cache lifecycle was strictly bound to an individual request: GPU memory was allocated when a request arrived and released in full when the request finished**.

This can lead to **redundant computation across requests**. In online LLM services, many requests often share the same system prompt, few-shot examples, tool descriptions, or conversation history from the same turn (for example, every request may begin with “You are an expert”). If every request independently performs a full prefill, a large amount of redundant computation is incurred, increasing TTFT.

Prefix Cache was introduced to solve this problem. Its central idea is to change the ownership of the KV Cache: instead of binding the cache to an individual request, associate it with a reusable prefix sequence. After a request finishes, its KV blocks can remain in a shared cache pool, and an eviction policy determines when they are released.

**This naturally raises a key question: why can only a prefix be reused, rather than an arbitrary segment at any position?**

---

### 2.2 The Core Principle of Prefix Cache

For each token, the hidden state used to generate it can attend only to the information at that position and to its left. This constraint is enforced by causal attention.

**During training**, a forward pass over a complete sequence must use a causal mask: position $i$ can see only positions $0 \ldots i$ and cannot see future tokens:

$$
h_i = \mathrm{Attention}(q_i,K_{0:i},V_{0:i})
$$

This is one-way self-attention with a causal mask.

**During inference**:

- **Prefill**: the whole input prompt is processed in parallel. Because later prompt tokens are already present in the sequence, a causal mask is required.
- **Decode**: only the current new token is processed at each step. Future tokens have not been generated, so causality is enforced by the process itself: historical KV is available before the current token is computed. An explicit triangular mask is therefore usually unnecessary.

Starting with the first attention layer, $h_i$ has already integrated information from the beginning of the sequence up to the current position. The K and V tensors produced by later layers therefore carry information about this prefix. As a result, when any preceding token changes, the KV states at subsequent positions will also change.

> **Note**: *When explaining why Prefix Cache is feasible, we consider both the Prefill and Decode stages. In practice, Prefix Cache is used mainly during Prefill: the system matches a cached prefix, skips its repeated computation, and processes only the remaining suffix. During Decode, the system does not perform a new prefix match; it reuses the existing KV states and appends newly generated KV states.*

For example:

- Request A: `["weather", "is", "good"]`
- Request B: `["mood", "is", "good"]`

Although the two requests share the same suffix `["is", "good"]`, and “is” and “good” occur at the same sequence positions, the left context of “is” is “weather” in A and “mood” in B. **Because Transformer hidden states progressively integrate left-context information across layers**, the hidden state at this position will gradually diverge, causing the K and V states at subsequent layers to differ as well. Directly reusing the KV Cache for this suffix across requests therefore cannot guarantee the same computation under the original context and may produce incorrect inference results.

*Note: After Q and K are obtained through linear projections, positional encoding (such as RoPE, which acts on Q and K) is usually applied. Positions are counted from the beginning of the sequence. When **the prefix lengths differ**, the absolute positions of subsequent tokens change, and their K states change as well.*

Even without considering positional encoding, hidden states in later layers differ because their left contexts differ, so the safely reusable part is still a completely identical prefix. With positional mechanisms such as RoPE, Q/K also explicitly depend on token positions, which means that the same token at different absolute positions cannot be reused directly.

Now that we have explained why only prefixes can be reused, we can introduce RadixAttention. The implementation of cross-request KV Cache reuse through Prefix Cache will be explained later together with the relevant RadixAttention mechanisms.

---

### 2.3 RadixAttention and Attention

The name RadixAttention can easily give the impression that it modifies the attention computation inside the model. In fact, the attention formula and computation flow remain unchanged; RadixAttention is an efficient KV Cache management and scheduling mechanism. Attention still uses the original formula, while RadixAttention focuses on organizing and scheduling the KV Cache:

- **KV Cache Pool**: stores the actual K and V tensors on the GPU;
- **Radix Tree**: maps token-id sequences to physical indices in the pool and supports efficient prefix matching, insertion, node splitting, and eviction;
- **Cache-aware scheduler**: orders requests according to their match results, helping recently used shared prefixes remain in the cache and improving the hit rate.

**RadixAttention is therefore best understood as the caching and scheduling layer that prepares and manages reusable KV states for attention computation.**

## 3 Core Data Structure: Radix Tree

**The core data structure of RadixAttention is the Radix Tree.** It organizes the token-id sequences of requests as a compressed prefix tree, so that **the same prefix corresponds to only one copy of KV Cache**. At the same time, it supports policy-driven scheduling while efficiently managing the cache, significantly improving the cache hit rate and reducing GPU memory usage.

---

### 3.1 Data-Structure Requirements for Prefix Cache

What kind of data structure is needed for an efficient prefix cache? We can consider the following requirements:

- **Fast longest-prefix matching**: given a new request's token-id sequence, quickly find the longest common prefix shared with the existing cache and locate the reusable KV states;
- **Efficient dynamic insertion and splitting**: add a new sequence while changing as little of the existing structure as possible; when a new sequence matches only part of a compressed edge, split the node in place so that the common prefix and suffix are separated;
- **Prefix sharing with independent suffixes**: a common prefix should correspond to one shared KV Cache, while different suffixes after a branch should be stored and computed independently;
- **Compactness**: minimize metadata overhead and avoid creating a large number of nodes along long paths without branches;
- **Safe eviction**: when GPU memory is insufficient, release only the physical space occupied by KV blocks that are no longer referenced by any request.

Given these requirements, a **tree structure** is a natural way to manage KV Cache and implement Prefix Cache. A tree organized by token-id sequences naturally expresses prefix sharing, but different tree shapes have different costs:

| Structure | Edge label | Long single-branch path | Cost in prefix-cache scenarios |
|---|---|---|---|
| Ordinary tree / custom tree | No uniform constraint | Depends on the implementation | Difficult to guarantee efficient longest-prefix queries directly |
| Token-by-token Trie | One token | One node per token | Simple logically, but high node and pointer overhead |
| Radix Tree | A sequence of tokens | Compressed into one edge / node | Few nodes; splitting occurs only when a real branch is needed |

An ordinary Trie naturally supports prefix queries, but each edge generally corresponds to only one token. An LLM prompt may contain hundreds or even tens of thousands of tokens. If every token along a long path is represented by a separate node, the implementation incurs substantial object, pointer, and traversal overhead. A **Radix Tree compresses consecutive token-id segments without a branch into a single edge and creates extra structure only where a real split is needed, making it more suitable for Prefix Cache.**

---

### 3.2 Basic Components

A Radix Tree node contains a parent pointer, a dictionary of children, a `key_fn` used to extract a child index from a key tensor, and its own `_key`, `_value`, and length. The structure is illustrated below:

<div align="center">
  <img src="./images/8-1-RadixTreeNode Structure.png" alt="RadixTreeNode structure" width="800">
  <p><em>Figure 1. RadixTreeNode structure</em></p>
</div>

The definition can be represented as:

```python
class RadixTreeNode:
    def __init__(self, key_fn: KEY_FN) -> None:
        self._parent = None
        self.children = {}       # Dictionary of child nodes
        self.key_fn = key_fn     # Extract a child-sequence index from a key tensor
        self.ref_count = 0       # Reference count
        self.uuid = counter
        self.timestamp = tic or time.monotonic()  # Node access timestamp

        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int
```

Important points:

- `len(node.key) == len(node.value)`: every logical token has a corresponding physical cache position; the K/V entries are aligned one-to-one;
- in `__init__`, `_key`, `_value`, and `_length` are type annotations only and are not assigned actual values. They are assigned later through dedicated methods, which makes node insertion and splitting more flexible;
- the root node is empty and stores no cache data; each other node is distinguished by a monotonically increasing integer UUID.

The complete token-id sequence represented by a node is obtained by concatenating all `_key` segments along the path from the root to that node. The corresponding physical cache positions are obtained by concatenating the `_value` segments along the same path.

In addition to the node fields, several methods manage the tree structure:

| Method | Purpose |
|---|---|
| `set_key_value` | Set the node's key and value |
| `set_parent` | Set the node's parent |
| `length` / `parent` / `value` | Access the node length, parent, and value |
| `is_root` / `is_leaf` | Determine whether the node is the root or a leaf |
| `get_match_len` | Obtain the longest common prefix length with an input |
| `split_at` | Split the node at a specified position |

How do these components work together to optimize KV Cache management through Prefix Cache?

---

### 3.3 Inserting New Nodes, Longest-Prefix Matching, and Splitting

Core Prefix Cache management usually takes place during Prefill. When a new request arrives, the system performs a **longest common prefix match** between the request's token-id sequence and the token-id sequences corresponding to historical KV Cache entries, then determines the next operation from the result:

- **Continue matching**: if a segment of the new request's token-id sequence exactly matches the sequence in the current node, follow the common-prefix path downward and reuse as much existing KV Cache as possible.
- **Split**: if token ids diverge during matching and `0 < match_len < node.length`, split the current node at the first differing position. After the split, **the common prefix remains shared, while the divergent suffixes are stored independently**.

As long as the cache remains resident and the execution contexts are compatible, the token-id sequence represented by each node needs to be computed only once. *If the corresponding cache node is evicted, a later request that needs the same prefix must compute it again.*

A prompt is first converted into a token-id sequence by the tokenizer. For clarity, the example below works directly with token ids. Assume that three requests are inserted into the Radix Tree in the order A → B → C:

- **Request A token-id sequence**: [101, 11, 12, 21, 22]
- **Request B token-id sequence**: [101, 11, 12, 31, 32]
- **Request C token-id sequence**: [101, 11, 99]

We use these token ids directly in the example. Request A is processed first, and each token id corresponds to one physical slot:

$$A = [101, 11, 12, 21, 22]$$

The KV states produced by the linear projections are written to physical slots `[p0, p1, p2, p3, p4]`. Because the tree is empty, only one compressed node is created:

<div align="center">
  <img src="./images/8-2-radix tree build.png" alt="Initial construction of the Radix Tree" width="800">
  <p><em>Figure 2. Initial construction of the Radix Tree</em></p>
</div>

The second request, B, is $[101, 11, 12, 31, 32]$. Comparing it token by token with the existing node gives a longest common prefix of length 3, namely $[101,11,12]$. The divergence occurs inside the node, so the original node must be split at position 3:

<div align="center">
  <img src="./images/8-3-Radix Tree branching.png" alt="Radix Tree node split" width="800">
  <p><em>Figure 3. Radix Tree node split</em></p>
</div>

*Note: splitting reorganizes only the tree metadata. The shared prefix still points to the original physical slots `[p0, p1, p2]`; the shared portion does not need to be copied.*

The third request, C, is $[101, 11, 99]$. After entering the current first node, it matches only the first two token ids, $[101,11]$, so the node must be split again at position 2:

<div align="center">
  <img src="./images/8-4-Radix Tree multilevel.png" alt="Multi-level branching in a Radix Tree" width="800">
  <p><em>Figure 4. Multi-level branching in a Radix Tree</em></p>
</div>

At this point, all the key operations have appeared:

- **New-node insertion**: every request may require new nodes and physical slots. The first request is attached directly below the root, while later requests first undergo longest-prefix matching against token-id sequences already in the tree.
- **Longest-prefix matching**: requests A, B, and C share the longest common prefix `[101,11]`, so the corresponding KV Cache can be reused without recomputation.
- **Node splitting**: from the perspective of request A, the original `[101, 11, 12, 21, 22]` is eventually divided into `[101,11]`, `[12]`, and `[21,22]`.

This example covers the **core workflow** for constructing a Radix Tree, which forms the basis of Prefix Cache management. Next, we examine how these operations are implemented.

---

### 3.4 Implementing get_match_len() and split_at()

Before implementing **longest-prefix matching** (`get_match_len`) and **node splitting** (`split_at`), a node needs two basic capabilities:

- bind a token-id sequence (key) to its physical indices (value) and record its length through `set_key_value()`;
- attach its key correctly to the parent's `children` dictionary through `set_parent()`, establishing links in both directions.

**These helper methods are prerequisites for the subsequent operations.** Their concrete implementations are covered in the coding section associated with this chapter. Here, without considering cache eviction (and therefore without counters or timestamps), we analyze how longest-prefix matching and node splitting can be implemented using the node data and parent-child relationships already established.

---

The core idea of **longest-prefix matching** is as follows. Let `x` and `y` be two token-id sequences: `x` is the sequence already stored in a node, and `y` is the sequence from a new request. Starting from the beginning, compare the token ids one by one until either:

- the end of the shorter sequence is reached; or
- the token ids at the current position differ.

When traversal stops, `i` is the **length of the longest common prefix**. If traversal stops because the token ids differ, `i` is also the index of the first difference and can be used as the split position.

The corresponding pseudocode is:

```text
function get_match_len(x, y) -> int:
    i ← 0
    n ← min(length(x), length(y))
    while i < n and x[i] = y[i]:
        i ← i + 1
    return i
```

*Note: the project implementation uses a C++ routine for longest-prefix matching.*

---

The core idea of **node splitting** is to divide a partially matched node into two segments. Suppose `get_match_len()` returns a match length `pos`, meaning that the first `pos` tokens match completely. The system calls `split_at(pos)`:

- create a new node that takes over the first half (`[:pos]`) of the original key/value, which represents the reusable common prefix;
- replace the original node with the new node under the original parent;
- trim the original node to its second half (`[pos:]`), attach it beneath the new node as a child, and return the new node so that matching can continue from it.

The corresponding pseudocode is:

```text
function split_at(node, pos) -> RadixTreeNode:
    assert 0 < pos < node.length          # Bounds check

    old_parent ← node.parent
    new_node ← create RadixTreeNode()     # Store the reusable common prefix

    # The new node takes the first half and is attached to the original parent
    new_node.key   ← node.key[:pos]
    new_node.value ← node.value[:pos]
    new_node.parent ← old_parent          # Also updates old_parent.children

    # The original node becomes the second half and is attached below the new node
    node.key   ← node.key[pos:]
    node.value ← node.value[pos:]
    node.parent ← new_node                # Also updates new_node.children

    return new_node                       # Continue matching from this new node
```

Together with ordinary tree construction and the remaining basic methods, `get_match_len` and `split_at` form a complete `RadixTreeNode` implementation.

## 4 How the Radix Tree Supports Prefix Cache

Section 3 focused on constructing the Radix Tree and answered “How is the tree built?” Section 4 assumes that the tree structure has already been established and analyzes how it works with the KV Cache Pool, scheduler, and physical-memory manager to implement cache matching, reuse, protection, and eviction. This answers the question raised at the end of Section 3.2.

### 4.1 Connecting the Radix Tree to the KV Cache Pool

Three key types of objects connect the Radix Tree to the KV Cache:

| Object | Stored content | Location and role |
|---|---|---|
| Radix Tree key | Token-id sequence | Logical index used to determine whether two requests share a prefix |
| Radix Tree value | Token-slot / page indices | Address indices pointing to the corresponding storage positions in the KV Cache Pool |
| KV Cache Pool | K and V tensors for every layer | The actual data in GPU memory used by attention computation |

*Note: K/V storage is generally paged to simplify KV Cache management. SGLang uses this approach.*

The Radix Tree does not store the large K and V tensors themselves. It maintains only the mapping between token-id sequences and physical addresses. **The actual KV Cache data resides in GPU memory**, where it can directly participate in computation. In simple terms, **the key represents “sequence content,” while the value represents “storage address.”**

How does the Radix Tree use this mapping to reuse the KV Cache and avoid redundant computation across requests? When cache space is insufficient, how does it select entries for eviction and free space for new requests? We start with a simple example to understand these two processes.

---

### 4.2 Cache Lifecycle from Arrival to Request Completion

**Preliminary terminology and implementation conventions**

- **page**: the basic unit for KV Cache allocation and reclamation; one page generally contains `page_size` token slots;
- **token slot**: the physical storage position or index corresponding to one token in the KV Cache Pool;
- **page table**: records the mapping from logical token positions in a request to physical pages. The exact mapping granularity depends on the implementation;
- **free list**: stores physical pages that are currently unallocated and available for new KV states. Depending on the implementation, it may be called `free_pages`, `release_pages`, or something similar;
- **ref_count**: records the number of active requests or cache handles referencing a Radix Tree node. Lock and unlock operations generally update reference counts along the path from the current node toward the root;
- **handle**: a cache handle that records information about the matched Radix Tree node. It is subsequently used to retrieve physical indices and perform lock/unlock operations.

Assume that the Radix Tree structure has stabilized. This section focuses on matching, locking, borrowing, and returning cache resources along the tree, connecting the responsibilities involved throughout a request's lifecycle.

In the following example, assume that a historical request has left a shared prefix $S$ in the Radix Tree, and that each token id corresponds to one physical slot:

- $S = [900, 10, 11, 12]$
- KV for $S$: [p0, p1, p2, p3]
- Tree node corresponding to $S$: `ref_count = 0`, with an old `last_access_time`

The two requests waiting to be processed are:

| Request | Prompt | Match result | Remaining computation |
|---|---|---|---|
| R1 | [900, 10, 11, 12, 31, 32] | cache_hit_len = 4, value = [p0, p1, p2, p3] | [31, 32] |
| R2 | [900, 10, 11, 12, 41, 42, 43] | cache_hit_len = 4, value = [p0, p1, p2, p3] | [41, 42, 43] |

Assume that the KV Cache Pool has limited capacity available in its **free list**. We first admit one request into a running batch; the other follows the same process.

**Step 1: Match produces candidate information.** The match result for R1/R2 includes `cache_hit_len = 4`, a node handle, and the physical indices [p0, p1, p2, p3]. At this point, R1/R2 have not locked these pages. In theory, these paths may still be evicted, so “matched” does not mean “safely held.”

> A **handle** mainly stores two pieces of information:
>
> - the tree node it points to (the matched node);
> - the length of the shared prefix currently held by the request.
>
> A handle acts like a “key”: it records which node was matched and how much of the prefix was locked, providing information for subsequent operations such as writing the page table.

**Step 2: [Schedule decides which request runs first](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py).** The scheduler compares the hit length, suffix length, and resource requirements of R1 and R2. SGLang's scheduling policies fall into two categories:

- **Prefix-aware**: `lpm`, `dfs-weight`, `hrrn`, `shortest-prefill-first`, and others. Depending on the policy, requests with longer shared prefixes, higher DFS weights, or shorter uncached workloads are prioritized.
- **Prefix-cache agnostic**: `fcfs` (first come, first served), `lof` (longest output first), `random`, `routing-key`, and others.

Assume that `shortest-prefill-first` is used. R1 has a shorter uncached portion; both requests share the same prefix, but R1 needs only two new slots. R1 is therefore selected first, while R2 remains in the waiting queue. Scheduling may also perform an in-batch prefix check to prevent similar prefixes from being computed repeatedly within the same batch.

**Step 3: Lock converts a candidate into a protected reference.** Before R1 enters the computation batch, the matched node is passed to a lock operation, which increments `ref_count` along the path from that node toward the root. When a node's `ref_count` changes from 0 to 1, the corresponding token count moves from `evictable_size` to `protected_size`, and its state in the evictable-leaf collection is updated. The shared prefix [900, 10, 11, 12] is therefore protected. Its matched indices are written into the request's page table. *Locking only protects the cache; it does not allocate new pages.*

**Step 4: Allocate only the missing portion.** R1 has an unmatched suffix of length 2, so [p6, p7] is allocated from the **free list**. The page table becomes [p0, p1, p2, p3, p6, p7]. If insufficient free space is available, the allocator first asks the CacheManager to perform `evict`. Only leaf nodes with a current reference count of 0 may be evicted. Once enough physical space is returned, allocation continues.

*Note: this process does **not** allocate the already matched physical space [p0, p1, p2, p3] again.*

**Step 5: Insert turns newly computed results into shared cache.** After Prefill finishes, the tree is updated using R1's token-id sequence and page table. The existing $S$ node remains shared, and only the following child is added:

$$
[31, 32] \to [p6, p7]
$$

This operation writes a token-id-to-physical-address index and does not copy the KV states of $S$. **If `insert` discovers that part of the prefix has already been added to the tree by another request—a race condition—the indices redundantly occupied by the current request are immediately freed, preventing duplicate data from remaining in the pool and causing a memory leak.**

**Step 6: Unlock and request-completion handling.** After R1 finishes its computation, the matching path is passed to the corresponding unlock operation, which decrements `ref_count`:

- if a node's `ref_count` drops to 0, it becomes evictable and contributes to `evictable_size`;
- if `ref_count > 0`, another request is still using the node, so it cannot be evicted.

*Note: [p6, p7], which were successfully inserted into the tree, are not freed immediately. They have merely lost the protection associated with this request. Whether they are removed from the tree depends on whether later memory pressure triggers eviction.*

**Step 7: Eviction under memory pressure and freeing at request completion**

- **evict** (when memory is insufficient): if a later request such as R2 cannot allocate enough space, it calls `prefix_cache.evict`. The method selects only leaf nodes whose reference counts are 0, removes them from the tree according to the eviction policy, and returns their physical page indices. For example, if the [31,32] node is no longer referenced, it may be selected and [p6, p7] released.
- **free** (direct return without a policy decision):
  - free duplicate pages immediately when a race is detected during insertion;
  - when a request truly finishes, free only the non-shared token-id tail that cannot be inserted into the tree, making space available for other requests.

**A successfully shared prefix page is not freed when the request finishes. It is only unlocked and remains available until later memory pressure triggers eviction.**

> **Note**: SGLang's `evict` removes leaf nodes with `ref_count == 0` according to their priority and directly releases their physical space. In mini-sglang, `evict` removes leaf nodes with `ref_count == 0` according to the policy and returns a tensor of their physical indices; the scheduling layer performs the actual release.

The overall process can be summarized as follows:

<div align="center">
  <img src="./images/8-5-RadixAttention Lifecycle.png" alt="Radix Cache management lifecycle" width="800">
  <p><em>Figure 5. Radix Cache management lifecycle</em></p>
</div>

Prefix Cache “hit” is therefore only the beginning of the lifecycle:

- **Match** finds the addresses;
- **Lock** keeps the addresses valid while they are in use;
- **Insert** adds new results to the shared index and reclaims pages allocated redundantly;
- **evict** and **free** determine whether the physical space occupied by a page is released.

*Note: Prefill and Decode use the same page table. By default, the prompt and output of a finished request are inserted into the Radix Tree, with a split at the prompt boundary so that the generated portion can be evicted independently when needed. It is therefore inaccurate to say that “only the prompt is cached while Decode is always private and immediately returned.”*

---

### 4.3 Prefix Cache Management Implementation

Section 4.2 traced the full $Match \to Lock \to Allocate \to Insert \to Unlock \to evict/free$ lifecycle from the perspective of a request. These steps are implemented through the key state and helper methods inside the Radix Cache. This section examines the implementation details, using **mini-sglang** as the reference. We begin with the core management configuration:

```python
class RadixPrefixCache:
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size          # Paging granularity; aligns KV Cache units
        self.key_fn = _get_key_fn(self.page_size)            # Extract page-granularity keys for node indexing
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)          # Root of the Radix Tree
        self.root_node.ref_count = 1                         # Protect the root from deletion
```

These fields clarify the core methods required for lifecycle management:

| Method | Purpose |
|---|---|
| `_tree_walk` | Walk downward from the root through `children` and return the node and length corresponding to the longest matched prefix |
| `insert_prefix` | Align to page boundaries and insert the unmatched remainder into the tree, splitting nodes when necessary |
| `lock_handle` | Increment or decrement reference counts along the matched path and control whether nodes may be evicted |
| `_collect_leave_nodes_for_evict`, `evict` | Collect leaf nodes with **`ref_count == 0`** as eviction candidates, then evict them using LRU to make space for later requests |

After `match_prefix` completes, the system has a node in the tree. To obtain the complete physical-index sequence corresponding to the **matched prefix**, the implementation provides `RadixCacheHandle`. Starting at the current node, it follows `parent` pointers toward the root, collects the `value` of each node, reverses the collection, and concatenates it into a one-dimensional tensor. The detailed implementation is omitted here.

---

Let us examine the ideas behind some of the more difficult methods.

The core idea of **node-by-node matching** is to start at the root and follow matching children downward until either no matching child exists or the full input sequence has been matched:

- each matching operation compares one compressed edge. Because **physical storage is managed in pages**, the match length is rounded down to a multiple of `page_size`, and that aligned result determines whether the current node must be split;
- to avoid comparing an already traversed prefix again, each iteration compares only the unmatched suffix of `input_ids`;
- to support later LRU eviction, the access timestamp of the relevant node is updated so that its latest access time is recorded.

The method finally returns the node at which matching stopped and the cumulative match length. This provides the correct starting point for `insert_prefix` and `lock_handle`.

The corresponding pseudocode is:

```text
function _tree_walk(input_ids) -> Tuple[RadixTreeNode, int]:
    prefix_len ← 0
    node ← root node
    tic ← current timestamp
    while prefix_len < input_ids.length:
        child ← node.children[key(input_ids[prefix_len:])]
        if child does not exist:
            return node, prefix_len
        node ← child
        match_len ← computed match length
        match_len ← round down to a multiple of page_size
        prefix_len ← prefix_len + match_len
        if match_len < node.length:          # A divergence requires a split
            node.split_at(match_len)
            node.timestamp ← tic
            return node, prefix_len
    node.timestamp ← tic
    return node, prefix_len
```

---

The core idea of **inserting a new sequence into the prefix tree** is to reuse as much existing prefix as possible and add only the **unmatched suffix as a new node**:

- first round the input length down to a multiple of `page_size` so that only complete pages are handled and cache management does not become unnecessarily fragmented;
- call `_tree_walk` to find the longest common prefix in the Radix Tree, obtaining the node where matching ended and the match length `prefix_len`;
- if `prefix_len < insert_len`, an unmatched suffix remains:
  - create a new node containing the unmatched token ids and their physical indices;
  - attach the new node below the node where matching ended;
  - add the new node's length to `evictable_size`, making it eligible for future eviction.

Finally, return the matched-prefix length and a cache handle pointing to the final node.

The corresponding pseudocode is:

```text
function insert_prefix(input_ids, indices) -> InsertResult:
    # Align to complete pages
    insert_len ← round the input length down to a multiple of page_size
    keep the first insert_len tokens and their corresponding indices

    # Find the longest common prefix
    node, prefix_len ← tree_walk(input_ids)

    # If an unmatched suffix remains, add it as a new node
    if prefix_len < insert_len:
        new_node ← create a new RadixTreeNode
        new_node stores Key input_ids[prefix_len:] and Value indices[prefix_len:]
        attach new_node as a child of node
        evictable_size ← new_node.length + evictable_size
        node ← new_node

    return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))
```

---

The core idea of **selecting eviction candidates** is to first use `_collect_leave_nodes_for_evict` to collect all leaf nodes whose `ref_count == 0`, then build a heap ordered by node access timestamps so that the least recently used node is evicted first:

- `ref_count == 0` is a necessary condition for eviction and prevents the removal of a shared prefix that is still in use;
- eviction proceeds until the requested `size` has been reclaimed, and `size` must not exceed the current `evictable_size`;
- the root does not participate in eviction. It represents the whole Radix Tree and does not correspond to actual cache data.

During `evict`, removing a leaf may turn its parent into a leaf. If that parent also has `ref_count == 0`, it is added to the candidate heap so that eviction can continue upward.

The corresponding pseudocode is:

```text
function evict(size) -> torch.Tensor:

    candidates ← all leaf nodes with ref_count = 0
    candidates ← construct a min-heap ordered by last access time

    evicted_indices ← []
    evicted_size ← 0

    while evicted_size < size:

        node ← pop the least recently used leaf

        evicted_indices.append(node.value)
        evicted_size ← evicted_size + node.length

        remove node from its parent's children

        if parent becomes a leaf and parent.ref_count = 0:
            add parent to the candidate heap

    return a tensor containing the physical KV Cache indices of evicted nodes
```

---

These methods complete the core management layer of Prefix Cache. Consider a typical situation under mini-sglang's KV Cache management, particularly its LRU-based eviction policy. After one request finishes, a later request may arrive when physical space is already insufficient, forcing the system to evict nodes from the Radix Tree. This can create the following problems:

- node A may be evicted immediately before a request that could have reused it arrives. The avoidable cache miss then increases TTFT;
- a cached prefix does not inherently contain information about a conversation topic or session identity, making it difficult to identify and retain prefixes related to an active session. This can trigger otherwise avoidable recomputation and increase the cost of individual requests.

To address these problems, SGLang's latest Unified Radix Cache uses a unified token-keyed Radix Tree with pluggable components for FULL Attention, SWA, and Mamba. It also natively supports HiCache multi-tier storage and session-aware eviction. The session-aware policy provides soft protection based on session activity and the “session identity” of a prefix: it first evicts nodes that are not referenced by any active session and considers referenced nodes only when necessary. This helps reduce TTFT, improve the cache hit rate, and avoid unnecessary recomputation.

The detailed design and implementation are beyond the scope of this chapter. Interested readers can refer directly to this [article](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache).

---

## 5 Summary and Exercises

### 5.1 Summary

This chapter first used requests A, B, and C to introduce Radix Tree construction and the basic principles of longest-prefix matching and node splitting. Then, assuming that the tree had already been built, it used R1 and R2 to analyze the complete Prefix Cache lifecycle—from request matching, scheduling, locking, and physical-space allocation to insertion, unlocking, and cache eviction. Finally, it connected these operations to the mini-sglang implementation, explaining details such as page alignment, reference counting, and duplicate-page reclamation, while briefly comparing the corresponding extensions in current SGLang.

### 5.2 Exercises

**1. Why can eviction in a Radix Cache begin only with leaf nodes, rather than directly evicting an internal node or the root?**

> Hint: recall the central Prefix Cache optimization—prefix reuse. Directly evicting an internal node would break every descendant path that uses it as a prefix, preventing other requests from hitting already computed shared KV states.

**2. During CacheManager processing, under what circumstances can a page leak occur, and what problems would it cause?**

> Hint: pages allocated to a request during Prefill may correspond to a prefix that another request has already inserted into the Radix Tree before the current request performs its own insertion.

**3. During cache scheduling, when are `evict` and `free` used, and what is the essential difference between them?**

> Hint: in the relevant SGLang implementations, both ultimately release physical pages, but at different layers and at different points in the lifecycle—one evicts tree nodes according to a policy, while the other directly returns pages.

## References

- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/html/2312.07104v2)
- [mini-sglang: radix_cache.py](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/kvcache/radix_cache.py)
- [mini-sglang: CacheManager](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/scheduler/cache.py)
- [From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving](https://arxiv.org/pdf/2607.02574)
- [SGLang: radix_cache.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- [SGLang: schedule_policy.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)
- [mini-sglang: cache.py](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/scheduler/cache.py)
- [SGLang: radix_cache.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- [Unified Radix Cache](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache)
