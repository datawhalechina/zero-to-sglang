# Chapter 4 KV Cache: The Core Data Structure of Inference

&emsp;&emsp;For a fixed model, batch size, and data type, the GPU memory occupied by the KV Cache grows linearly with the current sequence length. In long-running agent tasks, context lengths often reach 32K; some tasks may even expand to 128K or 1M. When multiple requests are processed concurrently, the KV Cache can quickly consume a large amount of GPU memory and become a bottleneck for inference throughput and latency.

## 1 Learning Objectives

&emsp;&emsp;The preceding chapters introduced the basic structure of LLMs, the inference process, and the fundamentals of GPU architecture. Building on that foundation, this chapter focuses on the KV Cache and addresses the following questions:

- Starting from the Attention formula, why is a KV Cache needed during autoregressive Decode? Which computations are reused by caching K and V, and which computations must still be performed?
- What is the Cache lifecycle of a single request from Prefill to Decode? When is the Cache created, read, and appended to?
- How can the GPU memory occupied by the KV Cache at any point be estimated from the number of model layers, the number of KV heads, the hidden dimension of each head, the current sequence length, the batch size, and the data type?
- Using a real model configuration, how can we calculate the GPU memory occupied at the end of Prefill, the additional memory required by one Decode step, and the memory required at the maximum context length?
- Why does naive Cache management suffer from GPU memory fragmentation and wasted space in multi-request scenarios?

## 2 The KV Cache in the Attention Mechanism

&emsp;&emsp;The KV Cache stores the Keys and Values used by Attention. Caching them avoids repeatedly computing the K and V of historical tokens during Decode, significantly improving efficiency.

### 2.1 Where the KV Cache Comes From

&emsp;&emsp;Before Attention is computed, the hidden state of each token is linearly projected to obtain its Query (Q), Key (K), and Value (V):

- **Q**: The information that the current token seeks from the context. It is used to compute the degree of match with each Key.
- **K**: The feature representation of each token. It is used to compute attention scores and determine which historical tokens should receive attention.
- **V**: The information actually carried by each token. The Values are weighted by the attention scores and aggregated to form the contextual representation of the current token.


**Analyzing the origin of the KV Cache:**

<div align="center">
  <img src="./images/4-1-Creating the KV Cache.png" alt="Figure 1. Creating the KV Cache during Prefill" width="800">
  <p><em>Figure 1. Creating the KV Cache</em></p>
</div>

&emsp;&emsp;During **1. Prefill**, the entire input sequence is processed in a single forward pass. The text is converted into token IDs by the tokenizer, then transformed into embeddings and passed through the transformer layers. Each layer computes Q, K, and V for every token, performs causal Attention, and saves the K and V tensors to form the initial KV Cache. **Q does not need to be cached because it is used only for the current Attention computation and will not be reused during subsequent Decode steps.**


<div align="center">
  <img src="./images/4-2-Appending to the KV Cache.png" alt="Figure 2. Appending to the KV Cache during Decode" width="800">
  <p><em>Figure 2. Appending to the KV Cache</em></p>
</div>

&emsp;&emsp;During **2. Decode**, the initial KV Cache is extended as new tokens are generated. At each step, the token sampled in the previous step is used as input, and its Q, K, and V are computed in every layer. The new K and V are appended to the historical KV Cache of the corresponding transformer layer. The new Q then performs causal self-attention against all Keys, both historical and current, and computes a weighted sum over all Values to produce the Attention output for the current token. The updated KV Cache is used in subsequent steps and grows progressively as generation continues.

In summary, the KV Cache is produced as follows: **Prefill computes and caches the K and V of the entire input sequence at once; Decode computes only the Q, K, and V of each new token and appends the new K and V to the Cache.**

### 2.2 How the KV Cache Works

&emsp;&emsp;Decode typically processes only one new token at a time. Suppose the token at position $t$ is currently being processed. Its hidden state $h_t$ is first obtained through embedding, and then projected to produce $q_t$, $k_t$, and $v_t$:

$$
q_t=h_tW_q, \quad k_t=h_tW_k, \quad v_t=h_tW_v
$$

Here, $K_{1:t}$ and $V_{1:t}$ contain all Keys and Values from the first token through the token at the current position $t$. The KV Cache is stored separately for each transformer layer because different layer weights produce different K and V tensors:

$$
\text{KV Cache} = [ (K^{(1)}, V^{(1)}), (K^{(2)}, V^{(2)}), \dots, (K^{(L)}, V^{(L)}) ]
$$

where $L$ is the number of transformer layers.

&emsp;&emsp;Once position $t$ has been processed, so that $K_{1:t}$ and $V_{1:t}$ are already in the Cache, the new token at position $t+1$ is embedded and linearly projected to compute $q_{t+1}$, $k_{t+1}$, and $v_{t+1}$. The new K and V are then appended:

$$
K_{1:t+1} = \text{concat}(K_{1:t}, k_{t+1}), \quad V_{1:t+1} = \text{concat}(V_{1:t}, v_{t+1})
$$


&emsp;&emsp;The Attention computation for the current token is then completed:

$$
o_{t+1} = \text{softmax}\left( \frac{q_{t+1} K_{1:t+1}^\top}{\sqrt{d_h}}  \right) V_{1:t+1}
$$

*Without a KV Cache, generating every new token would require all historical tokens to be linearly projected again to recompute their K and V, creating a large amount of redundant computation and reducing Attention inference efficiency.*


&emsp;&emsp;After each transformer layer completes the Attention computation, $o_{t+1}$ passes through the FFN, residual connections, and other operations. The result becomes the input to the next layer. Once all layers have been processed, the final hidden state passes through the LM Head to produce logits. A token ID is then selected according to the sampling strategy and embedded again as the input to the next Decode step.

*Note: Sampling produces a token ID because the vocabulary mapping is already represented by the vocabulary indices of the logits. The tokenizer converts token IDs back into natural language only when the final text is produced.*

&emsp;&emsp;For a complete inference request, how is the KV Cache built, read, and appended to across the Prefill and Decode stages?

## 3 The KV Cache Lifecycle

&emsp;&emsp;To illustrate how the KV Cache works, this section manually simulates how it is built, read, and appended to during one complete request using an extremely small LLM. The model has a vocabulary size of 10, represents each token with a four-dimensional vector, contains only one transformer layer, and uses single-head attention.

**Input prompt:** The user enters "I love AI." After tokenization and embedding, this produces three vectors:

- Token 1 ("I"): $[1, 0, 1, 0]$
- Token 2 ("love"): $[0, 1, 1, 0]$
- Token 3 ("AI"): $[1, 1, 0, 1]$

**Simplifying assumptions:**

- The linear projections are $W_Q = W_K = W_V = I$, where $I$ is the identity matrix.
- Positional encoding, output projection, residual connections, RMSNorm, and other components are temporarily omitted.

&emsp;&emsp;These components are still present in real models, but the simplifications above **do not change the essential behavior of the KV Cache**. The following walkthrough presents one complete lifecycle from Prefill to Decode.

### 3.1 Prefill

&emsp;&emsp;Prefill processes all tokens in the prompt at once. The input matrix $X \in \mathbb{R}^{3 \times 4}$ is:

$$
X=
\begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 1 & 0 \\
1 & 1 & 0 & 1
\end{bmatrix}
$$

&emsp;&emsp;Applying the linear projections $Q_{1:3}=X W_Q$, $K_{1:3}=X W_K$, and $V_{1:3}=X W_V$ gives:

$$
Q_{1:3}=K_{1:3}=V_{1:3}=X
$$

&emsp;&emsp;The model writes $K_{1:3}$ and $V_{1:3}$ to the KV Cache for the current request, then performs self-attention with a causal mask:

$$
S=\frac{Q K^\top}{\sqrt{d_h}}+M
$$

Here, $d_h=4$, and $M$ is the causal mask. Future positions that must not be exposed are assigned $-\infty$, ensuring that a token cannot attend to future tokens. Substituting the values gives:

$$
S=
\begin{bmatrix}
1 & -\infty & -\infty \\
0.5 & 1 & -\infty \\
0.5 & 0.5 & 1.5
\end{bmatrix}
$$

&emsp;&emsp;Applying Softmax to each row produces the Attention weights:

$$
A=\text{softmax}(S)=
\begin{bmatrix}
1 & 0 & 0 \\
0.378 & 0.622 & 0 \\
0.212 & 0.212 & 0.576
\end{bmatrix}
$$

&emsp;&emsp;The Attention output $O$ is:

$$
O=A V=
\begin{bmatrix}
1 & 0 & 1 & 0 \\
0.378 & 0.622 & 1 & 0 \\
0.788 & 0.788 & 0.424 & 0.576
\end{bmatrix}
$$

&emsp;&emsp;In a real model, $O$ also passes through output projection, residual connections, normalization, the FFN, and other operations to produce the final hidden states $h_1,h_2,h_3$ at each position. **Only the hidden state at the final position**, $h_3$, **is used to predict the next token**:

$$
z = W_{\mathrm{LM}} h_3 + b,\quad z \in \mathbb{R}^{10}
$$

$z$ contains the logits over the vocabulary. Applying Softmax produces a probability distribution. Suppose some of the results are:

| Candidate token | Probability |
|---|---|
| are | 0.05 |
| great | 0.08 |
| models | 0.12 |
| very | 0.63 |

&emsp;&emsp;With greedy decoding, "very," the token with the highest probability, becomes the first generated token, which is the fourth token in the sequence. With random sampling, a token is sampled from the distribution and need not be the highest-probability token.

**At this point, Prefill has accomplished two things:**

- It has built the initial KV Cache for all tokens in the prompt.
- It has used the hidden state at the **final position** of the prompt to predict the first new token.

### 3.2 Decode

&emsp;&emsp;After Decode begins, the model uses the discrete token "very" that was just produced as the input to the next step. The input to the model is **not** the $h_3$ from Prefill. Instead, it is the vector obtained by embedding the token ID of "very." For ease of calculation, assume:

$$
x_4=\text{embedding}(\text{"very"})=
\begin{bmatrix} 0.82 & 0.726 & 0.548 & 0 \end{bmatrix}
$$

&emsp;&emsp;Under the weights chosen for this example, projecting $x_4$ gives:

$$
q_4=k_4=v_4=x_4=
\begin{bmatrix} 0.82 & 0.726 & 0.548 & 0 \end{bmatrix}
$$

&emsp;&emsp;The newly computed $k_4$ and $v_4$ are appended to the historical KV Cache:

$$
K_{1:4}=
\begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 1 & 0 \\
1 & 1 & 0 & 1 \\
0.82 & 0.726 & 0.548 & 0
\end{bmatrix}
,\quad
V_{1:4}=K_{1:4}
$$

&emsp;&emsp;The current $q_4$ computes Attention against all Keys cached through position 4, then computes a weighted sum of all Values:

$$
a_4=\text{softmax}\left(\frac{q_4 K_{1:4}^\top}{\sqrt{d_h}}\right),\quad
o_4=a_4 V_{1:4}
$$

*Note: The Query in a single Decode step has only one row, so no additional mask is required.*

&emsp;&emsp;After subsequent processing, $o_4$ produces $h_4$, which passes through the LM Head and sampling to produce the fifth token, assumed to be "much" in this example. Each subsequent token is generated by repeating the following sequence:

$$
\text{new token ID}
\rightarrow
\text{embedding}
\rightarrow
q_t,k_t,v_t
\rightarrow
\text{append }k_t,v_t
\rightarrow
\text{use }q_t\text{ to read historical KV entries}
\rightarrow
\text{predict the next token}
$$

&emsp;&emsp;Conceptually, the KV Cache in ordinary autoregressive Decode is appended to incrementally as generation proceeds. In real systems, mechanisms such as prefix caching, sliding windows, speculative decoding, and Cache eviction may involve paging, sharing, rollback, or reclamation. Therefore, **appending does not always correspond to the growth of one contiguous region of GPU memory**.


### 3.3 Request Completion

&emsp;&emsp;The generation ends when the LLM generates an EOS token, reaches the maximum output length, or the request is canceled. The inference engine decrements the reference counts of the blocks/pages held by that request. Any block whose reference count falls to zero is returned to the global pool of free GPU memory blocks.

&emsp;&emsp;If optimizations such as Prefix Caching or RadixAttention are enabled, the Cache data structure continues to reference blocks corresponding to reusable prefixes, including newly written KV entries from the current request that later requests may reuse. Alternatively, these blocks may remain in the GPU KV pool until an eviction policy returns them to the free pool.

&emsp;&emsp;The lifecycle of the KV Cache for a request can therefore be summarized as:

$$
\text{allocation}
 \rightarrow
 \text{bulk write during Prefill}\text{ (some computation may be skipped on a prefix hit)}
 \rightarrow
 \text{repeated reads and appends during Decode}
 \rightarrow
 \text{return to the free pool or retain for reuse}
$$

*Note: A block/page is the unit used for paged KV Cache management.*

## 4 GPU Memory Usage of a Naive KV Cache

&emsp;&emsp;The preceding sections explained the importance of the KV Cache and how it changes throughout request processing. Its GPU memory usage is the main reason inference engines devote substantial effort to optimizing it. The symbols used in the following analysis are defined below:

- $B$: Number of requests, or batch size.
- $T$: Number of valid tokens in each request.
- $L$: Number of transformer layers.
- $n_{\mathrm{kv}}$: Number of KV heads.
- $d_h$: Dimension of each head.
- $s$: Number of bytes occupied by each data element.
- $l$: Length of the sequence generated during Decode.



**1. GPU memory occupied by the KV Cache built during Prefill**

Each layer must store one K and one V for every token, so the total GPU memory usage is:

$$
\text{KV}_{bytes} = B \times T \times L \times 2 \times n_{\mathrm{kv}} \times d_h \times s
$$

If the requests have different lengths $T_1, T_2, \dots, T_B$, the usage is:

$$
\text{KV}_{bytes} = L \times 2 \times n_{\mathrm{kv}} \times d_h \times s \times \sum_{b=1}^{B} T_b
$$


**2. GPU memory occupied by the KV Cache built during Decode**

Each layer must store one K and one V for every newly generated token, so the additional GPU memory usage is:

$$
KV_{bytes} = B \times l \times L \times 2 \times n_{\mathrm{kv}} \times d_h \times s
$$

If the requests generate different numbers of tokens, a more accurate formula is:

$$
KV_{bytes} = L \times 2 \times n_{\mathrm{kv}} \times d_h \times s \times \sum_{b=1}^{B} l_b
$$

The **total** GPU memory occupied by the KV Cache after Decode is:

$$
KV_{bytes} = B \times (T + l) \times L \times 2 \times n_{\mathrm{kv}} \times d_h \times s
$$

If the generated request sequences have different lengths, replace $T+l$ with $\sum_{b=1}^{B} l_b + T$.

*Note: The factor of 2 in these formulas already accounts for both the K and V tensors. In actual engines, Prompt Caching, Prefix Caching, and other factors also affect GPU memory usage in both stages. For simplicity, these optimizations are not considered here, and all values above represent theoretical usage.*

---

&emsp;&emsp;Based on the preceding analysis, consider the common Qwen3-4B-Thinking-2507 configuration as an example. This model has $L=36$ transformer layers, $n_{\mathrm{kv}}=8$ KV heads, a per-head dimension of $d_h=128$, and uses the bf16 data type. To focus on the KV Cache, the following calculations include only its theoretical **dense** GPU memory usage.

**1. Prefill**

&emsp;&emsp;When processing one request with a sequence length, or context length, of $T=100$, the KV Cache occupies:

$$
\mathrm{Size_{prefill}} = 2 \times 36 \times 100 \times 8 \times 128 \times 2 = 14{,}745{,}600\ \mathrm{B} \approx 14.06\ \mathrm{MB}
$$


**2. Decode**

&emsp;&emsp;Starting from the Prefill result, Decode generates tokens one at a time. For each new token, where $l=1$, the **additional** memory usage is:

$$
\mathrm{Size_{new}} = 2 \times 36 \times 1 \times 8 \times 128 \times 2 \approx 0.141\ \mathrm{MB}
$$

The **total** usage after one Decode step, including the existing Prefill Cache and the new entry, is:

$$
\mathrm{Size_{one}} = 2 \times 36 \times (100+1) \times 8 \times 128 \times 2 \approx 14.20\ \mathrm{MB}
$$

*Note: Decode appends one token at a time, so total usage grows linearly with the generated length $l$.*

Suppose Decode ultimately generates $l=200$ tokens. The total KV Cache usage is then:

$$
\mathrm{Size_{total}} = 2 \times 36 \times (100+200) \times 8 \times 128 \times 2 \approx 42.19 \ \mathrm{MB}
$$


&emsp;&emsp;Applying the calculation above to scenarios with different Prefill lengths gives a more intuitive comparison. All of the following calculations use $B=1$ and theoretical dense GPU memory usage:

<div align="center">
  <img src="./images/4-3-KV Cache proportions for a single request in different scenarios.png" alt="Figure 2. KV Cache proportions for a single request with different prompts" width="800">
  <p><em>Figure 2. KV Cache proportions for a single request in different scenarios</em></p>
</div>

&emsp;&emsp;As a single Qwen3-4B-Thinking-2507 request grows from 4K to 32K tokens, its idealized KV Cache grows from approximately $0.56 \ \mathrm{GB}$ to approximately $4.50 \ \mathrm{GB}$. If eight full 32K-context requests are processed simultaneously, the KV Cache alone occupies approximately $36 \ \mathrm{GB}$. This does not include model weights or other overhead. The model weights occupy approximately:

$$
\text{weight} = 2 \times 4 \times 10^{9} \approx 7.45\ \mathrm{GB}
$$

&emsp;&emsp;For a single short-context request, model weights remain the main consumer of GPU memory. When **the context is very long or many requests run concurrently**, the KV Cache quickly surpasses the weights and becomes the dominant consumer. Temporary allocations such as intermediate activations during Decode are generally much smaller. The data therefore shows that the KV Cache becomes a long-context bottleneck mainly for the following reasons:

- **Activations during inference are usually temporary.** In optimized implementations, the system retains only the intermediate results needed by the current layer and current computation, then reuses or releases their memory when the computation finishes. In contrast, the KV Cache must persist across Decode steps and grows approximately linearly with both the number of layers and the number of tokens.

- In real implementations, activation memory is also affected by FlashAttention, fused operators, temporary workspaces, and batch shapes, but its scale is generally much smaller than that of the long-lived KV Cache.

**In summary:** Activations are primarily a relatively small temporary workspace that flows through the current computation, whereas the KV Cache is historical state that each request must retain in GPU memory for an extended period.


&emsp;&emsp;In real-world scenarios, however, LLM inference services usually run continuously and process multiple requests concurrently. GPU memory remains occupied by model weights, the KV Cache, workspaces, temporary activations, and other data, making out-of-memory errors more likely. The next section focuses on GPU memory and the problems of traditional Cache management in multi-request scenarios.

## 5 GPU Memory and Traditional Cache Management Problems with Multiple Requests

&emsp;&emsp;For a single request, storing its KV Cache in one contiguous tensor makes allocation, appending, and release relatively simple. In a multi-request scenario, however, if a separate contiguous GPU memory region is still allocated for every request and each region grows with the sequence length, traditional contiguous allocation creates the following problems:

- **GPU memory fragmentation and wasted reservations.** Requests have different lengths and lifetimes, so dynamic allocation and release can leave free regions whose sizes do not match subsequent requests. Reserving space according to the maximum context length consumes memory in advance for tokens that have not yet been generated. Expanding a contiguous region may also trigger reallocation and data copying.
- **Inability to share repeated prefixes.** Multiple requests often use the same system prompt or conversation prefix. Independent Caches store duplicate copies of the same K and V tensors. The longer the common prefix, the more memory is wasted by duplication.
- **Complex scheduling and reclamation.** Request arrival, completion, preemption, and resumption all require synchronized management of GPU memory capacity, Cache locations, priorities, and latency targets. When Caches are scattered or must be expanded, reclamation, relocation, and metadata management incur additional overhead.
- **Poor support for efficient Continuous Batching.** Continuous Batching requires new requests to enter and completed requests to leave dynamically during each Decode iteration. Contiguous storage can support this mechanism, but it requires additional tracking of the addresses and lengths of variable-length requests. If the batching kernel uses a regular rectangular tensor, requests usually need to be padded to the maximum length, causing unnecessary computation and wasted space. Reorganizing requests to maintain a compact layout instead introduces gather operations, reordering, or data movement overhead.

These problems show that KV Cache management cannot treat each request only as an independent contiguous tensor. Instead, it requires finer-grained block/page allocation, reclamation, and scheduling, while reusing identical prefixes whenever possible.

## 6 Summary and Exercises

### 6.1 Summary

&emsp;&emsp;This chapter analyzed the KV Cache lifecycle and GPU memory usage starting from Attention, then examined the memory and management pressure it creates in long-context and multi-request scenarios. Efficient KV Cache organization and management are therefore major optimization priorities for inference engines.


### 6.2 Exercises

**1. During Full Attention inference, which computations produce the KV Cache, and what purpose does it serve?**

> Hint: Use the Attention formula and explain the Prefill and Decode stages separately.


**2. Describe the complete lifecycle of the KV Cache from the time a request is received until generation finishes.**

> Hint: Organize your answer around allocation, writes during Prefill, reads and appends during Decode, and request completion.


**3. Calculate GPU memory usage. Suppose a single request has an input context length of $T=200$ and ultimately produces an output sequence of length $l=100$. Use the Qwen2.5-1.5B configuration to calculate:**

- The KV Cache usage at the end of Prefill.
- The additional usage for each token generated during Decode.
- The total usage when generation finishes.

Briefly compare the values and explain what they show.

> Hint: Distinguish between the total Prefill usage, the Decode increment, and the final total usage.


**4. In a multi-request concurrent setting, what major problems arise if naive management still allocates contiguous GPU memory separately for each request?**

> Hint: Consider GPU memory fragmentation, wasted reservations, and the overhead of dynamically adding and removing requests.


**5. The KV Cache grows linearly with generated sequence length until the request ends, placing substantial pressure on GPU memory. Could some or all of the KV Cache be placed in CPU memory? What are the potential benefits and costs?**

> Hint: Consider the benefits of reducing GPU memory pressure and expanding capacity, as well as the impact of added latency on Decode throughput.


## References

- [Hicache](https://www.lmsys.org/blog/2025-09-10-sglang-hicache)
- [Unified Radix Cache](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache)
- [Qwen3-4B-Thinking-2507 Config](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507/blob/main/config.json)
