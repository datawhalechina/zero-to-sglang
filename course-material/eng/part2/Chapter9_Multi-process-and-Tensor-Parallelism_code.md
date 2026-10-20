# Chapter 9 Multi-process and Tensor Parallelism 

In the preceding conceptual chapter, we distinguished the responsibilities of **multi-process execution** and **tensor parallelism**: multi-process execution organizes cooperation among multiple processes, while tensor parallelism partitions the model weights and computation of the same layer across multiple ranks. This chapter examines the mini-SGLang source code and connects these two mechanisms through the complete execution path.

We recommend reading the [conceptual chapter](<./Chapter9_Multi-process-and-Tensor-Parallelism.md>) first to understand the principles, then using this chapter to study the implementation.

## 1 Learning Objectives

After completing this chapter, you should be able to:

- Distinguish the roles of ZMQ request broadcasting, Gloo control communication, and NCCL tensor communication, and explain how they work together.
- Explain the weight dimensions, input and output shapes, and communication points of column-parallel and row-parallel layers.
- Describe the sharding rules for QKV, the vocabulary, and the KV Cache, including the implementation constraints on KV head replication and vocabulary divisibility.
- Trace a request from reception to next-token generation, explaining how all ranks maintain the same request order and computation state.

## 2 Locating the Source Code

The main source files covered in this chapter are under **python/minisgl/**:

| File | Purpose |
| ------------------------------------------ | ---------------------------------------------------- |
| server/launch.py | Parses TP and other arguments, starts the API Server, creates Scheduler, Tokenizer, and Detokenizer subprocesses, and waits for all processes to become ready |
| distributed/info.py | Stores the current process's rank and world size so that layers can query TP information through `get_tp_info()` |
| distributed/impl.py | Provides a common interface for All-Reduce and All-Gather using Torch Distributed and PyNCCL |
| engine/engine.py | Binds the GPU (`cuda:{rank}`), initializes communication, loads and shards model weights, allocates the KV Cache, and runs forward passes and sampling |
| scheduler/io.py | Handles Tokenizer/Detokenizer communication on rank 0 only, broadcasts requests to other ranks, and synchronizes the message count for the current iteration |
| scheduler/scheduler.py | Runs the main scheduling loop: updates request states, selects a prefill/decode batch, prepares inputs, and calls the Engine for a forward pass |
| scheduler/decode.py | Manages running decode requests; sorts them by uid in `schedule_next_batch` to keep batch order consistent across ranks |
| kernel/pynccl.py | Creates and broadcasts the NCCL Unique ID on rank 0 and initializes the PyNCCL communicator |
| kernel/csrc/src/pynccl.cu | Calls NCCL through TVM FFI to enqueue all-reduce and all-gather operations on a CUDA stream |
| models/weight.py | Shards checkpoint weights by parameter name using **column/row/vocabulary** partitioning, then merges local projection shards such as QKV and gate/up |

## 3 From Startup to Output

Next, we will follow a request through the entire workflow, from system startup to the final output, to understand how the components cooperate.

---

### 3.1 Creating Processes

When launching a model for inference from the command line, choose the tensor-parallel size (`tp_size`) based on the model's total parameter count and the available GPU memory. For example, to run Qwen3-32B on three RTX 5090 GPUs:

```bash
python -m minisgl --model "Qwen/Qwen3-32B" --tp-size 2
```

> **Note:** If neither `--tensor-parallel-size` nor `--tp-size` is specified, the default is single-GPU execution.

In `parse_args` in server/args.py, the following is constructed first:

```python
kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
```

After the parent process parses the arguments, `server_args.tp_info` is initially `DistributedInfo(rank=0, size=tp_size)`.

Next, after parsing the arguments, `launch_server` passes `start_subprocess` to `run_api_server` as a callback:

```python
# server/launch.py (excerpt)
def start_subprocess() -> None:
    mp.set_start_method("spawn", force=True)
    world_size = server_args.tp_info.size
    ack_queue: mp.Queue[str] = mp.Queue()

    # Start a scheduler for each rank
    for i in range(world_size):
        new_args = replace(
            server_args,
            tp_info=DistributedInfo(i, world_size),
        )
        mp.Process(
            target=_run_scheduler,
            args=(new_args, ack_queue),
            daemon=False,
            name=f"minisgl-TP{i}-scheduler",
        ).start()

    # Then start one detokenizer and num_tokenizers tokenizers
    ...
    # Wait for acknowledgments (only the rank 0 scheduler sends an ack)
    # Expected total: 1 (primary scheduler) + 1 (detokenizer) + num_tokenizers
    for _ in range(num_tokenizers + 2):
        logger.info(ack_queue.get())

run_api_server(server_args, start_subprocess, run_shell=run_shell)
```

For each rank **i**, the parent process uses `dataclasses.replace` to replace `tp_info` with `DistributedInfo(i, world_size)`, then starts the corresponding scheduler subprocess using **spawn**.

`start_subprocess` returns only after receiving **[num_tokenizers + 2](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/server/launch.py#L105)** acknowledgments through `ack_queue`.

```python
# server/api_server.py (excerpt)
def run_api_server(config, start_backend, run_shell) -> None:
    global _GLOBAL_STATE
    ...
    host = config.server_host
    port = config.server_port

    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(...),
        send_tokenizer=ZmqAsyncPushQueue(...),
    )

    # Start the backend, including the subprocess for each rank
    # This invokes start_subprocess defined above
    start_backend()
    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
```

`run_api_server` first establishes the frontend ZMQ channels, then calls `start_backend` (that is, `start_subprocess`) to create the subprocesses. It enters **uvicorn.run** only after that callback returns. Model and communication initialization take place between process creation and the point at which the server can accept HTTP requests.

---

### 3.2 Initializing the Engine and Communication

`Scheduler.__init__` first creates `Engine(config)`, then initializes the Table, Cache, Decode, and Prefill managers in sequence, and initializes I/O through `SchedulerIOMixin`:

```python
# scheduler/scheduler.py (excerpt)
self.engine = Engine(config)
...
self.table_manager = TableManager(...)
self.cache_manager = CacheManager(...)
self.decode_manager = DecodeManager(...)
self.prefill_manager = PrefillManager(...)
...
super().__init__(config, self.engine.tp_cpu_group)  # IO mixin
```

The Engine initializes the key components as follows:

```python
# engine/engine.py (excerpt)
set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)  # Must be set first
self.device = torch.device(f"cuda:{config.tp_info.rank}")
torch.cuda.set_device(self.device)
self.stream = torch.cuda.Stream()
torch.cuda.set_stream(self.stream)

self.tp_cpu_group = self._init_communication(config)

# Initialize the model: build its structure on the meta device, then load weights
with torch.device("meta"), torch_dtype(config.dtype):
    self.model = create_model(config.model_config)
self.model.load_state_dict(self._load_weight_state_dict(config))
```

The order matters:

- **Set TP information before creating model layers.** Each layer uses `get_tp_info()` to determine its local weight shapes.
- **The device index refers to the visible devices.** The implementation directly uses `cuda:{config.tp_info.rank}`.
- **Build the model structure on the meta device first, then load the weights.**

Communication initialization has two paths; PyNCCL is used by default:

```python
# engine/engine.py (excerpt)
def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
    # Both size == 1 and the default use_pynccl setting take the Gloo path;
    # the early return that skips PyNCCL creation is inside enable_pynccl_distributed
    if config.tp_info.size == 1 or config.use_pynccl:
        torch.distributed.init_process_group(
            backend="gloo",
            rank=config.tp_info.rank,
            world_size=config.tp_info.size,
            ...
        )
        tp_cpu_group = torch.distributed.group.WORLD
        max_bytes = (...)
        enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            rank=config.tp_info.rank,
            world_size=config.tp_info.size,
            ...
        )
        tp_cpu_group = torch.distributed.new_group(backend="gloo")
    return tp_cpu_group
```

| Configuration | Default PyTorch Process Group | GPU Collective Communication | CPU Control Communication |
|------|-------------------|--------------|--------------|
| tp_size=1 | Gloo | None | Gloo WORLD |
| tp_size>1 (default) | Gloo | Separate PyNCCL communicator | Gloo WORLD |
| tp_size>1 --disable-pynccl | NCCL | torch.distributed NCCL | Separate Gloo group |

Single-GPU execution still initializes a Gloo process group for communication such as `_sync_get_memory`, but does not create a PyNCCL communicator. A PyNCCL communicator is created only on the default multi-GPU path.

```python
# distributed/impl.py (excerpt)
class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)
```

`enable_pynccl_distributed` appends `PyNCCLDistributedImpl` to `plugins`, so subsequent **all_reduce/all_gather** calls use the last plugin. Model layers depend only on the common `DistributedCommunicator` interface and do not need to know whether the underlying implementation uses PyNCCL or torch.distributed.

---

### 3.3 Loading Weights and Model State

During construction, parallel layers have already used `get_tp_info` to determine the parameter sizes required by the current rank. models/weight.py determines which slices to read from the checkpoint. Each rank **independently** executes `load_weight`, opening safetensors files in a streaming fashion and applying the [sharding rules](<./Chapter 9_Multi-process and Tensor Parallelism.md>) as it reads: **shard first, then merge as needed**.

```python
# models/weight.py (excerpt)
_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

_MERGE_GROUPS = {...}
_SLOT_NAMES = {...}

# Extract the current rank's shard according to the TP rules
def _shard_tensor(key, value, r, n, num_kv_heads):
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            # GQA: partition by head rather than simply splitting evenly
            ...
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        # Shard the vocabulary dimension by rank (div_ceil)
        ...
    else:
        # Keep weights such as RMSNorm intact
        return value

# Merge/stack as needed during streaming loading
def load_weight(...):
    tp_info = get_tp_info()
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}

    for file in tqdm(...):
        with safetensors.safe_open(...) as f:
            for name in f.keys():
                # Skip multimodal weights
                ...
                raw = f.get_tensor(name)
                name = name.removeprefix("language_model.")
                # First extract the shard for the current rank
                tensor = _shard_tensor(...)
                del raw

                # Merge as needed (q/k/v -> qkv_proj, gate/up -> gate_up_proj)
                if (info := _get_merge_info(name)) is None:
                    out = (name, tensor)
                else:
                    # Collect slots according to the merge rules; continue until
                    # all slots are available, then concatenate them
                    ...
                    out = (merged_key, torch.cat(parts, dim=0))

                # For MoE, stack weights with the same name from each expert
                # (already sharded and merged) by index along the expert dimension
                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    # Collect experts; continue until all are available,
                    # then stack and yield (packing only; no further sharding)
                    ...
                    yield packed_key, torch.stack(experts, dim=0)
                else:
                    # Yield dense weights or regular merged weights directly
                    yield out[0], out[1]
    # Verify that the checkpoint has been fully loaded
    assert not merge_buf, ...
    assert not expert_buf, ...
```

**The order is shard first, then merge**: the current rank's q/k/v shards are concatenated into **qkv_proj**, and its gate/up shards into **gate_up_proj**. Concatenating the full tensors first and then splitting them evenly would break the pairing required by local Attention / SwiGLU computation. MoE models also need to **pack** the processed weights with the same name from each expert into a single tensor, ordered by expert index, and yield it along with the other weights. Dense models yield their weights directly without this packing step.

> **Note:** Streaming loading aims to keep peak CPU memory usage within the space needed for one full tensor plus a small merge buffer, reducing memory spikes during loading.

---

The corresponding runtime implementation of the vocabulary layers is:

```python
# layers/embedding.py (excerpt)
class VocabParallelEmbedding(BaseOP):
    def __init__(self, num_embeddings, embedding_dim):
        ...

    @nvtx_annotate("Embedding")
    def forward(self, x):
        y = indexing(weights=self.weight, indices=x,
                     vocab_range=self.vocab_range if self.tp_size > 1 else None)
        return self._comm.all_reduce(y) if self.tp_size > 1 else y


class ParallelLMHead(VocabParallelEmbedding):
    def load_state_dict(...):
        if not self.tied_embedding:
            return super().load_state_dict(...)
        # When embeddings are tied, pop lm_head.weight / bias (shared weights)
        ...

    def state_dict(...):
        ...

    @nvtx_annotate("LMHead")
    def forward(self, x):
        # During prefill, select only the last token of each sequence
        if batch.is_prefill:
            x = x[batch.attn_metadata.get_last_indices(bs)].contiguous()
        module = self.tied_embedding or self
        logits = F.linear(x, module.weight, self.bias)
        if self.tp_size == 1:
            return logits
        output = self._comm.all_gather(logits)
        ...  # Reshape and truncate to the actual num_embeddings
        return output[:, :self.num_embeddings]
```

- `VocabParallelEmbedding`: Indexes the local weights using `vocab_range`, then applies **all_reduce** to the output when **tp_size > 1**.
- `ParallelLMHead`: Computes local logits, then uses **all_gather** to reconstruct the full vocabulary logits and truncates them when **tp_size > 1**.

The Engine uses `self.model.load_state_dict(...)` to populate the structure created on the meta device with the sharded and merged tensors. It then completes initialization of the KV Cache, page table, Attention/MoE backends, Sampler, and CUDA Graph.

Only after all initialization is complete is the system ready to serve requests.

---

### 3.4 Receiving and Broadcasting Requests

Only rank 0 interacts with the Tokenizer / Detokenizer. All other ranks receive request messages broadcast by rank 0, ensuring that every rank sees the same set of requests in the same order. Single-GPU execution follows the same principle, without synchronization across ranks:

```python
# scheduler/io.py (excerpt)
def _recv_msg_single_rank(self, blocking: bool = False):
    pending_msgs: List[BaseBackendMsg] = []
    # Single GPU: pull directly from the Tokenizer
    ...

# Rank 0: receive messages from the Tokenizer, broadcast the count using Gloo,
# and send the contents to the other ranks using ZMQ PUB
def _recv_msg_multi_rank0(self, blocking: bool = False):
    pending_msgs: List[BaseBackendMsg] = []
    if blocking:
        # Receive one raw message, publish it to the other ranks, then decode locally
        ...

    pending_raw_msgs: List[bytes] = []
    while not self._recv_from_tokenizer.empty():
        pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

    # Use Gloo to broadcast only the message count
    src_tensor = torch.tensor(len(pending_raw_msgs))
    self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

    # Publish each message through ZMQ PUB and decode it locally
    for raw in pending_raw_msgs:
        self._send_into_ranks.put_raw(raw)
        pending_msgs.append(self._recv_from_tokenizer.decode(raw))
    return pending_msgs

def _recv_msg_multi_rank1(self, blocking: bool = False):
    pending_msgs: List[BaseBackendMsg] = []
    if blocking:
        self.run_when_idle()
        pending_msgs.append(self._recv_from_rank0.get())

    # Other ranks receive the count, then read the same messages from ZMQ SUB
    dst_tensor = torch.tensor(-1)
    self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()
    for _ in range(int(dst_tensor.item())):
        pending_msgs.append(self._recv_from_rank0.get())
    return pending_msgs

def _reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]):
    # Only rank 0 sends replies to the Detokenizer;
    # _reply_tokenizer_rank1 is a no-op on the other ranks
    ...
```

**The same request set + the same request order + the same token/length state** form a batch that ranks can process cooperatively. **This is a prerequisite for correct cooperation in the subsequent RadixAttention prefix caching.** Each rank independently updates its local Prefill/Decode/Cache state from the same messages and executes the same scheduling logic. The runtime does not broadcast a complete Batch object on every iteration.

The complete path from startup and request reception to output generation was covered in [Chapter 2: The Journey of a Request](https://datawhalechina.github.io/zero-to-sglang/ch/part2/%E7%AC%AC2%E7%AB%A0_%E4%B8%80%E4%B8%AA%E8%AF%B7%E6%B1%82%E7%9A%84%E6%97%85%E7%A8%8B_%E4%BB%A3%E7%A0%81.html), so we will not repeat it here.

## 4 Mapping to SGLang

The preceding workflow shows how mini-SGLang starts multiple Schedulers, coordinates the same batch of requests, and completes a forward pass through weight sharding and collective communication. SGLang divides these responsibilities into more modules, but the components can still be roughly mapped as follows:

| Function | mini-SGLang (python/minisgl/) | SGLang (python/sglang/srt/) |
|------|----------------------------------|---------------------------|
| Multi-process startup | server/launch.py | entrypoints/engine.py |
| Scheduling and model execution | scheduler/scheduler.py, engine/engine.py | managers/scheduler.py, managers/tp_worker.py, model_executor/model_runner |
| Request reception and coordination | scheduler/io.py, scheduler/scheduler.py | managers/scheduler.py |
| Communication groups and tensor communication | distributed/impl.py, kernel/pynccl.py | distributed/parallel_state.py |
| Column/Row/QKV Parallel | layers/linear.py, models/weight.py | layers/linear.py |
| Vocabulary sharding and loading | layers/embedding.py, models/weight.py | layers/vocab_parallel_embedding.py |

This table lists only selected modules for reference. SGLang has more files and more complex configurations and communication paths involving DP/PP/EP, among others, but the main workflow is shared with mini-SGLang. You can first use mini-SGLang to understand that workflow, then map it to the corresponding SGLang modules.

## 5 Summary and Exercises

### 5.1 Summary

Following the mini-SGLang source code, this chapter traced the workflow from multi-process startup and weight loading according to TP rank and sharding rules to receiving and coordinating the same batch of requests. The communication mechanisms used in cooperation across ranks—Gloo for synchronizing message counts, ZMQ for broadcasting message contents, and PyNCCL/NCCL for tensor collectives during forward passes—work together to keep rank states consistent and ensure correct final computation results.

---

### 5.2 Exercises

1. Each rank starts a Scheduler. What purpose does it serve?

   > Hint: Consider only multi-GPU execution.

2. How are weights loaded? How do the ranks coordinate to process the same batch of requests?

   > Hint: Consider single-GPU and multi-GPU execution separately.

3. Why are weights loaded in a streaming fashion, and why are they yielded only after merging/stacking when needed?

   > Hint: Focus on peak CPU memory usage during loading and the weight memory requirements of runtime layers, such as fused QKV and MoE layers.

## References

- [mini-SGLang](https://github.com/sgl-project/mini-sglang)
- [SGLang](https://github.com/sgl-project/sglang)
- [Chapter 2: The Journey of a Request](https://datawhalechina.github.io/zero-to-sglang/eng/part2/Chapter2_Inside-SGLang_code.html)
