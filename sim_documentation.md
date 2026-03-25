# Megatron Simulator — Full Technical Documentation

> **File:** `sim.py`  
> **Purpose:** Discrete-event simulation of distributed LLM training under Megatron-style 3D parallelism, producing per-message and per-compute-event time-series traces in CSV format.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Summary](#2-architecture-summary)
3. [Enumerations](#3-enumerations)
4. [Data Classes](#4-data-classes)
   - 4.1 [Message](#41-message)
   - 4.2 [ComputeEvent](#42-computeevent)
   - 4.3 [NetworkConfig](#43-networkconfig)
   - 4.4 [GPUConfig](#44-gpuconfig)
   - 4.5 [ParallelConfig](#45-parallelconfig)
   - 4.6 [ModelConfig](#46-modelconfig)
5. [MegatronSimulator Class](#5-megatronsimulator-class)
   - 5.1 [Constructor and Initialization](#51-constructor-and-initialization)
   - 5.2 [Topology Setup](#52-topology-setup)
   - 5.3 [Compute Duration Model](#53-compute-duration-model)
   - 5.4 [Communication Time Model](#54-communication-time-model)
   - 5.5 [TP Layer Simulation](#55-tp-layer-simulation)
   - 5.6 [Special Layer Simulations](#56-special-layer-simulations)
   - 5.7 [Pipeline Stage Simulation](#57-pipeline-stage-simulation)
   - 5.8 [Pipeline Schedules](#58-pipeline-schedules)
   - 5.9 [Full Iteration](#59-full-iteration)
   - 5.10 [CSV Output (flush_to_csv)](#510-csv-output-flush_to_csv)
6. [Communication Model: Deep Dive](#6-communication-model-deep-dive)
7. [Compute Model: Deep Dive](#7-compute-model-deep-dive)
8. [Pipeline Schedules: Deep Dive](#8-pipeline-schedules-deep-dive)
9. [Output CSV Schemas](#9-output-csv-schemas)
10. [Configuration File Formats](#10-configuration-file-formats)
11. [CLI Entry Point and Interactive Flow](#11-cli-entry-point-and-interactive-flow)
12. [Known Limitations and TODOs](#12-known-limitations-and-todos)
13. [Worked Example](#13-worked-example)

---

## 1. Overview

`sim.py` is a Python-based discrete-event simulator for **large-scale distributed LLM training**. It models a Megatron-LM-style training stack with all three levels of parallelism:

| Parallelism | Abbreviation | Description |
|---|---|---|
| Tensor Parallel | TP | Splits weight tensors within a single layer across GPUs |
| Pipeline Parallel | PP | Splits model depth (layers) across GPUs via microbatch pipelining |
| Data Parallel | DP | Replicates the model across GPU groups, each consuming different data |

The simulator is **not** a real communication library. It estimates time using analytical models for:
- Roofline-based compute (FLOPs vs. memory bandwidth),
- Multi-tier network bandwidth (NVLink / intra-rack InfiniBand / cross-rack InfiniBand with oversubscription),
- Collective communication algorithms (ring, tree, point-to-point),
- Hardware bottlenecks (NIC bandwidth, PCIe bandwidth),
- Congestion, jitter, protocol overhead, and compute-communication overlap.

**Primary outputs** are two CSV trace files (one for network messages, one for compute events) that can be used for analysis, cost modelling, or visualisation.

---

## 2. Architecture Summary

```
sim.py
│
├── Enums
│   ├── ParallelType   — TP, PP, DP
│   ├── CollectiveType — ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER, ALL_TO_ALL, BROADCAST, P2P
│   └── ComputeType    — FORWARD, BACKWARD, OPTIMIZER, EMBEDDING, OUTPUT, GRAD_CLIP, ...
│
├── Data Classes (configuration & events)
│   ├── NetworkConfig  — bandwidth, latency, topology, NIC, PCIe, congestion, jitter
│   ├── GPUConfig      — TFLOPS, memory, NVLink, MFU, thermal throttle
│   ├── ParallelConfig — TP/PP/DP sizes, microbatch settings, schedule type
│   ├── ModelConfig    — layers, hidden size, FFN, attention heads, vocab size
│   ├── Message        — one communication event (collective or P2P)
│   └── ComputeEvent   — one compute kernel event per rank
│
└── MegatronSimulator
    ├── __init__            — sync NVLink, init state
    ├── setup_topology      — map every rank → {pp_stage, tp_rank, dp_group, node, rack}
    │                         pre-compute TP groups, DP all-reduce groups
    ├── _compute_duration   — roofline model (compute-bound vs memory-bound + jitter)
    ├── _record_compute_event
    ├── calculate_communication_time  — full 9-factor comm model
    ├── simulate_tp_layer   — compute + TP all-reduce per layer, with overlap
    ├── simulate_embedding_layer
    ├── simulate_output_layer
    ├── simulate_stage      — one PP stage for one microbatch
    ├── run_gpipe           — GPipe schedule
    ├── run_1f1b            — 1F1B schedule
    ├── run_full_iteration  — one training step (fwd + bwd + DP all-reduce + optimizer)
    └── flush_to_csv        — write messages & compute events to CSV, free memory
```

---

## 3. Enumerations

### `ParallelType`
Identifies the axis of parallelism for any given operation.

| Value               | String | Meaning                                |
| ------------------- | ------ | -------------------------------------- |
| `TENSOR_PARALLEL`   | `"TP"` | Within-layer weight sharding           |
| `PIPELINE_PARALLEL` | `"PP"` | Layer-depth sharding via microbatches  |
| `DATA_PARALLEL`     | `"DP"` | Replica-based gradient synchronisation |


---

### `CollectiveType`
Identifies the communication primitive for each `Message`.

| Value            | String             | Usage in simulator                                                      |
| ---------------- | ------------------ | ----------------------------------------------------------------------- |
| `ALL_REDUCE`     | `"all_reduce"`     | TP all-reduce after each layer; DP gradient sync; grad-clip scalar sync |
| `ALL_GATHER`     | `"all_gather"`     | Not directly emitted (modelled inside ALL_REDUCE for ring)              |
| `REDUCE_SCATTER` | `"reduce_scatter"` | Not directly emitted (modelled inside ALL_REDUCE for ring)              |
| `POINT_TO_POINT` | `"p2p"`            | Pipeline-parallel inter-stage activation / gradient tensors             |

---

### `ComputeType`
Labels each `ComputeEvent`.

| Value        | When emitted                                                          |
| ------------ | --------------------------------------------------------------------- |
| `FORWARD`    | Forward pass through a transformer layer (including recompute events) |
| `BACKWARD`   | Backward pass through a transformer layer                             |
| `OPTIMIZER`  | Adam-like optimizer step per DP group                                 |
| `EMBEDDING`  | Input embedding layer (stage 0 only)                                  |
| `OUTPUT`     | LM head / output projection (last stage only)                         |
| `GRAD_CLIP`  | Gradient norm computation before clipping                             |


---

## 4. Data Classes

### 4.1 `Message`

Represents a single logical communication event (one collective or one P2P transfer).

```python
@dataclass
class Message:
    msg_id: str                    # Unique identifier string
    src_rank: int                  # Source rank (or first rank for collectives; -1 if N/A)
    dst_rank: int                  # Destination rank (-1 for broadcast collectives)
    size_bytes: int                # Payload size in bytes
    collective_type: CollectiveType
    start_time: float              # Simulation time in seconds when comm begins
    end_time: float                # Simulation time in seconds when comm completes
    stage: str                     # Semantic label: e.g. "tp_all_reduce", "pp_forward", "dp_gradient_allreduce"
    layer_id: int                  # Which transformer layer (-1 for non-layer ops)
    participating_ranks: str       # Comma-separated list of all ranks in this collective
```

**`duration()`** — Returns `end_time - start_time` in seconds.

> One `Message` is emitted per **logical collective**, not per rank pair. For a TP all-reduce across 4 ranks, exactly one `Message` is recorded with `src_rank = ranks[0]` and `participating_ranks = "0,1,2,3"`. This keeps trace files tractable at scale.

---

### 4.2 `ComputeEvent`

Represents a single compute kernel execution on one rank.

```python
@dataclass
class ComputeEvent:
    event_id: str
    rank: int
    compute_type: ComputeType
    start_time: float              # seconds
    end_time: float                # seconds
    layer_id: int
    flop_count: int                # Raw FLOPs (not TFLOPS)
    memory_accessed: int           # Bytes accessed (for roofline model)
```

One `ComputeEvent` is emitted **per rank** in the participating group. For a TP group of 4, a forward pass emits 4 `ComputeEvent` objects.

---

### 4.3 `NetworkConfig`

Fully describes the interconnect topology, bandwidth, latency, and protocol parameters.

| Field | Default | Description |
|---|---|---|
| `nvlink_bw` | 600.0 | NVLink bandwidth in GB/s (overridden by `GPUConfig.nvlink_bw` at runtime) |
| `infiniband_bw` | 50.0 | Per-link InfiniBand bandwidth in GB/s |
| `latency_nvlink_us` | 0.5 | NVLink base latency in microseconds |
| `latency_ib_us` | 2.0 | InfiniBand base latency in microseconds |
| `num_gpus_per_node` | 8 | GPUs per physical node (affects intra-node routing) |
| `topology` | `"fat_tree"` | Topology label (informational; fat-tree is the modelled topology) |
| `overlap_factor` | 0.8 | Fraction of comm time that can overlap with compute |
| `collective_algo` | `"auto"` | `"ring"`, `"tree"`, or `"auto"` (auto selects based on `ring_tree_threshold_bytes`) |
| `nodes_per_rack` | 4 | Physical nodes per rack (used to determine same-rack vs. cross-rack routing) |
| `oversubscription_ratio` | 2.0 | Spine bandwidth oversubscription for cross-rack traffic (1.0 = full bisection) |
| `latency_cross_rack_us` | 5.0 | Legacy cross-rack latency field (fallback only; replaced by per-hop model) |
| `nic_bw_gbps` | 50.0 | Per-GPU NIC bandwidth in GB/s (400 Gb/s NDR ≈ 50 GB/s) |
| `num_nics_per_gpu` | 1 | NICs per GPU (rail-optimised = 1; multi-rail = 2+) |
| `pcie_bw_gbps` | 64.0 | PCIe Gen5 x16 host-device bandwidth in GB/s |
| `header_bytes` | 128 | Protocol/header overhead added per chunk |
| `chunk_size_bytes` | 8,388,608 | NCCL-style pipeline chunk size (8 MB default) |
| `bidirectional` | `True` | Whether links are full-duplex |
| `switch_latency_us` | 0.3 | Additional latency per switch hop |
| `num_hops_intra_rack` | 1 | Switch hops within a rack (ToR only) |
| `num_hops_cross_rack` | 3 | Switch hops across racks (ToR → Spine → ToR) |
| `ring_tree_threshold_bytes` | 1,048,576 | Auto-algo threshold: ring if message > threshold, tree otherwise |
| `latency_jitter_us` | 0.5 | Random ±jitter added to each communication latency (microseconds) |
| `congestion_penalty_factor` | 0.1 | Per-flow bandwidth degradation: 10% per additional concurrent flow on shared links |

**`from_json(path)`** — Class method. Loads a `NetworkConfig` from a JSON file, ignoring unrecognised keys (e.g. `name`, `description`).

---

### 4.4 `GPUConfig`

Describes the compute and memory characteristics of a single GPU.

| Field | Default | Description |
|---|---|---|
| `peak_tflops` | 989.0 | Peak BF16 TFLOPS (H100 SXM = 989 TFLOPS) |
| `memory_gb` | 80 | HBM capacity in GB |
| `memory_bw_gbps` | 3350.0 | HBM bandwidth in GB/s |
| `compute_efficiency` | 0.45 | Model FLOP Utilisation (MFU). Typical real-world range: 0.30–0.60 |
| `dtype_factor` | 1.0 | Precision multiplier: 1.0 = BF16, 2.0 = FP8, 4.0 = FP4 |
| `nvlink_bw` | 900.0 | NVLink bandwidth in GB/s (V100=300, A100=600, H100=900). Set to 0 for PCIe-only GPUs |
| `num_sms` | 132 | Number of streaming multiprocessors. Used to scale compute-comm overlap |
| `l2_cache_mb` | 50.0 | L2 cache size in MB (reserved for future memory-hierarchy modelling) |
| `tdp_watts` | 700 | Thermal Design Power in watts |
| `thermal_throttle_factor` | 0.95 | Sustained throughput as a fraction of peak (models thermal throttling) |

**`from_json(path)`** — Same pattern as `NetworkConfig.from_json`.

**NVLink sync logic** (in `MegatronSimulator.__init__`):
- If `gpu_config.nvlink_bw > 0`, the network config's `nvlink_bw` is overridden with the GPU's value.
- If `gpu_config.nvlink_bw == 0` (PCIe-only GPU), the simulator falls back to `memory_bw_gbps * 0.5` as the effective intra-node bandwidth (a conservative PCIe approximation).

---

### 4.5 `ParallelConfig`

Defines the 3D parallelism decomposition and microbatch schedule.

| Field | Default | Description |
|---|---|---|
| `tp_size` | 4 | Tensor parallel degree (number of GPUs per TP group) |
| `pp_size` | 8 | Pipeline parallel degree (number of pipeline stages) |
| `dp_size` | 4 | Data parallel degree (number of model replicas) |
| `sp_size` | 1 | Sequence parallel degree (reserved) |
| `micro_batch_size` | 2 | Samples per microbatch per DP rank |
| `num_microbatches` | 16 | Number of microbatches per global batch |
| `virtual_pp_size` | 1 | Virtual pipeline stages per physical stage (reserved for interleaved 1F1B) |
| `schedule_type` | `"1F1B"` | Pipeline schedule: `"1F1B"` or `"GPipe"` |

**`world_size()`** — Returns `tp_size * pp_size * dp_size`.

> **Global batch size** = `dp_size × micro_batch_size × num_microbatches`.

---

### 4.6 `ModelConfig`

Describes the transformer architecture.

| Field | Default | Description |
|---|---|---|
| `num_layers` | 32 | Total transformer layers (must be divisible by `pp_size`) |
| `hidden_size` | 4096 | Model hidden dimension H |
| `num_attention_heads` | 32 | Number of attention heads |
| `seq_length` | 2048 | Input sequence length S |
| `vocab_size` | 50257 | Vocabulary size |
| `ffn_hidden_size` | 16384 | FFN intermediate dimension (typically 4× hidden) |
| `activation_checkpoint_ratio` | 0.5 | Fraction of layers with activation checkpointing. 0.0 = disabled |

**`total_params()`** — Returns a hardcoded `8,800,000,000` (8.8B parameters). This is an approximation for an LLaMA-style 8B model; it is used for DP gradient all-reduce size and optimizer step calculations.

> **Note:** `total_params()` is not derived analytically from the architecture fields. If you change the model configuration significantly, update this value manually.

---

## 5. MegatronSimulator Class

### 5.1 Constructor and Initialization

```python
MegatronSimulator(
    parallel_config: ParallelConfig,
    model_config: ModelConfig,
    network_config: NetworkConfig,
    gpu_config: GPUConfig
)
```

**Internal state initialized:**

| Attribute | Type | Description |
|---|---|---|
| `rank_times` | `defaultdict(float)` | Current simulated time for each rank (seconds) |
| `messages` | `List[Message]` | Accumulated message events (flushed to CSV per iteration) |
| `compute_events` | `List[ComputeEvent]` | Accumulated compute events (flushed to CSV per iteration) |
| `task_completion_times` | `dict` | Maps `(microbatch, 'fwd'/'bwd', stage)` → completion time; used for dependency checking in the pipeline schedule |
| `_active_flows` | `int` | Counter of currently active network flows; used for congestion modelling |
| `rank_to_stage` | `dict` | Maps each rank integer to its topology metadata (see §5.2) |
| `current_time` | `float` | The final simulated time after `run_full_iteration` completes |

---

### 5.2 Topology Setup

**`setup_topology()`** is called from `__init__` and builds all rank-to-group mappings upfront.

**Per-rank metadata** stored in `rank_to_stage[rank]`:

| Key | Formula | Description |
|---|---|---|
| `dp_group` | `rank // (tp_size * pp_size)` | Which data parallel replica this rank belongs to |
| `pp_stage` | `(rank % (tp_size * pp_size)) // tp_size` | Which pipeline stage this rank sits at |
| `tp_rank` | `rank % tp_size` | Position within the tensor parallel group |
| `global_rank` | `rank` | Echo of the rank itself |
| `node_id` | `rank // num_gpus_per_node` | Physical node index |
| `rack_id` | `node_id // nodes_per_rack` | Physical rack index |

**Pre-computed group caches** (to avoid repeated list comprehensions in the hot path):

- `_stage_ranks[stage]` — all ranks at a given pipeline stage (across all DP groups)
- `_stage_dp_groups[stage][dp_id]` — list of TP ranks in a specific DP group at a specific stage
- `_dp_allreduce_groups[(pp_stage, tp_rank)]` — ranks that participate in the same DP all-reduce (same stage and same TP position, different DP group)

---

### 5.3 Compute Duration Model

```python
_compute_duration(flop_count: float, memory_bytes: float) -> float
```

Implements the **roofline model**:

```
effective_tflops = peak_tflops × dtype_factor × compute_efficiency × thermal_throttle_factor

compute_bound_time = flop_count / (effective_tflops × 1e12)
memory_bound_time  = memory_bytes / (memory_bw_gbps × 1e9)

base_time = max(compute_bound_time, memory_bound_time)
jitter    = uniform(−0.05, +0.05) × base_time
duration  = max(base_time + jitter, 1e-6)
```

- **Compute-bound:** the kernel is limited by peak FLOP throughput.
- **Memory-bound:** the kernel is limited by HBM read/write bandwidth (occurs for small batch sizes, layer norms, etc.).
- **Jitter:** ±5% uniform noise is added to simulate real-world kernel variance.
- **Floor:** `1e-6` seconds (1 µs) prevents zero-duration events.

The `dtype_factor` field allows modelling different precision regimes without changing the base TFLOPS:
- `1.0` → BF16 (baseline)
- `2.0` → FP8 (doubles throughput)
- `4.0` → FP4 (quadruples throughput)

---

### 5.4 Communication Time Model

```python
calculate_communication_time(
    size_bytes: int,
    collective: CollectiveType,
    participating_ranks: List[int]
) -> float  # seconds
```

This is the most detailed component of the simulator. It applies **9 factors** in order:

#### Factor 1: Network Tier Selection

The participating ranks are examined to determine which network tier is used:

| Condition | Bandwidth | Base Latency | Switch Hops |
|---|---|---|---|
| All ranks on same node | `nvlink_bw` | `latency_nvlink_us` | 0 |
| All ranks in same rack (different nodes) | `infiniband_bw` | `latency_ib_us` | `num_hops_intra_rack` |
| Ranks span multiple racks | `infiniband_bw / oversubscription_ratio` | `latency_ib_us` | `num_hops_cross_rack` |

The oversubscription ratio reduces effective cross-rack bandwidth (e.g. 2.0:1 means spine links are shared among 2× the traffic they can sustain at full bisection bandwidth).

#### Factor 2: Per-Hop Switch Latency

```
hop_latency = num_hops × switch_latency_us / 1,000,000
total_latency = base_latency + hop_latency
```

This replaces a flat cross-rack latency with a more physically accurate model: each switch adds approximately 300 ns.

#### Factor 3: Latency Jitter

```
jitter = uniform(−latency_jitter_us, +latency_jitter_us) / 1,000,000
total_latency = max(0, total_latency + jitter)
```

Prevents unrealistically perfect latency values.

#### Factor 4: Effective Bandwidth (NIC and PCIe Bottlenecks)

```
nic_bw      = nic_bw_gbps × num_nics_per_gpu
effective_bw = min(link_bw, nic_bw, pcie_bw)
```

This correctly models that the NIC (not the link) is often the binding constraint in modern rail-optimised clusters. For example:
- A 400 Gb/s NDR link with 1 NIC per GPU = 50 GB/s NIC limit, even if the fabric supports more.
- PCIe Gen5 x16 = 64 GB/s, which can bottleneck host-to-device transfers.

#### Factor 5: Bidirectional Link Awareness

If `bidirectional = True` (full-duplex) and the collective is not P2P, the bandwidth is unchanged (each direction gets the full link rate). If `bidirectional = False` (half-duplex), bandwidth for bidirectional collectives is halved.

#### Factor 6: Congestion Penalty

```
if _active_flows > 1 and not same_node:
    congestion_divisor = 1 + congestion_penalty_factor × (_active_flows − 1)
    effective_bw = effective_bw / congestion_divisor
```

`_active_flows` is a global counter of concurrently active network flows. Each `simulate_tp_layer`, `simulate_stage` (P2P), and `run_full_iteration` (DP AR) bracket their communication with `_active_flows += 1` / `_active_flows -= 1`. A congestion factor of 0.1 means each additional concurrent flow reduces bandwidth by 10%.

> **Limitation:** This is a simple approximation. Real congestion is flow-pair and topology dependent. It models the average case on a shared fabric.

#### Factor 7: Protocol Overhead

```
num_chunks                = ceil(size_bytes / chunk_size_bytes)
total_bytes_with_overhead = size_bytes + (num_chunks × header_bytes)
```

Each NCCL-style chunk adds a fixed protocol header. For an 8 MB chunk size and 128-byte header, overhead is negligible for large messages but measurable for small ones.

#### Factor 8: Collective Algorithm Selection and Transfer Time

**Point-to-Point:**
```
time = total_latency + (size_gb / effective_bw)
```

**All-Reduce (ring algorithm):**
```
lat_cost = 2 × (num_ranks − 1) × total_latency
bw_cost  = 2 × (num_ranks − 1) / num_ranks × size_gb / effective_bw
time     = lat_cost + bw_cost
```
This follows the standard alpha-beta model for ring all-reduce. The factor of 2 accounts for the reduce-scatter phase followed by the all-gather phase, each traversing the ring once.

**All-Reduce (tree algorithm):**
```
log_n    = log2(num_ranks)
lat_cost = 2 × log_n × total_latency
bw_cost  = 2 × log_n × size_gb / effective_bw
time     = lat_cost + bw_cost
```
A binary tree reduces in `log_n` steps in each direction.

**Algorithm auto-selection:**
```
if size_bytes > ring_tree_threshold_bytes:  # default 1 MB
    use ring   (efficient for large messages)
else:
    use tree   (lower latency for small messages)
```

**All other collectives** (All-Gather, Reduce-Scatter, etc.) fall back to:
```
time = total_latency × num_ranks + size_gb / effective_bw
```

#### Factor 9: Minimum Bandwidth Guard

`effective_bw` is clamped to at least `0.001` GB/s before division to prevent division by zero under extreme congestion.

---

### 5.5 TP Layer Simulation

```python
simulate_tp_layer(
    ranks: List[int],     # TP group (all GPUs sharing this layer shard)
    layer_id: int,
    is_forward: bool,
    start_time: float
) -> float                # returns end time for this layer across all TP ranks
```

**Step-by-step:**

1. **Compute FLOPs and memory** for a single transformer layer, divided by TP degree:
   ```
   layer_flops_fwd = (8 × B × S × H² + 4 × B × S × H × FFN) / tp
   weight_bytes    = ((8 × H² + 4 × H × FFN) × 2) / tp       # BF16
   activation_bytes = B × S × H × 2 × 4
   ```

2. **Activation checkpointing recompute**:
   - If `activation_checkpoint_ratio > 0` and `layer_id % interval == 0`, the layer is checkpointed.
   - During backward, a recomputed forward pass is inserted before the actual backward.
   - Interval = `1 / activation_checkpoint_ratio`. At ratio 0.5, every 2nd layer is recomputed.

3. **Main compute:**
   - Forward: `flop_count = layer_flops_fwd`, `memory_accessed = memory_accessed_fwd`
   - Backward: `flop_count = layer_flops_fwd × 2` (grad w.r.t. inputs + grad w.r.t. weights), `memory_accessed = memory_accessed_fwd × 2.5`

4. **TP All-Reduce:**
   - Size: `2 × H² × 2 + B × S × H × 2` bytes (weight synchronisation + activation tensor)
   - Called with `_active_flows += 1` / `_active_flows -= 1` for congestion tracking

5. **Compute-Communication Overlap:**
   ```
   sm_overlap_bonus   = min(num_sms / 132.0, 1.2)   # normalised to H100 SM count
   effective_overlap  = min(overlap_factor × sm_overlap_bonus, 0.95)
   effective_ar_time  = ar_duration × (1 − effective_overlap)
   rank_end_time      = max(comp_end, start_time + effective_ar_time)
   ```
   GPUs with more SMs can dedicate more of them to communication while others continue computing, so the overlap bonus scales with SM count. The 95% cap prevents complete hiding of communication cost.

6. **Emit Message:** One `Message` per logical all-reduce (not one per rank).

---

### 5.6 Special Layer Simulations

#### `simulate_embedding_layer`

Simulates the input embedding lookup on pipeline stage 0.

- Embedding table is vocab-parallel across TP ranks: shard size = `vocab_size // tp`
- FLOPs: `2 × B × S × H` (lookup + positional)
- Memory: `vocab_shard × H × 2 + B × S × H × 2` bytes
- Backward multiplies FLOPs by 2 and memory by 1.5
- **No TP all-reduce** is emitted (embeddings are gathered without communication in the common case)

#### `simulate_output_layer`

Simulates the LM head (output projection to vocabulary) on the last pipeline stage.

- Projection size: `B × S × H → B × S × vocab_shard`
- FLOPs: `2 × B × S × H × vocab_shard`
- Memory: same structure as embedding layer
- Backward multiplies FLOPs by 2, memory by 1.5
- **No TP all-reduce** is emitted

---

### 5.7 Pipeline Stage Simulation

```python
simulate_stage(mb: int, stage: int, is_forward: bool)
```

Simulates one pipeline stage for one microbatch in one direction.

**Dependency resolution:**

| Direction | Dependencies |
|---|---|
| Forward | All ranks at this stage must be idle; previous stage's forward for this microbatch must be complete |
| Backward | All ranks at this stage must be idle; next stage's backward for this microbatch must be complete; this stage's forward for this microbatch must also be complete |

```python
start_time = max(all rank_times at this stage, dependency from adjacent stage)
```

**Layer simulation loop:**
- Stage 0: embedding layer → N transformer layers (→ output layer if also last stage)
- Middle stages: N transformer layers only
- Last stage: N transformer layers → output layer

Each DP group is simulated independently. Since all DP replicas are structurally identical and start at the same time, their end times are averaged via `max(dp_end_times)`.

**Inter-stage P2P communication:**

After compute, a P2P message is sent to transfer activations (forward) or gradients (backward) to the adjacent stage:
- **Forward:** current stage → next stage
- **Backward:** current stage → previous stage
- Size: hardcoded **64 MB** (approximating `micro_batch_size × seq_len × hidden × 2` for BF16, with overhead)
- One `Message` per DP group is emitted

The P2P communication end time is stored in `task_completion_times[(mb, 'fwd'/'bwd', stage)]` and used as a dependency for subsequent stages.

---

### 5.8 Pipeline Schedules

#### `run_gpipe()`

Implements the **GPipe** (flush) schedule:

```
Phase 1 (Warmup):  All microbatches, all stages, FORWARD
Phase 2 (Flush):   All microbatches, all stages (reversed), BACKWARD
```

All forward microbatches complete before any backward begins. This maximises pipeline fill but results in a "bubble" at both ends and requires holding all microbatch activations in memory simultaneously.

```python
for mb in range(num_microbatches):
    for stage in range(pp_size):
        simulate_stage(mb, stage, True)

for mb in range(num_microbatches):
    for stage in range(pp_size - 1, -1, -1):
        simulate_stage(mb, stage, False)
```

#### `run_1f1b()`

Implements the **1F1B** (one-forward-one-backward) schedule, which interleaves forward and backward passes to keep all pipeline stages busy and limit the number of in-flight activations to `pp_size` (rather than `num_microbatches`).

**Schedule construction per stage:**
1. **Warmup phase:** `num_stages - stage - 1` forward-only microbatches (to fill the pipe from the first stage down)
2. **Steady state:** alternating (forward microbatch N+warmup, backward microbatch N) pairs
3. **Cooldown phase:** remaining backward-only microbatches (to drain the pipe)

**Execution loop:** A ready-task dispatcher iterates all stages and dispatches the next task at each stage if and only if its dependencies are satisfied. This continues until all tasks are complete.

```
while completed_tasks < total_tasks:
    for each stage:
        if next_task dependencies are met → simulate_stage(...), mark complete
```

If no stage makes progress (e.g. circular dependency — should not occur in a correct 1F1B schedule), the loop breaks.

---

### 5.9 Full Iteration

```python
run_full_iteration(iteration: int)
```

Executes one complete training step:

1. **Forward + Backward passes** via `run_1f1b()` or `run_gpipe()`

2. **DP Gradient All-Reduce:**
   - All DP groups execute their gradient synchronisation simultaneously (concurrently modelled via `_active_flows`)
   - Size: 512 MB per group (hardcoded approximation for 8.8B parameters with TP/PP sharding)
   - One `Message` emitted per `(pp_stage, tp_rank)` DP group

3. **Gradient Clipping:**
   - Gradient norm computation: `params_per_rank × 2` FLOPs, `params_per_rank × 2 × 2` bytes
   - All-reduce of scalar grad norm (8 bytes) across DP ranks
   - One `Message` per DP group

4. **Optimizer Step:**
   - Adam-like update: `params_per_rank × 8` FLOPs
   - Memory accessed: `param_bytes × 6` bytes (params + momentum + variance in BF16 and master copy)
   - One `ComputeEvent` per rank in each DP group

After all of the above, `current_time` is set to `max(rank_times.values())`.

**Console output:**
```
  Iteration 0 (1F1B) finished — sim time: 2.3456s  wall: 0.18s
```

---

### 5.10 CSV Output (`flush_to_csv`)

```python
flush_to_csv(msg_path: str, compute_path: str, write_header: bool)
```

Writes all accumulated `messages` and `compute_events` to CSV, then clears them from memory to prevent unbounded growth across iterations.

- First iteration: `write_header=True` writes the column headers and opens with `'w'`
- Subsequent iterations: `write_header=False` appends with `'a'`
- Times are converted from seconds to **milliseconds** in output

---

## 6. Communication Model: Deep Dive

The following table summarises the full decision tree inside `calculate_communication_time`:

```
Input: size_bytes, collective type, participating_ranks

┌─────────────────────────────────────────────────────────────────────┐
│ 1. Tier                                                             │
│    same_node? → NVLink BW, NVLink latency, 0 switch hops           │
│    same_rack? → IB BW, IB latency, num_hops_intra_rack switch hops │
│    cross_rack? → IB BW / oversub, IB latency, num_hops_cross_rack  │
├─────────────────────────────────────────────────────────────────────┤
│ 2. Switch latency: hops × switch_latency_us                        │
│ 3. Jitter: ±latency_jitter_us                                      │
├─────────────────────────────────────────────────────────────────────┤
│ 4. Bottleneck BW: min(link_bw, nic_bw × num_nics, pcie_bw)        │
│ 5. Bidirectionality: halve BW if half-duplex + bidirectional coll  │
│ 6. Congestion: BW /= (1 + penalty × (active_flows − 1))           │
├─────────────────────────────────────────────────────────────────────┤
│ 7. Protocol: add header_bytes per chunk                            │
├─────────────────────────────────────────────────────────────────────┤
│ 8. Algorithm:                                                       │
│    P2P       → lat + data/BW                                       │
│    All-Reduce ring → 2(n−1)lat + 2(n−1)/n × data/BW               │
│    All-Reduce tree → 2log_n × lat + 2log_n × data/BW              │
│    Other     → n×lat + data/BW                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Compute Model: Deep Dive

### FLOP Counts

For a transformer layer of hidden size H, FFN size F, batch B, sequence S:

| Operation | FLOPs (forward) |
|---|---|
| QKV projection | `6 × B × S × H²` |
| Output projection | `2 × B × S × H²` |
| FFN (two linear layers) | `4 × B × S × H × F` |
| **Total (approx)** | `8×B×S×H² + 4×B×S×H×F` |

Under TP with degree `tp`, each rank computes `1/tp` of these FLOPs.

Backward pass FLOPs are modelled as **2×** the forward FLOPs (computing gradient w.r.t. inputs and gradient w.r.t. weights).

### Memory Access

For a forward pass (BF16, 2 bytes per element):

```
weight_bytes     = (8H² + 4HF) × 2 / tp
activation_bytes = B × S × H × 2 × 4
total            = weight_bytes + activation_bytes
```

Backward is modelled as `2.5×` forward memory access (reads activations, reads weights, writes gradients).

### Roofline

```
effective_TFLOPS = peak_tflops × dtype_factor × MFU × thermal_throttle
compute_time     = flops / (effective_TFLOPS × 1e12)
memory_time      = memory_bytes / (memory_bw_gbps × 1e9)
kernel_time      = max(compute_time, memory_time)
```

Transformer layers at large batch sizes are typically **compute-bound**; at small batch or for layer norms they are **memory-bound**.

---

## 8. Pipeline Schedules: Deep Dive

### GPipe: Pipeline Bubble

With `num_microbatches = M` and `pp_size = P`:

```
Pipeline bubble fraction ≈ (P − 1) / M
```

For M=16, P=8 → ~44% of time is bubble. GPipe is simple but wasteful.

### 1F1B: Steady-State Overlap

1F1B eliminates most of the bubble in steady state. The number of in-flight activations is bounded at `P` (rather than `M` for GPipe), reducing peak memory. The remaining bubble comes only from the warmup and cooldown phases.

```
Warmup  = P − 1 forward passes (fill)
Steady  = M − (P−1) forward-backward pairs (full utilisation)
Cooldown = P − 1 backward passes (drain)

Bubble fraction ≈ (P − 1) / (M + P − 1) ≈ (P − 1) / M  for large M
```

The simulator's 1F1B implementation dispatches tasks in topological order using a readiness check, not a predetermined linear sequence. This correctly handles stalls when dependencies are not yet met.

---

## 9. Output CSV Schemas

### `messages_timeseries*.csv`

| Column | Type | Description |
|---|---|---|
| `message_id` | string | Unique message identifier (e.g. `ar_l0_tp0123`) |
| `src_rank` | int | Source rank (or first rank of collective; -1 if not applicable) |
| `dst_rank` | int | Destination rank (-1 for collectives) |
| `size_bytes` | int | Payload size in bytes |
| `size_mb` | float | Payload size in megabytes |
| `collective_type` | string | One of: `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `broadcast`, `p2p` |
| `start_time_ms` | float | Simulated start time in milliseconds |
| `end_time_ms` | float | Simulated end time in milliseconds |
| `duration_ms` | float | Duration in milliseconds |
| `stage` | string | Semantic label: `tp_all_reduce`, `pp_forward`, `pp_backward`, `dp_gradient_allreduce`, `dp_grad_clip` |
| `layer_id` | int | Transformer layer index (-1 for non-layer ops) |
| `pipeline_stage_src` | int | PP stage of `src_rank` (-1 if N/A) |
| `pipeline_stage_dst` | int | PP stage of `dst_rank` (-1 if N/A) |
| `dp_group_src` | int | DP group of `src_rank` (-1 if N/A) |
| `dp_group_dst` | int | DP group of `dst_rank` (-1 if N/A) |
| `participating_ranks` | string | Comma-separated list of all ranks in the collective |

### `compute_timeseries*.csv`

| Column | Type | Description |
|---|---|---|
| `event_id` | string | Unique event identifier |
| `rank` | int | GPU rank |
| `compute_type` | string | One of: `forward`, `backward`, `optimizer`, `embedding`, `output`, `grad_clip` |
| `start_time_ms` | float | Simulated start time in milliseconds |
| `end_time_ms` | float | Simulated end time in milliseconds |
| `duration_ms` | float | Duration in milliseconds |
| `layer_id` | int | Transformer layer index (-1 for non-layer ops) |
| `flop_count` | int | Raw FLOPs for this operation |
| `tflops` | float | `flop_count / 1e12` |
| `memory_accessed_bytes` | int | Bytes read/written from HBM |
| `pipeline_stage` | int | PP stage this rank is assigned to |
| `dp_group` | int | DP group this rank belongs to |
| `tp_rank` | int | TP position within the TP group |

### Output File Naming Convention

```
Traces/[i{iters}]messages_timeseries{YYYYMMDD_HHMMSS}[{N}gpu].csv
Traces/[i{iters}]compute_timeseries{YYYYMMDD_HHMMSS}[{N}gpu].csv
```

Files are written to a `Traces/` directory created automatically in the working directory.

---

## 10. Configuration File Formats

Network and GPU configurations are loaded from JSON files placed in `./Network/` and `./GPU/` directories respectively.

### Network JSON Format

```json
{
    "name": "HDR InfiniBand Fat Tree (2:1)",
    "description": "Standard 200 Gb/s HDR IB cluster with 2:1 oversubscription",
    "infiniband_bw": 25.0,
    "latency_ib_us": 2.5,
    "num_gpus_per_node": 8,
    "nodes_per_rack": 4,
    "oversubscription_ratio": 2.0,
    "nic_bw_gbps": 25.0,
    "num_nics_per_gpu": 1,
    "pcie_bw_gbps": 32.0,
    "switch_latency_us": 0.3,
    "num_hops_intra_rack": 1,
    "num_hops_cross_rack": 3,
    "congestion_penalty_factor": 0.1,
    "latency_jitter_us": 0.5
}
```

- All fields are optional; omitted fields fall back to `NetworkConfig` defaults.
- `name` and `description` are display-only and are ignored by `from_json`.

### GPU JSON Format

```json
{
    "name": "NVIDIA H100 SXM5",
    "description": "Hopper-generation GPU with 989 TFLOPS BF16, NVLink 4",
    "peak_tflops": 989.0,
    "memory_gb": 80,
    "memory_bw_gbps": 3350.0,
    "compute_efficiency": 0.45,
    "dtype_factor": 1.0,
    "nvlink_bw": 900.0,
    "num_sms": 132,
    "tdp_watts": 700,
    "thermal_throttle_factor": 0.95
}
```

---

## 11. CLI Entry Point and Interactive Flow


**Preset configurations:**

| Name | TP | PP | DP | Total GPUs | Notes |
|---|---|---|---|---|---|
| `128 GPUs` | 4 | 8 | 4 | 128 | 16 nodes |
| `64 GPUs` | 4 | 4 | 4 | 64 | 8 nodes |
| `32 GPUs` | 4 | 4 | 2 | 32 | 4 nodes |
| `16 GPUs` | 4 | 2 | 2 | 16 | 2 nodes |
| `8 GPUs (TP=4)` | 4 | 2 | 1 | 8 | 1 node |
| `8 GPUs (TP=8)` | 8 | 1 | 1 | 8 | 1 node |
| `256 GPUs` | 4 | 8 | 8 | 256 | 32 nodes |
| `512 GPUs` | 8 | 8 | 8 | 512 | 64 nodes |
| `Custom` | — | — | — | — | Prompt for all values |

**Validation:**
- `num_layers % pp_size == 0` is enforced (assertion)
- If `tp > 8`, a warning is shown (cross-node TP = slower communication)

**Output:**
- `Traces/` directory is created if it does not exist
- One `messages_timeseries*.csv` and one `compute_timeseries*.csv` are written
- Per-iteration timing is printed to console

---

## 13. Worked Example

**Configuration:** 32 GPUs, TP=4, PP=4, DP=2, H100, NDR IB, 1F1B schedule, 1 iteration.

**Topology mapping:**
- `world_size = 4 × 4 × 2 = 32`
- Ranks 0–3: DP group 0, PP stage 0, TP ranks 0–3 (all on node 0)
- Ranks 4–7: DP group 0, PP stage 1, TP ranks 0–3 (all on node 0)
- Ranks 16–19: DP group 1, PP stage 0, TP ranks 0–3 (node 2)
- etc.

**Per-layer TP communication (rank 0–3, layer 0, forward):**
- `ar_size = 2 × 4096² × 2 + 2 × 2048 × 4096 × 2 ≈ 67 MB + 33 MB = 100 MB`
- All 4 ranks are on the same node → NVLink tier → `nvlink_bw = 900 GB/s`
- Ring algorithm (100 MB > 1 MB threshold): `2 × (4−1) / 4 × 0.093 GB / 900 GB/s ≈ 0.155 µs`
- After overlap: `effective_ar_time = 0.155 µs × (1 − 0.87) ≈ 20 ns`

**DP all-reduce (cross-node):**
- `dp_ar_size = 512 MB`
- DP ranks are on different nodes but same rack (for a 4-node rack configuration)
- Same-rack IB tier → `infiniband_bw = 50 GB/s`, 1 switch hop
- Ring: `2 × (2−1) / 2 × 0.477 / 50 ≈ 9.5 ms`

**Total iteration time** is dominated by:
1. Compute: forward + backward through 8 layers per stage × 4 stages × 16 microbatches
2. Pipeline bubble: `(PP−1) / num_microbatches = 3/16 ≈ 19%` overhead
3. DP gradient sync: ~10 ms

**Output files** contain one row per message event (TP AR, P2P, DP AR) and one row per compute event (per rank, per layer).


---

## Abbreviations

| Abbreviation   | Definition                                                                         |
| -------------- | ---------------------------------------------------------------------------------- |
| AR             | All-Reduce                                                                         |
| BF16           | Brain Float 16 — a 16-bit floating-point format used for training                  |
| BW             | Bandwidth                                                                          |
| CLI            | Command-Line Interface                                                             |
| CSV            | Comma-Separated Values                                                             |
| DP             | Data Parallel / Data Parallelism                                                   |
| dtype          | Data type (e.g. BF16, FP8, FP4)                                                    |
| FFN            | Feed-Forward Network — the MLP sublayer in a transformer block                     |
| FLOP           | Floating-Point Operation                                                           |
| TFLOPS         | Tera Floating-Point Operations Per Second (10¹² FLOPS)                             |
| FP4            | 4-bit floating-point precision                                                     |
| FP8            | 8-bit floating-point precision                                                     |
| FP32           | 32-bit (single-precision) floating-point                                           |
| GB             | Gigabyte (10⁹ bytes)                                                               |
| GB/s           | Gigabytes per second                                                               |
| Gb/s           | Gigabits per second                                                                |
| HBM            | High Bandwidth Memory — the on-chip DRAM used by modern GPUs                       |
| HDR            | High Data Rate — InfiniBand generation rated at 200 Gb/s                           |
| IB             | InfiniBand                                                                         |
| JSON           | JavaScript Object Notation                                                         |
| L2             | Level-2 cache                                                                      |
| LLM            | Large Language Model                                                               |
| LM head        | Language Model head — the final output projection to vocabulary logits             |
| MB             | Megabyte (10⁶ bytes)                                                               |
| MFU            | Model FLOP Utilisation — ratio of observed to theoretical peak FLOPS               |
| MLP            | Multi-Layer Perceptron                                                             |
| MoE            | Mixture of Experts                                                                 |
| ms             | Milliseconds                                                                       |
| NCCL           | NVIDIA Collective Communications Library                                           |
| NDR            | Next Data Rate — InfiniBand generation rated at 400 Gb/s                           |
| NIC            | Network Interface Card                                                             |
| ns             | Nanoseconds                                                                        |
| NVLink         | NVIDIA's high-speed GPU-to-GPU interconnect                                        |
| NVSwitch       | NVIDIA's all-to-all NVLink switch fabric                                           |
| OOM            | Out of Memory                                                                      |
| P2P            | Point-to-Point (communication)                                                     |
| PCIe           | Peripheral Component Interconnect Express — the host-to-GPU bus                    |
| PP             | Pipeline Parallel / Pipeline Parallelism                                           |
| QKV            | Query, Key, Value — the three projections in self-attention                        |
| RDMA           | Remote Direct Memory Access                                                        |
| RoCE           | RDMA over Converged Ethernet                                                       |
| SM             | Streaming Multiprocessor — the basic compute unit of an NVIDIA GPU                 |
| SP             | Sequence Parallel / Sequence Parallelism                                           |
| TDP            | Thermal Design Power                                                               |
| ToR            | Top-of-Rack switch                                                                 |
| TP             | Tensor Parallel / Tensor Parallelism                                               |
| µs             | Microseconds                                                                       |
| 1F1B           | One-Forward-One-Backward — a pipeline schedule that interleaves fwd and bwd passes |
| 3D parallelism | The combination of TP, PP, and DP used in large-scale training                     |