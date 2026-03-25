# MegatronLM-Sim — Training-phase simulator

An interactive simulator for Megatron-style TP/PP/DP training. It produces time-series traces for compute and communication, including embedding/output heads, activation checkpointing recompute, gradient clipping, and optimizer overhead.

## Model defaults

| Parameter        | Value       |
|------------------|-------------|
| Parameters       | 8.8 Billion |
| Hidden Size      | 4096        |
| Layers           | 32          |
| Attention Heads  | 32          |
| Sequence Length  | 2048        |
| FFN Intermediate | 16384       |
| Vocabulary Size  | 50257       |

## Parallelism (typical preset)

| Dimension         | Degree | Notes                                |
|-------------------|--------|--------------------------------------|
| Tensor Parallel   | 4      | Intra-node, usually NVLink           |
| Pipeline Parallel | 8      | Inter-node, usually InfiniBand       |
| Data Parallel     | 4      | Gradient all-reduce across DP groups |
| **Total GPUs**    | **128**| 4 × 8 × 4                            |
| Global Batch Size | 128    | 1 sample per GPU in the default preset|

## What’s modeled
- Embedding compute on stage 0 (vocab-parallel shards across TP ranks).
- Output head / LM projection on the final PP stage (also TP-sharded).
- Activation checkpointing: backward can recompute forward work (controlled by `activation_checkpoint_ratio`).
- Gradient clipping: global grad-norm compute plus an 8-byte all-reduce per DP group.
- Optimizer step: Adam-like update cost on each rank’s parameter shard.
- Communication realism: NVLink / intra-rack IB / cross-rack with oversubscription, per-hop latency, chunk/header overhead, congestion, ring vs tree all-reduce selection, compute/comm overlap factor.

## How to run
1. From the repo root, launch: `python sim.py`
2. Pick a preset or enter custom TP/PP/DP, micro-batch, microbatches, and schedule (1F1B or GPipe).
3. Choose a network topology JSON from `Network/` and a GPU config from `GPU/` (or use defaults if none).
4. Enter iteration count; traces are written under `Traces/` with timestamped filenames.

## Trace files

**Messages** (`messages_timeseries*.csv`)

| Column               | Description                                                            |
|----------------------|------------------------------------------------------------------------|
| msg_id               | Unique message id                                                      |
| src_rank / dst_rank  | Source / destination GPU rank (-1 for collective dst)                  |
| size_bytes / size_mb | Message size                                                           |
| collective_type      | `all_reduce`, `p2p`, `all_gather`, `reduce_scatter`, etc.              |
| start_time_ms        | Start timestamp                                                        |
| end_time_ms          | End timestamp                                                          |
| duration_ms          | Transfer duration                                                      |
| stage                | Logical stage (e.g., `tp_all_reduce`, `pp_forward`, `dp_grad_clip`)    |
| layer_id             | Transformer layer id, `-1` for non-layer ops                           |
| pipeline_stage_src/dst| Pipeline stage ids for src/dst                                        |
| dp_group_src/dst     | Data-parallel group ids for src/dst                                    |
| participating_ranks  | Comma-separated ranks in the collective                                |

**Compute** (`compute_timeseries*.csv`)

| Column                 | Description                                         |
|------------------------|-----------------------------------------------------|
| event_id               | Unique compute event id                             |
| rank                   | GPU rank                                            |
| compute_type           | `forward`, `backward`, `embedding`, `output`, `grad_clip`, `optimizer` |
| start_time_ms / end_time_ms | Start / end timestamps                         |
| duration_ms            | Compute duration                                    |
| layer_id               | Layer index, `-1` for non-layer ops                 |
| flop_count / tflops    | FLOPs and TFLOPs                                    |
| memory_accessed_bytes  | Bytes read/written                                   |
| pipeline_stage         | Pipeline stage id                                   |
| dp_group               | Data-parallel group id                              |
| tp_rank                | Tensor-parallel rank                                |

## Naming and locations
- Traces: `Traces/[i{iters}]messages_timeseries<timestamp>[<gpus>gpu].csv` and matching compute file.
- Configs: network JSONs in `Network/`, GPU JSONs in `GPU/`.

## Key knobs to mention
- `activation_checkpoint_ratio` in `ModelConfig` (0 disables recompute).
- `overlap_factor` in network config (compute/comm overlap for collectives).
- TP/PP/DP sizes and microbatch count strongly affect pipeline timing and DP all-reduce volume.
