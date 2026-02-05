###A simulator for the Nvidia Megatron LM training process

## 1. Model Architecture

| Parameter          | Value       |
|--------------------|-------------|
| Parameters         | 8.8 Billion |
| Hidden Size        | 4096        |
| Layers             | 32          |
| Attention Heads    | 32          |
| Sequence Length    | 2048        |
| FFN Intermediate   | 16384       |
| Vocabulary Size    | 50257       |

## 2. Parallelism Configuration

| Dimension          | Degree | Notes                          |
|--------------------|--------|--------------------------------|
| Tensor Parallel    | 4      | Intra-node, usually NVLink     |
| Pipeline Parallel  | 8      | Inter-node, usually InfiniBand |
| Data Parallel      | 4      | Gradient all-reduce            |
| **Total GPUs**     | **128**| 4 × 8 × 4                      |
| Global Batch Size  | 128    | 1 sample per GPU               |

## 3. Simulation Trace Files

### File 1: messages_timeseries.csv
- Rows: 359,424
- Columns: 14

| Column               | Type   | Description                                           |
|----------------------|--------|-------------------------------------------------------|
| msg_id               | string |                                                       |
| src_rank             | int    | Source GPU (0–127)                                    |
| dst_rank             | int    | Destination GPU (0–127)                               |
| size_bytes           | int    | Message size in bytes                                 |
| size_mb              | float  | Size in MB                                            |
| collective_type      | enum   | all_reduce, p2p, all_gather, reduce_scatter           |
| start_time_ms        | float  | Start timestamp                                       |
| end_time_ms          | float  | End timestamp                                         |
| duration_ms          | float  | Transfer duration (ms)                                |
| stage                | string | e.g. attn_fwd, mlp_fwd, pp_forward_mbX, optimizer...  |
| layer_id             | int    | Transformer layer ID (-1 = non-layer)                 |
| pipeline_stage_src   | int    | Source pipeline stage (0–7)                           |
| pipeline_stage_dst   | int    | Destination pipeline stage (0–7)                      |
| dp_group_src         | int    | Source data-parallel group (0–3)                      |
| dp_group_dst         | int    | Destination data-parallel group (0–3)                 |

### File 2: compute_timeseries.csv
- Rows: 65,792
- Columns: 12

| Column               | Type   | Description                              |
|----------------------|--------|------------------------------------------|
| event_id             | string |                                          |
| rank                 | int    | GPU rank (0–127)                         |
|
