import json
import time
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from collections import defaultdict
import os
import datetime

# --- Constants and Enums ---

class ParallelType(Enum):
    TENSOR_PARALLEL = "TP"
    PIPELINE_PARALLEL = "PP"
    DATA_PARALLEL = "DP"
    SEQUENCE_PARALLEL = "SP"

class CollectiveType(Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"
    POINT_TO_POINT = "p2p"

class ComputeType(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"
    EMBEDDING = "embedding"
    ATTENTION = "attention"
    MLP = "mlp"
    LAYER_NORM = "layer_norm"

# --- Data Classes ---

@dataclass
class Message:
    msg_id: str
    src_rank: int
    dst_rank: int
    size_bytes: int
    collective_type: CollectiveType
    start_time: float
    end_time: float = 0.0
    stage: str = ""
    layer_id: int = -1
    
    def duration(self) -> float:
        return self.end_time - self.start_time

@dataclass
class ComputeEvent:
    event_id: str
    rank: int
    compute_type: ComputeType
    start_time: float
    end_time: float
    layer_id: int
    flop_count: int
    memory_accessed: int

@dataclass
class NetworkConfig:
    bandwidth_gbps: float = 800.0          # more modern (H200 / Blackwell era)
    latency_us: float = 0.6
    num_nics: int = 8
    topology: str = "fat_tree"
    
@dataclass
class GPUConfig:
    peak_tflops: float = 1979.0            # ~H200 FP8 / BF16 rough
    memory_gb: int = 141
    memory_bw_gbps: float = 4800.0
    
@dataclass
class ParallelConfig:
    tp_size: int = 4
    pp_size: int = 8
    dp_size: int = 4
    sp_size: int = 1
    micro_batch_size: int = 2
    num_microbatches: int = 16
    virtual_pp_size: int = 1
    
    def world_size(self) -> int:
        return self.tp_size * self.pp_size * self.dp_size

@dataclass
class ModelConfig:
    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    seq_length: int = 2048
    vocab_size: int = 50257
    ffn_hidden_size: int = 16384
    
    def total_params(self) -> int:
        return 8800000000

# --- Simulator Class ---

class AstraSimMegatronSimulator:
    def __init__(self, 
                 parallel_config: ParallelConfig,
                 model_config: ModelConfig,
                 network_config: NetworkConfig,
                 gpu_config: GPUConfig):
        self.para_cfg = parallel_config
        self.model_cfg = model_config
        self.net_cfg = network_config
        self.gpu_cfg = gpu_config
        
        self.current_time = 0.0
        self.messages: List[Message] = []
        self.compute_events: List[ComputeEvent] = []
        self.logs: List[Dict] = []
        
        self.rank_to_stage = {}
        self.setup_topology()
        
    def setup_topology(self):
        world_size = self.para_cfg.world_size()
        for rank in range(world_size):
            dp_group = rank // (self.para_cfg.tp_size * self.para_cfg.pp_size)
            stage_in_dp = (rank % (self.para_cfg.tp_size * self.para_cfg.pp_size)) // self.para_cfg.tp_size
            tp_rank = rank % self.para_cfg.tp_size
            
            self.rank_to_stage[rank] = {
                'dp_group': dp_group,
                'pp_stage': stage_in_dp,
                'tp_rank': tp_rank,
                'global_rank': rank
            }
    
    def calculate_communication_time(self, size_bytes: int, collective: CollectiveType, 
                                     num_ranks: int) -> float:
        size_gb = size_bytes / (1024 ** 3)
        bandwidth_gbps = self.net_cfg.bandwidth_gbps

        if collective == CollectiveType.POINT_TO_POINT:
            # Simple: latency + transfer time (one direction)
            latency_ms = self.net_cfg.latency_us / 1000
            transfer_ms = (size_gb * 8) / (bandwidth_gbps / 1000)
            return latency_ms + transfer_ms * 1.15  # slight overhead

        elif collective == CollectiveType.ALL_REDUCE:
            # Rough model: 2×reduce-scatter + all-gather like behavior
            base_latency_ms = 0.8
            alpha = 1.1 if num_ranks <= 8 else 1.4 + (num_ranks / 128) * 0.6
            transfer_cost = (size_gb * 8 * 2) / (bandwidth_gbps / 1000)  # 2x traffic
            return base_latency_ms + transfer_cost * alpha + random.uniform(-0.08, 0.12)

        else:
            # fallback
            return 0.5 + size_gb * 12.0

    def simulate_collective(self, collective_type: CollectiveType, 
                           size_bytes: int, 
                           participating_ranks: List[int],
                           stage: str = "",
                           layer_id: int = -1) -> List[Message]:
        messages = []
        base_time = self.current_time
        num_ranks = len(participating_ranks)
        duration = self.calculate_communication_time(size_bytes, collective_type, num_ranks)
        
        if collective_type == CollectiveType.ALL_REDUCE:
            # Ring-like modeling (simplified)
            for i, rank in enumerate(participating_ranks):
                next_rank = participating_ranks[(i + 1) % num_ranks]
                msg = Message(
                    msg_id=f"ar_{stage}_l{layer_id}_r{rank}_{base_time:.3f}",
                    src_rank=rank,
                    dst_rank=next_rank,
                    size_bytes=size_bytes,
                    collective_type=collective_type,
                    start_time=base_time,
                    end_time=base_time + duration,
                    stage=stage,
                    layer_id=layer_id
                )
                messages.append(msg)
        elif collective_type == CollectiveType.POINT_TO_POINT:
            # Simplified: pair-wise
            for i in range(0, len(participating_ranks)-1, 2):
                src = participating_ranks[i]
                dst = participating_ranks[i+1]
                msg = Message(
                    msg_id=f"p2p_{stage}_l{layer_id}_r{src}_to_{dst}_{base_time:.3f}",
                    src_rank=src,
                    dst_rank=dst,
                    size_bytes=size_bytes,
                    collective_type=collective_type,
                    start_time=base_time,
                    end_time=base_time + duration,
                    stage=stage,
                    layer_id=layer_id
                )
                messages.append(msg)
        
        self.messages.extend(messages)
        self.current_time += duration
        return messages

    def simulate_tensor_parallel_layer(self, rank: int, layer_id: int, is_forward: bool):
        start = self.current_time
        
        # Realistic durations (H200 GPU)
        if is_forward:
            attn_time = random.uniform(0.75, 1.05)      # ~0.9 s
            mlp_time  = random.uniform(2.1, 2.9)        # ~2.5 s
        else:
            attn_time = random.uniform(1.0, 1.4)        # backward heavier
            mlp_time  = random.uniform(2.8, 3.8)

        # Attention compute
        self.compute_events.append(ComputeEvent(
            event_id=f"attn_{'fwd' if is_forward else 'bwd'}_l{layer_id}_r{rank}",
            rank=rank,
            compute_type=ComputeType.ATTENTION,
            start_time=start,
            end_time=start + attn_time,
            layer_id=layer_id,
            flop_count=280_000_000_000,   # rough
            memory_accessed=1_400_000_000
        ))
        self.current_time = start + attn_time

        # TP all-reduce after attention
        stage_info = self.rank_to_stage[rank]
        tp_group = [r for r in range(self.para_cfg.world_size()) 
                    if self.rank_to_stage[r]['pp_stage'] == stage_info['pp_stage'] 
                    and self.rank_to_stage[r]['dp_group'] == stage_info['dp_group']]
        self.simulate_collective(CollectiveType.ALL_REDUCE, 4*1024*1024, tp_group,
                                f"attn_{'fwd' if is_forward else 'bwd'}", layer_id)

        # MLP compute
        mlp_start = self.current_time
        self.compute_events.append(ComputeEvent(
            event_id=f"mlp_{'fwd' if is_forward else 'bwd'}_l{layer_id}_r{rank}",
            rank=rank,
            compute_type=ComputeType.MLP,
            start_time=mlp_start,
            end_time=mlp_start + mlp_time,
            layer_id=layer_id,
            flop_count=820_000_000_000,
            memory_accessed=2_800_000_000
        ))
        self.current_time = mlp_start + mlp_time

        # TP all-reduce after MLP (usually larger)
        self.simulate_collective(CollectiveType.ALL_REDUCE, 16*1024*1024, tp_group,
                                f"mlp_{'fwd' if is_forward else 'bwd'}", layer_id)

    def simulate_pipeline_forward(self, mb: int):
        layers_per_stage = 4  # 32 layers / 8 stages
        for stage in range(self.para_cfg.pp_size):
            for rank in range(self.para_cfg.world_size()):
                if self.rank_to_stage[rank]['pp_stage'] == stage:
                    for l in range(layers_per_stage):
                        layer_id = stage * layers_per_stage + l
                        self.simulate_tensor_parallel_layer(rank, layer_id, True)
            
            # Pipeline bubble / P2P forward
            if stage < self.para_cfg.pp_size - 1:
                p2p_ranks = []
                for r in range(self.para_cfg.world_size()):
                    if self.rank_to_stage[r]['pp_stage'] in [stage, stage+1]:
                        p2p_ranks.append(r)
                # Activation size rough
                self.simulate_collective(CollectiveType.POINT_TO_POINT, 2*1024*1024*4,
                                       p2p_ranks[:2], "pp_forward", stage)

    def simulate_pipeline_backward(self, mb: int):
        layers_per_stage = 4
        for stage in range(self.para_cfg.pp_size - 1, -1, -1):
            for rank in range(self.para_cfg.world_size()):
                if self.rank_to_stage[rank]['pp_stage'] == stage:
                    for l in range(layers_per_stage):
                        layer_id = stage * layers_per_stage + l
                        self.simulate_tensor_parallel_layer(rank, layer_id, False)

    def simulate_optimizer_step(self):
        # Each DP-group all-reduce ~30–80 ms realistic
        # Total optimizer sync usually 2–12 seconds on this scale
        single_ar_duration = random.uniform(0.035, 0.085)
        
        for l in range(759):
            group_idx = 0
            for pp in range(self.para_cfg.pp_size):
                for tp in range(self.para_cfg.tp_size):
                    if l == 758 and group_idx >= 8:
                        break
                    dp_group = [r for r in range(self.para_cfg.world_size())
                               if self.rank_to_stage[r]['pp_stage'] == pp
                               and self.rank_to_stage[r]['tp_rank'] == tp]
                    
                    size = 1316100 * 4
                    
                    self.current_time += single_ar_duration * random.uniform(0.92, 1.08)
                    
                    self.simulate_collective(
                        CollectiveType.ALL_REDUCE,
                        size,
                        dp_group,
                        "optimizer_grad_sync",
                        l
                    )
                    group_idx += 1

    def run_full_iteration(self, iteration: int, run_opt: bool = False):
        t0 = time.time()
        print(f"  Iteration {iteration} started at wall-clock {time.strftime('%H:%M:%S')}")
        
        for mb in range(self.para_cfg.num_microbatches):
            self.simulate_pipeline_forward(mb)
            self.simulate_pipeline_backward(mb)
        
        if run_opt:
            print("  Starting optimizer step...")
            self.simulate_optimizer_step()
        
        wall_elapsed = time.time() - t0
        sim_duration = self.current_time / 1000
        print(f"  Iteration {iteration} finished — sim time: {sim_duration:.1f}s  wall: {wall_elapsed:.1f}s")

if __name__ == "__main__":
    p_cfg, m_cfg = ParallelConfig(), ModelConfig()
    print(f"Configuration Summary:\n  World Size: {p_cfg.world_size()} GPUs\n  Model Parameters: {m_cfg.total_params()/1e9:.2f}B")
    sim = AstraSimMegatronSimulator(p_cfg, m_cfg, NetworkConfig(), GPUConfig())
    
    #Variable for current date/time for output file naming
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(2):
        print(f"\nRunning Iteration {i}...")
        sim.run_full_iteration(i, run_opt=(i == 1))
    
    # Save Messages CSV to traces folder
    os.makedirs("Traces", exist_ok=True)
    msg_data = []
    for m in sim.messages:
        msg_data.append({
            'message_id': m.msg_id,
            'src_rank': m.src_rank,
            'dst_rank': m.dst_rank,
            'size_bytes': m.size_bytes,
            'size_mb': m.size_bytes/1e6,
            'collective_type': m.collective_type.value,
            'start_time_ms': m.start_time,
            'end_time_ms': m.end_time,
            'duration_ms': m.duration(),
            'stage': m.stage,
            'layer_id': m.layer_id,
            'pipeline_stage_src': sim.rank_to_stage[m.src_rank]['pp_stage'],
            'pipeline_stage_dst': sim.rank_to_stage[m.dst_rank]['pp_stage'],
            'dp_group_src': sim.rank_to_stage[m.src_rank]['dp_group'],
            'dp_group_dst': sim.rank_to_stage[m.dst_rank]['dp_group']
        })
        
    pd.DataFrame(msg_data).to_csv(f'Traces/messages_timeseries{now}.csv', index=False)

    # Save Compute CSV to traces folder
    os.makedirs("Traces", exist_ok=True)
    compute_data = []
    for e in sim.compute_events:
        compute_data.append({
            'event_id': e.event_id,
            'rank': e.rank,
            'compute_type': e.compute_type.value,
            'start_time_ms': e.start_time,
            'end_time_ms': e.end_time,
            'duration_ms': e.end_time - e.start_time,
            'layer_id': e.layer_id,
            'flop_count': e.flop_count,
            'tflops': e.flop_count / 1e12,
            'memory_accessed_bytes': e.memory_accessed,
            'pipeline_stage': sim.rank_to_stage[e.rank]['pp_stage'],
            'dp_group': sim.rank_to_stage[e.rank]['dp_group'],
            'tp_rank': sim.rank_to_stage[e.rank]['tp_rank']
        })
    pd.DataFrame(compute_data).to_csv(f'Traces/compute_timeseries{now}.csv ', index=False)

    final_sim_time = sim.current_time / 1000
    print(f"\nSimulation Complete.")
    print(f"  Final simulated time: {final_sim_time:.1f} seconds")
    print(f"  Total Messages: {len(sim.messages):,}")
