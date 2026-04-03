import sys
import os
import math
from typing import List
from sim import (
    MegatronSimulator, ParallelConfig, ModelConfig, 
    NetworkConfig, GPUConfig, ParallelType, CollectiveType
)

def run_validation(pp_size: int, num_microbatches: int):
    tp_size = 8
    dp_size = 1

    num_layers = 8 * pp_size

    # Model config (GPT-3 175B style or similar to Megatron-LM study)
    m_cfg = ModelConfig(
        num_layers=num_layers,
        hidden_size=15360,
        num_attention_heads=128,
        seq_length=2048,
        vocab_size=51200,
        ffn_hidden_size=4 * 15360,
        activation_checkpoint_ratio=1.0 
    )

    p_cfg = ParallelConfig(
        tp_size=tp_size,
        pp_size=pp_size,
        dp_size=dp_size,
        micro_batch_size=1,
        num_microbatches=num_microbatches,
        schedule_type='1F1B'
    )

    net_cfg = NetworkConfig(
        nvlink_bw=600.0,
        infiniband_bw=50.0,
        num_gpus_per_node=8,
        overlap_factor=0.8
    )

    gpu_cfg = GPUConfig(
        peak_tflops=312.0,
        compute_efficiency=0.57, # Tweaked to match Megatron-LM's observed efficiency
        dtype_factor=1.0,
        nvlink_bw=600.0,
        thermal_throttle_factor=1.0
    )

    sim = MegatronSimulator(p_cfg, m_cfg, net_cfg, gpu_cfg)
    sim.run_full_iteration(0)
    total_flops = sum(e.flop_count for e in sim.compute_events)
    total_time = sim.current_time
    num_gpus = tp_size * pp_size * dp_size
    tflops_per_gpu = total_flops / (total_time * num_gpus * 1e12)

    return tflops_per_gpu

if __name__ == "__main__":
    pp_stages = [1, 2, 4, 8]
    batch_sizes = [8, 128]

    print(f"{'PP':<5} {'Batch':<10} {'TFLOPS/GPU':<15}")
    print("-" * 35)
    
    results = {}
    for bs in batch_sizes:
        results[bs] = []
        for pp in pp_stages:
            tflops = run_validation(pp, bs)
            results[bs].append(tflops)
            print(f"{pp:<5} {bs:<10} {tflops:<15.2f}")

