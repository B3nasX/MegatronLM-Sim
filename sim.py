import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
from collections import defaultdict
import os
import csv
import json
import datetime
import math

#TODO: Optimizer step X, Activation checkpointing X, Output layers? X, Gradient clipping? X

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
    OUTPUT = "output"
    GRAD_CLIP = "grad_clip"
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
    participating_ranks: str = ""  # comma-separated list of all ranks involved
    
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
    flop_count: int = 0
    memory_accessed: int = 0

@dataclass
class NetworkConfig:
    # Bandwidths in GB/s
    nvlink_bw: float = 600.0
    infiniband_bw: float = 50.0
    
    latency_nvlink_us: float = 0.5
    latency_ib_us: float = 2.0
    
    num_gpus_per_node: int = 8
    topology: str = "fat_tree"
    overlap_factor: float = 0.8
    collective_algo: str = 'auto'
    
    # Fat tree topology parameters
    nodes_per_rack: int = 4                  # Physical nodes per rack
    oversubscription_ratio: float = 2.0      # Spine oversubscription (1.0 = full bisection)
    latency_cross_rack_us: float = 5.0       # Cross-rack latency (legacy, used as fallback)
    
    # --- NIC and PCIe limits ---
    nic_bw_gbps: float = 50.0               # Per-GPU NIC bandwidth (400 Gb/s NDR = 50 GB/s)
    num_nics_per_gpu: int = 1                # NICs per GPU (rail-optimized = 1 dedicated NIC)
    pcie_bw_gbps: float = 64.0              # PCIe Gen5 x16 bandwidth 
    
    # --- Protocol overhead ---
    header_bytes: int = 128                  # Per-message protocol/header overhead
    chunk_size_bytes: int = 8_388_608        # NCCL chunk/pipeline size (8 MB default)
    
    # --- Bidirectional links ---
    bidirectional: bool = True               # Whether links are full-duplex
    
    # --- Per-hop switch latency ---
    switch_latency_us: float = 0.3           # Latency added per switch hop
    num_hops_intra_rack: int = 1             # Hops within a rack (ToR only)
    num_hops_cross_rack: int = 3             # Hops across racks (ToR -> Spine -> ToR)
    
    # --- Algorithm tuning ---
    ring_tree_threshold_bytes: int = 1_048_576  # Auto algo: ring above this, tree below
    
    # --- Jitter ---
    latency_jitter_us: float = 0.5           # Random +/- jitter per message (microseconds)
    
    # --- Congestion ---
    congestion_penalty_factor: float = 0.1   # BW degradation per additional concurrent flow (0.1 = 10% loss per flow)

    @classmethod
    def from_json(cls, path: str) -> 'NetworkConfig':
        """Load a NetworkConfig from a JSON file. Unknown keys (like 'name', 'description') are ignored."""
        with open(path, 'r') as f:
            data = json.load(f)
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

@dataclass
class GPUConfig:
    peak_tflops: float = 989.0       # Peak TFLOPS 
    memory_gb: int = 80              # HBM capacity
    memory_bw_gbps: float = 3350.0   # HBM bandwidth
    
    compute_efficiency: float = 0.45  # Model FLOP Utilization (MFU). Real-world: 30-60%
    dtype_factor: float = 1.0         # Multiplier for precision: 1.0=BF16, 2.0=FP8, 4.0=FP4
    
    nvlink_bw: float = 900.0          # NVLink bandwidth in GB/s (V100=300, A100=600, H100=900)
    
    num_sms: int = 132                # Streaming multiprocessors (affects compute/comm overlap)
    l2_cache_mb: float = 50.0         # L2 cache size in MB (affects memory-bound kernels)
    
    tdp_watts: int = 700              # TDP in watts. Sustained load may throttle ~5-10%
    thermal_throttle_factor: float = 0.95  # Sustained throughput as fraction of peak (0.95 = 5% throttle)

    @classmethod
    def from_json(cls, path: str) -> 'GPUConfig':
        """Load a GPUConfig from a JSON file. Unknown keys (like 'name', 'description') are ignored."""
        with open(path, 'r') as f:
            data = json.load(f)
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
    
@dataclass
class ParallelConfig:
    tp_size: int = 4
    pp_size: int = 8
    dp_size: int = 4
    sp_size: int = 1
    micro_batch_size: int = 2
    num_microbatches: int = 16
    virtual_pp_size: int = 1
    schedule_type: str = '1F1B'
    
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
    activation_checkpoint_ratio: float = 0.5  # Fraction of layers checkpointed (0.0 = off)
    
    def total_params(self) -> int:
        return 8800000000

# --- Simulator Class ---

class MegatronSimulator:
    def __init__(self, 
                 parallel_config: ParallelConfig,
                 model_config: ModelConfig,
                 network_config: NetworkConfig,
                 gpu_config: GPUConfig):
        self.para_cfg = parallel_config
        self.model_cfg = model_config
        self.net_cfg = network_config
        self.gpu_cfg = gpu_config
        
        # Sync NVLink bandwidth: the GPU determines the actual NVLink speed
        # If the GPU uses NVLink, override the network config value
        # If nvlink_bw = 0, fall back to PCIe bandwidth for intra-node
        if gpu_config.nvlink_bw > 0:
            self.net_cfg.nvlink_bw = gpu_config.nvlink_bw
        else:
            # PCIe GPUs: intra-node communication goes over PCIe, not NVLink
            self.net_cfg.nvlink_bw = gpu_config.memory_bw_gbps * 0.5 
        
        self.rank_times = defaultdict(float)
        self.messages: List[Message] = []
        self.compute_events: List[ComputeEvent] = []
        self.task_completion_times = {} 
        
        # Congestion tracking: count of active network flows at any point
        # Incremented before comm, decremented after comm completes
        self._active_flows: int = 0
        
        self.rank_to_stage = {}
        self.setup_topology()
        
    def setup_topology(self):
        world_size = self.para_cfg.world_size()
        for rank in range(world_size):
            dp_group = rank // (self.para_cfg.tp_size * self.para_cfg.pp_size)
            stage_in_dp = (rank % (self.para_cfg.tp_size * self.para_cfg.pp_size)) // self.para_cfg.tp_size
            tp_rank = rank % self.para_cfg.tp_size
            node_id = rank // self.net_cfg.num_gpus_per_node
            rack_id = node_id // self.net_cfg.nodes_per_rack
            
            self.rank_to_stage[rank] = {
                'dp_group': dp_group,
                'pp_stage': stage_in_dp,
                'tp_rank': tp_rank,
                'global_rank': rank,
                'node_id': node_id,
                'rack_id': rack_id
            }

        # Pre-compute rank groups to avoid repeated list comprehensions
        # _stage_ranks[stage] = all ranks at that pipeline stage (across all DP groups)
        self._stage_ranks = defaultdict(list)
        # _stage_dp_groups[stage][dp_id] = list of TP ranks in that DP group at that stage
        self._stage_dp_groups = defaultdict(lambda: defaultdict(list))
        for rank, info in self.rank_to_stage.items():
            self._stage_ranks[info['pp_stage']].append(rank)
            self._stage_dp_groups[info['pp_stage']][info['dp_group']].append(rank)

        # Pre-compute DP groups keyed by (pp_stage, tp_rank) for DP all-reduce
        self._dp_allreduce_groups = defaultdict(list)
        for rank, info in self.rank_to_stage.items():
            self._dp_allreduce_groups[(info['pp_stage'], info['tp_rank'])].append(rank)

        # Pre-compute whether each rank group is same-node (for comm time)
        self._same_node_cache = {}

    def _compute_duration(self, flop_count: float, memory_bytes: float) -> float:
        """Estimate compute duration using compute vs memory roofline with jitter."""
        gpu = self.gpu_cfg
        effective_tflops = (
            gpu.peak_tflops
            * gpu.dtype_factor
            * gpu.compute_efficiency
            * gpu.thermal_throttle_factor
        )
        compute_bound = flop_count / (effective_tflops * 1e12)
        memory_bound = memory_bytes / (gpu.memory_bw_gbps * 1e9)
        base = max(compute_bound, memory_bound)
        jitter = random.uniform(-0.05, 0.05) * base
        return max(base + jitter, 1e-6)

    def _record_compute_event(
        self,
        ranks: List[int],
        compute_type: ComputeType,
        start_time: float,
        duration: float,
        layer_id: int,
        flop_count: int,
        memory_accessed: int,
        prefix: str,
    ) -> float:
        end_time = start_time + duration
        for r in ranks:
            self.compute_events.append(
                ComputeEvent(
                    event_id=f"{prefix}_l{layer_id}_r{r}",
                    rank=r,
                    compute_type=compute_type,
                    start_time=start_time,
                    end_time=end_time,
                    layer_id=layer_id,
                    flop_count=flop_count,
                    memory_accessed=memory_accessed,
                )
            )
            self.rank_times[r] = max(self.rank_times[r], end_time)
        return end_time
    """
    Computes communication time with full accuracy model:
      1. 3-tier bandwidth: NVLink / intra-rack IB / cross-rack IB (with oversubscription)
      2. Per-hop switch latency instead of flat cross-rack latency
      3. NIC bandwidth bottleneck (per-GPU NIC can't exceed nic_bw)
      4. PCIe bottleneck (GPU <-> NIC limited by PCIe Gen5)
      5. Protocol overhead: header bytes per chunk + chunked pipelining
      6. Bidirectional link awareness
      7. Congestion penalty from concurrent flows sharing links
      8. Latency jitter for realistic variance
      9. Configurable ring/tree algorithm threshold
    """
    def calculate_communication_time(self, size_bytes: int, collective: CollectiveType, 
                                     participating_ranks: List[int]) -> float:
        num_ranks = len(participating_ranks)
        if num_ranks <= 1:
            return 0.0

        net = self.net_cfg

        # --- 1. Determine tier: same-node / same-rack / cross-rack ---
        nodes = set(self.rank_to_stage[r]['node_id'] for r in participating_ranks)
        racks = set(self.rank_to_stage[r]['rack_id'] for r in participating_ranks)
        same_node = len(nodes) == 1
        same_rack = len(racks) == 1

        if same_node:
            link_bw = net.nvlink_bw                                     # GB/s
            base_lat = net.latency_nvlink_us / 1_000_000.0              # seconds
            num_hops = 0                                                # NVSwitch, no switch hops
        elif same_rack:
            link_bw = net.infiniband_bw                                 # GB/s
            base_lat = net.latency_ib_us / 1_000_000.0
            num_hops = net.num_hops_intra_rack
        else:
            link_bw = net.infiniband_bw / net.oversubscription_ratio    # GB/s (reduced at spine)
            base_lat = net.latency_ib_us / 1_000_000.0                 # base wire latency
            num_hops = net.num_hops_cross_rack

        # --- 2. Per-hop switch latency ---
        hop_lat = num_hops * (net.switch_latency_us / 1_000_000.0)     # seconds
        total_lat = base_lat + hop_lat

        # --- 3. Latency jitter ---
        jitter = random.uniform(-net.latency_jitter_us, net.latency_jitter_us) / 1_000_000.0
        total_lat = max(0.0, total_lat + jitter)

        # --- 4. Bandwidth: min of link BW, NIC BW, PCIe BW ---
        # NIC: each GPU has num_nics_per_gpu NICs of nic_bw_gbps each
        nic_bw = net.nic_bw_gbps * net.num_nics_per_gpu                # GB/s per GPU
        pcie_bw = net.pcie_bw_gbps                                     # GB/s
        effective_bw = min(link_bw, nic_bw, pcie_bw)                   # GB/s

        # --- 5. Bidirectional: for collectives that use both directions simultaneously ---
        # All-reduce uses bidirectional (reduce-scatter + all-gather); P2P uses one direction
        if net.bidirectional and collective != CollectiveType.POINT_TO_POINT:
            effective_bw = effective_bw  # already full-duplex, no change
        elif not net.bidirectional and collective != CollectiveType.POINT_TO_POINT:
            # Half-duplex: bidirectional collectives share the link
            effective_bw = effective_bw * 0.5

        # --- 6. Congestion penalty from concurrent flows ---
        if self._active_flows > 1 and not same_node:
            # Each additional concurrent flow degrades shared link bandwidth
            # Formula: bw_effective = bw / (1 + penalty * (num_flows - 1))
            congestion_divisor = 1.0 + net.congestion_penalty_factor * (self._active_flows - 1)
            effective_bw = effective_bw / congestion_divisor

        # Prevent zero/negative bandwidth
        effective_bw = max(effective_bw, 0.001)

        # --- 7. Protocol overhead: header per chunk + chunked pipelining ---
        # Total data includes header overhead per chunk
        num_chunks = max(1, math.ceil(size_bytes / net.chunk_size_bytes))
        total_bytes_with_overhead = size_bytes + (num_chunks * net.header_bytes)
        size_gb = total_bytes_with_overhead / (1024 ** 3)

        # --- 8. Compute transfer time based on collective type ---
        if collective == CollectiveType.POINT_TO_POINT:
            # Simple: latency + data / bandwidth
            transfer_time = size_gb / effective_bw
            return total_lat + transfer_time

        elif collective == CollectiveType.ALL_REDUCE:
            algo = net.collective_algo
            if algo == 'auto':
                algo = 'ring' if size_bytes > net.ring_tree_threshold_bytes else 'tree'

            if algo == 'ring':
                # Ring all-reduce: 2(n-1) latency steps, 2(n-1)/n bandwidth cost
                lat_cost = 2.0 * (num_ranks - 1) * total_lat
                bw_cost = 2.0 * (num_ranks - 1) / num_ranks * size_gb / effective_bw
            else:  # tree
                log_n = math.log2(max(num_ranks, 2))
                lat_cost = 2.0 * log_n * total_lat
                bw_cost = 2.0 * log_n * size_gb / effective_bw

            # Chunked pipelining: with many chunks, latency and BW overlap
            # Approximation: full latency for first chunk, then BW-dominated
            if num_chunks > 1:
                # Pipeline effect: total ≈ latency_startup + bw_cost (chunks overlap latency)
                return lat_cost + bw_cost
            else:
                return lat_cost + bw_cost

        # Fallback heuristic for other collectives
        return total_lat * num_ranks + size_gb / effective_bw
    """
    Simulates compute for a TP layer, including all-reduce.
    Compute time is derived from the GPU config:
      1. FLOPs per layer calculated from model dimensions and TP split
      2. Compute-bound time = flops / (effective TFLOPS with MFU, dtype, thermal throttle)
      3. Memory-bound time = memory_accessed / memory_bandwidth
      4. Actual time = max(compute-bound, memory-bound) + jitter
    All-reduce time is calculated and can overlap with compute.
    """
    def simulate_tp_layer(self, ranks: List[int], layer_id: int, is_forward: bool, start_time: float) -> float:
        gpu = self.gpu_cfg
        model = self.model_cfg
        tp = self.para_cfg.tp_size

        B = self.para_cfg.micro_batch_size
        S = model.seq_length
        H = model.hidden_size
        FFN = model.ffn_hidden_size

        # FLOPs for full layer (forward), split across TP ranks
        layer_flops_fwd = (8 * B * S * H * H + 4 * B * S * H * FFN) // tp

        bytes_per_param = 2  # BF16
        weight_bytes = ((8 * H * H + 4 * H * FFN) * bytes_per_param) // tp
        activation_bytes = B * S * H * bytes_per_param * 4
        memory_accessed_fwd = weight_bytes + activation_bytes

        current_start = start_time
        checkpointed = False
        if self.model_cfg.activation_checkpoint_ratio > 0:
            interval = max(1, int(1 / max(1e-6, self.model_cfg.activation_checkpoint_ratio)))
            checkpointed = (layer_id % interval) == 0

        # --- Optional activation recompute during backward ---
        if checkpointed and not is_forward:
            recompute_duration = self._compute_duration(layer_flops_fwd, memory_accessed_fwd)
            current_start = self._record_compute_event(
                ranks,
                ComputeType.FORWARD,
                current_start,
                recompute_duration,
                layer_id,
                layer_flops_fwd,
                memory_accessed_fwd,
                prefix="recompute",
            )

        # --- Main compute ---
        flop_count = layer_flops_fwd if is_forward else layer_flops_fwd * 2
        memory_accessed = memory_accessed_fwd if is_forward else int(memory_accessed_fwd * 2.5)
        comp_duration = self._compute_duration(flop_count, memory_accessed)
        comp_end = self._record_compute_event(
            ranks,
            ComputeType.FORWARD if is_forward else ComputeType.BACKWARD,
            current_start,
            comp_duration,
            layer_id,
            flop_count,
            memory_accessed,
            prefix="fwd" if is_forward else "bwd",
        )
        
        # TP all-reduce: 2 * hidden^2 per layer
        # With sequence parallelism overhead: scale by seq_length
        ar_size = 2 * H * H * bytes_per_param + B * S * H * bytes_per_param
        # Track congestion: this TP group starts an all-reduce flow
        self._active_flows += 1
        ar_duration = self.calculate_communication_time(ar_size, CollectiveType.ALL_REDUCE, ranks)
        self._active_flows -= 1
        ar_end = start_time + ar_duration

        # effective_ar_duration is seconds overlap_factor reduces active comm time
        # More SMs = better overlap (can dedicate SMs to comm while others compute)
        sm_overlap_bonus = min(gpu.num_sms / 132.0, 1.2)  # normalize to H100's 132 SMs, cap at 1.2x
        effective_overlap = min(self.net_cfg.overlap_factor * sm_overlap_bonus, 0.95)  # cap at 95%
        effective_ar_duration = ar_duration * (1.0 - effective_overlap)
        rank_end = max(comp_end, start_time + effective_ar_duration)

        # Emit ONE message per logical all-reduce (not per rank)
        # Use first rank as representative src; participating_ranks tracks all members
        self.messages.append(Message(
            msg_id=f"ar_l{layer_id}_tp{'_'.join(str(r) for r in ranks)}",
            src_rank=ranks[0],
            dst_rank=-1,
            size_bytes=ar_size,
            collective_type=CollectiveType.ALL_REDUCE,
            start_time=start_time,
            end_time=ar_end,
            stage="tp_all_reduce",
            layer_id=layer_id,
            participating_ranks=','.join(str(r) for r in ranks)
        ))
        for r in ranks:
            self.rank_times[r] = max(self.rank_times[r], rank_end)

        return rank_end

    def simulate_embedding_layer(self, ranks: List[int], layer_id: int, is_forward: bool, start_time: float) -> float:
        """Simulate embeddings (shared on stage 0) assuming vocab-parallel shards across TP ranks."""
        model = self.model_cfg
        tp = self.para_cfg.tp_size
        B = self.para_cfg.micro_batch_size
        S = model.seq_length
        H = model.hidden_size
        vocab_shard = max(1, model.vocab_size // tp)
        bytes_per_param = 2

        flop_count = 2 * B * S * H
        weight_bytes = vocab_shard * H * bytes_per_param
        activation_bytes = B * S * H * bytes_per_param
        memory_accessed = weight_bytes + activation_bytes
        if not is_forward:
            flop_count *= 2
            memory_accessed = int(memory_accessed * 1.5)

        duration = self._compute_duration(flop_count, memory_accessed)
        return self._record_compute_event(
            ranks,
            ComputeType.EMBEDDING,
            start_time,
            duration,
            layer_id,
            flop_count,
            memory_accessed,
            prefix="embed_fwd" if is_forward else "embed_bwd",
        )

    def simulate_output_layer(self, ranks: List[int], layer_id: int, is_forward: bool, start_time: float) -> float:
        """Simulate final LM head / output projection on last PP stage."""
        model = self.model_cfg
        tp = self.para_cfg.tp_size
        B = self.para_cfg.micro_batch_size
        S = model.seq_length
        H = model.hidden_size
        vocab_shard = max(1, model.vocab_size // tp)
        bytes_per_param = 2

        # Dense projection to vocab shard
        flop_count = 2 * B * S * H * vocab_shard
        weight_bytes = vocab_shard * H * bytes_per_param
        activation_bytes = B * S * H * bytes_per_param
        memory_accessed = weight_bytes + activation_bytes
        if not is_forward:
            flop_count *= 2
            memory_accessed = int(memory_accessed * 1.5)

        duration = self._compute_duration(flop_count, memory_accessed)
        return self._record_compute_event(
            ranks,
            ComputeType.OUTPUT,
            start_time,
            duration,
            layer_id,
            flop_count,
            memory_accessed,
            prefix="out_fwd" if is_forward else "out_bwd",
        )

    """
    Simulates a stage of the pipeline for a given microbatch and direction.
    Determines which ranks are in this stage and their dependencies.
    For forward, depends on previous stage's forward completion. For backward, depends on next stage's backward and current stage's forward.
    Computes the start time based on dependencies and simulates each layer in the stage sequentially, updating rank times and recording messages for inter-stage communication.
    """
    def simulate_stage(self, mb: int, stage: int, is_forward: bool):
        # Use pre-computed rank groups
        all_ranks_at_stage = self._stage_ranks[stage]
        dp_groups_map = self._stage_dp_groups[stage]
        
        dep_times = [max([self.rank_times[r] for r in all_ranks_at_stage])]
        
        if is_forward:
            if stage > 0:
                dep_times.append(self.task_completion_times.get((mb, 'fwd', stage-1), 0))
        else:
            if stage < self.para_cfg.pp_size - 1:
                dep_times.append(self.task_completion_times.get((mb, 'bwd', stage+1), 0))
            dep_times.append(self.task_completion_times.get((mb, 'fwd', stage), 0))
            
        start_time = max(dep_times)
        
        layers_per_stage = self.model_cfg.num_layers // self.para_cfg.pp_size

        # Simulate compute + TP all-reduce for each DP group independently
        dp_end_times = []
        for dp_id, tp_ranks in dp_groups_map.items():
            current_t = start_time
            if stage == 0:
                current_t = self.simulate_embedding_layer(tp_ranks, -1, is_forward, current_t)
            for l in range(layers_per_stage):
                layer_id = stage * layers_per_stage + l
                current_t = self.simulate_tp_layer(tp_ranks, layer_id, is_forward, current_t)
            if stage == self.para_cfg.pp_size - 1:
                current_t = self.simulate_output_layer(tp_ranks, self.model_cfg.num_layers, is_forward, current_t)
            dp_end_times.append(current_t)
        
        # All DP groups finish at approximately the same time 
        current_t = max(dp_end_times)
            
        p2p_duration = 0
        # Inter-stage activations: micro_batch * seq_len * hidden * sizeof(bf16)
        # = 2 * 2048 * 4096 * 2 = ~32 MB, with overhead 64 MB
        p2p_size = 64 * 1024 * 1024  # 64 MB
        if is_forward and stage < self.para_cfg.pp_size - 1:
            next_stage_ranks = self._stage_ranks[stage + 1]
            self._active_flows += 1
            p2p_duration = self.calculate_communication_time(p2p_size, CollectiveType.POINT_TO_POINT, [all_ranks_at_stage[0], next_stage_ranks[0]])
            self._active_flows -= 1
            # Emit ONE P2P message per DP group (one rank sends activations to next stage)
            for dp_id, tp_ranks in dp_groups_map.items():
                next_dp_ranks = self._stage_dp_groups[stage + 1][dp_id]
                self.messages.append(Message(
                    msg_id=f"p2p_fwd_mb{mb}_s{stage}_dp{dp_id}",
                    src_rank=tp_ranks[0],
                    dst_rank=next_dp_ranks[0],
                    size_bytes=p2p_size,
                    collective_type=CollectiveType.POINT_TO_POINT,
                    start_time=current_t,
                    end_time=current_t + p2p_duration,
                    stage="pp_forward",
                    layer_id=stage,
                    participating_ranks=f"{tp_ranks[0]},{next_dp_ranks[0]}"
                ))
        elif not is_forward and stage > 0:
            prev_stage_ranks = self._stage_ranks[stage - 1]
            self._active_flows += 1
            p2p_duration = self.calculate_communication_time(p2p_size, CollectiveType.POINT_TO_POINT, [all_ranks_at_stage[0], prev_stage_ranks[0]])
            self._active_flows -= 1
            # Emit ONE P2P message per DP group (one rank sends gradients to prev stage)
            for dp_id, tp_ranks in dp_groups_map.items():
                prev_dp_ranks = self._stage_dp_groups[stage - 1][dp_id]
                self.messages.append(Message(
                    msg_id=f"p2p_bwd_mb{mb}_s{stage}_dp{dp_id}",
                    src_rank=tp_ranks[0],
                    dst_rank=prev_dp_ranks[0],
                    size_bytes=p2p_size,
                    collective_type=CollectiveType.POINT_TO_POINT,
                    start_time=current_t,
                    end_time=current_t + p2p_duration,
                    stage="pp_backward",
                    layer_id=stage,
                    participating_ranks=f"{tp_ranks[0]},{prev_dp_ranks[0]}"
                ))
            
        p2p_end = current_t + p2p_duration
        self.task_completion_times[(mb, 'fwd' if is_forward else 'bwd', stage)] = p2p_end
        for r in all_ranks_at_stage:
            self.rank_times[r] = max(self.rank_times[r], p2p_end)
    """
    Runs the GPipe schedule: all forward microbatches through all stages, then all backward microbatches in reverse stage order.
    """
    def run_gpipe(self):
        for mb in range(self.para_cfg.num_microbatches):
            for stage in range(self.para_cfg.pp_size):
                self.simulate_stage(mb, stage, True)
        for mb in range(self.para_cfg.num_microbatches):
            for stage in range(self.para_cfg.pp_size - 1, -1, -1):
                self.simulate_stage(mb, stage, False)
    """
    Runs the 1F1B schedule: interleaves forward and backward microbatches across stages to maximize overlap.
    For each stage, it first schedules the warmup forward microbatches that only depend on previous stages.
    Then it schedules the steady-state microbatches where each forward depends on the previous stage's forward and each backward depends on the next stage's backward and current stage's forward.
    Finally, it schedules the cooldown backward microbatches that only depend on the next stage's backward.
    It uses a loop to check for ready tasks across all stages and simulates them as they become ready, ensuring correct dependencies are respected.
    This approach allows for better utilization of resources by overlapping computation and communication across stages.
    """
    def run_1f1b(self):
        num_mb = self.para_cfg.num_microbatches
        num_stages = self.para_cfg.pp_size
        schedule = []
        for stage in range(num_stages):
            stage_tasks = []
            warmup_steps = num_stages - stage - 1
            for mb in range(min(warmup_steps, num_mb)):
                stage_tasks.append((mb, stage, True))
            for mb in range(num_mb - warmup_steps):
                stage_tasks.append((mb + warmup_steps, stage, True))
                stage_tasks.append((mb, stage, False))
            for mb in range(num_mb - warmup_steps, num_mb):
                stage_tasks.append((mb, stage, False))
            schedule.append(stage_tasks)
            
        stage_task_idx = [0] * num_stages
        total_tasks = sum(len(s) for s in schedule)
        completed_tasks = 0
        while completed_tasks < total_tasks:
            progress = False
            for stage in range(num_stages):
                if stage_task_idx[stage] < len(schedule[stage]):
                    mb, st, is_fwd = schedule[stage][stage_task_idx[stage]]
                    can_run = True
                    if is_fwd:
                        if stage > 0 and (mb, 'fwd', stage-1) not in self.task_completion_times:
                            can_run = False
                    else:
                        if stage < num_stages - 1 and (mb, 'bwd', stage+1) not in self.task_completion_times:
                            can_run = False
                        if (mb, 'fwd', stage) not in self.task_completion_times:
                            can_run = False
                    if can_run:
                        self.simulate_stage(mb, st, is_fwd)
                        stage_task_idx[stage] += 1
                        completed_tasks += 1
                        progress = True
            if not progress: break

    def run_full_iteration(self, iteration: int):
        t0 = time.time()
        if self.para_cfg.schedule_type == '1F1B':
            self.run_1f1b()
        else:
            self.run_gpipe()

        # After all backward passes, each DP group must synchronize gradients.
        # Size: model_params / (tp_size * pp_size) * sizeof(bf16) per rank
        # For 8B params, tp=4, pp=8: 8.8B / 32 * 2 bytes ≈ 550 MB per rank
        dp_ar_size = 512 * 1024 * 1024  # 512 MB gradient all-reduce
        dp_size = self.para_cfg.dp_size

        if dp_size > 1:
            # All DP groups do gradient all-reduce simultaneously 
            num_dp_groups = len(self._dp_allreduce_groups)
            self._active_flows += num_dp_groups
            for (pp_stage, tp_rank), dp_ranks in self._dp_allreduce_groups.items():
                dp_start = max(self.rank_times[r] for r in dp_ranks)
                dp_ar_duration = self.calculate_communication_time(
                    dp_ar_size, CollectiveType.ALL_REDUCE, dp_ranks)

                dp_ar_end = dp_start + dp_ar_duration

                # Emit ONE message per logical DP all-reduce group
                self.messages.append(Message(
                    msg_id=f"dp_ar_iter{iteration}_pp{pp_stage}_tp{tp_rank}",
                    src_rank=dp_ranks[0],
                    dst_rank=-1,
                    size_bytes=dp_ar_size,
                    collective_type=CollectiveType.ALL_REDUCE,
                    start_time=dp_start,
                    end_time=dp_ar_end,
                    stage="dp_gradient_allreduce",
                    layer_id=-1,
                    participating_ranks=','.join(str(r) for r in dp_ranks)
                ))
                for r in dp_ranks:
                    self.rank_times[r] = max(self.rank_times[r], dp_ar_end)
            self._active_flows -= num_dp_groups

        # Gradient clipping norm all-reduce and optimizer step
        params_per_rank = self.model_cfg.total_params() / (self.para_cfg.tp_size * self.para_cfg.pp_size)
        param_bytes = params_per_rank * 2  # bf16

        for (pp_stage, tp_rank), dp_ranks in self._dp_allreduce_groups.items():
            group_start = max(self.rank_times[r] for r in dp_ranks)

            # Compute global grad norm (approximate as scan over params)
            grad_norm_flops = int(params_per_rank * 2)
            grad_norm_mem = int(param_bytes * 2)
            grad_norm_duration = self._compute_duration(grad_norm_flops, grad_norm_mem)
            grad_norm_end = self._record_compute_event(
                dp_ranks,
                ComputeType.GRAD_CLIP,
                group_start,
                grad_norm_duration,
                layer_id=-1,
                flop_count=grad_norm_flops,
                memory_accessed=grad_norm_mem,
                prefix=f"dp{pp_stage}_tp{tp_rank}_gradnorm",
            )

            # All-reduce of scalar grad norm across DP ranks (8 bytes)
            grad_norm_size = 8
            self._active_flows += 1
            grad_norm_ar = self.calculate_communication_time(grad_norm_size, CollectiveType.ALL_REDUCE, dp_ranks)
            self._active_flows -= 1
            grad_norm_ar_end = grad_norm_end + grad_norm_ar
            self.messages.append(
                Message(
                    msg_id=f"gradclip_ar_pp{pp_stage}_tp{tp_rank}",
                    src_rank=dp_ranks[0],
                    dst_rank=-1,
                    size_bytes=grad_norm_size,
                    collective_type=CollectiveType.ALL_REDUCE,
                    start_time=grad_norm_end,
                    end_time=grad_norm_ar_end,
                    stage="dp_grad_clip",
                    layer_id=-1,
                    participating_ranks=','.join(str(r) for r in dp_ranks),
                )
            )

            # Optimizer step (Adam-like ~8 FLOPs per parameter)
            opt_flops = int(params_per_rank * 8)
            opt_mem = int(param_bytes * 6)
            opt_duration = self._compute_duration(opt_flops, opt_mem)
            opt_end = self._record_compute_event(
                dp_ranks,
                ComputeType.OPTIMIZER,
                grad_norm_ar_end,
                opt_duration,
                layer_id=-1,
                flop_count=opt_flops,
                memory_accessed=opt_mem,
                prefix=f"optimizer_pp{pp_stage}_tp{tp_rank}",
            )
            for r in dp_ranks:
                self.rank_times[r] = max(self.rank_times[r], opt_end)

        self.current_time = max(self.rank_times.values())
        wall_elapsed = time.time() - t0
        # current_time is in seconds
        print(f"  Iteration {iteration} ({self.para_cfg.schedule_type}) finished — sim time: {self.current_time:.4f}s  wall: {wall_elapsed:.2f}s")

    def flush_to_csv(self, msg_path: str, compute_path: str, write_header: bool):
        """Write accumulated events to CSV with csv.writer, then clear lists to prevent memory throttle."""
        _r2s = self.rank_to_stage

        # --- Messages ---
        with open(msg_path, 'w' if write_header else 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['message_id','src_rank','dst_rank','size_bytes','size_mb',
                            'collective_type','start_time_ms','end_time_ms','duration_ms',
                            'stage','layer_id','pipeline_stage_src','pipeline_stage_dst',
                            'dp_group_src','dp_group_dst','participating_ranks'])
            for m in self.messages:
                sr, dr = m.src_rank, m.dst_rank
                w.writerow([
                    m.msg_id, sr, dr, m.size_bytes, m.size_bytes / 1e6,
                    m.collective_type.value,
                    m.start_time * 1000.0, m.end_time * 1000.0, m.duration() * 1000.0,
                    m.stage, m.layer_id,
                    _r2s[sr]['pp_stage'] if sr != -1 else -1,
                    _r2s[dr]['pp_stage'] if dr != -1 else -1,
                    _r2s[sr]['dp_group'] if sr != -1 else -1,
                    _r2s[dr]['dp_group'] if dr != -1 else -1,
                    m.participating_ranks
                ])

        # --- Compute ---
        with open(compute_path, 'w' if write_header else 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['event_id','rank','compute_type','start_time_ms','end_time_ms',
                            'duration_ms','layer_id','flop_count','tflops',
                            'memory_accessed_bytes','pipeline_stage','dp_group','tp_rank'])
            for e in self.compute_events:
                w.writerow([
                    e.event_id, e.rank, e.compute_type.value,
                    e.start_time * 1000.0, e.end_time * 1000.0,
                    (e.end_time - e.start_time) * 1000.0,
                    e.layer_id, e.flop_count, e.flop_count / 1e12,
                    e.memory_accessed,
                    _r2s[e.rank]['pp_stage'], _r2s[e.rank]['dp_group'], _r2s[e.rank]['tp_rank']
                ])

        # Free memory
        self.messages.clear()
        self.compute_events.clear()

if __name__ == "__main__":
    m_cfg = ModelConfig()

    # --- Parallelism configuration ---
    print("\n=== Parallelism Configuration ===")
    print(f"  Model: {m_cfg.num_layers} layers, hidden={m_cfg.hidden_size}, FFN={m_cfg.ffn_hidden_size}")
    print()
    presets = [
        {"name": "128 GPUs  (TP=4, PP=8, DP=4)",   "tp": 4, "pp": 8, "dp": 4, "mbs": 2, "nmb": 16},
        {"name": "64 GPUs   (TP=4, PP=4, DP=4)",    "tp": 4, "pp": 4, "dp": 4, "mbs": 2, "nmb": 16},
        {"name": "32 GPUs   (TP=4, PP=4, DP=2)",    "tp": 4, "pp": 4, "dp": 2, "mbs": 2, "nmb": 16},
        {"name": "16 GPUs   (TP=4, PP=2, DP=2)",    "tp": 4, "pp": 2, "dp": 2, "mbs": 2, "nmb": 16},
        {"name": "8 GPUs    (TP=4, PP=2, DP=1)",    "tp": 4, "pp": 2, "dp": 1, "mbs": 4, "nmb": 8},
        {"name": "8 GPUs    (TP=8, PP=1, DP=1)",    "tp": 8, "pp": 1, "dp": 1, "mbs": 4, "nmb": 8},
        {"name": "256 GPUs  (TP=4, PP=8, DP=8)",    "tp": 4, "pp": 8, "dp": 8, "mbs": 2, "nmb": 32},
        {"name": "512 GPUs  (TP=8, PP=8, DP=8)",    "tp": 8, "pp": 8, "dp": 8, "mbs": 1, "nmb": 64},
        {"name": "Custom",                           "tp": 0, "pp": 0, "dp": 0, "mbs": 0, "nmb": 0},
    ]
    for i, p in enumerate(presets):
        total = p["tp"] * p["pp"] * p["dp"]
        if p["name"] == "Custom":
            print(f"  [{i}] Custom (enter your own TP, PP, DP)")
        else:
            nodes = max(1, total // 8)
            print(f"  [{i}] {p['name']}  →  {total} GPUs = {nodes} nodes")
    print()
    p_choice = int(input(f"Select parallelism config [0-{len(presets)-1}]: "))
    preset = presets[p_choice]

    if preset["name"] == "Custom":
        print("\n  Enter parallelism dimensions:")
        tp = int(input("    TP size (tensor parallel, keep ≤ GPUs/node): "))
        pp = int(input("    PP size (pipeline parallel, must divide num_layers): "))
        dp = int(input("    DP size (data parallel): "))
        mbs = int(input("    Micro-batch size: "))
        nmb = int(input("    Number of microbatches: "))
        schedule = input("    Schedule [1F1B / GPipe] (default 1F1B): ").strip() or '1F1B'
    else:
        tp, pp, dp = preset["tp"], preset["pp"], preset["dp"]
        mbs, nmb = preset["mbs"], preset["nmb"]
        schedule = '1F1B'

    # Validate
    total_gpus = tp * pp * dp
    assert m_cfg.num_layers % pp == 0, f"PP={pp} must divide num_layers={m_cfg.num_layers}"
    assert tp <= 8 or input(f"  TP={tp} > 8 GPUs/node — this will use cross-node TP (slow). Continue? [y/N]: ").lower() == 'y'

    p_cfg = ParallelConfig(
        tp_size=tp, pp_size=pp, dp_size=dp,
        micro_batch_size=mbs, num_microbatches=nmb,
        schedule_type=schedule
    )
    num_nodes = max(1, total_gpus // 8)
    print(f"\n  Config: TP={tp}  PP={pp}  DP={dp} = {total_gpus} GPUs ({num_nodes} nodes)")
    print(f"    Schedule: {schedule} | Micro-batch: {mbs} | Microbatches: {nmb}")
    print(f"    Layers per PP stage: {m_cfg.num_layers // pp}")
    print(f"    Global batch size: {dp * mbs * nmb} samples")

    # --- Network topology selection ---
    net_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Network")
    net_files = sorted([f for f in os.listdir(net_dir) if f.endswith('.json')])
    
    if not net_files:
        print("No network config files found in Network/. Using defaults.")
        net_cfg = NetworkConfig()
    else:
        print("\n=== Available Network Topologies ===")
        for i, fname in enumerate(net_files):
            path = os.path.join(net_dir, fname)
            with open(path, 'r') as f:
                data = json.load(f)
            name = data.get('name', fname)
            desc = data.get('description', '')
            print(f"  [{i}] {name}")
            if desc:
                print(f"      {desc}")
        print()
        choice = int(input(f"Select network topology [0-{len(net_files)-1}]: "))
        net_path = os.path.join(net_dir, net_files[choice])
        net_cfg = NetworkConfig.from_json(net_path)
        print(f"  Loaded: {net_files[choice]}")
        print(f"  Topology: {net_cfg.topology} | IB BW: {net_cfg.infiniband_bw} GB/s | "
              f"Oversub: {net_cfg.oversubscription_ratio}:1 | Cross-rack latency: {net_cfg.latency_cross_rack_us} us")

    # --- GPU selection ---
    gpu_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPU")
    gpu_files = sorted([f for f in os.listdir(gpu_dir) if f.endswith('.json')])
    
    if not gpu_files:
        print("No GPU config files found in GPU/. Using defaults.")
        gpu_cfg = GPUConfig()
    else:
        print("\n=== Available GPUs ===")
        for i, fname in enumerate(gpu_files):
            path = os.path.join(gpu_dir, fname)
            with open(path, 'r') as f:
                data = json.load(f)
            name = data.get('name', fname)
            desc = data.get('description', '')
            mem = data.get('memory_gb', '?')
            tflops = data.get('peak_tflops', '?')
            print(f"  [{i}] {name}  ({mem} GB, {tflops} TFLOPS)")
            if desc:
                print(f"      {desc}")
        print()
        choice = int(input(f"Select GPU [0-{len(gpu_files)-1}]: "))
        gpu_path = os.path.join(gpu_dir, gpu_files[choice])
        gpu_cfg = GPUConfig.from_json(gpu_path)
        print(f"  Loaded: {gpu_files[choice]}")
        print(f"  TFLOPS: {gpu_cfg.peak_tflops} | Memory: {gpu_cfg.memory_gb} GB | "
              f"Mem BW: {gpu_cfg.memory_bw_gbps} GB/s")

    sim = MegatronSimulator(p_cfg, m_cfg, net_cfg, gpu_cfg)

    iters = int(input("""
                                                         .............. ...........                                  .                      
           .                                       ..........                        ....                       .         .           .  .  
   .                                           .....  .                                ......               .       .                       
                             .             ...........................      ....      .........                                             
                                         ..........................................................        .                                
                     .                ................................................................                                      
                                    ....................................................................                                    
                                  ........................................................................                                  
      .                         ...........................................................................                                 
                               ..............................................................................                               
   .                         .................................................................................    .                         
  .                         ...................................................................................                      .      
                 .        ......................................................................................                       .    
                         .........................................................................................                          
                        ............................................................................................                        
 .                     .............................:::::------:::.........::::::::::................................  .                    
                      ............................::-=+++++++++++=-:::::::-===+++====-::..............................                 .    
       .              ..........................:-=+*+=-:--:....:===-::-====-:::::::--==:.............................         .    .       
                     ..........................:=++=-::-----:::..:-==-===-:::--::::::.:-==::..........................                      
     ..             ..........................:=+++=---==+**++=::::-=++-:.::-+****+=--:.:-=-:.........................                  .   
                    ..........................:=++++==+*#######+-::-==-:::-=*#%%%%%#*=-::-==-:........................                      
                   ..........................:-=+---=+#%%%%%%%%#+-----:::-=*%%@@@@@%%*=--:-=--:.......................            .         
                   ..........................:-=+-:--+#%%@@@@@%#*=----:::-=#%%@@@@@@%*=-::-=--:........................                     
                   ..........................:-=+=::-=*#%%@@%%%*+=-----::-=*#%%@@@%%#+=::.:---:.........................                    
                   ..........................:--==---=+**#####*+=------::--=+*#%%##*+=-:::--:::.........................              .     
                  ...........................::--===-===++++++===--------::-==++++++==-::--::...........................                .   
                  ............................::--===================--=---:---======-----::.............................                   
                  .............................:--=================--:---==-----------==--:..............................                   
    .             ...............................:--==============-::..::--============-::...............................               .   
                  .................................:-====++++==--::.......::--======-:::.................................               .   
                  ...................................::---::::::..............:::::......................................                   
 .                .......................................................................................................                   
              .   .......................................................................................................         .         
                   ......................................................................................................                   
                   .....................................................................................................                    
             .      ....................................................................................................                    
                    ...............:::::::::.............................................:::::::.......................       .             
                     ...........-+*#########*******++++******+++++++++++++++++++******#########**=:....................                     
                     ...........:=+*************************************************************+=:...................             .        
.                     ............:--==+++++++++++++++++++++++++++++++====+++++=++++++++++++==-::....................                       
                       .................::::--=======+======+++=======================--:::.........................                        
                        ..........................................................................................     .                    
.                        ........................................................................................                    . ..   
     .                    ......................................................................................             .              
                      .     ...................................................................................                             
                              ...............................................................................                     .         
                    .       .  ............................................................................   .           .                 
     .                           .........................................................................           .                      
    .                              .....................................................................                                    
                                     .................................................................                                      
                                       .............................................................                                        
                   .                      .......................................................                                           
              .                    .      .  ..................................................                                      .      
                .                               ...........................................                           .                .    
.                                              .     .................................                               .                      
                                                              ................              .                       .                       
    How many iterations boss? 
    """))
    
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("Traces", exist_ok=True)
    msg_path = f'Traces/[i{iters}]messages_timeseries{now}[{total_gpus}gpu].csv'
    compute_path = f'Traces/[i{iters}]compute_timeseries{now}[{total_gpus}gpu].csv'

    for i in range(iters):
        print(f"\nRunning Iteration {i} with {p_cfg.schedule_type}...")
        sim.run_full_iteration(i)
        # Flush this iteration's events to disk and free memory
        sim.flush_to_csv(msg_path, compute_path, write_header=(i == 0))
    
    # sim.current_time is in seconds
    final_sim_time = sim.current_time
    print(f"\nSimulation Complete.")
    print(f"  Final simulated time: {final_sim_time:.1f} seconds")
    print(f"  Total messages written to: {msg_path}")
    print(f"  Total compute written to:  {compute_path}")
