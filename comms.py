import math
import random
import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any

import plotly.graph_objects as go
import plotly.express as px

class CollectiveType(Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"
    POINT_TO_POINT = "p2p"

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

class CommunicationModel:
    def __init__(self, network_config: NetworkConfig, rank_to_stage: Dict[int, Any]):
        self.net_cfg = network_config
        self.rank_to_stage = rank_to_stage

    def calculate_communication_time(self, size_bytes: int, collective: CollectiveType, 
                                     participating_ranks: List[int], active_flows: int = 0) -> float:
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
        if active_flows > 1 and not same_node:
            # Each additional concurrent flow degrades shared link bandwidth
            # Formula: bw_effective = bw / (1 + penalty * (num_flows - 1))
            congestion_divisor = 1.0 + net.congestion_penalty_factor * (active_flows - 1)
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
            if num_chunks > 1:
                # Pipeline effect: total ≈ latency_startup + bw_cost (chunks overlap latency)
                return lat_cost + bw_cost
            else:
                return lat_cost + bw_cost

        # Fallback heuristic for other collectives
        return total_lat * num_ranks + size_gb / effective_bw
    def plot_message_timeseries(self, messages: List[Message], title: str = "Communication Timeline (Messages)", 
                                output_html: str = "comm_timeline.html"):
        if not messages:
            print("No messages to plot.")
            return

        # Prepare data
        data = []
        for msg in messages:
            data.append({
                "Message ID": msg.msg_id,
                "Collective": msg.collective_type.value,
                "Start (s)": msg.start_time,
                "End (s)": msg.end_time,
                "Duration (s)": msg.duration(),
                "Stage": msg.stage,
                "Layer": msg.layer_id,
                "Size (MB)": round(msg.size_bytes / (1024*1024), 2),
                "Src Rank": msg.src_rank,
                "Participating": msg.participating_ranks[:100] + "..." if len(msg.participating_ranks) > 100 else msg.participating_ranks
            })

        import pandas as pd
        df = pd.DataFrame(data)

       # === Alternate colors for neighbouring rows ===
        df = df.sort_values(by="Start (s)")  # Sort by start time for logical ordering
        df["RowColor"] = ["Blue" if i % 2 == 0 else "Red" for i in range(len(df))]

        fig = go.Figure() # Haha go figure

        for color_name, group in df.groupby("RowColor"):
            line_color = "rgba(31, 119, 180, 0.5)" if color_name == "Blue" else "rgba(214, 39, 40, 0.5)"
            for _, row in group.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["Start (s)"], row["End (s)"]],
                    y=[row["Message ID"], row["Message ID"]],
                    mode="lines",
                    line=dict(color=line_color, width=12),
                    name=color_name,
                    hovertemplate=(
                        f"<b>{row['Message ID']}</b><br>"
                        f"Collective: {row['Collective']}<br>"
                        f"Stage: {row['Stage']}<br>"
                        f"Layer: {row['Layer']}<br>"
                        f"Size: {row['Size (MB)']} MB<br>"
                        f"Duration: {row['Duration (s)']:.6f} s<br>"
                        f"Time: %{{x}} s"
                    ),
                    showlegend=True
                ))

        # Get full time range
        max_time = df["End (s)"].max()

        fig.update_layout(
            title=title,
            xaxis_title="Time (seconds) — Full timeline",
            yaxis_title="Messages",
            width=3400,
            height=950,
            template="plotly_white",
            hovermode="closest",
            xaxis=dict(
                rangeslider=dict(visible=True, thickness=0.06),
                type="linear",
                autorange=True,
                tickformat=".5f"
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=8)
            ),
        )

        fig.update_yaxes(categoryorder="category ascending")

        fig.write_html(output_html)
        print(f"Timeline plot saved to: {output_html}")
        print("Full timeline shown with alternating blue/red rows for neighbouring messages.")
