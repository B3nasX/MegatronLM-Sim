import argparse
import pandas as pd
import os
from collections import defaultdict


def list_csv_files(traces_dir="."):
    """List all CSV files in the Traces directory"""
    if not os.path.isdir(traces_dir):
        print(f"Error: Directory '{traces_dir}' not found.")
        return []

    files = [f for f in os.listdir(traces_dir) if f.endswith(".csv")]
    files.sort()
    return files


import re as _re


def _find_matching_compute_file(messages_filename, traces_dir):
    """Find the compute_timeseries file that matches the given messages file.

    Matching strategy:
      1. Extract the timestamp suffix (e.g. '20260302_133350') and iter prefix
         (e.g. '[i251]') from the messages filename and look for a compute file
         with the same suffix.
      2. If no suffix match, try matching just the iter prefix '[iN]'.
      3. Last resort: return the last sorted candidate.
    """
    candidates = [f for f in list_csv_files(traces_dir) if 'compute_timeseries' in f.lower()]
    if not candidates:
        return None

    base = messages_filename
    # Extract timestamp suffix like '20260302_133350' (digits_digits before .csv)
    m = _re.search(r'(\d{8}_\d{6})', base)
    if m:
        ts_suffix = m.group(1)
        for c in candidates:
            if ts_suffix in c:
                return c

    # Try matching iter prefix like '[i251]'
    m = _re.search(r'(\[i\d+\])', base)
    if m:
        iter_prefix = m.group(1)
        matches = [c for c in candidates if c.startswith(iter_prefix.replace('messages', 'compute')) or iter_prefix in c]
        if matches:
            return matches[-1]

    # Fallback: last sorted candidate
    return candidates[-1]


def basic_compute_summary(df):
    """Simple summary when user selects a compute_timeseries file"""
    print("\n[ COMPUTE TIMESERIES SUMMARY ]")
    print(f"  Total rows (events):     {len(df):,}")
    print(f"  Columns:                 {list(df.columns)}")
    
    if 'start_time_ms' in df.columns and 'end_time_ms' in df.columns:
        start = df['start_time_ms'].min()
        end = df['end_time_ms'].max()
        duration_sec = (end - start) / 1000
        print(f"  Simulated time range:    {start:.2f} → {end:.2f} ms")
        print(f"  Total simulated duration: {duration_sec:.2f} seconds")
    
    print(f"  Unique tasks/nodes:      {df.get('node_id', pd.Series()).nunique() or 'N/A'}")
    print("\nFirst few rows (preview):")
    print(df.head(3))


def run_analysis():
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                       SIMULATION DATA ANALYSIS                               ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")

    parser = argparse.ArgumentParser(description="Analyze a compute/message timeseries CSV or choose from the Traces/ directory interactively.")
    parser.add_argument("--file", "-f", help="Path to a CSV file to analyze (skips interactive selection).")
    parser.add_argument("--traces-dir", "-d", default="Traces", help="Directory to list CSVs from when running interactively.")
    args = parser.parse_args()

    if args.file:
        file_path = args.file
        if not os.path.isfile(file_path):
            print(f"Error: file not found: {file_path}")
            return
        selected_file = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            print(f"\n→ Loaded: {selected_file} ({len(df):,} rows) from {file_path}")
        except Exception as e:
            print(f"Error loading file: {e}")
            parser = argparse.ArgumentParser(description="Analyze a compute/message timeseries CSV or choose from the workspace root interactively.")
            parser.add_argument("--file", "-f", help="Path to a CSV file to analyze (skips interactive selection).")
            parser.add_argument("--traces-dir", "-d", default=".", help="Directory to list CSVs from when running interactively (default: workspace root).")
            return
    else:
        traces_dir = args.traces_dir
        csv_files = list_csv_files(traces_dir)

        if not csv_files:
            print(f"No CSV files found in '{traces_dir}' folder.")
            return

        print("\nAvailable trace files:")
        print("-" * 70)
        for i, fname in enumerate(csv_files, 1):
            print(f"  [{i:2d}]  {fname}")
        print("-" * 70)

        while True:
            choice = input("\nEnter number to analyze (or 'q' to quit): ").strip().lower()
            if choice in ['q', 'quit', 'exit']:
                print("Exiting.")
                return

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(csv_files):
                    selected_file = csv_files[idx]
                    break
                print(f"Please enter a number between 1 and {len(csv_files)}")
            except ValueError:
                print("Please enter a valid number or 'q'")

        file_path = os.path.join(traces_dir, selected_file)

        try:
            df = pd.read_csv(file_path)
            print(f"\n→ Loaded: {selected_file} ({len(df):,} rows)")
        except Exception as e:
            print(f"Error loading file: {e}")
            return

    # ────────────────────────────────────────────────
    # Decide analysis type based on filename
    # ────────────────────────────────────────────────
    is_messages = "messages_timeseries" in selected_file.lower()

    if not is_messages:
        basic_compute_summary(df)
        print("\n" + "═" * 80)
        print("✓ Summary complete.")
        return

    # ────────────────────────────────────────────────
    # Full analysis for messages_timeseries files
    # ────────────────────────────────────────────────

    msg_df = df  # alias for clarity
    # remember where the file came from so we can search for matching compute files
    base_dir = os.path.dirname(file_path) or '.'

    # Coerce common time/size columns to numeric (tolerant)
    for col in ['start_time_ms', 'end_time_ms', 'duration_ms', 'size_mb', 'size_bytes']:
        if col in msg_df.columns:
            msg_df[col] = pd.to_numeric(msg_df[col], errors='coerce')

    # Normalize mixed time units: detect and rescale extreme outliers that are likely in the wrong unit
    def _normalize_time_col(col_name):
        """Heuristic normalization for mixed time-unit columns.

        Detects wide scale differences (e.g., a mixture of values in ms and values 1000x larger)
        and downscales the larger cluster by 1000 iteratively until spread is reasonable.
        Returns True if any normalization was applied.
        """
        if col_name not in msg_df.columns:
            return False
        s = msg_df[col_name].dropna()
        if s.empty:
            return False
        changed = False
        for _ in range(4):
            s = msg_df[col_name].dropna()
            if s.empty:
                break
            q10 = s.quantile(0.10)
            q90 = s.quantile(0.90)
            if q10 <= 0:
                break
            if q90 / q10 > 1000.0:
                # threshold to split clusters
                threshold = q10 * 1000.0
                mask = msg_df[col_name] > threshold
                if mask.any():
                    msg_df.loc[mask, col_name] = msg_df.loc[mask, col_name] / 1000.0
                    changed = True
                    continue
            break
        return changed

    norm_start = _normalize_time_col('start_time_ms')
    norm_end = _normalize_time_col('end_time_ms')
    norm_dur = _normalize_time_col('duration_ms')
    if norm_start or norm_end or norm_dur:
        print('  Note: detected mixed time units in CSV and applied heuristic normalization (dividing outlier values by 1000).')

    # Basic timeline — try message timestamps first, fall back to compute file later
    simulated_start = msg_df['start_time_ms'].min() if 'start_time_ms' in msg_df.columns else float('nan')
    simulated_end = msg_df['end_time_ms'].max() if 'end_time_ms' in msg_df.columns else float('nan')
    total_simulated_duration_ms = simulated_end - simulated_start

    # Counts & volume
    # Each row is ONE logical operation. size_bytes = logical data size per collective.
    # For all-reduce among N ranks, actual wire traffic ≈ 2*(N-1)/N * size (ring algo).
    # For P2P, wire traffic = size_bytes. We report the logical volume (application-level).
    total_messages = len(msg_df)
    total_comm_volume_mb = msg_df['size_mb'].sum() if 'size_mb' in msg_df.columns else (msg_df['size_bytes'].sum() / (1024.0 * 1024.0) if 'size_bytes' in msg_df.columns else 0.0)
    total_comm_volume_tb = total_comm_volume_mb / (1024.0 * 1024.0)

    # Communication type breakdown (be tolerant if column missing)
    if 'collective_type' in msg_df.columns:
        comm_counts = msg_df['collective_type'].value_counts()
        all_reduce_count = comm_counts.get('all_reduce', 0)
        p2p_count = comm_counts.get('p2p', 0)
    else:
        all_reduce_count = 0
        p2p_count = 0
    pct_all_reduce = (all_reduce_count / total_messages * 100) if total_messages > 0 else 0.0

    # If timestamps are missing, attempt to load a compute file to recover simulated time
    compute_df = None
    if pd.isna(simulated_start) or pd.isna(simulated_end):
        try:
            cand = _find_matching_compute_file(selected_file, base_dir)
            if cand:
                cand_path = os.path.join(base_dir, cand)
                compute_df = pd.read_csv(cand_path)
                for c in ['start_time_ms', 'end_time_ms', 'duration_ms']:
                    if c in compute_df.columns:
                        compute_df[c] = pd.to_numeric(compute_df[c], errors='coerce')
                simulated_start = compute_df['start_time_ms'].min() if 'start_time_ms' in compute_df.columns else simulated_start
                simulated_end = compute_df['end_time_ms'].max() if 'end_time_ms' in compute_df.columns else simulated_end
                total_simulated_duration_ms = simulated_end - simulated_start
        except Exception:
            compute_df = None

    print(f"\n[1] SIMULATION TIMELINE")
    print(f"  Start Timestamp:         {simulated_start if pd.notna(simulated_start) else 'N/A'} ms")
    print(f"  End Timestamp:           {simulated_end if pd.notna(simulated_end) else 'N/A'} ms")
    if pd.notna(total_simulated_duration_ms):
        print(f"  Total Simulated Time:    {total_simulated_duration_ms/1000:.2f} seconds")
    else:
        print(f"  Total Simulated Time:    N/A")

    print(f"\n[2] RESOURCE METRICS")
    print(f"  Total Messages (ops):    {total_messages:,}")
    if total_comm_volume_tb >= 0.01:
        print(f"  Total Data Transferred:  {total_comm_volume_tb:.2f} TB  ({total_comm_volume_mb:,.0f} MB)")
    else:
        print(f"  Total Data Transferred:  {total_comm_volume_mb:,.1f} MB")

    print(f"\n[3] COMMUNICATION BREAKDOWN")
    print(f"  All-Reduce Operations:   {all_reduce_count:,} ({pct_all_reduce:.1f}%)")
    print(f"  P2P Operations:          {p2p_count:,} ({100 - pct_all_reduce:.1f}%)")

    # ────────────────────────────────────────────────
    # [4] PERFORMANCE ANALYSIS (interval-based)
    # ────────────────────────────────────────────────
    # To correctly handle overlapping compute and comm, we merge time intervals
    # per rank rather than summing durations (which double-counts overlaps).
    print(f"\n[4] PERFORMANCE ANALYSIS")

    # load compute durations if not already
    if compute_df is None:
        try:
            cand = _find_matching_compute_file(selected_file, base_dir)
            if cand:
                cand_path = os.path.join(base_dir, cand)
                print(f"  (Matched compute file: {cand})")
                compute_df = pd.read_csv(cand_path)
                for c in ['start_time_ms', 'end_time_ms', 'duration_ms']:
                    if c in compute_df.columns:
                        compute_df[c] = pd.to_numeric(compute_df[c], errors='coerce')
        except Exception:
            compute_df = None

    def _merge_intervals(intervals):
        """Merge overlapping [start, end] intervals and return total covered time."""
        if not intervals:
            return 0.0, []
        intervals.sort()
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        total = sum(e - s for s, e in merged)
        return total, merged

    # ── Build per-rank INTERVALS for compute and comm ──
    per_rank_compute_intervals = defaultdict(list)  # rank -> [(start_ms, end_ms), ...]
    per_rank_comm_intervals = defaultdict(list)
    per_rank_all_intervals = defaultdict(list)  # compute + comm merged for busy time

    num_ranks = 1

    # Compute intervals from compute_df
    if compute_df is not None and 'start_time_ms' in compute_df.columns and 'end_time_ms' in compute_df.columns:
        if 'rank' in compute_df.columns:
            num_ranks = max(1, int(compute_df['rank'].nunique()))
        for _, row in compute_df.iterrows():
            rk = int(row.get('rank', 0))
            s = row['start_time_ms']
            e = row['end_time_ms']
            if pd.notna(s) and pd.notna(e) and e > s:
                per_rank_compute_intervals[rk].append((s, e))
                per_rank_all_intervals[rk].append((s, e))

    # Comm intervals from msg_df using participating_ranks
    has_participating = 'participating_ranks' in msg_df.columns
    if 'start_time_ms' in msg_df.columns and 'end_time_ms' in msg_df.columns:
        for _, row in msg_df.iterrows():
            s = row['start_time_ms']
            e = row['end_time_ms']
            if pd.isna(s) or pd.isna(e) or e <= s:
                continue
            if has_participating:
                pr_str = str(row.get('participating_ranks', ''))
                if pr_str and pr_str != 'nan' and pr_str.strip():
                    for rk_str in pr_str.split(','):
                        try:
                            rk = int(float(rk_str.strip()))
                            per_rank_comm_intervals[rk].append((s, e))
                            per_rank_all_intervals[rk].append((s, e))
                        except (ValueError, TypeError):
                            pass
                    continue
            # Fallback: attribute to src_rank
            sr = row.get('src_rank', -1)
            if pd.notna(sr) and int(sr) >= 0:
                rk = int(sr)
                per_rank_comm_intervals[rk].append((s, e))
                per_rank_all_intervals[rk].append((s, e))

    all_ranks_set = set(per_rank_compute_intervals.keys()) | set(per_rank_comm_intervals.keys())
    if all_ranks_set:
        num_ranks = max(num_ranks, len(all_ranks_set))
    all_ranks = sorted(all_ranks_set) if all_ranks_set else list(range(num_ranks))

    # Compute merged times per rank
    rank_stats = []
    for rk in all_ranks:
        comp_time, _ = _merge_intervals(per_rank_compute_intervals.get(rk, []))
        comm_time, _ = _merge_intervals(per_rank_comm_intervals.get(rk, []))
        busy_time, _ = _merge_intervals(per_rank_all_intervals.get(rk, []))
        rank_stats.append({
            'rank': rk,
            'compute_ms': comp_time,
            'comm_ms': comm_time,
            'busy_ms': busy_time,
        })

    stats_df = pd.DataFrame(rank_stats)
    if stats_df.empty:
        stats_df = pd.DataFrame({'rank': [0], 'compute_ms': [0], 'comm_ms': [0], 'busy_ms': [0]})

    avg_compute_ms = stats_df['compute_ms'].mean()
    avg_comm_ms = stats_df['comm_ms'].mean()
    avg_busy_ms = stats_df['busy_ms'].mean()

    total_compute_sum_ms = stats_df['compute_ms'].sum()
    total_comm_sum_ms = stats_df['comm_ms'].sum()

    print(f"  Number of ranks:                                   {num_ranks}")
    print(f"  Total compute time (sum across all ranks):         {total_compute_sum_ms:.2f} ms")
    print(f"  Total communication time (sum across all ranks):   {total_comm_sum_ms:.2f} ms")
    print(f"  Per-rank avg compute time (merged):                {avg_compute_ms:.2f} ms")
    print(f"  Per-rank avg communication time (merged):          {avg_comm_ms:.2f} ms")
    print(f"  Per-rank avg busy time (compute+comm merged):      {avg_busy_ms:.2f} ms")

    comm_series = stats_df.set_index('rank')['comm_ms']
    if comm_series.size > 0:
        print("\n  Per-rank communication time distribution:")
        print(f"    Avg:    {comm_series.mean():.2f} ms")
        print(f"    Median: {comm_series.median():.2f} ms")
        print(f"    P90:    {comm_series.quantile(0.9):.2f} ms")

    if pd.notna(total_simulated_duration_ms) and total_simulated_duration_ms > 0:
        wall = total_simulated_duration_ms
        pct_compute = 100.0 * avg_compute_ms / wall
        pct_comm = 100.0 * avg_comm_ms / wall
        pct_busy = 100.0 * min(avg_busy_ms, wall) / wall
        pct_idle = max(0.0, 100.0 - pct_busy)
        # Overlap = how much compute and comm ran concurrently
        overlap_ms = max(0.0, avg_compute_ms + avg_comm_ms - avg_busy_ms)
        pct_overlap = 100.0 * overlap_ms / wall if wall > 0 else 0.0

        print(f"\n  Per-rank wall-time breakdown (wall = {wall:.1f} ms):")
        print(f"    Compute:            {pct_compute:.1f}%  ({avg_compute_ms:.1f} ms)")
        print(f"    Communication:      {pct_comm:.1f}%  ({avg_comm_ms:.1f} ms)")
        print(f"    Compute/Comm overlap: {pct_overlap:.1f}%  ({overlap_ms:.1f} ms)")
        print(f"    Busy (merged):      {pct_busy:.1f}%  ({min(avg_busy_ms, wall):.1f} ms)")
        print(f"    Idle/Bubble:        {pct_idle:.1f}%  ({max(0.0, wall - avg_busy_ms):.1f} ms)")

    # ────────────────────────────────────────────────
    # GPU wait (per-rank detail) — uses interval-merged busy time
    try:
        if all_ranks_set and pd.notna(total_simulated_duration_ms) and total_simulated_duration_ms > 0:
            wall = float(total_simulated_duration_ms)
            waits = []
            for row in rank_stats:
                rk = row['rank']
                busy = min(row['busy_ms'], wall)
                wait = max(0.0, wall - busy)
                waits.append({
                    'rank': rk,
                    'compute_ms': row['compute_ms'],
                    'comm_ms': row['comm_ms'],
                    'busy_ms': busy,
                    'wait_ms': wait,
                    'wait_pct': wait / wall * 100.0
                })
            gpu_wait_df = pd.DataFrame(waits).set_index('rank')
            mean_wait_pct = gpu_wait_df['wait_pct'].mean()
            median_wait_pct = gpu_wait_df['wait_pct'].median()
            max_wait = gpu_wait_df['wait_pct'].max()
            top_wait_ranks = gpu_wait_df['wait_pct'].nlargest(5)

            print("\n  GPU wait (per-rank) — fraction of wall time GPU idle:")
            print(f"    Mean wait:   {mean_wait_pct:.2f}%")
            print(f"    Median wait: {median_wait_pct:.2f}%")
            print(f"    Max wait:    {max_wait:.2f}%")
            print("    Top ranks by wait%:")
            for rk, val in top_wait_ranks.items():
                print(f"      rank {int(rk):>4d}: {val:.2f}% (wait={gpu_wait_df.loc[rk,'wait_ms']:.1f}ms, comp={gpu_wait_df.loc[rk,'compute_ms']:.1f}ms, comm={gpu_wait_df.loc[rk,'comm_ms']:.1f}ms)")
    except Exception as e:
        print(f"  GPU wait: could not be computed ({e})")

    # ────────────────────────────────────────────────
    # [5] THROUGHPUT & ITERATION
    # ────────────────────────────────────────────────
    print(f"\n[5] THROUGHPUT & ITERATION")
    iterations = None
    if 'stage' in msg_df.columns:
        iterations = msg_df['stage'].dropna().unique()
    else:
        for c in msg_df.columns:
            if 'iter' in c.lower():
                iterations = msg_df[c].dropna().unique()
                break

    if iterations is not None and len(iterations) > 0 and 'start_time_ms' in msg_df.columns:
        key_col = 'stage' if 'stage' in msg_df.columns else next((c for c in msg_df.columns if 'iter' in c.lower()), None)
        grouped = msg_df.groupby(key_col).agg({'start_time_ms': 'min', 'end_time_ms': 'max'})
        grouped['iter_time_ms'] = grouped['end_time_ms'] - grouped['start_time_ms']
        iter_times = grouped['iter_time_ms'].dropna()
        avg_iter_ms = iter_times.mean()
        print(f"  Detected {len(iter_times)} iterations (by {key_col}). Avg iteration time: {avg_iter_ms/1000.0:.3f} s")
        batch_size = None
        for c in msg_df.columns:
            if 'batch' in c.lower():
                batch_size = msg_df[c].dropna().iloc[0]
                break
        if batch_size is not None:
            try:
                throughput = float(batch_size) / (avg_iter_ms / 1000.0)
                print(f"  Estimated throughput (using batch_size={batch_size}): {throughput:,.1f} samples/sec")
            except Exception:
                print("  Found batch column but couldn't compute throughput (bad value)")
        else:
            print("  No batch size found in CSV; cannot estimate samples/sec automatically.")
    else:
        print("  Iteration boundaries not detected automatically.")
        if pd.notna(total_simulated_duration_ms) and total_simulated_duration_ms > 0:
            print("  If you provide the global batch size, throughput = batch_size / (avg iteration time).")

    print("\n" + "═" * 80)
    print("✓ Analysis complete. (Extend with real metrics from data)")


if __name__ == "__main__":
    run_analysis()
