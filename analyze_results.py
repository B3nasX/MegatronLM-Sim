import pandas as pd
import os

def list_csv_files(traces_dir="Traces"):
    """List all CSV files in the Traces directory"""
    if not os.path.isdir(traces_dir):
        print(f"Error: Directory '{traces_dir}' not found.")
        return []

    files = [f for f in os.listdir(traces_dir) if f.endswith(".csv")]
    files.sort()
    return files


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

    traces_dir = "Traces"
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

    # Basic timeline
    simulated_start = msg_df['start_time_ms'].min()
    simulated_end = msg_df['end_time_ms'].max()
    total_simulated_duration_ms = simulated_end - simulated_start

    # Counts & volume
    total_messages = len(msg_df)
    total_comm_volume_mb = msg_df['size_mb'].sum()
    total_comm_volume_tb = total_comm_volume_mb / (1024 * 1024)

    # Communication type breakdown
    comm_counts = msg_df['collective_type'].value_counts()
    all_reduce_count = comm_counts.get('all_reduce', 0)
    p2p_count = comm_counts.get('p2p', 0)
    pct_all_reduce = (all_reduce_count / total_messages * 100) if total_messages > 0 else 0.0

    print(f"\n[1] SIMULATION TIMELINE")
    print(f"  Start Timestamp:         {simulated_start:.2f} ms")
    print(f"  End Timestamp:           {simulated_end:.2f} ms")
    print(f"  Total Simulated Time:    {total_simulated_duration_ms/1000:.2f} seconds")
    # If you saved wall-clock time somewhere, load and show it here

    print(f"\n[2] RESOURCE METRICS")
    print(f"  Total Messages:          {total_messages:,}")
    # You'll need to load the matching compute file if you want total compute events
    print(f"  Total Data Transferred:  {total_comm_volume_tb:.2f} TB")

    print(f"\n[3] COMMUNICATION BREAKDOWN")
    print(f"  All-Reduce Operations:   {all_reduce_count:,} ({pct_all_reduce:.1f}%)")
    print(f"  P2P Operations:          {p2p_count:,} ({100 - pct_all_reduce:.1f}%)")

    print(f"\n[4] PERFORMANCE ANALYSIS  (TODO: replace with real calculations)")
    # Example placeholders — replace with actual logic:
    #   - sum duration where type=communication
    #   - sum wait/bubble time
    #   - sum compute durations from matching compute file
    print(f"  Communication Time:      [calculate % of total time]")
    print(f"  Pipeline Bubble (Wait):  [calculate % of total time]")
    print(f"  Compute Efficiency:      [calculate % of total time]")

    print(f"\n[5] THROUGHPUT & ITERATION  (TODO: replace with real calculations)")
    # You'll likely need to:
    #   - Group by iteration / step
    #   - Find time per iteration
    #   - Estimate samples/second based on global batch size
    print(f"  Calculated Throughput:   [samples/second]")
    print(f"  Avg. Iteration Time:     [seconds]")

    print("\n" + "═" * 80)
    print("✓ Analysis complete. (Extend with real metrics from data)")


if __name__ == "__main__":
    run_analysis()
