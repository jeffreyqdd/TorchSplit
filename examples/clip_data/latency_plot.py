import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

monolithic_path = Path(__file__).parent / "monolithic/data"
optimized_path = Path(__file__).parent / "optimized/data"
output_dir = Path(__file__).parent / "visualizations"
output_dir.mkdir(exist_ok=True)

label_a = "Monolithic Deployment"
label_b = "Optimized Deployment"

# Collect metrics from individual results files
qps_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
qps = []
p50_a = []
p99_a = []
p50_b = []
p99_b = []
throughput_a = []
throughput_b = []

for qps_target in qps_levels:
    csv_a = monolithic_path / f"results_qps_{qps_target}.csv"
    csv_b = optimized_path / f"results_qps_{qps_target}.csv"

    if csv_a.exists() and csv_b.exists():
        df_a = pd.read_csv(csv_a)
        df_b = pd.read_csv(csv_b)

        qps.append(qps_target)
        p50_a.append(df_a["latency_s"].quantile(0.50))
        p99_a.append(df_a["latency_s"].quantile(0.99))
        p50_b.append(df_b["latency_s"].quantile(0.50))
        p99_b.append(df_b["latency_s"].quantile(0.99))
        
        # Calculate throughput: successful requests / total duration
        successful_a = (df_a["ok"] == True).sum()
        duration_a = df_a["end"].max() - df_a["start"].min()
        throughput_a.append(successful_a / duration_a if duration_a > 0 else 0)
        
        successful_b = (df_b["ok"] == True).sum()
        duration_b = df_b["end"].max() - df_b["start"].min()
        throughput_b.append(successful_b / duration_b if duration_b > 0 else 0)

qps = np.array(qps)
p50_a = np.array(p50_a)
p99_a = np.array(p99_a)
p50_b = np.array(p50_b)
p99_b = np.array(p99_b)
throughput_a = np.array(throughput_a)
throughput_b = np.array(throughput_b)

print("\nPerformance Improvement Summary:")
print(f"{'QPS':<10} {'Latency Improvement (%)':<25} {'Throughput Improvement (%)':<25}")
print("-" * 60)

for i in range(len(qps)):
    # Latency improvement (lower is better): (Monolithic - Optimized) / Monolithic
    lat_improv = ((p50_a[i] - p50_b[i]) / p50_a[i]) * 100
    
    # Throughput improvement (higher is better): (Optimized - Monolithic) / Monolithic
    # Handle division by zero if monolithic throughput is 0
    if throughput_a[i] > 0:
        thr_improv = ((throughput_b[i] - throughput_a[i]) / throughput_a[i]) * 100
    else:
        thr_improv = float('inf') if throughput_b[i] > 0 else 0
        
    print(f"{qps[i]:<10} {lat_improv:<25.2f} {thr_improv:<25.2f}")
print("-" * 60)
print()

err_a = np.vstack(
    [
        np.zeros_like(p50_a),
        p99_a - p50_a,
    ]
)

err_b = np.vstack(
    [
        np.zeros_like(p50_b),
        p99_b - p50_b,
    ]
)

plt.figure(figsize=(10, 6))

plt.errorbar(
    qps,
    p50_a,
    yerr=err_a,
    fmt="-o",
    capsize=4,
    label=label_a,
    color="blue",
)

plt.errorbar(
    qps,
    p50_b,
    yerr=err_b,
    fmt="-o",
    capsize=4,
    label=label_b,
    color="green",
)

plt.xlabel("Send Rate (qps)")
plt.ylabel("Latency (seconds)")
plt.xscale("log", base=2)
plt.xticks(qps_levels)
plt.title("Latency vs Send Rate")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

# Create throughput plot
plt.figure(figsize=(10, 6))

x = np.arange(len(qps))
width = 0.35

plt.bar(x - width/2, throughput_a, width, label=label_a, color="blue")
plt.bar(x + width/2, throughput_b, width, label=label_b, color="green")

plt.xlabel("Send Rate (qps)")
plt.ylabel("Throughput (requests/sec)")
plt.xticks(x, qps)
plt.title("Achieved Throughput vs Send Rate")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5, axis="y")

plt.tight_layout()
throughput_file = output_dir / "throughput_comparison.png"
plt.savefig(throughput_file, dpi=300, bbox_inches="tight")
print(f"Saved throughput plot to {throughput_file}")
plt.close()

plt.figtext(
    0.5,
    -0.08,
    "Line denotes median latency; tick above denotes 99th percentile.",
    ha="center",
    fontsize=10,
)

output_file = output_dir / "latency_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Saved plot to {output_file}")
plt.close()


# Parse GPU utilization data
def parse_gpu_utilization(file_path):
    """Parse GPU utilization .dat file and return average utilization % (excluding zeros)"""
    utilizations = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Extract first % value (GPU compute %)
            if '%' in line:
                # Find first percentage value
                parts = line.split(',')
                for part in parts:
                    part = part.strip()
                    if '%' in part:
                        try:
                            val = float(part.replace('%', '').strip())
                            if val > 0:  # Only include non-zero utilization
                                utilizations.append(val)
                            break
                        except ValueError:
                            pass
    
    return np.mean(utilizations) if utilizations else 0


def parse_gpu_utilization_over_time(file_path):
    """Parse GPU utilization .dat file and return time series data per GPU"""
    gpu_data = {}  # {gpu_id: [(timestamp, utilization), ...]}
    
    with open(file_path, 'r') as f:
        current_timestamp = None
        gpu_id = 0
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check if line has timestamp
            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
                
                # Check if first part is a timestamp (all digits)
                if parts[0].isdigit():
                    current_timestamp = int(parts[0])
                    gpu_id = 0
                
                # Extract GPU utilization %
                if '%' in line:
                    for part in parts:
                        part = part.strip()
                        if '%' in part:
                            try:
                                val = float(part.replace('%', '').strip())
                                if gpu_id not in gpu_data:
                                    gpu_data[gpu_id] = []
                                if current_timestamp is not None:
                                    gpu_data[gpu_id].append((current_timestamp, val))
                                gpu_id += 1
                                break
                            except ValueError:
                                pass
    
    return gpu_data


def parse_gpu_memory_over_time(file_path):
    """Parse GPU memory .dat file and return time series data per GPU"""
    gpu_data = {}  # {gpu_id: [(timestamp, memory_mb), ...]}
    
    with open(file_path, 'r') as f:
        current_timestamp = None
        gpu_id = 0
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check if line has timestamp
            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
                
                # Check if first part is a timestamp (all digits)
                if parts[0].isdigit():
                    current_timestamp = int(parts[0])
                    gpu_id = 0
                
                # Extract GPU memory (used memory in MiB)
                if 'MiB' in line:
                    mib_values = []
                    for part in parts:
                        part = part.strip()
                        if 'MiB' in part:
                            try:
                                val = float(part.replace('MiB', '').strip())
                                mib_values.append(val)
                            except ValueError:
                                pass
                    
                    # We want used memory (3rd MiB value)
                    if len(mib_values) >= 2 and current_timestamp is not None:
                        if gpu_id not in gpu_data:
                            gpu_data[gpu_id] = []
                        gpu_data[gpu_id].append((current_timestamp, mib_values[1]))  # Used memory
                        gpu_id += 1
    
    return gpu_data


# Calculate GPU utilization for both approaches
gpu_util_a = parse_gpu_utilization(monolithic_path.parent / "gpu_data/gpu_utilization.dat")
gpu_util_b = parse_gpu_utilization(optimized_path.parent / "gpu_data/gpu_utilization.dat")

# Create GPU utilization comparison plot
gpu_data_a = parse_gpu_utilization_over_time(monolithic_path.parent / "gpu_data/gpu_utilization.dat")
gpu_data_b = parse_gpu_utilization_over_time(optimized_path.parent / "gpu_data/gpu_utilization.dat")

def smooth_gpu_data(gpu_data):
    """Convert per-GPU data to smoothed average and min/max over time"""
    # Collect all timestamps
    all_timestamps = set()
    for gpu_id, data in gpu_data.items():
        for timestamp, _ in data:
            all_timestamps.add(timestamp)
    
    all_timestamps = sorted(all_timestamps)
    
    # For each timestamp, calculate average, min, max across GPUs
    averages = []
    maxes = []
    
    for ts in all_timestamps:
        utils_at_ts = []
        for gpu_id, data in gpu_data.items():
            # Find utilization at this timestamp (use closest if exact match not found)
            for data_ts, util in data:
                if data_ts == ts:
                    utils_at_ts.append(util)
                    break
        
        if utils_at_ts:
            averages.append(np.mean(utils_at_ts))
            maxes.append(np.max(utils_at_ts))
    
    # Apply rolling average for smoothing
    window = max(5, len(averages) // 20)  # Use 5% of data points or min 5
    if len(averages) > window:
        averages = pd.Series(averages).rolling(window=window, center=True).mean().values
        maxes = pd.Series(maxes).rolling(window=window, center=True).mean().values
    
    # Normalize timestamps
    timestamps = np.array(all_timestamps) - all_timestamps[0]
    
    return timestamps, averages, maxes

# Smooth data
ts_a, avg_a, max_a = smooth_gpu_data(gpu_data_a)
ts_b, avg_b, max_b = smooth_gpu_data(gpu_data_b)

# Plot GPU utilization over time - overlaid with trimmed bounds
fig, ax = plt.subplots(figsize=(12, 6))

# Find non-zero regions to trim white space - use threshold of 0.5% to exclude idle periods
def find_data_bounds(data):
    """Find where data starts and ends having meaningful values"""
    # Use a threshold to exclude idle periods
    threshold = 0.5
    meaningful_indices = np.where(data > threshold)[0]
    if len(meaningful_indices) == 0:
        return 0, len(data)
    # Return exact bounds without padding
    return meaningful_indices[0], meaningful_indices[-1] + 1

start_a, end_a = find_data_bounds(avg_a)
start_b, end_b = find_data_bounds(avg_b)
start = min(start_a, start_b)
end = max(end_a, end_b)

# Monolithic
ax.fill_between(ts_a[start:end], 0, max_a[start:end], alpha=0.2, color="blue", label=f"{label_a} (Max)")
ax.plot(ts_a[start:end], avg_a[start:end], linewidth=2, color="blue", label=f"{label_a} (Average)")

# Optimized
ax.fill_between(ts_b[start:end], 0, max_b[start:end], alpha=0.2, color="green", label=f"{label_b} (Max)")
ax.plot(ts_b[start:end], avg_b[start:end], linewidth=2, color="green", label=f"{label_b} (Average)")

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("GPU Utilization (%)")
ax.set_title("GPU Utilization Over Time (Smoothed)")
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc='upper right')
ax.set_ylim(0, 100)
ax.set_xlim(min(ts_a[start], ts_b[start]), max(ts_a[start:end][-1], ts_b[start:end][-1]))

plt.tight_layout()
gpu_time_file = output_dir / "gpu_utilization_over_time.png"
plt.savefig(gpu_time_file, dpi=300, bbox_inches="tight")
print(f"Saved GPU utilization over time plot to {gpu_time_file}")
plt.close()
