import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib
import numpy as np

# Set Times New Roman font for all text
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # Increased base font size

# Create output directory if it doesn't exist
os.makedirs("graphs/results/metrics", exist_ok=True)

# Define the directory structure based on your image
base_dir = "output"
metrics_dir = os.path.join(base_dir, "metrics")
algorithms = ["dbscan", "divisive", "gmm", "hierarchical", "kmeans", "meanshift", "optics"]

# Metrics we want to extract and plot
target_metrics = [
    "process_resident_memory_bytes",
    "process_cpu_seconds_total",
    "request_latency_seconds_sum",
    "request_throughput_total",
    "request_response_time_seconds"
]

def parse_metrics_file(file_path):
    """Parse a metrics file and extract values for each edge"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split metrics by edge
    edge_sections = re.split(r'Metrics from edge_\d+:', content)
    edge_sections = [s.strip() for s in edge_sections if s.strip()]
    
    edge_data = []
    for edge_num, section in enumerate(edge_sections, 1):
        edge_metrics = {}
        # Extract all metric lines
        metric_lines = [line.strip() for line in section.split('\n') if line.strip() and not line.startswith('#')]
        
        for line in metric_lines:
            if '{' in line:
                # Handle metrics with labels
                metric_name = line.split('{')[0].strip()
                value_part = line.split('}')[-1].strip()
                value = float(value_part)
            else:
                # Handle metrics without labels
                parts = line.split()
                metric_name = parts[0]
                value = float(parts[1])
            
            if metric_name in target_metrics:
                edge_metrics[metric_name] = value
        
        if edge_metrics:
            edge_metrics['edge_num'] = edge_num
            edge_data.append(edge_metrics)
    
    return edge_data

def aggregate_metrics(edge_data):
    """Aggregate metrics across all edges for a single file"""
    if not edge_data:
        return None
    
    # Calculate averages across all edges
    df = pd.DataFrame(edge_data)
    aggregated = df.mean(numeric_only=True).to_dict()
    aggregated['edge_count'] = len(edge_data)
    
    return aggregated

# Main processing
all_data = []

for algorithm in algorithms:
    algorithm_dir = os.path.join(metrics_dir, algorithm)
    if not os.path.exists(algorithm_dir):
        continue
    
    # Process each metrics file for this algorithm
    for filename in os.listdir(algorithm_dir):
        if filename.endswith('_metrics.txt'):
            edge_count = int(filename.split('_')[0])
            file_path = os.path.join(algorithm_dir, filename)
            
            try:
                edge_data = parse_metrics_file(file_path)
                aggregated = aggregate_metrics(edge_data)
                if aggregated:
                    aggregated['algorithm'] = algorithm
                    all_data.append(aggregated)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Plot each metric
for metric in target_metrics:
    plt.figure(figsize=(10, 6))  # Slightly taller figure to accommodate legend
    
    # Group by algorithm and plot
    for algorithm in algorithms:
        algorithm_data = df[df['algorithm'] == algorithm]
        if not algorithm_data.empty:
            algorithm_data = algorithm_data.sort_values('edge_count')
            plt.plot(algorithm_data['edge_count'], algorithm_data[metric], 
                    label=algorithm, marker='o', linewidth=2 ,linestyle='--')
    
    # Set x-ticks to show even numbers from 2 to 20
    plt.xticks(np.arange(2, 21, 2))
    
    plt.xlabel('Number of Edges', fontsize=18)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=18)
    plt.title(f'{metric.replace("_", " ").title()} vs Number of Edges', fontsize=21, pad=20 , fontweight="bold")
    plt.grid(True)
    
    # Adjust legend position - 4 items per row below the plot
    plt.legend(bbox_to_anchor=(0.46, -0.15), loc='upper center', 
              ncol=4, fontsize=18)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)  # Increase bottom margin
    
    # Save the plot
    safe_metric_name = metric.replace('_', '-')
    plot_filename = os.path.join("graphs", "results", "metrics", f"{safe_metric_name}.png")
    plt.savefig(plot_filename, bbox_inches='tight', dpi=400)
    plt.close()
    
    print(f"Saved plot for {metric} to {plot_filename}")

print("All plots generated successfully!")