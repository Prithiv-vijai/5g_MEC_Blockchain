import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set Times New Roman font for all text
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14  # Increased base font size

# Create output directories if they don't exist
os.makedirs("../graphs/results/metrics", exist_ok=True)
os.makedirs("../output/post", exist_ok=True)  # Directory for CSV output

# Define the directory structure
base_dir = "../output"
docker_metrics_dir = os.path.join(base_dir, "docker_metrics")
algorithms = ["kmeans", "meanshift", "optics"]

# Define specific colors and markers for each algorithm
algorithm_styles = {
        'kmeans': {'color': '#1f77b4', 'marker': 'o', 'label': 'K-Means'},
        'optics': {'color': '#d62728', 'marker': '^', 'label': 'OPTICS'},
        'meanshift': {'color': '#9467bd', 'marker': 'v', 'label': 'Mean-Shift'},
    }


def parse_docker_metrics_file(file_path):
    """Parse a docker metrics file and extract values for each edge"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract all metric lines
    metric_lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    edge_data = []
    current_edge = None
    edge_metrics = {}
    
    for line in metric_lines:
        # Check if this is a new edge section
        edge_match = re.match(r'Average (.*) for edge_(\d+):', line)
        if edge_match:
            metric_type = edge_match.group(1).lower()
            edge_num = int(edge_match.group(2))
            
            # If we have a new edge number, save the previous edge's metrics
            if current_edge is not None and current_edge != edge_num:
                edge_metrics['edge_num'] = current_edge
                edge_data.append(edge_metrics)
                edge_metrics = {}
            
            current_edge = edge_num
            
            # Extract the metric value
            value_part = line.split(':')[-1].strip()
            if 'cpu' in metric_type.lower():
                value = float(value_part.replace('%', ''))
                edge_metrics['cpu_usage'] = value
            elif 'memory' in metric_type.lower():
                value = float(value_part.replace('MiB', '').strip())
                edge_metrics['memory_usage'] = value
    
    # Add the last edge's metrics
    if edge_metrics and current_edge is not None:
        edge_metrics['edge_num'] = current_edge
        edge_data.append(edge_metrics)
    
    return edge_data

def aggregate_docker_metrics(edge_data):
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
    algorithm_dir = os.path.join(docker_metrics_dir, algorithm)
    if not os.path.exists(algorithm_dir):
        continue
    
    # Process each docker metrics file for this algorithm
    for filename in os.listdir(algorithm_dir):
        if filename.endswith('_docker_metrics.txt'):
            edge_count = int(filename.split('_')[0])
            file_path = os.path.join(algorithm_dir, filename)
            
            try:
                edge_data = parse_docker_metrics_file(file_path)
                aggregated = aggregate_docker_metrics(edge_data)
                if aggregated:
                    aggregated['algorithm'] = algorithm
                    all_data.append(aggregated)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Save consolidated metrics to CSV
csv_path = os.path.join("output", "post", "docker_consolidated_metrics.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved consolidated metrics to {csv_path}\n")

def plot_metrics(df, metric, ylabel, title, filename):
    """Plot metrics (CPU or Memory usage) with specific algorithm styles"""
    if metric in df.columns:
        plt.figure(figsize=(10, 7))
        
        for algorithm in algorithms:
            algorithm_data = df[df['algorithm'] == algorithm]
            if not algorithm_data.empty:
                algorithm_data = algorithm_data.sort_values('edge_count')
                style = algorithm_styles.get(algorithm, {'color': '#000000', 'marker': 'o'})
                plt.plot(algorithm_data['edge_count'], algorithm_data[metric], 
                        label=style['label'],
                        color=style['color'],
                        marker=style['marker'],
                        linestyle='--',
                        markersize=8,
                        linewidth=2.5,
                        alpha=0.8)
        
        # Set x-ticks to show even numbers from 2 to 20
        plt.xticks(np.arange(2, 21, 2))
        
        plt.xlabel('Number of Edges', fontsize=22)
        plt.ylabel(ylabel, fontsize=22)
        plt.title(title, fontsize=22, pad=20, fontweight="bold")
        
        # Adjust legend position - 4 items per row below the plot
        plt.legend(bbox_to_anchor=(0.46, -0.15), loc='upper center', 
                  ncol=4, fontsize=22)
        
        plt.grid(True)
        
        # Adjust layout to make room for the legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        
        plot_filename = os.path.join("../graphs", "results", "metrics", filename)
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {metric} plot to {plot_filename}")
    else:
        print(f"No {metric} data found in the metrics files")

# Plot CPU Usage
plot_metrics(df, 'cpu_usage', 'CPU Usage (%)', 'Average CPU Usage vs Number of Edges', 'cpu-usage.png')

# Plot Memory Usage
plot_metrics(df, 'memory_usage', 'Memory Usage (MiB)', 'Average Memory Usage vs Number of Edges', 'memory-usage.png')

print("\nDocker metrics processing complete!")
