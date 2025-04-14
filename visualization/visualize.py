import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from typing import Dict, List

warnings.filterwarnings('ignore')

# Define directories
pre_data_folder = "../output/pre/kmeans"
post_data_folder = "../output/post/kmeans"
out_folder = "../output/post/"
output_folder = "../graphs/results/decentralization"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(post_data_folder, exist_ok=True)

# ==============================================
# PART 1: DATA CALCULATION AND CONSOLIDATED CSV
# ==============================================

def calculate_and_save_metrics() -> pd.DataFrame:
    """Calculate all metrics and save to a consolidated CSV file."""
    # Mapping for word to number conversion
    word_to_num = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
        'nineteen': 19, 'twenty': 20
    }

    def extract_edge_count(file_name: str) -> int:
        """Extract edge count from filename."""
        edge_part = file_name.split('_')[0].lower()
        return int(edge_part) if edge_part.isdigit() else word_to_num.get(edge_part)

    # Load all data files
    pre_files = {str(extract_edge_count(f)): f for f in os.listdir(pre_data_folder) 
                if f.endswith('.csv') and extract_edge_count(f)}
    post_files = {str(extract_edge_count(f)): f for f in os.listdir(post_data_folder) 
                 if f.endswith('.csv') and extract_edge_count(f)}

    # Initialize consolidated metrics dataframe - ADD THROUGHPUT COLUMN
    metrics = pd.DataFrame(columns=[
        'Cluster_Count', 'Latency', 'Signal_Strength', 'Distance',
        'Resource_Allocation', 'Allocated_Bandwidth', 'Throughput'
    ])

    # Process baseline data (10 clusters)
    baseline_df = pd.read_csv(os.path.join(pre_data_folder, '10_cluster.csv')).dropna()
    metrics.loc[0] = {
        'Cluster_Count': 1,
        'Latency': baseline_df['Latency'].mean(),
        'Signal_Strength': baseline_df['Signal_Strength'].mean(),
        'Distance': baseline_df['Distance_meters'].mean(),
        'Resource_Allocation': None,  # Will be filled from post data
        'Allocated_Bandwidth': None,  # Will be filled from post data
        'Throughput': None            # Will be calculated later
    }

    # Process pre files (updated metrics)
    for edge_count, file in pre_files.items():
        df = pd.read_csv(os.path.join(pre_data_folder, file)).dropna()
        metrics.loc[len(metrics)] = {
            'Cluster_Count': int(edge_count),
            'Latency': df['Updated_Latency'].mean(),
            'Signal_Strength': df['Updated_Signal_Strength'].mean(),
            'Distance': df['New_Distance'].mean(),
            'Resource_Allocation': None,
            'Allocated_Bandwidth': None,
            'Throughput': None
        }

    # Process post files (resource allocation and bandwidth)
    for edge_count, file in post_files.items():
        df = pd.read_csv(os.path.join(post_data_folder, file)).dropna()
        
        # Update corresponding row in metrics
        idx = metrics[metrics['Cluster_Count'] == int(edge_count)].index
        if not idx.empty:
            metrics.loc[idx, 'Resource_Allocation'] = df['New_Resource_Allocation'].mean()
            metrics.loc[idx, 'Allocated_Bandwidth'] = df['New_Allocated_Bandwidth'].sum()
        else:
            # If no pre data exists, create new row
            metrics.loc[len(metrics)] = {
                'Cluster_Count': int(edge_count),
                'Latency': None,
                'Signal_Strength': None,
                'Distance': None,
                'Resource_Allocation': df['New_Resource_Allocation'].mean(),
                'Allocated_Bandwidth': df['New_Allocated_Bandwidth'].sum(),
                'Throughput': None
            }

    # Fill baseline resource allocation and bandwidth from 10_cluster post data
    if '10' in post_files:
        post_10 = pd.read_csv(os.path.join(post_data_folder, post_files['10'])).dropna()
        metrics.loc[metrics['Cluster_Count'] == 1, 'Resource_Allocation'] = post_10['Old_Resource_Allocation'].mean()
        metrics.loc[metrics['Cluster_Count'] == 1, 'Allocated_Bandwidth'] = post_10['Allocated_Bandwidth'].sum()

    # CALCULATE THROUGHPUT FOR ALL ROWS
    # Throughput formula: (Allocated Bandwidth) / ( Latency)
    for idx, row in metrics.iterrows():
        if pd.notnull(row['Allocated_Bandwidth']) and pd.notnull(row['Latency']):
            metrics.at[idx, 'Throughput'] = row['Allocated_Bandwidth'] / ( row['Latency'])

    # Sort by cluster count and save
    metrics = metrics.sort_values('Cluster_Count').reset_index(drop=True)

    csv_path = os.path.join(out_folder, 'decentralized_consolidated_metrics.csv')
    metrics.to_csv(csv_path, index=False)
    
    print(f"Consolidated metrics saved to {csv_path}")
    return metrics
# ==============================================
# PART 2: PLOTTING FROM CONSOLIDATED CSV
# ==============================================

def plot_from_consolidated_csv():
    """Generate all plots from the consolidated CSV file."""
    # Set visual styling
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'xtick.labelsize': 14,
        'ytick.labelsize': 14
    })

    # Load consolidated data
    csv_path = os.path.join(out_folder, 'decentralized_consolidated_metrics.csv')
    metrics = pd.read_csv(csv_path)
    
    # Convert cluster count to string for categorical plotting
    metrics['Cluster_Count'] = metrics['Cluster_Count'].astype(str)

    def plot_metric(metric: str, ylabel: str, title: str, filename: str, inverted: bool = False):
        plt.figure(figsize=(10, 7))
        ax = sns.barplot(x='Cluster_Count', y=metric, data=metrics, palette="viridis")
        
        # Get y-axis limits for proper label positioning
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        offset = y_range * 0.05  # 5% of y-range as offset
        value_offset = y_range * 0.001  # Additional offset for actual value label
        
        # Add percentage change annotations and actual values
        baseline = metrics[metric].iloc[0]
        for i, row in metrics.iterrows():
            current_value = row[metric]
            
            
            if inverted:
                # For inverted metrics, place value at the top
                ax.text(i, current_value - value_offset, f"{current_value:.1f}", 
                    ha='center', va='top',
                    fontsize=16, color='black',
                    bbox=dict(facecolor='white', alpha=0.8, 
                            edgecolor='none', pad=1))
            else:
                ax.text(i, current_value + value_offset, f"{current_value:.1f}",
            ha='center', va='bottom',
            fontsize=16, color='black')
            
            if i == 0:  # Skip baseline for percentage change
                continue
                
            change = ((current_value - baseline) / baseline) * 100
            
            # Percentage change label placement
            if metric == 'Resource_Allocation':
                change_text = f"+{change:.1f}%" if change >= 0 else f"{change:.1f}%"
                label_y = current_value * 0.98  # 98% of bar height to keep text inside
                ax.text(i, label_y, change_text, 
                        ha='center', va='top',
                        fontsize=16, color='red',
                        bbox=dict(facecolor='white', alpha=0.8, 
                                edgecolor='none', pad=1))
            elif inverted:  # For inverted metrics like Signal Strength
                label_y = current_value + offset
                va = 'bottom'
                change_text = f"+{abs(change):.1f}%"
                ax.text(i, label_y, change_text,
                        ha='center', va=va,
                        fontsize=16, color='red',
                        bbox=dict(facecolor='white', alpha=0.8,
                                edgecolor='none', pad=1))
            else:  # Default case
                if change < 0:
                    label_y = current_value - offset
                    va = 'top'
                    change_text = f"{change:+.1f}%"
                else:
                    label_y = current_value + offset
                    va = 'bottom'
                    change_text = f"+{change:.1f}%"
                
                ax.text(i, label_y, change_text, 
                        ha='center', va=va,
                        fontsize=16, color='red',
                        bbox=dict(facecolor='white', alpha=0.8, 
                                edgecolor='none', pad=1))
        
        ax.set(xlabel='Number of Edge Nodes', ylabel=ylabel, title=title)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{filename}.png"), dpi=300)
        plt.close()
    
    def plot_throughput():
        plt.figure(figsize=(10, 7))
        ax = sns.barplot(x='Cluster_Count', y='Throughput', data=metrics, palette="viridis")
        
        # Set consistent y-axis limits with some padding
        max_throughput = metrics['Throughput'].max()
        min_throughput = metrics['Throughput'].min()
        y_padding = (max_throughput - min_throughput) * 0.15  # 15% padding
        ax.set_ylim([min_throughput - y_padding, max_throughput + y_padding])
        
        # Get updated y-axis limits after setting them
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        offset = y_range * 0.05  # 5% of y-range as offset
        value_offset = y_range * 0.02  # Additional offset for actual value label
        
        # Add percentage change annotations with boundary checks
        baseline = metrics['Throughput'].iloc[0]
        for i, row in metrics.iterrows():
            current_value = row['Throughput']
            
            ax.text(i, current_value + value_offset, f"{current_value:.1f}",
           ha='center', va='bottom',
           fontsize=15, color='black')
            
            if i == 0:  # Skip baseline
                continue
                
            change = ((current_value - baseline) / baseline) * 100
            
            # Determine label position with boundary checks
            if change < 0:  # Degradation (label below bar)
                label_y = max(ymin + offset, current_value - offset)  # Ensure doesn't go below ymin
                va = 'top'
                change_text = f"{change:+.1f}%"
            else:  # Improvement (label above bar)
                label_y = min(ymax - offset, current_value + offset)  # Ensure doesn't go above ymax
                va = 'bottom'
                change_text = f"+{change:.1f}%"
            
            ax.text(i, label_y, change_text, 
                ha='center', va=va,
                fontsize=16, color='red',
                bbox=dict(facecolor='white', alpha=0.8, 
                            edgecolor='none', pad=1))
        
        ax.set(xlabel='Number of Edge Nodes', 
            ylabel='Throughput (KBps/ms)',
            title='Average Throughput Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "throughput_comparison.png"), dpi=300)
        plt.close()



    
    # Generate individual plots
    plot_throughput()
    plot_metric('Latency', 'Latency (ms)', 'Average Latency Comparison', 'latency_comparison')
    plot_metric('Signal_Strength', 'Signal Strength (dBm)', 
               'Average Signal Strength Comparison', 'signal_strength_comparison',
               inverted=True)  # Signal strength is inverted (higher is better)
    plot_metric('Distance', 'Distance (meters)', 'Average Distance Comparison', 'distance_comparison')
    plot_metric('Resource_Allocation', 'Resource Allocation (%)', 
               'Average Resource Allocation Comparison', 'resource_allocation_comparison')

    # Bandwidth plot (special handling for size)
    plt.figure(figsize=(10, 7))
    ax = sns.barplot(x='Cluster_Count', y='Allocated_Bandwidth', data=metrics, palette="viridis")

    # Get y-axis limits
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    offset = y_range * 0.05
    value_offset = y_range * 0.001  # Additional offset for actual value label

    # Add percentage change annotations and actual values
    baseline = metrics['Allocated_Bandwidth'].iloc[0]
    for i, row in metrics.iterrows():
        current_value = row['Allocated_Bandwidth']
        
        # Only show value label for first bar in scientific notation
        if i == 0:
            ax.text(i, current_value + value_offset, f"{current_value:.2e}",
                ha='center', va='bottom',
                fontsize=14, color='black')
        
        if i == 0:
            continue
        change = ((current_value - baseline) / baseline) * 100
        
        if change < 0:  # Improvement (reduction)
            label_y = current_value - offset
            va = 'top'
            change_text = f"{change:+.1f}%"
        else:  # Degradation (increase)
            label_y = current_value + offset
            va = 'bottom'
            change_text = f"+{change:.1f}%"
        
        ax.text(i, label_y, change_text, 
            ha='center', va=va,
            fontsize=16, color='red',
            bbox=dict(facecolor='white', alpha=0.8, 
                        edgecolor='none', pad=1))

    # Format y-axis in scientific notation
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))

    ax.set(xlabel='Number of Edge Nodes', 
        ylabel='Total Allocated Bandwidth (KBps)',
        title='Total Allocated Bandwidth Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "total_allocated_bandwidth.png"), dpi=300)
    plt.close()

    print(f"All plots saved to {output_folder}")

# ==============================================
# MAIN EXECUTION
# ==============================================

if __name__ == "__main__":
    # Step 1: Calculate and save all metrics
    consolidated_metrics_path = '../output/post/decentralized_consolidated_metrics.csv'
    
    # Step 1: Calculate and save all metrics (only if file doesn't exist)
    if not os.path.exists(consolidated_metrics_path):
        metrics_df = calculate_and_save_metrics()
    else:
        print(f"Using existing consolidated metrics file: {consolidated_metrics_path}")
    
    # Step 2: Generate plots from consolidated data
    plot_from_consolidated_csv()