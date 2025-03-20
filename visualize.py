import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# Define directories
pre_data_folder = "output/pre"
post_data_folder = "output/post"
output_folder = "graphs/results"
os.makedirs(output_folder, exist_ok=True)

# Fetch pre and post edge files dynamically
pre_edge_files = {}
post_edge_files = {}

# Extract numeric edge count from filenames and map them to file names
edge_mapping = {
    'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
}

for file in os.listdir(pre_data_folder):
    if file.endswith('.csv'):
        edge_count = file.split('_')[0].lower()
        numeric_edge_count = edge_mapping.get(edge_count)
        if numeric_edge_count:
            pre_edge_files[str(numeric_edge_count)] = file  

for file in os.listdir(post_data_folder):
    if file.endswith('.csv'):
        edge_count = file.split('_')[0].lower()
        numeric_edge_count = edge_mapping.get(edge_count)
        if numeric_edge_count:
            post_edge_files[str(numeric_edge_count)] = file

# Load '1' Edge values from the three_cluster dataset
df_old = pd.read_csv(os.path.join(pre_data_folder, 'three_cluster.csv')).dropna()
avg_latency = {'1': df_old['Latency'].mean()}
avg_signal_strength = {'1': df_old['Signal_Strength'].mean()}
avg_distance = {'1': df_old['Distance_meters'].mean()}

# Read data for new configurations and compute updated averages
for label, file in pre_edge_files.items():
    df = pd.read_csv(os.path.join(pre_data_folder, file)).dropna()
    avg_latency[label] = df['Updated_Latency'].mean()
    avg_signal_strength[label] = df['Updated_Signal_Strength'].mean()
    avg_distance[label] = df['New_Distance'].mean()

# Convert to pandas Series for plotting (ensuring proper order)
avg_latency = pd.Series(avg_latency).sort_index(key=lambda x: x.astype(int))
avg_signal_strength = pd.Series(avg_signal_strength).sort_index(key=lambda x: x.astype(int))
avg_distance = pd.Series(avg_distance).sort_index(key=lambda x: x.astype(int))

# Load datasets for post-processing analysis
data = {edge_count: pd.read_csv(os.path.join(post_data_folder, file)) for edge_count, file in post_edge_files.items()}
edge_counts = sorted(data.keys(), key=int)  # Sort numerically: ['3', '5', '7', '10']

# Compute statistics for resource allocation analysis
avg_old_res_alloc = data["3"]["Old_Resource_Allocation"].mean() 
avg_new_res_alloc = [data[edge]["New_Resource_Allocation"].mean() for edge in edge_counts]
boost_percentage = [(val - avg_old_res_alloc) / avg_old_res_alloc * 100 for val in avg_new_res_alloc]

# Calculate total allocated bandwidth in Gbps
baseline_bandwidth = data["3"]["Allocated_Bandwidth"].sum() 
total_new_alloc_bandwidth = [baseline_bandwidth] + [df["New_Allocated_Bandwidth"].sum() for df in data.values()]

# Calculate percentage reduction compared to baseline
percentage_reduction = [None] + [(baseline_bandwidth - val) / baseline_bandwidth * 100 for val in total_new_alloc_bandwidth[1:]]

# Set font configurations
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

# Function to add improvement labels for updated values
def add_improvement_labels(ax, old_value, new_values):
    for i, new_value in enumerate(new_values, start=1):
        improvement = ((old_value - new_value) / old_value) * 100
        y_position = new_value + (0.05 * abs(new_value)) if new_value > 0 else new_value - (0.05 * abs(new_value))
        ax.text(i, y_position, f"+{abs(improvement):.2f}% boost", ha='center', fontsize=13, color='green', fontweight='bold')

# Function to plot and save graphs
def plot_and_save(data, ylabel, title, filename):
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=data.index, y=data.values, palette="coolwarm", alpha=1.0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Number of Edge Nodes')
    ax.set_title(title)
    add_improvement_labels(ax, data['1'], data.values[1:])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{filename}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Generate and save graphs
plot_and_save(avg_latency, 'Latency (ms)', 'Average Latency Comparison', 'latency_comparison')
plot_and_save(avg_signal_strength, 'Signal Strength (dBm)', 'Average Signal Strength Comparison', 'signal_strength_comparison')
plot_and_save(avg_distance, 'Distance (meters)', 'Average Distance Comparison', 'distance_comparison')

# Plot and save total allocated bandwidth comparison
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=["1"] + edge_counts, y=total_new_alloc_bandwidth, palette="coolwarm")
plt.xlabel("Number of Edge Nodes", fontsize=18, fontweight='bold')
plt.ylabel("Total Allocated Bandwidth (KBps)", fontsize=18, fontweight='bold')
plt.title("Total Allocated Bandwidth Comparison", fontsize=20, fontweight='bold')

# Add percentage reduction labels
for i, reduction in enumerate(percentage_reduction[1:], start=1):
    if reduction is not None:
        ax.text(i, total_new_alloc_bandwidth[i] - 0.1, f"-{abs(reduction):.2f}%", ha='center', fontsize=13, color='red', fontweight='bold')

plt.savefig(os.path.join(output_folder, "total_allocated_bandwidth.png"), dpi=300, bbox_inches='tight')
plt.close()

print("All graphs have been saved in the 'graphs' folder.")
