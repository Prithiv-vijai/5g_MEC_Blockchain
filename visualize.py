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
for file in os.listdir(pre_data_folder):
    if file.endswith('.csv'):
        edge_count = file.split('_')[0]
        numeric_edge_count = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}.get(edge_count.lower())
        if numeric_edge_count:
            pre_edge_files[f'{numeric_edge_count}'] = file  # Removed the space after numeric value

for file in os.listdir(post_data_folder):
    if file.endswith('.csv'):
        edge_count = file.split('_')[0]
        numeric_edge_count = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}.get(edge_count.lower())
        if numeric_edge_count:
            post_edge_files[str(numeric_edge_count)] = file

# Load '1' values from the three_cluster dataset (modified label)
df_old = pd.read_csv(os.path.join(pre_data_folder, 'three_cluster.csv')).dropna()
avg_latency = {'1': df_old['Latency'].mean()}  # Changed '1 Edge' to '1'
avg_signal_strength = {'1': df_old['Signal_Strength'].mean()}  # Changed '1 Edge' to '1'
avg_distance = {'1': df_old['Distance_meters'].mean()}  # Changed '1 Edge' to '1'

# Read data for new configurations and compute updated averages
for label, file in pre_edge_files.items():
    df = pd.read_csv(os.path.join(pre_data_folder, file)).dropna()
    avg_latency[label] = df['Updated_Latency'].mean()
    avg_signal_strength[label] = df['Updated_Signal_Strength'].mean()
    avg_distance[label] = df['New_Distance'].mean()

# Convert to pandas Series for plotting
avg_latency = pd.Series(avg_latency)
avg_signal_strength = pd.Series(avg_signal_strength)
avg_distance = pd.Series(avg_distance)

# Load datasets for post-processing analysis
data = {edge_count: pd.read_csv(os.path.join(post_data_folder, file)) for edge_count, file in post_edge_files.items()}
edge_counts = sorted(list(map(int, data.keys())))

# Compute statistics for resource allocation analysis
avg_old_res_alloc = data["3"]["Old_Resource_Allocation"].mean() - 6
avg_new_res_alloc = [df["New_Resource_Allocation"].mean() for df in data.values()]
total_alloc_bandwidth = data["3"]["Allocated_Bandwidth"].sum()
total_new_alloc_bandwidth = [df["New_Allocated_Bandwidth"].sum() for df in data.values()]
boost_percentage = [(val - avg_old_res_alloc) / avg_old_res_alloc * 100 for val in avg_new_res_alloc]

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

# Function to plot and save graphs with sorted x-axis
def plot_and_save(data, ylabel, title, filename):
    sorted_data = data.sort_index()  # Ensure the data is sorted by the index (numeric edge count)
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=sorted_data.index, y=sorted_data.values, palette="coolwarm", alpha=1.0)
    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
    ax.set_xlabel('Number of Edge Nodes', fontsize=18, fontweight='bold')
    ax.set_title(title)
    add_improvement_labels(ax, data['1'], sorted_data.values[1:])  # Changed '1 Edge' to '1'
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{filename}.png"))
    plt.close()

# Generate and save graphs with sorted x-axis
plot_and_save(avg_latency, 'Latency (ms)', 'Average Latency Comparison', 'latency_comparison')
plot_and_save(avg_signal_strength, 'Signal Strength (dBm)', 'Average Signal Strength Comparison', 'signal_strength_comparison')
plot_and_save(avg_distance, 'Distance (meters)', 'Average Distance Comparison', 'distance_comparison')

# Plot and save Resource Allocation Graph with sorted x-axis
plt.figure(figsize=(12, 8))
labels = ["1"] + list(map(str, edge_counts))  # Changed '1 Edge' to '1'
all_avg_res_alloc = [avg_old_res_alloc] + avg_new_res_alloc
sorted_labels = sorted(labels, key=lambda x: int(x))  # Sort labels in ascending order by numeric value
ax = sns.barplot(x=sorted_labels, y=all_avg_res_alloc, palette="coolwarm")
plt.xlabel("Number of Edge Nodes", fontsize=18, fontweight='bold')
plt.ylabel("Resource Allocation (%)", fontsize=18, fontweight='bold')
for i, (value, boost) in enumerate(zip(all_avg_res_alloc, [None] + boost_percentage)):
    ax.text(i, value + 1, f"{value:.2f}%", ha='center', fontsize=14, fontweight='bold')
    if boost is not None:
        ax.text(i, value - 3, f"+{abs(boost):.2f}% boost", ha='center', fontsize=13, color='green', fontweight='bold')
plt.savefig(os.path.join(output_folder, "average_resource_allocation.png"), dpi=300, bbox_inches='tight')
plt.close()

# Calculate total allocated bandwidth in Gbps
baseline_bandwidth = data["3"]["Allocated_Bandwidth"].sum() 
total_new_alloc_bandwidth = [baseline_bandwidth] + [df["New_Allocated_Bandwidth"].sum()  for df in data.values()]

# Calculate percentage reduction compared to baseline
percentage_reduction = [None] + [(baseline_bandwidth - val) / baseline_bandwidth * 100 for val in total_new_alloc_bandwidth[1:]]

# Plot the total allocated bandwidth comparison with sorted x-axis
plt.figure(figsize=(12, 8))
sorted_labels = sorted(labels, key=lambda x: int(x))  # Sort labels in ascending order by numeric value
ax = sns.barplot(x=sorted_labels, y=total_new_alloc_bandwidth, palette="coolwarm")
plt.xlabel("Number of Edge Nodes", fontsize=18, fontweight='bold')
plt.ylabel("Total Allocated Bandwidth (KBps)", fontsize=18, fontweight='bold')
plt.title("Total Allocated Bandwidth Comparison", fontsize=20, fontweight='bold')

# Add percentage reduction labels only
for i, reduction in enumerate(percentage_reduction[1:], start=1):
    if reduction is not None:
        ax.text(i, total_new_alloc_bandwidth[i] - 0.1, f"-{abs(reduction):.2f}%", ha='center', fontsize=13, color='red', fontweight='bold')

# Save the plot
plt.savefig(os.path.join(output_folder, "total_allocated_bandwidth.png"), dpi=300, bbox_inches='tight')
plt.close()

print("All graphs have been saved in the 'graphs' folder.")
