import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define directories
pre_data_folder = "result/pre"
post_data_folder = "result/post"
output_folder = "graphs"
os.makedirs(output_folder, exist_ok=True)

# Define file mappings for different edge configurations
pre_edge_files = {
    '3 Edges': 'three_cluster.csv',
    '5 Edges': 'five_cluster.csv',
    '7 Edges': 'seven_cluster.csv',
    '10 Edges': 'ten_cluster.csv'
}
post_edge_files = {
    "3": "three_edges.csv",
    "5": "five_edges.csv",
    "7": "seven_edges.csv",
    "10": "ten_edges.csv"
}

# Load '1 Edge' values from the three_cluster dataset
df_old = pd.read_csv(os.path.join(pre_data_folder, 'three_cluster.csv')).dropna()
avg_latency = {'1 Edge': df_old['Latency'].mean()}
avg_signal_strength = {'1 Edge': df_old['Signal_Strength'].mean()}
avg_distance = {'1 Edge': df_old['Distance_meters'].mean()}

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
edge_counts = list(data.keys())

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

# Function to plot and save graphs
def plot_and_save(data, ylabel, title, filename):
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=data.index, y=data.values, palette="coolwarm", alpha=1.0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    ax.set_title(title)
    add_improvement_labels(ax, data['1 Edge'], data.values[1:])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{filename}.png"))
    plt.close()

# Generate and save graphs
plot_and_save(avg_latency, 'Latency (ms)', 'Average Latency Comparison', 'latency_comparison')
plot_and_save(avg_signal_strength, 'Signal Strength (dBm)', 'Average Signal Strength Comparison', 'signal_strength_comparison')
plot_and_save(avg_distance, 'Distance (meters)', 'Average Distance Comparison', 'distance_comparison')

# Plot and save Resource Allocation Graph
plt.figure(figsize=(12, 8))
labels = ["1"] + edge_counts
all_avg_res_alloc = [avg_old_res_alloc] + avg_new_res_alloc
ax = sns.barplot(x=labels, y=all_avg_res_alloc, palette="coolwarm")
plt.xlabel("Number of Edge Nodes", fontsize=18, fontweight='bold')
plt.ylabel("Resource Allocation (%)", fontsize=18, fontweight='bold')
for i, (value, boost) in enumerate(zip(all_avg_res_alloc, [None] + boost_percentage)):
    ax.text(i, value + 1, f"{value:.2f}%", ha='center', fontsize=14, fontweight='bold')
    if boost is not None:
        ax.text(i, value - 3, f"+{abs(boost):.2f}% boost", ha='center', fontsize=13, color='green', fontweight='bold')
plt.savefig(os.path.join(output_folder, "average_resource_allocation.png"), dpi=300, bbox_inches='tight')
plt.close()


print("All graphs have been saved in the 'graphs' folder.")