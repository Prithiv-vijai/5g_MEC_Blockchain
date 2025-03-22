import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# Define directories
pre_data_folder = "output/pre/dbscan"
post_data_folder = "output/post/dbscan"
output_folder = "graphs/results"
os.makedirs(output_folder, exist_ok=True)

# Mapping words to numbers for filenames with word-based edge counts
word_to_num = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20
}

# Fetch pre and post edge files dynamically
pre_edge_files = {}
post_edge_files = {}

def extract_edge_count(file_name):
    edge_part = file_name.split('_')[0].lower()
    return int(edge_part) if edge_part.isdigit() else word_to_num.get(edge_part)

for file in os.listdir(pre_data_folder):
    if file.endswith('.csv'):
        edge_count = extract_edge_count(file)
        if edge_count:
            pre_edge_files[str(edge_count)] = file  

for file in os.listdir(post_data_folder):
    if file.endswith('.csv'):
        edge_count = extract_edge_count(file)
        if edge_count:
            post_edge_files[str(edge_count)] = file

# Load '10' Edge values from the three_cluster dataset
df_old = pd.read_csv(os.path.join(pre_data_folder, '10_cluster.csv')).dropna()
avg_latency = {'1': df_old['Latency'].mean()}
avg_signal_strength = {'1': df_old['Signal_Strength'].mean()}
avg_distance = {'1': df_old['Distance_meters'].mean()}

# Read data for new configurations and compute updated averages
for label, file in pre_edge_files.items():
    df = pd.read_csv(os.path.join(pre_data_folder, file)).dropna()
    avg_latency[label] = df['Updated_Latency'].mean()
    avg_signal_strength[label] = df['Updated_Signal_Strength'].mean()
    avg_distance[label] = df['New_Distance'].mean()

# Convert to pandas Series for plotting
avg_latency = pd.Series(avg_latency).sort_index(key=lambda x: x.astype(int))
avg_signal_strength = pd.Series(avg_signal_strength).sort_index(key=lambda x: x.astype(int))
avg_distance = pd.Series(avg_distance).sort_index(key=lambda x: x.astype(int))

# Load datasets for post-processing analysis
data = {edge_count: pd.read_csv(os.path.join(post_data_folder, file)) for edge_count, file in post_edge_files.items()}
edge_counts = sorted(data.keys(), key=int)

# Compute statistics for resource allocation analysis
avg_old_res_alloc = data["10"]["Old_Resource_Allocation"].mean() 
avg_new_res_alloc = [data[edge]["New_Resource_Allocation"].mean() for edge in edge_counts]
boost_percentage = [(val - avg_old_res_alloc) / avg_old_res_alloc * 100 for val in avg_new_res_alloc]

# Calculate total allocated bandwidth in Gbps
baseline_bandwidth = data["10"]["Allocated_Bandwidth"].sum()
total_new_alloc_bandwidth = [baseline_bandwidth] + [data[edge]["New_Allocated_Bandwidth"].sum() for edge in edge_counts]

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

# Function to plot and save graphs with percentage change labels
def plot_and_save(data, ylabel, title, filename, baseline_value):
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=data.index, y=data.values, palette="coolwarm", alpha=1.0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Number of Edge Nodes')
    ax.set_title(title)

    # Add percentage change labels
    for i, val in enumerate(data.values):
        if i == 0:
            continue
        percentage_change = ((val - baseline_value) / baseline_value) * 100
        ax.text(i, val + 0.1, f"{percentage_change:+.2f}%", ha='center', fontsize=13, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{filename}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Use '1' as the baseline for percentage change calculations
baseline_latency = avg_latency['1']
baseline_signal_strength = avg_signal_strength['1']
baseline_distance = avg_distance['1']

# Generate and save graphs with percentage labels
plot_and_save(avg_latency, 'Latency (ms)', 'Average Latency Comparison', 'latency_comparison', baseline_latency)
plot_and_save(avg_signal_strength, 'Signal Strength (dBm)', 'Average Signal Strength Comparison', 'signal_strength_comparison', baseline_signal_strength)
plot_and_save(avg_distance, 'Distance (meters)', 'Average Distance Comparison', 'distance_comparison', baseline_distance)

# Corrected section for total allocated bandwidth comparison
edge_counts = sorted(data.keys(), key=int)  # Ensure correct numeric sorting
total_new_alloc_bandwidth = [baseline_bandwidth] + [data[edge]["New_Allocated_Bandwidth"].sum() for edge in edge_counts]

# Plot and save total allocated bandwidth comparison with percentage reduction labels
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=["1"] + edge_counts, y=total_new_alloc_bandwidth, palette="coolwarm")
plt.xlabel("Number of Edge Nodes", fontsize=18, fontweight='bold')
plt.ylabel("Total Allocated Bandwidth (KBps)", fontsize=18, fontweight='bold')
plt.title("Total Allocated Bandwidth Comparison", fontsize=20, fontweight='bold')

# Add percentage reduction labels for bandwidth
for i, reduction in enumerate(percentage_reduction[1:], start=1):
    if reduction is not None:
        ax.text(i, total_new_alloc_bandwidth[i] - 0.1, f"-{abs(reduction):.2f}%", ha='center', fontsize=13, color='red', fontweight='bold')

plt.savefig(os.path.join(output_folder, "total_allocated_bandwidth.png"), dpi=300, bbox_inches='tight')
plt.close()

print("All graphs have been saved in the 'graphs/results' folder.")
