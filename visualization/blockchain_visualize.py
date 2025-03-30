import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
# Global Settings
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_palette("tab10")  # Better color differentiation

# Folder Creation
if not os.path.exists("../graphs/blockchain"):
    os.makedirs("../graphs/blockchain")

# Function to Extract Edge Count from Filename
def get_edges_from_filename(filename):
    return int(filename.split('_')[0])

# Function to Load Data
def load_data(folder):
    data = {}
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            edges = get_edges_from_filename(file)
            df = pd.read_csv(os.path.join(folder, file))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['relative_time'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
            data[edges] = df
    return data

# Save Plot Function
def save_plot(fig, name):
    path = f"../graphs/blockchain/{name}.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ {name}.png saved successfully")

# Plot 1: Average Transactions per Account (Fixed + Sorted)
def plot_avg_transactions(data):
    avg_tx = {edges: df['from'].value_counts().mean() for edges, df in data.items()}
    avg_tx_df = pd.DataFrame(avg_tx.items(), columns=["Edges", "Avg_Transactions"])
    avg_tx_df = avg_tx_df.sort_values(by="Edges", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Edges", y="Avg_Transactions", data=avg_tx_df, ax=ax, palette="viridis", legend=False)
    plt.title("Average Transactions per Account")
    plt.xlabel("Number of Edges")
    plt.ylabel("Average Transactions")
    plt.grid(True)
    for i, v in enumerate(avg_tx_df["Avg_Transactions"]):
        ax.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    save_plot(fig, "Avg_Transactions")

# Plot 2: Average Gas Usage
def plot_avg_gas(data):
    avg_gas = {edges: df['gas_used'].mean() for edges, df in data.items()}
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(avg_gas.keys()), y=list(avg_gas.values()), ax=ax, hue=list(avg_gas.keys()), legend=False)
    plt.title("Average Gas Usage")
    plt.xlabel("Number of Edges")
    plt.ylabel("Average Gas Used")
    plt.grid(True)
    for i, v in enumerate(avg_gas.values()):
        ax.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    save_plot(fig, "Avg_Gas_Usage")

# Plot 3: Transaction Frequency Over Relative Time
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tx_frequency(data):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set font to Times New Roman for all text elements
    plt.rcParams.update({'font.family': 'Times New Roman'})  # Apply Times New Roman globally
    
    # Increase font size for titles, labels, ticks, and legend
    title_fontsize = 18
    label_fontsize = 18
    tick_fontsize = 16
    legend_fontsize = 18

    for edges, df in data.items():
        df['relative_time'] = df['relative_time'] - df['relative_time'].min()
        tx_counts = df.groupby('relative_time').size().reset_index(name='transaction_count')
        sns.lineplot(x='relative_time', y='transaction_count', data=tx_counts, label=f"{edges} Edges", ax=ax)
        total_tx = len(df)
        total_time = df['relative_time'].max() - df['relative_time'].min()
        avg_tps = total_tx / total_time if total_time > 0 else 0
        last_point = tx_counts.iloc[-1]
        ax.text(last_point['relative_time'], last_point['transaction_count'], f"{avg_tps:.2f} TPS", fontsize=18, color=ax.get_lines()[-1].get_color(), fontweight='bold')
    
    # Set title, labels, and grid with increased font size
    ax.set_title("Transaction Frequency Over Relative Time", fontsize=title_fontsize, fontweight='bold')
    ax.set_xlabel("Time (Seconds)", fontsize=label_fontsize,fontweight='bold')
    ax.set_ylabel("Transactions Count", fontsize=label_fontsize,fontweight='bold')

    # Increase tick label font size
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)

    # Set legend with increased font size
    ax.legend(fontsize=legend_fontsize)

    # Display grid with adjusted font
    ax.grid(True)

    save_plot(fig, "Tx_Frequency")

# Plot 4: Average Transaction Fee
def plot_avg_tx_fee(data):
    avg_fee = {edges: df['transaction_fee'].mean() for edges, df in data.items()}
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(avg_fee.keys()), y=list(avg_fee.values()), ax=ax, hue=list(avg_fee.keys()), legend=False)
    plt.title("Average Transaction Fee")
    plt.xlabel("Number of Edges")
    plt.ylabel("Average Fee")
    plt.grid(True)
    for i, v in enumerate(avg_fee.values()):
        ax.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    save_plot(fig, "Avg_Transaction_Fee")

# Plot 5: Throughput vs Time
def plot_throughput(data):
    fig, ax = plt.subplots(figsize=(15, 7))
    for edges, df in data.items():
        df['throughput'] = df['gas_used'] / df['relative_time']
        sns.lineplot(x='relative_time', y='throughput', data=df, label=f"{edges} Edges", ax=ax)
    ax.set_title("Throughput vs Time")
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("Throughput")
    ax.legend()
    ax.grid(True)
    save_plot(fig, "Throughput_vs_Time")

# Plot 6: Latency Distribution
def plot_latency_distribution(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    for edges, df in data.items():
        sns.histplot(df['latency'], bins=30, kde=True, label=f"{edges} Edges", ax=ax)
    ax.set_title("Latency Distribution")
    ax.set_xlabel("Latency")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)
    save_plot(fig, "Latency_Distribution")
# Plot 7: Gas Usage Distribution
def plot_gas_distribution(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    for edges, df in data.items():
        sns.histplot(df['gas_used'], bins=30, kde=True, label=f"{edges} Edges", ax=ax)
    ax.set_title("Gas Usage Distribution")
    ax.set_xlabel("Gas Used")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)
    save_plot(fig, "Gas_Usage_Distribution")

# Plot 8: Transaction Fee Distribution
def plot_fee_distribution(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    for edges, df in data.items():
        sns.histplot(df['transaction_fee'], bins=30, kde=True, label=f"{edges} Edges", ax=ax)
    ax.set_title("Transaction Fee Distribution")
    ax.set_xlabel("Transaction Fee")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)
    save_plot(fig, "Transaction_Fee_Distribution")

# Plot 9: Cumulative Transactions Over Time
def plot_cumulative_transactions(data):
    fig, ax = plt.subplots(figsize=(15, 7))
    for edges, df in data.items():
        df['cumulative_tx'] = np.arange(1, len(df) + 1)
        sns.lineplot(x='relative_time', y='cumulative_tx', data=df, label=f"{edges} Edges", ax=ax)
    ax.set_title("Cumulative Transactions Over Time")
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("Cumulative Transactions")
    ax.legend()
    ax.grid(True)
    save_plot(fig, "Cumulative_Transactions")

# Plot 10: Latency vs Gas Used (Scatter Plot)
def plot_latency_vs_gas(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    for edges, df in data.items():
        sns.scatterplot(x='latency', y='gas_used', data=df, label=f"{edges} Edges", ax=ax)
    ax.set_title("Latency vs Gas Used")
    ax.set_xlabel("Latency")
    ax.set_ylabel("Gas Used")
    ax.legend()
    ax.grid(True)
    save_plot(fig, "Latency_vs_Gas")


# Main Execution (Updated)
def main():
    folder = './output/blockchain'
    data = load_data(folder)
    # plot_avg_transactions(data)
    # plot_avg_gas(data)
    plot_tx_frequency(data)
    # plot_avg_tx_fee(data)
    # plot_throughput(data)
    # plot_latency_distribution(data)
    # plot_gas_distribution(data)
    # plot_fee_distribution(data)
    # plot_cumulative_transactions(data)
    # plot_latency_vs_gas(data)
    print("\n🎯 All graphs stored successfully inside **graphs/blockchain/** folder")

if __name__ == "__main__":
    main()