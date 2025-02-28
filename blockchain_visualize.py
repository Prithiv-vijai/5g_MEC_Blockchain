import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
# Global Settings
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_palette("tab10")  # Better color differentiation

# Folder Creation
if not os.path.exists("./graphs/blockchain"):
    os.makedirs("./graphs/blockchain")

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
    path = f"./graphs/blockchain/{name}.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ {name}.png saved successfully")

# Plot 1: Average Transactions per Account (Fixed + Sorted)
def plot_avg_transactions(data):
    # Calculate average transactions per account
    avg_tx = {edges: df['from'].value_counts().mean() for edges, df in data.items()}
    
    # Convert dictionary to DataFrame
    avg_tx_df = pd.DataFrame(avg_tx.items(), columns=["Edges", "Avg_Transactions"])
    
    # Sort the DataFrame by Edges (Ascending Order)
    avg_tx_df = avg_tx_df.sort_values(by="Edges", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot sorted bars
    sns.barplot(x="Edges", y="Avg_Transactions", data=avg_tx_df, ax=ax, palette="viridis", legend=False)

    plt.title("Average Transactions per Account")
    plt.xlabel("Number of Edges")
    plt.ylabel("Average Transactions")
    plt.grid(True)

    # Display Value Labels on Bars
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

    # Display Total Value on Bars
    for i, v in enumerate(avg_gas.values()):
        ax.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    save_plot(fig, "Avg_Gas_Usage")

# Plot 3: Transaction Frequency Over Relative Time (with Avg TPS at End)
def plot_tx_frequency(data):
    fig, ax = plt.subplots(figsize=(15, 7))

    for edges, df in data.items():
        df['relative_time'] = df['relative_time'] - df['relative_time'].min()  # Reset start to 0
        tx_counts = df.groupby('relative_time').size().reset_index(name='transaction_count')
        sns.lineplot(x='relative_time', y='transaction_count', data=tx_counts, label=f"{edges} Edges", ax=ax)

        # Calculate Average Transactions per Second (TPS)
        total_tx = len(df)
        total_time = df['relative_time'].max() - df['relative_time'].min()
        avg_tps = total_tx / total_time if total_time > 0 else 0

        # Display Avg TPS at the end of the line
        last_point = tx_counts.iloc[-1]
        ax.text(last_point['relative_time'], last_point['transaction_count'], f"{avg_tps:.2f} TPS", 
                fontsize=11, color=ax.get_lines()[-1].get_color(), fontweight='bold')

    ax.set_title("Transaction Frequency Over Relative Time")
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("Transactions Count")
    ax.legend()
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

    # Display Total Value on Bars
    for i, v in enumerate(avg_fee.values()):
        ax.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    save_plot(fig, "Avg_Transaction_Fee")

    
# Plot 6: Block Size Over Relative Time
def plot_block_size(data):
    fig, ax = plt.subplots(figsize=(15, 7))
    for edges, df in data.items():
        df['relative_time'] = df['relative_time'] - df['relative_time'].min()  # Reset start to 0
        sns.lineplot(x='relative_time', y='block_size', data=df, label=f"{edges} Edges", ax=ax)

    ax.set_title("Block Size Over Relative Time")
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("Block Size")
    ax.legend()
    ax.grid(True)

    save_plot(fig, "Block_Size")
    
    
    
def plot_pending_tx(data):
    fig, ax = plt.subplots(figsize=(15, 7))
    for edges, df in data.items():
        sns.lineplot(x='relative_time', y='pending_tx_count', data=df, label=f"{edges} Edges", ax=ax)

    ax.set_title("Pending Transactions Over Time")
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("Pending Transactions")
    ax.legend()
    ax.grid(True)

    save_plot(fig, "Pending_Tx")


def plot_block_utilization(data):
    fig, ax = plt.subplots(figsize=(15, 7))
    for edges, df in data.items():
        df['utilization'] = (df['block_gas_used'] / df['gas_limit']) * 100
        sns.lineplot(x='relative_time', y='utilization', data=df, label=f"{edges} Edges", ax=ax)

    ax.set_title("Block Utilization Percentage Over Time")
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("Block Utilization (%)")
    ax.legend()
    ax.grid(True)

    save_plot(fig, "Block_Utilization")






# Main Execution
def main():
    folder = './output/blockchain'
    data = load_data(folder)
    plot_avg_transactions(data)
    plot_avg_gas(data)
    plot_tx_frequency(data)
    plot_avg_tx_fee(data)
    plot_block_utilization(data)
    plot_pending_tx(data)
    plot_block_size(data)
    print("\n🎯 All graphs stored successfully inside **graphs/blockchain/** folder")

if __name__ == "__main__":
    main()
