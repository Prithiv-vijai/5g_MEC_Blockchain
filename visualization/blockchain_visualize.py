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

 

def main():
    folder = './output/blockchain'
    data = load_data(folder)
    plot_tx_frequency(data)

    print("\n🎯 All graphs stored successfully inside **graphs/blockchain/** folder")

if __name__ == "__main__":
    main()