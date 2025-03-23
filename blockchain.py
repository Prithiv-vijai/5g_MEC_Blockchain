import os
import json
import time
from web3 import Web3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib as mpl

# Global Settings
mpl.rcParams['font.family'] = 'Times New Roman'
coolwarm_colors = plt.get_cmap('coolwarm', 10)

graph_dir = "graphs/blockchain"
output_dir = "output/blockchain"
os.makedirs(graph_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Load Config
with open('mec_config.json', 'r') as config_file:
    config = json.load(config_file)

# Blockchain Setup
GANACHE_URL = config['ganache_url']
web3 = Web3(Web3.HTTPProvider(GANACHE_URL))
contract_address = config['contract_address']
abi = config['abi']
contract = web3.eth.contract(address=contract_address, abi=abi)

# Fetch Transactions
transactions = []
latest_block = web3.eth.block_number
block_timestamps = {}

for block_number in tqdm(range(latest_block + 1), desc="Processing Blocks"):
    try:
        block = web3.eth.get_block(block_number, full_transactions=True)
        timestamp = block.timestamp
        block_timestamps[block_number] = timestamp
        
        if block_number > 0:
            block_interval = timestamp - block_timestamps[block_number - 1]
        else:
            block_interval = 0
        
        for tx in block.transactions:
            try:
                receipt = web3.eth.get_transaction_receipt(tx.hash)
                gas_efficiency = (receipt.gasUsed / block.gasLimit) * 100

                # Latency Calculation
                if block_number > 1 and (block_number - 1) in block_timestamps:
                    submission_time = block_timestamps[block_number - 1]
                    latency = timestamp - submission_time
                else:
                    latency = None

                transactions.append({
                    "tx_hash": tx.hash.hex(),
                    "from": tx["from"],
                    "to": tx["to"],
                    "gas_used": receipt.gasUsed,
                    "gas_price": tx["gasPrice"],
                    "transaction_fee": receipt.gasUsed * tx["gasPrice"],
                    "block_number": block_number,
                    "block_size": block.size,
                    "block_gas_used": block.gasUsed,
                    "gas_efficiency": gas_efficiency,
                    "block_interval": block_interval,
                    "throughput": len(block.transactions),
                    "latency": latency,
                    "tx_size": block.size / len(block.transactions) if len(block.transactions) > 0 else 0,
                    "avg_gas_price_gwei": tx["gasPrice"] / 1e9,
                    "timestamp": timestamp
                })
            except:
                pass
    except:
        pass

# Convert to DataFrame
df_tx = pd.DataFrame(transactions)
df_tx['timestamp'] = pd.to_datetime(df_tx['timestamp'], unit='s')
df_tx['relative_time'] = (df_tx['timestamp'] - df_tx['timestamp'].min()).dt.total_seconds()

# Save DataFrame
num_accounts = df_tx['from'].nunique()
num_transactions = len(df_tx)
file_name = f"{num_accounts}_{num_transactions}_enhanced.csv"
file_path = os.path.join(output_dir, file_name)
df_tx.to_csv(file_path, index=False)

print(f"✅ Transactions saved to {file_path}")

# Summary Stats
print("\n🔍 Summary Stats:")
print(f"Throughput (Avg TPS): {df_tx['throughput'].mean():.2f}")
print(f"Avg Gas Efficiency: {df_tx['gas_efficiency'].mean():.2f}%")
print(f"Avg Latency: {df_tx['latency'].mean():.2f} sec")
print(f"Avg Transaction Size: {df_tx['tx_size'].mean():.2f} bytes")
print(f"Avg Gas Price: {df_tx['avg_gas_price_gwei'].mean():.2f} Gwei")
