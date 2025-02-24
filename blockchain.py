import os
import json
import time
from web3 import Web3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib as mpl

# Set global plot settings
mpl.rcParams['font.family'] = 'Times New Roman'
coolwarm_colors = plt.get_cmap('coolwarm', 10)

# Ensure directories exist
graph_dir = "graphs/blockchain"
output_dir = "output/blockchain"
os.makedirs(graph_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Load configuration
with open('mec_config.json', 'r') as config_file:
    config = json.load(config_file)

# Blockchain setup
GANACHE_URL = config['ganache_url']
web3 = Web3(Web3.HTTPProvider(GANACHE_URL))
accounts = web3.eth.accounts[:10]
contract_address = config['contract_address']
abi = config['abi']
contract = web3.eth.contract(address=contract_address, abi=abi)

# Fetch transactions with additional details
transactions = []
latest_block = web3.eth.block_number

for block_number in tqdm(range(latest_block + 1), desc="Processing Blocks"):
    try:
        block = web3.eth.get_block(block_number, full_transactions=True)
        timestamp = block.timestamp if hasattr(block, 'timestamp') else None
        for tx in block.transactions:
            try:
                receipt = web3.eth.get_transaction_receipt(tx.hash)
                transactions.append({
                    "tx_hash": tx.hash.hex(),
                    "from": tx["from"],
                    "to": tx["to"],
                    "gas_used": receipt.gasUsed,
                    "gas_price": tx["gasPrice"],  # Gas price in Wei
                    "transaction_fee": receipt.gasUsed * tx["gasPrice"],  # Fee in Wei
                    "block_number": block_number,
                    "block_size": block.size,  # Block size in bytes
                    "gas_limit": block.gasLimit,  # Block gas limit
                    "block_gas_used": block.gasUsed,  # Gas used in the block
                    "difficulty": block.difficulty,  # Mining difficulty
                    "nonce": tx.nonce,  # Nonce value
                    "pending_tx_count": len(web3.eth.get_block('pending').transactions),  # Pending transactions
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

# Save DataFrame to CSV
num_accounts = df_tx['from'].nunique()
num_transactions = len(df_tx)
file_name = f"{num_accounts}_{num_transactions}.csv"
file_path = os.path.join(output_dir, file_name)
df_tx.to_csv(file_path, index=False)

print(f"✅ Transactions saved to {file_path}")

