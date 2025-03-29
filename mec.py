import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Fetch environment variables
input_file = os.environ.get('INPUT_FILE')
output_folder = os.environ.get('OUTPUT_FOLDER')
output_file_name = os.environ.get('OUTPUT_FILE')
output_file = os.path.join(output_folder, output_file_name)

# Ensure directory exists
os.makedirs('graphs/mec', exist_ok=True)

# Remove existing output file
if os.path.exists(output_file):
    os.remove(output_file)

# Load dataset
df = pd.read_csv(input_file)
total_rows = len(df)

# Initialize progress bar
progress_bar = tqdm(total=total_rows, desc="Processing Users", unit="user")

# Define columns for input
input_columns_1 = ['Application_Type', 'Updated_Signal_Strength', 'Updated_Latency', 'Required_Bandwidth']
input_columns_2 = ['Updated_Signal_Strength', 'Updated_Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']
executor = ThreadPoolExecutor(max_workers=5)


# # Blockchain setup
 # GANACHE_URL = config['ganache_url']
 # web3 = Web3(Web3.HTTPProvider(GANACHE_URL))
 # accounts = web3.eth.accounts[:10]
 # contract_address = config['contract_address']
 # abi = config['abi']
 # contract = web3.eth.contract(address=contract_address, abi=abi)
 
 
 
 # def log_to_blockchain(user_id, allocated_bandwidth, container_id):
 #     try:
 #         web3.eth.default_account = accounts[container_id]
 #         tx_hash = contract.functions.addAllocation(int(user_id), int(float(allocated_bandwidth))).transact()
 #         web3.eth.wait_for_transaction_receipt(tx_hash)  # Ensure transaction completes
 #     except Exception as e:
 #         print(f"Error logging to blockchain: UserID={user_id}, ContainerID={container_id}, Error={e}")
 
 
def send_data_to_container(user_data_1, user_data_2, slice_type, updated_latency, signal_strength, updated_signal_strength, container_port):
    """Send data to container for processing."""
    url = f'http://localhost:{container_port}/predict'
    payload = {
        'user_data_1': user_data_1,
        'user_data_2': user_data_2,
        'slice_type': slice_type,
        'updated_latency': updated_latency,
        'signal_strength': signal_strength,
        'updated_signal_strength': updated_signal_strength
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error sending data to container: {e}")
        return None

# Process users with a loop
for frame in range(total_rows):
    user = df.iloc[frame]
    user_id = user['User_ID']
    container_port = int(user['Assigned_Edge_Node'])
    slice_type = str(user['Network_Slice']).strip()

    # Prepare input data for prediction
    user_data_1 = user[input_columns_1].to_dict()
    user_data_2 = user[input_columns_2].to_dict()

    # Send data to container and get predictions
    future = executor.submit(
        send_data_to_container,
        user_data_1,
        user_data_2,
        slice_type,
        user['Updated_Latency'],
        user['Signal_Strength'],
        user['Updated_Signal_Strength'],
        container_port
    )

    result = future.result()
    if result:
        new_allocated_bandwidth = result.get('new_allocated_bandwidth', 'Error')
        new_resource_allocation = result.get('new_resource_allocation', 'Error')
    else:
        new_allocated_bandwidth = 'Error'
        new_resource_allocation = 'Error'



    # Prepare data for CSV
    result_data = {
        'User_ID': int(user['User_ID']),
        'Assigned_Edge_Node': container_port,
        'Old_Resource_Allocation': user['Resource_Allocation'],
        'New_Resource_Allocation': new_resource_allocation,
        'Allocated_Bandwidth': user['Allocated_Bandwidth'],
        'New_Allocated_Bandwidth': new_allocated_bandwidth
    }

    # Save results to CSV
    file_exists = os.path.exists(output_file)
    pd.DataFrame([result_data]).to_csv(output_file, mode='a', header=not file_exists, index=False)

    progress_bar.update(1)  # Update progress bar

progress_bar.close()  # Close progress bar after completion

print(f"✔ Processing for {input_file} to {output_file} completed successfully.")
