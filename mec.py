import os
import requests
import pandas as pd
from web3 import Web3
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
from concurrent.futures import ThreadPoolExecutor
import matplotlib
from tqdm import tqdm  # Import tqdm for progress bar

matplotlib.use('Agg')  # Avoid GUI window
matplotlib.rcParams['font.family'] = 'Times New Roman'

# Fetch environment variables
input_file = os.environ.get('INPUT_FILE')
output_folder = os.environ.get('OUTPUT_FOLDER')
output_file_name = os.environ.get('OUTPUT_FILE')
output_file = os.path.join(output_folder, output_file_name)
output_file_name = output_file_name[:-4]

# Ensure directory exists
os.makedirs('graphs/mec', exist_ok=True)

# Remove existing output file
if os.path.exists(output_file):
    os.remove(output_file)

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

# Load dataset
df = pd.read_csv(input_file)
total_rows = len(df)

def log_to_blockchain(user_id, allocated_bandwidth, container_id):
    try:
        web3.eth.default_account = accounts[container_id]
        tx_hash = contract.functions.addAllocation(int(user_id), int(float(allocated_bandwidth))).transact()
        web3.eth.wait_for_transaction_receipt(tx_hash)  # Ensure transaction completes
    except Exception as e:
        print(f"Error logging to blockchain: UserID={user_id}, ContainerID={container_id}, Error={e}")

# Initialize progress bar
progress_bar = tqdm(total=total_rows, desc="Processing Users", unit="user")

# Get colors for slices
def get_slice_color(slice_type):
    return {'eMBB': 'red', 'URLLC': 'blue', 'mMTC': 'green'}.get(slice_type.strip(), 'black')

# Setup figure
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(-105, 105)
ax.set_ylim(-105, 105)
ax.set_title(f"MEC Visualization ({output_file_name})", fontsize=16, weight='bold')
ax.set_xlabel("Longitude", fontsize=14)
ax.set_ylabel("Latitude", fontsize=14)
ax.set_facecolor('#f0f0f0')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.8)

# Compute container positions
container_positions = df.groupby('Assigned_Edge_Node')[['x_coordinate', 'y_coordinate']].mean().to_dict(orient='index')

# Load images
image_cache = {
    "container": offsetbox.OffsetImage(plt.imread('files/c.png'), zoom=0.04),
    "user": offsetbox.OffsetImage(plt.imread('files/user.png'), zoom=0.03)
}

def place_image(ax, image_key, x, y):
    ax.add_artist(offsetbox.AnnotationBbox(image_cache[image_key], (x, y), frameon=False))

# Place containers
for node, position in container_positions.items():
    place_image(ax, "container", position['y_coordinate'], position['x_coordinate'])

def send_data_to_container(user_data, container_port, endpoint):
    url = f'http://localhost:{container_port}/{endpoint}'
    try:
        response = requests.post(url, json=user_data)
        response.raise_for_status()
        return response.json()
    except:
        return None

input_columns_1 = ['Updated_Signal_Strength', 'Updated_Latency', 'Required_Bandwidth']
input_columns_2 = ['Application_Type', 'Updated_Signal_Strength', 'Updated_Latency', 'Required_Bandwidth', 'Resource_Allocation']

executor = ThreadPoolExecutor(max_workers=5)

def enforce_qos(slice_type, allocated_bandwidth, latency):
    if slice_type == 'URLLC' and latency > 10:
        allocated_bandwidth *= 1 + ((latency - 10) / 10) * 0.1  
    elif slice_type == 'eMBB' and allocated_bandwidth < 10000:
        allocated_bandwidth *= 1 + ((10000 - allocated_bandwidth) / 10000) * 0.1  
    elif slice_type == 'mMTC' and allocated_bandwidth < 100:
        allocated_bandwidth *= 1 + ((100 - allocated_bandwidth) / 100) * 0.1  
    return allocated_bandwidth

# Process users with a loop
for frame in range(total_rows):
    user = df.iloc[frame]
    lat, lon = user['x_coordinate'], user['y_coordinate']
    user_id = user['User_ID']
    container_port = int(user['Assigned_Edge_Node'])

    slice_type = str(user['Network_Slice']).strip()
    color = get_slice_color(slice_type)

    if container_port in container_positions:
        user_data_1 = user[input_columns_1].to_dict()
        future1 = executor.submit(send_data_to_container, user_data_1, container_port, 'predict1')
        prediction1 = future1.result()
        new_resource_allocation = prediction1.get('prediction', ['Error'])[0] if prediction1 else 'Error'
        user_data_2 = user[input_columns_2].to_dict()
        
        #To TEST , remove while execution
        adjusted_signal_strength = user['Signal_Strength'] - abs(user['Updated_Signal_Strength'] - user['Signal_Strength'])
        user_data_2['Updated_Signal_Strength'] = adjusted_signal_strength
        user_data_2['Required_Bandwidth'] *= 0.8
        
        
        user_data_2['Resource_Allocation'] = new_resource_allocation

        future2 = executor.submit(send_data_to_container, user_data_2, container_port, 'predict2')
        prediction2 = future2.result()
        new_allocated_bandwidth = prediction2.get('prediction', ['Error'])[0] if prediction2 else 'Error'
        
        new_allocated_bandwidth = enforce_qos(slice_type, new_allocated_bandwidth, user_data_2['Updated_Latency'])

        container_id = list(container_positions.keys()).index(container_port)
        log_to_blockchain(user_id, new_allocated_bandwidth, container_id)

        place_image(ax, "user", lon, lat)

        x_start, y_start = lon, lat
        x_end, y_end = container_positions[container_port]['y_coordinate'], container_positions[container_port]['x_coordinate']
        ax.plot([x_start, x_end], [y_start, y_end], linestyle="-", linewidth=1, color=color, alpha=0.5)

        result_data = {
            'User_ID': int(user['User_ID']),
            'Assigned_Edge_Node': container_port,
            'Old_Resource_Allocation': user['Resource_Allocation'],
            'New_Resource_Allocation': new_resource_allocation,
            'Allocated_Bandwidth': user['Allocated_Bandwidth'],
            'New_Allocated_Bandwidth': new_allocated_bandwidth  
        }

        file_exists = os.path.exists(output_file)
        pd.DataFrame([result_data]).to_csv(output_file, mode='a', header=not file_exists, index=False)

        progress_bar.update(1)  # ✅ Update progress bar

progress_bar.close()  # ✅ Close progress bar after completion

# Save final image (uncomment if needed)
# plt.savefig(os.path.join('graphs/mec', f"mec_{output_file_name}.png"))
plt.close()

print(f"✔ Processing for {input_file} to {output_file} completed successfully.")
