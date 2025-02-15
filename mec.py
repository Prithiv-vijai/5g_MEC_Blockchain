import requests
import pandas as pd
from web3 import Web3
import json
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.offsetbox as offsetbox
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

output_folder = 'output'
output_file = os.path.join(output_folder, "output_container_data.csv")

# Delete existing output file if it exists to start fresh
if os.path.exists(output_file):
    os.remove(output_file)

# Load configuration from config.json
with open('mec_config.json', 'r') as config_file:
    config = json.load(config_file)

# Blockchain setup
GANACHE_URL = config['ganache_url']
web3 = Web3(Web3.HTTPProvider(GANACHE_URL))
accounts = web3.eth.accounts[:10]
contract_address = config['contract_address']
abi = config['abi']
contract = web3.eth.contract(address=contract_address, abi=abi)

def log_to_blockchain(user_id, allocated_bandwidth, container_id):
    try:
        web3.eth.default_account = accounts[container_id]
        tx_hash = contract.functions.addAllocation(int(user_id), int(float(allocated_bandwidth))).transact()
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Logged to blockchain (Container {container_id}): User {user_id}, Bandwidth {allocated_bandwidth}, TX: {receipt.transactionHash.hex()}")
    except Exception as e:
        print(f"Error logging to blockchain (Container {container_id}): {e}")

# Load processed dataset
df = pd.read_csv('clustered_user_data.csv')

def get_slice_color(slice_type):
    slice_colors = {'eMBB': 'red', 'URLLC': 'blue', 'mMTC': 'green'}
    return slice_colors.get(slice_type.strip(), 'black')

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(-105, 105)
ax.set_ylim(-105, 105)
ax.set_title("Multi Access Edge Computing - Visualization", fontsize=16, weight='bold')
ax.set_xlabel("Longitude", fontsize=14)
ax.set_ylabel("Latitude", fontsize=14)
ax.set_facecolor('#f0f0f0')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.8)

container_positions = df.groupby('Assigned_Edge_Node')[['x_coordinate', 'y_coordinate']].mean().to_dict(orient='index')

image_cache = {
    "container": offsetbox.OffsetImage(mpimg.imread('files/c.png'), zoom=0.04),
    "user": offsetbox.OffsetImage(mpimg.imread('files/user.png'), zoom=0.03)
}

def place_image(ax, image_key, x, y):
    imagebox = offsetbox.AnnotationBbox(image_cache[image_key], (x, y), frameon=False)
    ax.add_artist(imagebox)

for node, position in container_positions.items():
    place_image(ax, "container", position['y_coordinate'], position['x_coordinate'])

def send_data_to_container(user_data, container_port, endpoint):
    url = f'http://localhost:{container_port}/{endpoint}'
    try:
        response = requests.post(url, json=user_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error while sending data to {endpoint} on container {container_port}: {e}")
        return None

input_columns_1 = ['Updated_Signal_Strength', 'Updated_Latency', 'Required_Bandwidth' ,'Allocated_Bandwidth']
input_columns_2 = ['Application_Type','Updated_Signal_Strength', 'Updated_Latency', 'Required_Bandwidth', 'Resource_Allocation']

animated_lines = []
executor = ThreadPoolExecutor(max_workers=5)

def update(frame):
    user = df.iloc[frame]
    lat, lon = user['x_coordinate'], user['y_coordinate']
    user_id = user['User_ID']
    container_port = int(user['Assigned_Edge_Node'])

    slice_type = str(user['Network_Slice']).strip()
    color = get_slice_color(slice_type)

    if container_port in container_positions:
        # **First Prediction (Predict1)**
        user_data_1 = user[input_columns_1].to_dict()
        future1 = executor.submit(send_data_to_container, user_data_1, container_port, 'predict1')
        prediction1 = future1.result()
        new_resource_allocation = prediction1.get('prediction', ['Error'])[0] if prediction1 else 'Error'

        # **Second Prediction (Predict2)**
        adjusted_signal_strength = user['Signal_Strength'] - abs(user['Updated_Signal_Strength'] - user['Signal_Strength'])
        user_data_2 = user[input_columns_2].to_dict()
        user_data_2['Resource_Allocation'] = new_resource_allocation
        user_data_2['Updated_Signal_Strength'] = adjusted_signal_strength
        user_data_2['Required_Bandwidth'] *= 0.8
        future2 = executor.submit(send_data_to_container, user_data_2, container_port, 'predict2')
        prediction2 = future2.result()
        new_allocated_bandwidth = prediction2.get('prediction', ['Error'])[0] if prediction2 else 'Error'

        container_id = list(container_positions.keys()).index(container_port)
        log_to_blockchain(user_id, new_allocated_bandwidth, container_id)

        place_image(ax, "user", lon, lat)
        
        x_start, y_start = lon, lat
        x_end, y_end = container_positions[container_port]['y_coordinate'], container_positions[container_port]['x_coordinate']
        
        num_segments = 8
        alpha_values = np.linspace(0.2, 0.8, num_segments)[::-1]
        for i in range(num_segments):
            segment_x = np.linspace(x_start, x_end, num_segments)[i:i+2]
            segment_y = np.linspace(y_start, y_end, num_segments)[i:i+2]
            line, = ax.plot(segment_x, segment_y, linestyle="-", linewidth=1, color=color, alpha=alpha_values[i])
            animated_lines.append(line)

        old_resource_allocation = user['Resource_Allocation']

        result_data = {
            'User_ID': int(user['User_ID']),
            'Assigned_Edge_Node': container_port,
            'Old_Resource_Allocation': old_resource_allocation,
            'New_Resource_Allocation': new_resource_allocation,
            'Allocated_Bandwidth': user['Allocated_Bandwidth'],
            'New_Allocated_Bandwidth': new_allocated_bandwidth  
        }

        file_exists = os.path.exists(output_file)
        pd.DataFrame([result_data]).to_csv(output_file, mode='a', header=not file_exists, index=False)

handles = [plt.Line2D([0], [0], linestyle="dashed", linewidth=1, color=get_slice_color(slice_type), label=slice_type)
           for slice_type in ['eMBB', 'URLLC', 'mMTC']]
ax.legend(handles=handles, loc='upper right')

ani = animation.FuncAnimation(fig, update, frames=len(df), repeat=False, interval=200, blit=False)
plt.show()
