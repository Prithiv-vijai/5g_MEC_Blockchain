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

# Ensure 'output' folder exists
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


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

# Function to log allocation to blockchain
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

# Pre-caching colors for slices
def get_slice_color(slice_type):
    slice_colors = {'eMBB': 'red', 'URLLC': 'blue', 'mMTC': 'green'}
    return slice_colors.get(slice_type.strip(), 'black')

# Visualization setup
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(-105, 105)
ax.set_ylim(-105, 105)
ax.set_title("Multi Access Edge Computing - Visualization", fontsize=16, weight='bold')
ax.set_xlabel("Longitude", fontsize=14)
ax.set_ylabel("Latitude", fontsize=14)
ax.set_facecolor('#f0f0f0')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.8)

# Precompute container positions
container_positions = df.groupby('Assigned_Edge_Node')[['x_coordinate', 'y_coordinate']].mean().to_dict(orient='index')

# Pre-load images (Performance Optimization)
image_cache = {
    "container": offsetbox.OffsetImage(mpimg.imread('files/c.png'), zoom=0.04),
    "user": offsetbox.OffsetImage(mpimg.imread('files/user.png'), zoom=0.03)
}

# Function to place images at coordinates
def place_image(ax, image_key, x, y):
    imagebox = offsetbox.AnnotationBbox(image_cache[image_key], (x, y), frameon=False)
    ax.add_artist(imagebox)

# Plot edge node positions (containers)
for node, position in container_positions.items():
    place_image(ax, "container", position['y_coordinate'], position['x_coordinate'])

# Function to send data to the appropriate container
def send_data_to_container(user_data, container_port):
    url = f'http://localhost:{container_port}/predict'
    try:
        response = requests.post(url, json=user_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error while sending data to container {container_port}: {e}")
        return None

# Define input columns
input_columns = ['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Resource_Allocation']

# Store animated lines for updates
animated_lines = []

# ThreadPoolExecutor for parallel execution
executor = ThreadPoolExecutor(max_workers=5)

# Function to update user positions and create data flow effect
def update(frame):
    user = df.iloc[frame]
    lat, lon = user['x_coordinate'], user['y_coordinate']
    user_id = user['User_ID']
    container_port = int(user['Assigned_Edge_Node'])

    slice_type = str(user['Network_Slice']).strip()
    color = get_slice_color(slice_type)

    if container_port in container_positions:
        user_data = user[input_columns].to_dict()

        # Run data sending in parallel
        future = executor.submit(send_data_to_container, user_data, container_port)

        # Get prediction asynchronously
        prediction = future.result()
        allocated_bandwidth = prediction.get('prediction', ['Error'])[0] if prediction else 'Error'

        container_id = list(container_positions.keys()).index(container_port)
        log_to_blockchain(user_id, allocated_bandwidth, container_id)

        # Plot user image at correct coordinates
        place_image(ax, "user", lon, lat)

        # Flowing line effect: Multiple short segments appearing sequentially
        x_start, y_start = lon, lat
        x_end, y_end = container_positions[container_port]['y_coordinate'], container_positions[container_port]['x_coordinate']

        num_segments = 8
        alpha_values = np.linspace(0.2, 0.8, num_segments)[::-1]

        for i in range(num_segments):
            segment_x = np.linspace(x_start, x_end, num_segments)[i:i+2]
            segment_y = np.linspace(y_start, y_end, num_segments)[i:i+2]
            line, = ax.plot(segment_x, segment_y, linestyle="-", linewidth=1, color=color, alpha=alpha_values[i])
            animated_lines.append(line)

        # Save data to CSV with headers
        result_data = {
            'User_ID': int(user['User_ID']),
            'Allocated_Bandwidth': allocated_bandwidth,
            'Resource_Allocation': user['Resource_Allocation']
        }
        csv_file = f'output/container_{container_port}_data.csv'
        file_exists = os.path.exists(csv_file)
        pd.DataFrame([result_data]).to_csv(csv_file, mode='a', header=not file_exists, index=False)

# Create the legend correctly
handles = [plt.Line2D([0], [0], linestyle="dashed", linewidth=1, color=get_slice_color(slice_type), label=slice_type)
           for slice_type in ['eMBB', 'URLLC', 'mMTC']]
ax.legend(handles=handles, loc='upper right')

# Create animation with a "flowing" effect
ani = animation.FuncAnimation(fig, update, frames=len(df), repeat=False, interval=200, blit=False)

plt.show()
