import requests
import pandas as pd
from web3 import Web3
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.offsetbox as offsetbox

# Blockchain setup
GANACHE_URL = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(GANACHE_URL))
accounts = web3.eth.accounts[:10]

contract_address = "0x629739ff45aEcff36eAefe9db59F59329eE0b154"  # Replace with your deployed contract address
abi = [
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "userId",
          "type": "uint256"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "bandwidth",
          "type": "uint256"
        }
      ],
      "name": "AllocationAdded",
      "type": "event"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "name": "allocations",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "userId",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "bandwidth",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function",
      "constant": True
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "userId",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "bandwidth",
          "type": "uint256"
        }
      ],
      "name": "addAllocation",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "getAllocations",
      "outputs": [
        {
          "components": [
            {
              "internalType": "uint256",
              "name": "userId",
              "type": "uint256"
            },
            {
              "internalType": "uint256",
              "name": "bandwidth",
              "type": "uint256"
            }
          ],
          "internalType": "struct UserAllocation.Allocation[]",
          "name": "",
          "type": "tuple[]"
        }
      ],
      "stateMutability": "view",
      "type": "function",
      "constant": True
    }
  ]
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

# Load images
container_img = mpimg.imread('images/edge.png')
user_img = mpimg.imread('images/user.png')

# Resize images
container_img = offsetbox.OffsetImage(container_img, zoom=0.06)
user_img = offsetbox.OffsetImage(user_img, zoom=0.035)

# Visualization setup
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(-110, 110)
ax.set_ylim(-110, 110)
ax.set_title("Multi Access Edge Computing - Visualization", fontsize=16, weight='bold')
ax.set_xlabel("Longitude", fontsize=14)
ax.set_ylabel("Latitude", fontsize=14)
ax.set_facecolor('#f0f0f0')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Dictionary to map edge node IDs to their positions (extracted from dataset)
container_positions = df.groupby('Assigned_Edge_Node')[['x_coordinate', 'y_coordinate']].mean().to_dict(orient='index')

# Function to plot images at specific coordinates
def place_image(ax, image, x, y):
    imagebox = offsetbox.AnnotationBbox(image, (x, y), frameon=False)
    ax.add_artist(imagebox)

# Plot edge node positions from dataset
for node, position in container_positions.items():
    place_image(ax, container_img, position['y_coordinate'], position['x_coordinate'])

# Function to update user positions
def update(frame):
    user = df.iloc[frame]
    lat, lon = user['x_coordinate'], user['y_coordinate']
    user_id = user['User_ID']
    container_port = int(user['Assigned_Edge_Node'])

    if container_port in container_positions:
        user_data = user[input_columns].to_dict()
        prediction = send_data_to_container(user_data, container_port)
        allocated_bandwidth = prediction.get('prediction', ['Error'])[0] if prediction else 'Error'

        container_id = list(container_positions.keys()).index(container_port)
        log_to_blockchain(user_id, allocated_bandwidth, container_id)

        place_image(ax, user_img, lon, lat)
        ax.plot(
            [lon, container_positions[container_port]['y_coordinate']],
            [lat, container_positions[container_port]['x_coordinate']],
            'g:', linewidth=0.8, alpha=0.4
        )

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(df), repeat=False)
plt.show()
