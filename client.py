import requests
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.offsetbox as offsetbox
from web3 import Web3
import json

# Blockchain setup
GANACHE_URL = "http://127.0.0.1:7545"  # Ganache RPC URL
web3 = Web3(Web3.HTTPProvider(GANACHE_URL))
web3.eth.default_account = web3.eth.accounts[0]  # Set the default account for transactions

# Load deployed contract
#with open("build/contracts/UserAllocation.json") as f:
#    contract_data = json.load(f)

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

# Add user to blockchain
# Add user to blockchain
def log_to_blockchain(user_id, allocated_bandwidth):
    try:
        tx_hash = contract.functions.addAllocation(int(user_id), int(float(allocated_bandwidth))).transact()
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Logged to blockchain: User {user_id}, Bandwidth {allocated_bandwidth}, TX: {receipt.transactionHash.hex()}")
    except Exception as e:
        print(f"Error logging to blockchain: {e}")


# Load your dataset 
df = pd.read_csv('user_input.csv')



# Define the latitude and longitude ranges for each container (forming a triangle)
container_positions = {
    5001: {'lat': 60, 'lon': -60},  # Top-left vertex
    5002: {'lat': 60, 'lon': 60},   # Top-right vertex
    5003: {'lat': -60, 'lon': 0},   # Bottom vertex (center)
}

# Extract Latitude and Longitude for clustering
coordinates = df[['x_coordinate', 'y_coordinate']].values

# Apply DBSCAN clustering
db = DBSCAN(eps=0.5, min_samples=5)
df['Cluster'] = db.fit_predict(coordinates)

# Function to check which container the user belongs to
def find_nearest_container(lat, lon):
    nearest_container = None
    min_distance = float('inf')
    for port, position in container_positions.items():
        distance = ((lat - position['lat'])**2 + (lon - position['lon'])**2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            nearest_container = port
    return nearest_container

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

# Define the input columns to send to the container
input_columns = ['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Resource_Allocation']

# Setup for visualization
fig, ax = plt.subplots(figsize=(12, 8))  # Larger figure for better clarity
ax.set_xlim(-110, 110)
ax.set_ylim(-110, 110)
ax.set_title("Multi Access Edge Computing - Visualization (3 Edges)", fontsize=16, weight='bold')
ax.set_xlabel("Longitude", fontsize=14)
ax.set_ylabel("Latitude", fontsize=14)

# Enhance background color and grid
ax.set_facecolor('#f0f0f0')  # Light grey background
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Load images for visualization
container_img = mpimg.imread('images/edge.png')
user_img = mpimg.imread('images/user.png')

# Resize images
container_img = offsetbox.OffsetImage(container_img, zoom=0.06)  # Slightly bigger image for containers
user_img = offsetbox.OffsetImage(user_img, zoom=0.035)  # Adjust user image size

# Initialize CSV storage for each container
for port in container_positions.keys():
    csv_file = f'output/container_{port}_data.csv'
    pd.DataFrame(columns=['User_ID', 'Allocated_Bandwidth', 'Resource_Allocation']).to_csv(csv_file, index=False)

# Track already added user labels
added_labels = set()

# Function to place an image at a specific coordinate
def place_image(ax, image, x, y):
    imagebox = offsetbox.AnnotationBbox(image, (x, y), frameon=False)
    ax.add_artist(imagebox)

# Plot initial container positions
for port, position in container_positions.items():
    lat, lon = position['lat'], position['lon']
    place_image(ax, container_img, lon, lat)

# Function to update the user data
def update(frame):
    user = df.iloc[frame]
    lat, lon = user['x_coordinate'], user['y_coordinate']
    user_id = user['User_ID']

    # Find nearest container
    container_port = find_nearest_container(lat, lon)

    if container_port:
        # Send user data for prediction
        user_data = user[input_columns].to_dict()
        prediction = send_data_to_container(user_data, container_port)

        # Extract the predicted Allocated_Bandwidth if available
        predicted_allocated_bandwidth = prediction.get('prediction', ['Error'])[0] if prediction else 'Error'

        # Log the allocation to the blockchain
        log_to_blockchain(user_id, predicted_allocated_bandwidth)

        # Plot user position
        place_image(ax, user_img, lon, lat)
        if f"User {user_id}" not in added_labels:
            ax.scatter(lon, lat, c='blue', s=15, label=f'User {user_id}', alpha=0.7, edgecolors='black', linewidth=0.5)  # Enhanced user markers
            added_labels.add(f"User {user_id}")

        # Draw connection to container with thin translucent lines
        container_lat, container_lon = container_positions[container_port]['lat'], container_positions[container_port]['lon']
        ax.plot([lon, container_lon], [lat, container_lat], 'g:', linewidth=0.8, alpha=0.4)  # Dotted line with less opacity


        # Update the data with the predicted Allocated_Bandwidth and save to CSV
        result_data = {
            'User_ID': int(user['User_ID']),  # Use the original User_Id from the dataset
            'Allocated_Bandwidth': predicted_allocated_bandwidth,
            'Resource_Allocation': user['Resource_Allocation']
        }
        csv_file = f'output/container_{container_port}_data.csv'
        # Append to the respective container CSV file
        pd.DataFrame([result_data]).to_csv(csv_file, mode='a', header=False, index=False)

# Zoom functionality
def zoom(event):
    if event.key == "up":
        ax.set_xlim(ax.get_xlim()[0] * 0.9, ax.get_xlim()[1] * 0.9)
        ax.set_ylim(ax.get_ylim()[0] * 0.9, ax.get_ylim()[1] * 0.9)
    elif event.key == "down":
        ax.set_xlim(ax.get_xlim()[0] * 1.1, ax.get_xlim()[1] * 1.1)
        ax.set_ylim(ax.get_ylim()[0] * 1.1, ax.get_ylim()[1] * 1.1)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', zoom)

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(df), repeat=False)
plt.legend(loc="best", fontsize=12)
plt.tight_layout()
plt.show()


