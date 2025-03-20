import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Load environment variables
num_containers = int(os.environ.get("NUM_CONTAINERS", 10))  # Default: 10 containers
output_folder = os.environ.get("OUTPUT_FOLDER", "output")  # Default: 'output' directory
output_file = os.environ.get("OUTPUT_FILE", "clustered_user_data.csv")  # Default: 'clustered_user_data.csv'


# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load dataset
df = pd.read_csv('user_input.csv')

# Extract the coordinates from the user dataset
user_coordinates = df[['x_coordinate', 'y_coordinate']].values

# Perform KMeans clustering to find the centroids (new container positions)
kmeans = KMeans(n_clusters=num_containers, random_state=42)
kmeans.fit(user_coordinates)

# Get the cluster centroids (new container positions)
container_positions = {
    5001 + i: {'lat': centroid[1], 'lon': centroid[0]}
    for i, centroid in enumerate(kmeans.cluster_centers_)
}

# Function to find the nearest edge node (container) for a user
def find_nearest_edge(lat, lon):
    nearest_node = None
    min_distance = float('inf')

    for node, position in container_positions.items():
        distance = np.sqrt((lat - position['lat'])**2 + (lon - position['lon'])**2)
        if distance < min_distance:
            min_distance = distance
            nearest_node = node

    return nearest_node, min_distance

# Assign users to the nearest edge node (container) based on their coordinates
df[['Assigned_Edge_Node', 'New_Distance']] = df.apply(
    lambda row: find_nearest_edge(row['x_coordinate'], row['y_coordinate']), axis=1, result_type='expand'
)

# Normalize x and y coordinates to range between -99 and 99
x, y = df['x_coordinate'], df['y_coordinate']
df['x_coordinate'] = 198 * (x - x.min()) / (x.max() - x.min()) - 99
df['y_coordinate'] = 198 * (y - y.min()) / (y.max() - y.min()) - 99




# Constants for 5G mid-band (3.5 GHz)
frequency = 3.5e9  # 3.5 GHz in Hz
K = 36.6  # Reference path loss (dB) at d0 = 1 meter
n = 2  # Path loss exponent for urban environment
speed_of_light = 2.99e8  # Speed of light in m/s

# Reverse Calculation Functions using LDPL
def calculate_signal_strength(d, Pt=46, k=K, n=n):
    """
    Computes received signal strength using LDPL model.
    
    Parameters:
        d (float): Distance in meters
        Pt (float): Transmit power in dBm (default: 46 dBm for 5G macro tower)
        k (float): Reference path loss at d0=1m (default: 36.6 dB)
        n (float): Path loss exponent (default: 3.5 for urban)
    
    Returns:
        float: Received signal strength (Pr) in dBm
    """
    if d > 0:
        return Pt - (k + 10 * n * np.log10(d))
    return np.nan

def calculate_latency(d, speed=speed_of_light):
    """
    Computes one-way latency based on distance.
    
    Parameters:
        d (float): Distance in meters
        speed (float): Speed of light in m/s
    
    Returns:
        float: Latency in milliseconds
    """
    if d > 0:
        return (d / speed) * 1000  # Convert to ms
    return np.nan


# Apply the reverse calculations
df['Updated_Signal_Strength'] = df.apply(
    lambda row: calculate_signal_strength(row['New_Distance']) if row['New_Distance'] > 0 else row['Signal_Strength'],
    axis=1
)

df['Updated_Latency'] = df.apply(
    lambda row: calculate_latency(row['New_Distance']) if row['New_Distance'] > 0 else row['Latency'],
    axis=1
)
# Sort dataset by User_ID
df = df.sort_values(by='User_ID')

# Save updated dataset
output_path = os.path.join(output_folder, output_file)
df.to_csv(output_path, index=False)

# Output results
print(f"Dataset updated and saved as '{output_path}'")

#To print the container positions
# print("\nNew container positions (cluster centroids):")
# for node, position in container_positions.items():
#     print(f"Node {node}: Latitude: {position['lat']}, Longitude: {position['lon']}")
