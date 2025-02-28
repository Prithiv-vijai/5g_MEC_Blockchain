import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Load environment variables
num_containers = int(os.environ.get("NUM_CONTAINERS", 10))  # Default: 10 containers
output_folder = os.environ.get("OUTPUT_FOLDER", "output")  # Default: 'output' directory
output_file = os.environ.get("OUTPUT_FILE", "clustered_user_data.csv")  # Default: 'clustered_user_data.csv'
# Constants
frequency = 2.4e9  # 2.4 GHz in Hz (WiFi frequency)
K = 32.45  # Constant for 1 GHz and 1 meter
speed_of_light = 3.0e8  # Speed of light in m/s

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

# Define scaling factors for signal strength and latency adjustments based on distance
scaling_factor = 0.75
reference_factor = 0.26


# Reverse Calculation Functions
def calculate_signal_strength(d, f=frequency, k=K):
    if d > 0:
        return -(20 * np.log10(d) + 20 * np.log10(f) + k)
    return np.nan

def calculate_latency(d, speed=speed_of_light):
    if d > 0:
        return (d / speed) * 1000  # Latency in ms
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
