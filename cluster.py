import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('user_input.csv')

# Define number of containers (clusters)
num_containers = 10  # Change this value as needed

# Extract the coordinates from the user dataset
user_coordinates = df[['x_coordinate', 'y_coordinate']].values

# Perform KMeans clustering to find the centroids (new container positions)
kmeans = KMeans(n_clusters=num_containers, random_state=42)
kmeans.fit(user_coordinates)

# Get the cluster centroids (these will be the new container positions)
container_positions = {}
for i, centroid in enumerate(kmeans.cluster_centers_):
    container_positions[5001 + i] = {'lat': centroid[1], 'lon': centroid[0]}

# Function to find the nearest edge node (container)
def find_nearest_edge(lat, lon):
    nearest_node = None
    min_distance = float('inf')
    
    for node, position in container_positions.items():
        distance = np.sqrt((lat - position['lat'])**2 + (lon - position['lon'])**2)
        if distance < min_distance:
            min_distance = distance
            nearest_node = node
            
    return nearest_node, min_distance

# Assign users to nearest edge nodes (containers) based on their coordinates
df[['Assigned_Edge_Node', 'New_Distance']] = df.apply(
    lambda row: find_nearest_edge(row['x_coordinate'], row['y_coordinate']), axis=1, result_type='expand'
)

x = df['x_coordinate']
y = df['y_coordinate']
df['x_coordinate'] = 198 * (x - x.min()) / (x.max() - x.min()) - 99
df['y_coordinate'] = 198 * (y - y.min()) / (y.max() - y.min()) - 99

# Define a scaling factor to reduce the sensitivity of the adjustment (less than 1)
scaling_factor = 0.75  # Adjust this value to control the "looseness"

# Apply the transformation to update Signal Strength and Latency based on the new distance
df['Updated_Signal_Strength'] = df.apply(
    lambda row: row['Signal_Strength'] * (row['New_Distance'] / row['Distance_meters']) ** 0.1
    if row['New_Distance'] != 0 else row['Signal_Strength'], axis=1)

df['Updated_Latency'] = df.apply(
    lambda row: row['Latency'] * (row['New_Distance'] / row['Distance_meters']) ** scaling_factor 
    if row['Distance_meters'] != 0 else row['Latency'], axis=1)

# Save the updated dataset with new distances, signal strength, and latency
df.to_csv('clustered_user_data.csv', index=False)

print("Dataset updated and saved as 'clustered_user_data.csv'")

# Output the new container positions
print("\nNew container positions (cluster centroids):")
for node, position in container_positions.items():
    print(f"Node {node}: Latitude: {position['lat']}, Longitude: {position['lon']}")
