import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

# Load environment variables
num_containers_list = [12,14,16,18,20]  # Cluster counts to test
output_folder = os.environ.get("OUTPUT_FOLDER", "output/pre")  # Default: 'pre' directory
os.makedirs(output_folder, exist_ok=True)

# Load dataset
df = pd.read_csv('simulated_dataset.csv')
user_coordinates = df[['x_coordinate', 'y_coordinate']].values

# Constants
Pt = 46  # Transmit power in dBm
PL_d0 = 36.6  # Path loss at reference distance in dB
n = 3.5  # Path loss exponent for urban environment
d0 = 1  # Reference distance in meters
c = 1e8  # Effective propagation speed in m/s
k = 0.7  # Reduction factor for latency-based distance
scaling_factor = 0.75
reference_factor = 0.26

# Function to calculate Euclidean distance between two points
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Function to find the nearest edge node (container) for a user
def find_nearest_edge(lat, lon, container_positions):
    nearest_node = None
    min_distance = float('inf')

    for node, position in container_positions.items():
        distance = calculate_distance(lat, lon, position['lon'], position['lat'])
        if distance < min_distance:
            min_distance = distance
            nearest_node = node

    return nearest_node, min_distance

# Function to normalize coordinates to range between -99 and 99
def normalize_coordinates(df):
    x, y = df['x_coordinate'], df['y_coordinate']
    df['x_coordinate'] = 198 * (x - x.min()) / (x.max() - x.min()) - 99
    df['y_coordinate'] = 198 * (y - y.min()) / (y.max() - y.min()) - 99
    return df

# Function to update signal strength and latency
def update_signal_latency(df):
    df['Updated_Signal_Strength'] = df.apply(
        lambda row: row['Signal_Strength'] * (row['New_Distance'] / row['Distance_meters']) ** reference_factor
        if row['New_Distance'] != 0 else row['Signal_Strength'], axis=1
    )
    df['Updated_Latency'] = df.apply(
        lambda row: row['Latency'] * (row['New_Distance'] / row['Distance_meters']) ** scaling_factor
        if row['Distance_meters'] != 0 else row['Latency'], axis=1
    )
    return df


def assign_network_slice(application_type):
    """
    Assigns a network slice based on the application type.
    """
    if application_type in [0, 2, 5, 7]:  # Background_Download, File_Download, Streaming, Video_Streaming
        return 'eMBB'  # High bandwidth applications
    elif application_type in [1, 4, 6, 8, 9]:  # Emergency_Service, Online_Gaming, Video_Call, VoIP_Call, Voice_Call
        return 'URLLC'  # Low latency applications
    else:  # IoT_Temperature, Web_Browsing
        return 'mMTC'  # IoT & low bandwidth applications
    
    
# Function to run KMeans clustering
def run_kmeans(df, coordinates, n_clusters, output_path):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(coordinates)
    df['Cluster'] = labels

    # Get the cluster centroids (new container positions)
    container_positions = {
        5001 + i: {'lat': centroid[1], 'lon': centroid[0]}
        for i, centroid in enumerate(kmeans.cluster_centers_)
    }

    # Assign users to the nearest edge node and calculate new distance
    df[['Assigned_Edge_Node', 'New_Distance']] = df.apply(
        lambda row: find_nearest_edge(row['x_coordinate'], row['y_coordinate'], container_positions),
        axis=1, result_type='expand'
    )

    # Normalize coordinates
    df = normalize_coordinates(df)
    # Update signal strength and latency
    df = update_signal_latency(df)
    # Assign Network Slices based on Application Type
    df['Network_Slice'] = df['Application_Type'].apply(assign_network_slice)
    # Save the updated dataset
    df.to_csv(output_path, index=False)

# Function to run DBSCAN clustering
def run_dbscan(df, coordinates, n_clusters, output_path):
    # Estimate eps using k-nearest neighbors
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(coordinates)
    distances, _ = neigh.kneighbors(coordinates)
    distances = np.sort(distances[:, -1], axis=0)
    eps = distances[-n_clusters]  # Adjust eps to get approximately n_clusters

    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(coordinates)

    # Post-processing to enforce n_clusters
    unique_labels = np.unique(labels)
    if len(unique_labels) > n_clusters:
        # Merge smaller clusters into the largest cluster
        largest_cluster = np.argmax(np.bincount(labels[labels != -1]))
        labels[labels != largest_cluster] = largest_cluster
    elif len(unique_labels) < n_clusters:
        # Split the largest cluster into smaller clusters
        largest_cluster = np.argmax(np.bincount(labels[labels != -1]))
        largest_cluster_indices = np.where(labels == largest_cluster)[0]
        kmeans = KMeans(n_clusters=n_clusters - len(unique_labels) + 1, random_state=42)
        sub_labels = kmeans.fit_predict(coordinates[largest_cluster_indices])
        labels[largest_cluster_indices] = sub_labels + len(unique_labels)

    df['Cluster'] = labels

    # Calculate centroids for DBSCAN clusters
    unique_labels = np.unique(labels)
    centroids = np.array([coordinates[labels == label].mean(axis=0) for label in unique_labels if label != -1])

    # Get the cluster centroids (new container positions)
    container_positions = {
        5001 + i: {'lat': centroid[1], 'lon': centroid[0]}
        for i, centroid in enumerate(centroids)
    }

    # Assign users to the nearest edge node and calculate new distance
    df[['Assigned_Edge_Node', 'New_Distance']] = df.apply(
        lambda row: find_nearest_edge(row['x_coordinate'], row['y_coordinate'], container_positions),
        axis=1, result_type='expand'
    )

    # Normalize coordinates
    df = normalize_coordinates(df)
    # Update signal strength and latency
    df = update_signal_latency(df)
    # Assign Network Slices based on Application Type
    df['Network_Slice'] = df['Application_Type'].apply(assign_network_slice)
    # Save the updated dataset
    df.to_csv(output_path, index=False)
    
    
    
# Function to run Hierarchical Clustering
def run_hierarchical(df, coordinates, n_clusters, output_path):
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hierarchical.fit_predict(coordinates)
    df['Cluster'] = labels

    # Calculate centroids for hierarchical clusters
    centroids = np.array([coordinates[labels == label].mean(axis=0) for label in np.unique(labels)])

    # Get the cluster centroids (new container positions)
    container_positions = {
        5001 + i: {'lat': centroid[1], 'lon': centroid[0]}
        for i, centroid in enumerate(centroids)
    }

    # Assign users to the nearest edge node and calculate new distance
    df[['Assigned_Edge_Node', 'New_Distance']] = df.apply(
        lambda row: find_nearest_edge(row['x_coordinate'], row['y_coordinate'], container_positions),
        axis=1, result_type='expand'
    )

    # Normalize coordinates
    df = normalize_coordinates(df)
    # Update signal strength and latency
    df = update_signal_latency(df)
    # Assign Network Slices based on Application Type
    df['Network_Slice'] = df['Application_Type'].apply(assign_network_slice)
    # Save the updated dataset
    df.to_csv(output_path, index=False)
    
    
def run_divisive(df, coordinates, n_clusters, output_path):
    """
    Perform top-down (divisive) clustering on the given coordinates.
    """
    # Function to recursively split clusters
    def split_clusters(points, n_clusters):
        if n_clusters == 1:
            return np.zeros(len(points), dtype=int)  # All points belong to cluster 0
        
        # Split the points into 2 clusters using KMeans
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(points)
        
        # Recursively split each subcluster
        cluster1_indices = np.where(labels == 0)[0]
        cluster2_indices = np.where(labels == 1)[0]
        
        # Assign cluster IDs
        cluster_ids = np.zeros(len(points), dtype=int)
        cluster_ids[cluster1_indices] = split_clusters(points[cluster1_indices], n_clusters // 2)
        cluster_ids[cluster2_indices] = split_clusters(points[cluster2_indices], n_clusters - n_clusters // 2) + n_clusters // 2
        
        return cluster_ids

    # Perform divisive clustering
    labels = split_clusters(coordinates, n_clusters)
    df['Cluster'] = labels

    # Calculate centroids for divisive clusters
    centroids = np.array([coordinates[labels == label].mean(axis=0) for label in np.unique(labels)])

    # Get the cluster centroids (new container positions)
    container_positions = {
        5001 + i: {'lat': centroid[1], 'lon': centroid[0]}
        for i, centroid in enumerate(centroids)
    }

    # Assign users to the nearest edge node and calculate new distance
    df[['Assigned_Edge_Node', 'New_Distance']] = df.apply(
        lambda row: find_nearest_edge(row['x_coordinate'], row['y_coordinate'], container_positions),
        axis=1, result_type='expand'
    )

    # Normalize coordinates
    df = normalize_coordinates(df)
    # Update signal strength and latency
    df = update_signal_latency(df)
    # Assign Network Slices based on Application Type
    df['Network_Slice'] = df['Application_Type'].apply(assign_network_slice)
    # Save the updated dataset
    df.to_csv(output_path, index=False)
    
    
    
# Function to run Gaussian Mixture Models (GMM) clustering
def run_gmm(df, coordinates, n_clusters, output_path):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = gmm.fit_predict(coordinates)
    df['Cluster'] = labels

    # Calculate centroids for GMM clusters
    centroids = gmm.means_

    # Get the cluster centroids (new container positions)
    container_positions = {
        5001 + i: {'lat': centroid[1], 'lon': centroid[0]}
        for i, centroid in enumerate(centroids)
    }

    # Assign users to the nearest edge node and calculate new distance
    df[['Assigned_Edge_Node', 'New_Distance']] = df.apply(
        lambda row: find_nearest_edge(row['x_coordinate'], row['y_coordinate'], container_positions),
        axis=1, result_type='expand'
    )

    # Normalize coordinates
    df = normalize_coordinates(df)
    # Update signal strength and latency
    df = update_signal_latency(df)
    # Assign Network Slices based on Application Type
    df['Network_Slice'] = df['Application_Type'].apply(assign_network_slice)
    # Save the updated dataset
    df.to_csv(output_path, index=False)



# Function to run  OPTICS clustering
def run_optics(df, coordinates, n_clusters, output_path):
    optics = OPTICS(min_samples=5)
    labels = optics.fit_predict(coordinates)

    # Post-processing to enforce n_clusters
    unique_labels = np.unique(labels)
    if len(unique_labels) != n_clusters:
        # Use KMeans to enforce the exact number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(coordinates)

    # Ensure cluster IDs are in the range 5001 to 5001 + n_clusters - 1
    unique_labels = np.unique(labels)
    cluster_id_mapping = {label: 5001 + i for i, label in enumerate(unique_labels)}
    df['Cluster'] = [cluster_id_mapping[label] for label in labels]

    # Calculate centroids for OPTICS clusters
    unique_labels = np.unique(labels)
    centroids = np.array([coordinates[labels == label].mean(axis=0) for label in unique_labels if label != -1])

    # Get the cluster centroids (new container positions)
    container_positions = {
        5001 + i: {'lat': centroid[1], 'lon': centroid[0]}
        for i, centroid in enumerate(centroids)
    }

    # Assign users to the nearest edge node and calculate new distance
    df[['Assigned_Edge_Node', 'New_Distance']] = df.apply(
        lambda row: find_nearest_edge(row['x_coordinate'], row['y_coordinate'], container_positions),
        axis=1, result_type='expand'
    )

    # Normalize coordinates
    df = normalize_coordinates(df)
    # Update signal strength and latency
    df = update_signal_latency(df)
    # Assign Network Slices based on Application Type
    df['Network_Slice'] = df['Application_Type'].apply(assign_network_slice)
    # Save the updated dataset
    df.to_csv(output_path, index=False)
    
    

def run_meanshift(df, coordinates, n_clusters, output_path):
    # Estimate bandwidth using a heuristic (e.g., median distance)
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(coordinates)
    distances, _ = neigh.kneighbors(coordinates)
    bandwidth = np.median(distances[:, -1])  # Adjust bandwidth to get approximately n_clusters

    meanshift = MeanShift(bandwidth=bandwidth)
    labels = meanshift.fit_predict(coordinates)

    # Post-processing to enforce n_clusters
    unique_labels = np.unique(labels)
    if len(unique_labels) != n_clusters:
        # Use KMeans to enforce the exact number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(coordinates)

    # Ensure cluster IDs are in the range 5001 to 5001 + n_clusters - 1
    unique_labels = np.unique(labels)
    cluster_id_mapping = {label: 5001 + i for i, label in enumerate(unique_labels)}
    df['Cluster'] = [cluster_id_mapping[label] for label in labels]

    # Calculate centroids for Mean Shift clusters
    centroids = np.array([coordinates[labels == label].mean(axis=0) for label in np.unique(labels)])

    # Get the cluster centroids (new container positions)
    container_positions = {
        5001 + i: {'lat': centroid[1], 'lon': centroid[0]}
        for i, centroid in enumerate(centroids)
    }

    # Assign users to the nearest edge node and calculate new distance
    df[['Assigned_Edge_Node', 'New_Distance']] = df.apply(
        lambda row: find_nearest_edge(row['x_coordinate'], row['y_coordinate'], container_positions),
        axis=1, result_type='expand'
    )

    # Normalize coordinates
    df = normalize_coordinates(df)
    # Update signal strength and latency
    df = update_signal_latency(df)
    # Assign Network Slices based on Application Type
    df['Network_Slice'] = df['Application_Type'].apply(assign_network_slice)
    # Save the updated dataset
    df.to_csv(output_path, index=False)


# Create output folders for each algorithm
algorithms = {
    # 'kmeans': run_kmeans,
    'dbscan': run_dbscan,
    # 'hierarchical': run_hierarchical,
    # 'meanshift': run_meanshift,
    # 'optics': run_optics,
    # 'gmm': run_gmm,
    # 'divisive': run_divisive 
}

for algorithm_name in algorithms:
    algorithm_folder = os.path.join(output_folder, algorithm_name)
    os.makedirs(algorithm_folder, exist_ok=True)

# Run each algorithm for different cluster counts
for n_clusters in num_containers_list:
    print(f"\nRunning for {n_clusters} clusters...")
    
    # Define the output file name based on cluster count
    cluster_file_name = f"{n_clusters}_cluster.csv"
    
    for algorithm_name, algorithm_func in algorithms.items():
        # Define output file path
        output_file = os.path.join(output_folder, algorithm_name, cluster_file_name)
        
        # Print start of task
        print(f"  Running {algorithm_name.capitalize()}...")
        
        # Run the algorithm
        if algorithm_name in ['kmeans', 'hierarchical', 'divisive', 'gmm', 'dbscan', 'meanshift', 'optics']:
            algorithm_func(df.copy(), user_coordinates, n_clusters, output_file)
        
        # Print completion of task
        print(f"  Completed {algorithm_name.capitalize()} for {n_clusters} clusters.")
    
    # Print completion status for the current cluster count
    print(f"Completed all algorithms for {n_clusters} clusters.")

print("\nClustering completed and results saved.")