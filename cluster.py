import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, OPTICS
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

# Load environment variables
num_containers_list = [2,4,6,8,10,12,14,16,18,20]  # Cluster counts to test
output_folder = os.environ.get("OUTPUT_FOLDER", "output/pre")  # Default: 'pre' directory
os.makedirs(output_folder, exist_ok=True)

# Load dataset
df = pd.read_csv('simulated_dataset.csv')
user_coordinates = df[['x_coordinate', 'y_coordinate']].values

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

def update_signal_latency(df):
    # Constants (same as in your original setup)
    Pt = 46  # Transmit power in dBm
    PL_d0 = 36.6  # Path loss at reference distance in dB
    n = 3.5  # Path loss exponent (urban environment)
    c = 1e8  # Effective propagation speed in m/s
    d0 = 1  # Reference distance in meters
    
    # Update Signal Strength using inverse path loss model
    df['Updated_Signal_Strength'] = Pt - PL_d0 - (10 * n * np.log10(df['New_Distance']/d0))
    
    # Update Latency using inverse propagation delay model
    # Latency in milliseconds = (2 * distance) / (speed of light * 1000)
    df['Updated_Latency'] = (2 * df['New_Distance']) / (c) * 1000  # Convert to ms
    
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
    
    while True:
        # Calculate current centroids (exclude noise)
        centroids = np.array([coordinates[labels == label].mean(axis=0) 
                          for label in unique_labels if label != -1])
        
        # Exit condition
        if len(centroids) == n_clusters:
            break
            
        # Case 1: Too many clusters (merge closest pair)
        if len(centroids) > n_clusters:
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(centroids, centroids)
            np.fill_diagonal(dist_matrix, np.inf)
            i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            # Merge cluster j into i
            labels[labels == unique_labels[j]] = unique_labels[i]
            
        # Case 2: Too few clusters (split largest)
        elif len(centroids) < n_clusters:
            largest_cluster = np.argmax(np.bincount(labels[labels != -1]))
            largest_indices = np.where(labels == largest_cluster)[0]
            kmeans = KMeans(n_clusters=n_clusters - len(centroids) + 1, random_state=42)
            sub_labels = kmeans.fit_predict(coordinates[largest_indices])
            labels[largest_indices] = sub_labels + max(unique_labels) + 1
        
        unique_labels = np.unique(labels)

    # Final centroids (edge node positions)
    centroids = np.array([coordinates[labels == label].mean(axis=0) 
                        for label in unique_labels if label != -1])
    
    # Assign to DataFrame (same as your original code)
    df['Cluster'] = labels
    container_positions = {
        5001 + i: {'lat': centroid[1], 'lon': centroid[0]}
        for i, centroid in enumerate(centroids)
    }
    df[['Assigned_Edge_Node', 'New_Distance']] = df.apply(
        lambda row: find_nearest_edge(row['x_coordinate'], row['y_coordinate'], container_positions),
        axis=1, result_type='expand'
    )
    df = normalize_coordinates(df)
    df = update_signal_latency(df)
    df['Network_Slice'] = df['Application_Type'].apply(assign_network_slice)
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
    # Step 1: Run OPTICS clustering
    optics = OPTICS(min_samples=5)
    labels = optics.fit_predict(coordinates)
    
    # Step 2: Post-processing to enforce n_clusters
    unique_labels = np.unique(labels[labels != -1])  # Exclude noise (-1)
    
    # Case 1: Too many clusters (merge)
    if len(unique_labels) > n_clusters:
        # Calculate current centroids
        centroids = np.array([coordinates[labels == label].mean(axis=0) 
                            for label in unique_labels])
        
        # Iteratively merge closest clusters until we reach n_clusters
        while len(unique_labels) > n_clusters:
            # Compute pairwise distances between centroids
            dist_matrix = cdist(centroids, centroids)
            np.fill_diagonal(dist_matrix, np.inf)
            
            # Find two closest clusters
            i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            
            # Merge cluster j into cluster i
            labels[labels == unique_labels[j]] = unique_labels[i]
            
            # Recalculate centroids and unique labels
            unique_labels = np.unique(labels[labels != -1])
            centroids = np.array([coordinates[labels == label].mean(axis=0) 
                               for label in unique_labels])
    
    # Case 2: Too few clusters or noise present (use K-Means on the clustered points)
    if len(unique_labels) < n_clusters or -1 in labels:
        # Get indices of clustered points (non-noise)
        clustered_indices = np.where(labels != -1)[0]
        
        # If we have some clusters, use them as initialization
        if len(unique_labels) > 0:
            init_centers = np.array([coordinates[labels == label].mean(axis=0) 
                                   for label in unique_labels])
            # Pad with random points if needed
            if len(init_centers) < n_clusters:
                additional = coordinates[np.random.choice(
                    clustered_indices, 
                    n_clusters - len(init_centers),
                    replace=False
                )]
                init_centers = np.vstack([init_centers, additional])
            kmeans = KMeans(n_clusters=n_clusters, init=init_centers, n_init=1, random_state=42)
        else:
            # Pure noise case - regular K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        # Fit on all points (including former noise)
        labels = kmeans.fit_predict(coordinates)
    
    # Step 3: Assign final cluster IDs (5001, 5002, ...)
    unique_labels = np.unique(labels)
    cluster_id_mapping = {label: 5001 + i for i, label in enumerate(unique_labels)}
    df['Cluster'] = [cluster_id_mapping[label] for label in labels]

    # Step 4: Calculate centroids and container positions
    centroids = np.array([coordinates[labels == label].mean(axis=0) 
                         for label in unique_labels])
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
    # Step 1: Dynamic bandwidth estimation targeting n_clusters
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(coordinates)
    distances, _ = neigh.kneighbors(coordinates)
    
    # Sort the 5th-nearest neighbor distances and pick nth largest
    sorted_distances = np.sort(distances[:, -1])
    bandwidth = sorted_distances[-min(n_clusters, len(sorted_distances))] * 1.5  # Adjusted scaling
    
    # Step 2: Run MeanShift with automatic bandwidth selection if needed
    if bandwidth == 0:  # Fallback for very dense data
        meanshift = MeanShift()  # Let sklearn estimate bandwidth
    else:
        meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    
    labels = meanshift.fit_predict(coordinates)
    unique_labels = np.unique(labels)
    
    # Step 3: Post-processing to enforce exact n_clusters
    if len(unique_labels) != n_clusters:
        # Case 1: Too many clusters - merge smallest into nearest
        if len(unique_labels) > n_clusters:
            centroids = np.array([coordinates[labels == label].mean(axis=0) 
                                for label in unique_labels])
            
            # Compute cluster sizes and sort by size (ascending)
            cluster_sizes = np.array([np.sum(labels == label) for label in unique_labels])
            size_order = np.argsort(cluster_sizes)
            
            # Merge smallest clusters into their nearest neighbor
            for i in range(len(unique_labels) - n_clusters):
                target_label = unique_labels[size_order[i]]
                dists = cdist([centroids[size_order[i]]], centroids)
                dists[0, size_order[i]] = np.inf  # Ignore self
                nearest_idx = np.argmin(dists)
                labels[labels == target_label] = unique_labels[nearest_idx]
        
        # Case 2: Too few clusters - split largest using K-Means
        elif len(unique_labels) < n_clusters:
            largest_cluster = np.argmax(np.bincount(labels))
            largest_indices = np.where(labels == largest_cluster)[0]
            
            # Use K-Means to split the largest cluster
            kmeans = KMeans(n_clusters=n_clusters - len(unique_labels) + 1, 
                           random_state=42)
            sub_labels = kmeans.fit_predict(coordinates[largest_indices])
            
            # Reassign with new labels (offset to avoid conflicts)
            max_label = np.max(labels)
            labels[largest_indices] = sub_labels + max_label + 1
    
    # Step 4: Final cluster assignment and centroid calculation
    unique_labels = np.unique(labels)
    cluster_id_mapping = {label: 5001 + i for i, label in enumerate(unique_labels)}
    df['Cluster'] = [cluster_id_mapping[label] for label in labels]
    
    # Calculate final centroids
    centroids = np.array([coordinates[labels == label].mean(axis=0) 
                        for label in unique_labels])
    
    # Step 5: Container positions and edge assignments
    container_positions = {
        5001 + i: {'lat': centroid[1], 'lon': centroid[0]}
        for i, centroid in enumerate(centroids)
    }
    
    df[['Assigned_Edge_Node', 'New_Distance']] = df.apply(
        lambda row: find_nearest_edge(row['x_coordinate'], row['y_coordinate'], container_positions),
        axis=1, result_type='expand'
    )
    
    # Final processing
    df = normalize_coordinates(df)
    df = update_signal_latency(df)
    df['Network_Slice'] = df['Application_Type'].apply(assign_network_slice)
    df.to_csv(output_path, index=False)


# Create output folders for each algorithm
algorithms = {
    'kmeans': run_kmeans,
    'dbscan': run_dbscan,
    'hierarchical': run_hierarchical,
    'meanshift': run_meanshift,
    'optics': run_optics,
    'gmm': run_gmm,
    'divisive': run_divisive 
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