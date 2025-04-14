import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Define the output folder
output_folder = "../output/pre"
graphs_folder = "../graphs/results/cluster_plots"

# Create the graphs folder if it doesn't exist
os.makedirs(graphs_folder, exist_ok=True)

# List of algorithms and cluster counts
algorithms = ['kmeans', 'dbscan', 'hierarchical', 'meanshift', 'optics', 'gmm', 'divisive']
num_containers_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# Initialize a dictionary to store results
results = {
    'Algorithm': [],
    'Cluster_Count': [],
    'Average_Distance': [],
    'Silhouette_Score': [],
    'Davies_Bouldin_Index': [],
    'Calinski_Harabasz_Index': [],
    'Execution_Time': []
}

# Function to calculate metrics and store results
def analyze_results(df, algorithm, n_clusters):
    start_time = time.time()  # Start timing
    
    # Calculate average distance
    avg_distance = df['New_Distance'].mean()

    # Calculate clustering metrics
    if n_clusters > 1:  # Silhouette Score requires at least 2 clusters
        silhouette_avg = silhouette_score(df[['x_coordinate', 'y_coordinate']], df['Cluster'])
        calinski_harabasz = calinski_harabasz_score(df[['x_coordinate', 'y_coordinate']], df['Cluster'])
    else:
        silhouette_avg = np.nan
        calinski_harabasz = np.nan


    # Store results
    results['Algorithm'].append(algorithm)
    results['Cluster_Count'].append(n_clusters)
    results['Average_Distance'].append(avg_distance)
    results['Silhouette_Score'].append(silhouette_avg)
    results['Calinski_Harabasz_Index'].append(calinski_harabasz)

# Function to visualize clusters and save the plot
def visualize_clusters(df, algorithm, n_clusters):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x='x_coordinate', y='y_coordinate',
        hue='Cluster', palette='viridis',
        data=df, legend='full'
    )
    plt.title(f'{algorithm.capitalize()} Clustering (Clusters = {n_clusters})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # Save the plot in the algorithm-specific folder
    algorithm_folder = os.path.join(graphs_folder, algorithm)
    os.makedirs(algorithm_folder, exist_ok=True)
    plt.savefig(os.path.join(algorithm_folder, f'{n_clusters}_cluster.png'))
    plt.close()

# Function to analyze and visualize results for all algorithms
def analyze_and_visualize_clusters():
    for algorithm in algorithms:
        print(f"Starting analysis for {algorithm.capitalize()}...")
        for n_clusters in num_containers_list:
            print(f"Processing {algorithm.capitalize()} with {n_clusters} clusters...")
            
            # Define the input file path
            input_file = os.path.join(output_folder, algorithm, f"{n_clusters}_cluster.csv")
            
            # Check if the file exists
            if not os.path.exists(input_file):
                print(f"File not found: {input_file}. Skipping...")
                continue  # Skip to the next iteration
            
            try:
                # Load the clustered dataset
                df = pd.read_csv(input_file)
                
                # Analyze results
                analyze_results(df, algorithm, n_clusters)
                
                # Visualize clusters and save the plot
                visualize_clusters(df, algorithm, n_clusters)
                print(f"Completed processing {algorithm.capitalize()} with {n_clusters} clusters.")
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
        print(f"Completed analysis for {algorithm.capitalize()}.")

# Function to save results to a CSV file
def save_results_to_csv():
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save results to a CSV file
    results_csv_path = os.path.join(output_folder, 'clustering_analysis_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}.")

# Main function for Part 1: CSV generation and scatterplotting
def main_part1():
    # Step 1: Analyze and visualize clusters
    analyze_and_visualize_clusters()

    # Step 2: Save results to CSV
    save_results_to_csv()

    print("CSV generation and scatterplotting completed successfully.")

# Run Part 1
if __name__ == "__main__":
    main_part1()