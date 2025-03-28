import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set Times New Roman font for all text
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # Increased base font size

# Define the output folder
output_folder = "output/pre"
graphs_folder = "graphs/results/cluster_comparison"
os.makedirs(graphs_folder, exist_ok=True)

def load_and_process_results():
    """Load and process the clustering analysis results"""
    results_csv_path = os.path.join(output_folder, 'clustering_analysis_results.csv')
    if not os.path.exists(results_csv_path):
        raise FileNotFoundError(f"Results file not found at {results_csv_path}")
    
    results_df = pd.read_csv(results_csv_path)
    
    # Calculate average values for each algorithm
    avg_results = results_df.groupby('Algorithm').agg({
        'Average_Distance': 'mean',
        'Silhouette_Score': 'mean',
        'Calinski_Harabasz_Index': 'mean'
    }).reset_index()
    
    # Calculate ranks
    ranked_results = avg_results.copy()
    
    # For metrics where higher is better
    for metric in ['Silhouette_Score', 'Calinski_Harabasz_Index']:
        ranked_results[f'{metric}_Rank'] = ranked_results[metric].rank(ascending=False, method='min')
    
    # For metrics where lower is better
    ranked_results['Average_Distance_Rank'] = ranked_results['Average_Distance'].rank(ascending=True, method='min')
    
    return results_df, avg_results, ranked_results

def generate_comparison_graphs(results_df):
    """Generate comparison graphs from the results data"""
    # List of algorithms
    algorithms = results_df['Algorithm'].unique()

    # Define parameters to compare
    parameters = ['Average_Distance', 'Silhouette_Score', 'Calinski_Harabasz_Index']

    # Define specific colors and markers for each algorithm
    algorithm_styles = {
        'kmeans': {'color': '#1f77b4', 'marker': 'o', 'label': 'K-Means'},
        'hierarchical': {'color': '#ff7f0e', 'marker': 's', 'label': 'Hierarchical'},
        'dbscan': {'color': '#2ca02c', 'marker': 'D', 'label': 'DBSCAN'},
        'optics': {'color': '#d62728', 'marker': '^', 'label': 'OPTICS'},
        'meanshift': {'color': '#9467bd', 'marker': 'v', 'label': 'Mean-Shift'},
        'gmm': {'color': '#8c564b', 'marker': '<', 'label': 'GMM'},
        'divisive': {'color': '#e377c2', 'marker': '>', 'label': 'Divisive'}
    }

    # Create a plot for each parameter
    for parameter in parameters:
        print(f"\nGenerating plot for {parameter}...")
        plt.figure(figsize=(10, 7))
        
        # Plot a line for each algorithm
        for algorithm in algorithms:
            # Filter results for the current algorithm
            algorithm_results = results_df[results_df['Algorithm'] == algorithm]
            
            # Get color and marker for the algorithm
            style = algorithm_styles.get(algorithm, {'color': 'gray', 'marker': 'x'})
            
            # Plot the line with distinct style
            plt.plot(
                algorithm_results['Cluster_Count'],
                algorithm_results[parameter],
                label=style['label'],
                color=style['color'],
                linestyle='--',
                marker=style['marker'],
                markersize=8,
                linewidth=2.5,
                alpha=0.8
            )
        
        # Add labels and title
        plt.xlabel('Cluster Count', fontsize=22)
        plt.ylabel(parameter.replace('_', ' ').title(), fontsize=22)
        plt.title(f'{parameter.replace("_", " ").title()} vs Cluster Count', 
                 fontsize=22, pad=20, fontweight="bold")
        
        # Set grid and ticks
        plt.xticks(np.arange(2, 21, 2), fontsize=18)
        plt.yticks(fontsize=16)
        plt.grid(True)
        
        # Adjust legend position
        plt.legend(bbox_to_anchor=(0.46, -0.15), loc='upper center', 
                  ncol=3, fontsize=22)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        
        # Save the plot
        plot_path = os.path.join(graphs_folder, f'{parameter}_vs_cluster_count.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Plot saved to {plot_path}")

def save_rankings_to_csv(ranked_results, csv_folder):
    """Save the ranked results to a CSV file"""
    csv_path = os.path.join(csv_folder, 'algorithm_rankings.csv')
    ranked_results.to_csv(csv_path, index=False)
    print(f"\nAlgorithm rankings saved to {csv_path}")
    print("\nRanked Results:")
    print(ranked_results.to_string(index=False))

def main_part2():
    """Main function for comparative analysis"""
    try:
        # Load and process results
        results_df, avg_results, ranked_results = load_and_process_results()
        
        # Print average values
        print("\nAverage Values for Each Algorithm:")
        print(avg_results.to_string(index=False))
        
        # Generate comparison graphs
        generate_comparison_graphs(results_df)
        
        # Save rankings to CSV
        save_rankings_to_csv(ranked_results, output_folder)
        
        print("\nComparative analysis completed successfully.")
        
    except Exception as e:
        print(f"\nError in comparative analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main_part2()