import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the output folder
output_folder = "output/pre"
graphs_folder = "graphs/results"

# Function to calculate average values and rank algorithms
def calculate_averages_and_rank(results_df):
    # Group by algorithm and calculate the mean for each metric
    avg_results = results_df.groupby('Algorithm').mean().reset_index()

    # Rank algorithms for each metric
    ranked_results = avg_results.copy()
    for parameter in ['Average_Distance', 'Silhouette_Score', 'Davies_Bouldin_Index', 'Calinski_Harabasz_Index']:
        if parameter in ['Silhouette_Score', 'Calinski_Harabasz_Index']:
            # Higher values are better
            ranked_results[f'{parameter}_Rank'] = ranked_results[parameter].rank(ascending=False, method='min')
        else:
            # Lower values are better
            ranked_results[f'{parameter}_Rank'] = ranked_results[parameter].rank(ascending=True, method='min')
    
    return avg_results, ranked_results

# Function to generate comparison graphs from the results CSV
def generate_comparison_graphs():
    # Load the analysis results from the CSV file
    results_csv_path = os.path.join(output_folder, 'clustering_analysis_results.csv')
    results_df = pd.read_csv(results_csv_path)

    # Calculate average values and rank algorithms
    avg_results, ranked_results = calculate_averages_and_rank(results_df)

    # Print average values
    print("\nAverage Values for Each Algorithm:")
    print(avg_results.to_string(index=False))

    # Print ranked results for each parameter
    print("\nRanked Results for Each Parameter:")
    for parameter in ['Average_Distance', 'Silhouette_Score', 'Davies_Bouldin_Index', 'Calinski_Harabasz_Index']:
        print(f"\nRanking for {parameter.replace('_', ' ').title()}:")
        print(ranked_results[['Algorithm', parameter, f'{parameter}_Rank']].sort_values(by=f'{parameter}_Rank').to_string(index=False))

    # List of algorithms
    algorithms = results_df['Algorithm'].unique()

    # Define parameters to compare
    parameters = ['Average_Distance', 'Silhouette_Score', 'Davies_Bouldin_Index', 'Calinski_Harabasz_Index']

    # Create a plot for each parameter
    for parameter in parameters:
        print(f"\nGenerating plot for {parameter}...")
        plt.figure(figsize=(14, 8))  # Increase figure size
        sns.set(style="whitegrid")

        # Define colors, line styles, and markers
        colors = sns.color_palette("husl", n_colors=len(algorithms))
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
        markers = ['o', 's', 'D', '^', 'v', '<', '>']

        # Plot a line for each algorithm
        for i, algorithm in enumerate(algorithms):
            # Filter results for the current algorithm
            algorithm_results = results_df[results_df['Algorithm'] == algorithm]
            
            # Plot the line with distinct style and transparency
            plt.plot(
                algorithm_results['Cluster_Count'],
                algorithm_results[parameter],
                label=algorithm.capitalize(),
                color=colors[i],
                linestyle=line_styles[i],  # Distinct line style
                marker=markers[i],        # Distinct marker
                markersize=8,
                linewidth=2.5,
                alpha=0.8  # Add transparency
            )
        
        # Add labels, title, and legend
        plt.xlabel('Cluster Count', fontsize=14)
        plt.ylabel(parameter.replace('_', ' ').title(), fontsize=14)
        plt.title(f'{parameter.replace("_", " ").title()} vs Cluster Count', fontsize=16)
        plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)  # Move legend outside

        # Add grid for better readability
        plt.grid(True)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save the plot in the common graphs folder
        plot_path = os.path.join(graphs_folder, f'{parameter}_vs_cluster_count.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {plot_path}.")

# Main function for Part 2: Comparative analysis
def main_part2():
    # Generate comparison graphs
    generate_comparison_graphs()

    print("\nComparative analysis completed successfully.")
    # Data from your rankings
    data = {
        'Algorithm': ['meanshift', 'optics', 'kmeans', 'hierarchical', 'dbscan', 'divisive', 'gmm'],
        'Average_Distance': [497.007636, 497.007636, 497.023062, 498.348802, 500.641362, 511.803597, 521.537868],
        'Average_Distance_Rank': [1.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        'Silhouette_Score': [0.324483, 0.324483, 0.324483, 0.295187, 0.325184, 0.279784, 0.307504],
        'Silhouette_Score_Rank': [2.0, 2.0, 2.0, 6.0, 1.0, 7.0, 5.0],
        'Davies_Bouldin_Index': [0.902343, 0.902343, 0.902343, 0.923155, 0.890922, 0.984692, 13.624563],
        'Davies_Bouldin_Index_Rank': [2.0, 2.0, 2.0, 5.0, 1.0, 6.0, 7.0],
        'Calinski_Harabasz_Index': [7151.693353, 7151.693353, 7151.693353, 5990.016818, 7013.803287, 6244.380947, 5376.079385],
        'Calinski_Harabasz_Index_Rank': [1.0, 1.0, 1.0, 6.0, 4.0, 5.0, 7.0]
    }
    csv_folder = "output/pre"
    # Convert data to a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    csv_path = os.path.join(csv_folder, 'algorithm_rankings.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV file saved to {csv_path}")

# Run Part 2
if __name__ == "__main__":
    main_part2()