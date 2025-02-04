import pandas as pd
import numpy as np

def process_and_generate_coordinates(file1, file2, output_file):
    # Step 1: Load the two datasets into pandas DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Step 2: Merge the datasets on the User_ID column
    merged_df = pd.merge(df1, df2, on='User_ID', how='inner')

    # Step 3: Generate random angles for each user in radians
    np.random.seed(42)  # For reproducibility
    angles = np.random.uniform(0, 2 * np.pi, len(merged_df))

 
    # Step 5: Calculate x and y coordinates based on the normalized distance and random angles
    x = merged_df['Distance_meters'] * np.cos(angles)
    y = merged_df['Distance_meters'] * np.sin(angles)

    # Step 6: Assign the calculated x and y coordinates to the dataset
    merged_df['x_coordinate'] = x
    merged_df['y_coordinate'] = y

    # Step 7: Select the required columns for the final dataset
    columns_to_keep = [
        'User_ID', 'Application_Type', 'Signal_Strength', 'Latency',
        'Required_Bandwidth', 'Allocated_Bandwidth', 'Resource_Allocation', 'Distance_meters',
         'x_coordinate', 'y_coordinate'
    ]
    final_df = merged_df[columns_to_keep]

    # Step 8: Save the updated dataset to a new CSV file
    final_df.to_csv(output_file, index=False)
    print(f"Final dataset with x, y coordinates and normalized distance has been saved to {output_file}.")

# Example usage
file1 = r'augmented_dataset.csv'  # Path of your first dataset
file2 = r'distance.csv'  # Path of your second dataset
output_file = r'final_dataset.csv'  # Desired path for the final dataset

process_and_generate_coordinates(file1, file2, output_file)
