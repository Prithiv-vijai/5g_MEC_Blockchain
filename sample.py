import pandas as pd

# Load the dataset
input_file = "NS3/final_dataset.csv"
output_file = "user_input.csv"
output_size = 500  # Desired size of the output dataset

# Load the original dataset
data = pd.read_csv(input_file)

# Ensure the dataset has enough rows
if len(data) < output_size:
    raise ValueError("The dataset does not have enough rows to sample from.")

# Sample data randomly
sampled_data = data.sample(n=output_size, random_state=42, replace=False)

# Assign User_IDs from 1 to output_size
sampled_data['User_ID'] = range(1, output_size + 1)

# Save the sampled dataset
sampled_data.to_csv(output_file, index=False)

print(f"Updated and sampled dataset saved as {output_file}.")
