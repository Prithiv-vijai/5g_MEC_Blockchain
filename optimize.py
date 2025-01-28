import pandas as pd

# Load the dataset
input_file = "NS3/final_dataset.csv"
output_file = "user_input.csv"
output_size = 5000  # Desired size of the output dataset

# Load the original dataset
data = pd.read_csv(input_file)

# Update the Resource_Allocation column
data['Resource_Allocation'] = data['Resource_Allocation'] * 1.2
data['Resource_Allocation'] = data['Resource_Allocation'].clip(upper=100)

# Ensure equal sampling for each Application_Type
num_application_types = data['Application_Type'].nunique()
samples_per_type = output_size // num_application_types

# Sample data equally for each Application_Type
sampled_data = (
    data.groupby('Application_Type')
    .apply(lambda x: x.sample(n=samples_per_type, random_state=42, replace=True))
    .reset_index(drop=True)
)

# Save the sampled dataset
sampled_data.to_csv(output_file, index=False)

print(f"Updated and sampled dataset saved as {output_file}.")
