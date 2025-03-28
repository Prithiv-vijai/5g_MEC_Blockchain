import pandas as pd
import numpy as np
import os

# Constants
Pt = 46  # Transmit power in dBm
PL_d0 = 36.6  # Path loss at reference distance in dB
n = 3.5  # Adjusted path loss exponent (urban environment)
c = 1e8  # Effective propagation speed in m/s
d0 = 1  # Reference distance in meters


# Output filename
output_filename = 'simulated_dataset.csv'

# Check if the output file already exists
if os.path.exists(output_filename):
    pass
else:
    # Load the dataset
    augmented_dataset = pd.read_csv('augmented_dataset.csv')

    # Calculate signal strength-based distance
    augmented_dataset['d_signal_strength'] = d0 * 10**((Pt - augmented_dataset['Signal_Strength'] - PL_d0) / (10 * n))

    # Calculate latency-based distance with reduction factor
    augmented_dataset['d_latency'] = ((c * augmented_dataset['Latency'] * 1e-3) / 2)  # Latency in ms

    # Calculate combined distance
    augmented_dataset['Distance_meters'] = (augmented_dataset['d_signal_strength'] + augmented_dataset['d_latency']) / 2

    np.random.seed(42)  # For reproducibility
    angles = np.random.uniform(0, 2 * np.pi, len(augmented_dataset))

    x = augmented_dataset['Distance_meters'] * np.cos(angles)
    y = augmented_dataset['Distance_meters'] * np.sin(angles)

    augmented_dataset['x_coordinate'] = x
    augmented_dataset['y_coordinate'] = y

    # Drop intermediate columns
    augmented_dataset.drop(columns=['d_signal_strength', 'd_latency'], inplace=True)

    # Save the new dataset
    augmented_dataset.to_csv(output_filename, index=False)
    print(f"New dataset saved as '{output_filename}'")