import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import time

# Load the dataset
data = pd.read_csv('NS3/augmented_dataset.csv')

# Define features and target
X = data[['Signal_Strength', 'Latency', 'Required_Bandwidth' ,'Allocated_Bandwidth']]
y = data['Resource_Allocation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define LightGBM parameters based on the provided values
params = {
    'num_leaves': 31,
    'n_estimators': 250,
    'learning_rate': 0.06943367924214801,
    'max_depth': 18,
    'min_data_in_leaf': 25,
    'lambda_l1': 3.0,
    'lambda_l2': 2.0509620328168165,
    'objective': 'regression',
    'metric': 'rmse'
}

# Initialize the LightGBM Regressor
model = lgb.LGBMRegressor(
    num_leaves=params['num_leaves'],
    n_estimators=params['n_estimators'],
    learning_rate=params['learning_rate'],
    max_depth=params['max_depth'],
    min_child_samples=params['min_data_in_leaf'],
    reg_alpha=params['lambda_l1'],
    reg_lambda=params['lambda_l2']
)

# Train the model
print("Training LightGBM model...")
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print(f"Model training completed in {end_time - start_time:.2f} seconds.")

# Save the model to model.txt
print("Saving the model to model.txt...")
booster = model.booster_  # Get the underlying LightGBM Booster
booster.save_model('model.txt')
print("Model saved successfully!")

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
print("Model Evaluation:")

# Check sklearn version to decide on RMSE calculation
try:
    # Use the squared parameter for newer scikit-learn versions
    print(f"Train RMSE: {mean_squared_error(y_train, y_pred_train, squared=False):.4f}")
    print(f"Test RMSE: {mean_squared_error(y_test, y_pred_test, squared=False):.4f}")
except TypeError:
    # For older scikit-learn versions, compute RMSE manually
    train_rmse = mean_squared_error(y_train, y_pred_train) ** 0.5
    test_rmse = mean_squared_error(y_test, y_pred_test) ** 0.5
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

print(f"Train R2 Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test R2 Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
