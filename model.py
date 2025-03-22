import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import time

# Load the dataset
data = pd.read_csv('NS3/augmented_dataset.csv')

# Define feature sets and targets
# Model 1: Predicts Allocated Bandwidth
X1 = data[['Signal_Strength', 'Latency', 'Required_Bandwidth']]
y1 = data['Allocated_Bandwidth']

# Model 2: Predicts Resource Allocation
X2 = data[['Application_Type','Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y2 = data['Resource_Allocation']

# Split data
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

# LightGBM Parameters for Model 1 (Allocated Bandwidth prediction)
params_model1 = {
    'num_leaves': 36,  # Max Leaf Nodes
    'n_estimators': 250,  # Max Iteration
    'learning_rate': 0.0575,  # Learning Rate
    'max_depth': 31,  # Max Depth
    'min_data_in_leaf': 29,  # Min Samples Leaf
    'lambda_l1': 2.7798,  # L1 Regularization
    'lambda_l2': 2.4736,  # L2 Regularization
    'objective': 'regression',
    'metric': 'rmse'
}

# LightGBM Parameters for Model 2 (Resource Allocation prediction)
params_model2 = {
    'num_leaves': 31,  # Max Leaf Nodes
    'n_estimators': 250,  # Max Iteration
    'learning_rate': 0.0694,  # Learning Rate
    'max_depth': 18,  # Max Depth
    'min_data_in_leaf': 25,  # Min Samples Leaf
    'lambda_l1': 3.0,  # L1 Regularization
    'lambda_l2': 2.0510,  # L2 Regularization
    'objective': 'regression',
    'metric': 'rmse'
}

# Train Model 1 (Allocated Bandwidth prediction)
print("\nTraining Model 1 (Allocated Bandwidth prediction)...")
model1 = lgb.LGBMRegressor(**params_model1)
start_time = time.time()
model1.fit(X1_train, y1_train)
print(f"Model 1 trained in {time.time() - start_time:.2f} seconds.")

# Save Model 1
print("Saving Model 1 as model1.txt...")
model1.booster_.save_model('model1.txt')

# Train Model 2 (Resource Allocation prediction)
print("\nTraining Model 2 (Resource Allocation prediction)...")
model2 = lgb.LGBMRegressor(**params_model2)
start_time = time.time()
model2.fit(X2_train, y2_train)
print(f"Model 2 trained in {time.time() - start_time:.2f} seconds.")

# Save Model 2
print("Saving Model 2 as model2.txt...")
model2.booster_.save_model('model2.txt')

# Predictions
y1_pred_test = model1.predict(X1_test)  # Model 1 predictions
y2_pred_test = model2.predict(X2_test)  # Model 2 predictions

# Evaluation
print("\nModel 1 (Allocated Bandwidth prediction) Evaluation:")
print(f"Test RMSE: {mean_squared_error(y1_test, y1_pred_test, squared=False):.4f}")
print(f"Test R2 Score: {r2_score(y1_test, y1_pred_test):.4f}")
print(f"Test MAE: {mean_absolute_error(y1_test, y1_pred_test):.4f}")

print("\nModel 2 (Resource Allocation prediction) Evaluation:")
print(f"Test RMSE: {mean_squared_error(y2_test, y2_pred_test, squared=False):.4f}")
print(f"Test R2 Score: {r2_score(y2_test, y2_pred_test):.4f}")
print(f"Test MAE: {mean_absolute_error(y2_test, y2_pred_test):.4f}")