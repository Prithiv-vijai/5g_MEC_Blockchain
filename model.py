import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import time

# Load the dataset
data = pd.read_csv('NS3/augmented_dataset.csv')

# Define feature sets and targets
X1 = data[['Signal_Strength', 'Latency', 'Required_Bandwidth']]
y1 = data['Resource_Allocation']

X2 = data[['Application_Type','Signal_Strength', 'Latency', 'Required_Bandwidth', 'Resource_Allocation']]  
y2 = data['Allocated_Bandwidth']  

# Split data
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

# LightGBM Parameters
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

# Train first model
print("\nTraining Model 1...")
model1 = lgb.LGBMRegressor(**params)
start_time = time.time()
model1.fit(X1_train, y1_train)
print(f"Model 1 trained in {time.time() - start_time:.2f} seconds.")

# Save Model 1
print("Saving Model 1 as model1.txt...")
model1.booster_.save_model('model1.txt')

# Train second model
print("\nTraining Model 2...")
model2 = lgb.LGBMRegressor(**params)
start_time = time.time()
model2.fit(X2_train, y2_train)
print(f"Model 2 trained in {time.time() - start_time:.2f} seconds.")

# Save Model 2
print("Saving Model 2 as model2.txt...")
model2.booster_.save_model('model2.txt')

# Predictions
y1_pred_test = model1.predict(X1_test)
y2_pred_test = model2.predict(X2_test)

# Evaluation
print("\nModel 1 Evaluation:")
print(f"Test RMSE: {mean_squared_error(y1_test, y1_pred_test, squared=False):.4f}")
print(f"Test R2 Score: {r2_score(y1_test, y1_pred_test):.4f}")
print(f"Test MAE: {mean_absolute_error(y1_test, y1_pred_test):.4f}")

print("\nModel 2 Evaluation:")
print(f"Test RMSE: {mean_squared_error(y2_test, y2_pred_test, squared=False):.4f}")
print(f"Test R2 Score: {r2_score(y2_test, y2_pred_test):.4f}")
print(f"Test MAE: {mean_absolute_error(y2_test, y2_pred_test):.4f}")
