import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import time
import matplotlib.patches as mpatches

# Set output directory
output_dir = 'graphs/model_output/'
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv('NS3/augmented_dataset.csv')

# Define models
models = {
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'Hgbrt': HistGradientBoostingRegressor(random_state=42),
    'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1)
}

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Function to train and evaluate models
def evaluate_models(X, y, target_name):
    results = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    for name, model in models.items():
        start_time = time.time()
        y_pred = cross_val_predict(model, X_train, y_train, cv=kf)
        completion_time = time.time() - start_time
        
        results.append({
            'Model': name,
            'Target': target_name,
            'MSE': mean_squared_error(y_train, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_pred)),
            'MAE': mean_absolute_error(y_train, y_pred),
            'R2': r2_score(y_train, y_pred),
            'MAPE': mean_absolute_percentage_error(y_train, y_pred),
            'Completion_Time': completion_time
        })
    
    return pd.DataFrame(results)

# Evaluate for Resource_Allocation
X1 = df[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y1 = df['Resource_Allocation']


# Evaluate for Allocated_Bandwidth
X2 = df[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Resource_Allocation']]
y2 = df['Allocated_Bandwidth']

# results_df1 = evaluate_models(X1, y1, 'Resource_Allocation')
# results_df2 = evaluate_models(X2, y2, 'Allocated_Bandwidth')

file_path = 'model_performance_metrics.csv'
if os.path.exists(file_path):
    final_results_df = pd.read_csv(file_path)
else:    
    final_results_df = pd.concat([results_df1, results_df2], ignore_index=True)
    final_results_df.to_csv('model_performance_metrics.csv', index=False)

# Define metrics to plot
metrics_to_plot = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2', 'Completion_Time']

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Function to plot metrics for each target with proper highlighting and dynamic legend layout
def plot_metrics(target_df, target_name, filter_o_models=False):
    if filter_o_models:
        target_df = target_df[target_df['Model'].str.startswith('O_')]
    else:
        target_df = target_df[~target_df['Model'].str.startswith('O_')]

    if target_df.empty:
        return  # Skip if no models match

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    palette = sns.color_palette("coolwarm", len(target_df['Model'].unique()))
    model_colors = dict(zip(target_df['Model'].unique(), palette))

    for i, metric in enumerate(metrics_to_plot):
        sorted_df = target_df.sort_values(by=metric, ascending=(metric != 'R2'))
        
        ax = axes[i]
        sns.barplot(x='Model', y=metric, hue='Model', data=sorted_df, ax=ax, palette=model_colors, legend=False)
        ax.set_title(f'{metric}', fontsize=18, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14)
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_xlabel('')

        if metric == 'R2':
            ax.set_ylim(0.65, 1)
        highlight_count = 1 if filter_o_models else 4   
        bars = [bar for bar in ax.patches[:highlight_count]]  # Select the first N bars

        # Highlight all bars
        for bar in bars:
            bar.set_edgecolor('black')
            bar.set_linewidth(2)

    # Adjust legend with dynamic columns based on the flag
    ncol_value = 4 if filter_o_models else 3
    legend_patches = [mpatches.Patch(color=color, label=model.replace("O_", "")) for model, color in model_colors.items()]
    
    fig.subplots_adjust(bottom=0.2)
    fig.legend(handles=legend_patches, loc="lower center", ncol=ncol_value, fontsize=22, frameon=False)
    
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    suffix = "_Optimized" if filter_o_models else ""
    plt.savefig(os.path.join(output_dir, f'{target_name}{suffix}_model_metrics.png'))
    plt.close()
    
plot_metrics(final_results_df[final_results_df['Target'] == 'Resource_Allocation'], 'Resource_Allocation')
plot_metrics(final_results_df[final_results_df['Target'] == 'Allocated_Bandwidth'], 'Allocated_Bandwidth')
plot_metrics(final_results_df[final_results_df['Target'] == 'Resource_Allocation'], 'Resource_Allocation', filter_o_models=True)
plot_metrics(final_results_df[final_results_df['Target'] == 'Allocated_Bandwidth'], 'Allocated_Bandwidth', filter_o_models=True)

# Identify best models
best_models = {}
for metric in ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']:
    best_models[metric] = final_results_df.loc[
        final_results_df.groupby('Target')[metric].idxmin() if metric != 'R2' else final_results_df.groupby('Target')[metric].idxmax()
    ]

# Print best models grouped by target in a concise format
print("\nBest Models Summary")
print("=" * 50)

for target in final_results_df['Target'].unique():
    print(f"\n{target}:")
    for metric, df in best_models.items():
        row = df[df['Target'] == target]
        if not row.empty:
            row = row.iloc[0]
            model_name = row['Model'].replace("O_", "Optimized ")
            print(f"  {metric}: {model_name} ({row[metric]:.4f})")