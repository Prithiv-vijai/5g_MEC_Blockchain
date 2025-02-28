import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Set global font and style for IEEE standards
plt.rcParams.update({
    'font.family': 'Times New Roman',  
    'font.size': 18,
    'axes.titlesize': 25,
    'axes.labelsize': 25,
    'legend.fontsize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.figsize': (12, 8)
})

# Define colors
colors = {"Test": "#1f77b4", "Train": "#ff7f0e"}

# Metrics
metrics = ["MSE", "RMSE", "MAE", "R2", "MAPE"]
test_values = [1.2619, 1.1226, 0.9041, 0.9802, 0.0129]
train_values = [1.1719, 1.0826, 0.8341, 0.9838, 0.0113]

# Create DataFrame
plot_data = pd.DataFrame({
    "Metric": metrics * 2,
    "Value": test_values + train_values,
    "Data Type": ["Test"] * len(metrics) + ["Train"] * len(metrics)
})

# Plot
plt.figure(figsize=(12, 8))
ax = sns.barplot(x="Metric", y="Value", hue="Data Type", data=plot_data, palette=colors)

# Add Title
plt.title("Comparison of Model Performance Metrics", fontsize=25, fontweight='bold')

# Add values on top of each bar
for bar in ax.patches:
    value = bar.get_height()
    if value > 0.01:
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 0.05,
            f"{value:.2f}", ha="center", fontsize=18, color="black", fontweight='bold'
        )

# Bold X-axis labels
for label in ax.get_xticklabels():
    label.set_fontweight("bold")

# Remove x and y labels
ax.set_xlabel("")
ax.set_ylabel("")

# Extend Y-axis to 26
ax.set_ylim(0, 2)

# Legend inside the plot (top-right)
plt.legend(title=None, loc="upper right", fontsize=22, frameon=True)

# Adjust Layout
plt.tight_layout()

# Save Plot
plt.savefig("graphs/model_output/model_1_overfit_graph.png", dpi=300, bbox_inches="tight")


