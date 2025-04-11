import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

# Load
file_path = '/Users/lukassspazo/Year 3 Python/Machine Learning Final Project/SNaX.CSV'
try:
    data = pd.read_csv(file_path, on_bad_lines='skip', skipinitialspace=True)
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit(1)

# Clean
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Save directory
save_dir = '/Users/lukassspazo/Year 3 Python/Machine Learning Final Project'
os.makedirs(save_dir, exist_ok=True)

# Numeric features
numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()

# Histogram
n_numeric_features = len(numeric_features)
layout_rows = (n_numeric_features // 2) + (n_numeric_features % 2 > 0)
figsize_hist = (20, 6 * layout_rows)

# Plot histograms
plt.figure(figsize=figsize_hist)

for i, feature in enumerate(numeric_features):
    valid_data = data[feature].dropna()

    if not valid_data.empty:
        plt.subplot(layout_rows, 2, i + 1)
        plt.hist(valid_data, bins=15, alpha=0.7)

        # Titles and labels
        plt.title(f'Histogram of {feature}', fontsize=20, fontweight='bold')
        plt.xlabel(feature, fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)

    else:
        print(f"No valid data in column: {feature}")

# Layout
plt.tight_layout(pad=5.0)
plt.suptitle('Histograms of Numerical Features', fontsize=24, y=1.02, fontweight='bold')
plt.savefig(os.path.join(save_dir, 'histograms.png'), dpi=300)
plt.close()

# Correlation matrix
correlation_matrix = data[numeric_features].corr()
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, annot_kws={"size": 10})
plt.title('Correlation Matrix Heatmap', fontsize=20)
plt.savefig(os.path.join(save_dir, 'correlation_matrix.png'), dpi=300)
plt.close()

# Box plots
plt.figure(figsize=(24, 12))
valid_numeric_features = [feature for feature in numeric_features if data[feature].notna().any()]

if valid_numeric_features:
    for i, feature in enumerate(valid_numeric_features):
        plt.subplot(4, 4, i + 1)
        sns.boxplot(x=data[feature])
        plt.title(f'Box Plot of {feature}', fontsize=16)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'boxplots.png'), dpi=300)
    plt.close()
else:
    print("No valid numeric features available for box plots.")

# Categorical analysis
if 'type' in data.columns and 'Flux (10^-13 erg cm^-2 s^-1)' in data.columns:
    plt.figure(figsize=(16, 8))
    sns.boxplot(x='type', y='Flux (10^-13 erg cm^-2 s^-1)', data=data)
    plt.title('Box Plot of Flux by Type', fontsize=20)
    plt.xticks(rotation=45, fontsize=14)
    plt.ylabel('Flux (10^-13 erg cm^-2 s^-1)', fontsize=16)
    plt.xlabel('Type', fontsize=16)
    plt.savefig(os.path.join(save_dir, 'flux_by_type_boxplot.png'), dpi=300)
    plt.close()