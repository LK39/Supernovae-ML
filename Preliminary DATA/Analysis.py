import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the base path
base_path = r'C:\Users\pazol\Programms\Supernovae' 

# Initialize a list to hold all data
data = []

# Load data from both folders for analysis
for supernova_type in ['Type II', 'Type IIP']:
    folder_path = os.path.join(base_path, supernova_type, 'CSV Files')
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                if not {'_RAJ2000', '_DEJ2000', 'sed_freq', 'sed_flux', 'sed_eflux'}.issubset(df.columns):
                    print(f"Missing columns in {filename}, skipping.")
                    continue
                data.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

# Concatenate all dataframes into one
all_data = pd.concat(data, ignore_index=True)

print("\nSummary:")
print(all_data.info())

print("\nStatistical Summary:")
print(all_data.describe())

if 'Cluster' in all_data.columns:
    print("\nCluster Value Counts:")
    print(all_data['Cluster'].value_counts())

# Set up the visualizations
plt.figure(figsize=(16, 12))

# 1. Histograms for each feature
plt.subplot(2, 3, 1)
all_data['_RAJ2000'].hist(bins=30, color='skyblue', edgecolor='black')
plt.title('_RAJ2000 Distribution')
plt.xlabel('_RAJ2000')
plt.ylabel('Frequency')

plt.subplot(2, 3, 2)
all_data['_DEJ2000'].hist(bins=30, color='lightgreen', edgecolor='black')
plt.title('_DEJ2000 Distribution')
plt.xlabel('_DEJ2000')
plt.ylabel('Frequency')

plt.subplot(2, 3, 3)
all_data['sed_freq'].hist(bins=30, color='salmon', edgecolor='black')
plt.title('sed_freq Distribution')
plt.xlabel('sed_freq')
plt.ylabel('Frequency')

plt.subplot(2, 3, 4)
all_data['sed_flux'].hist(bins=30, color='orange', edgecolor='black')
plt.title('sed_flux Distribution')
plt.xlabel('sed_flux')
plt.ylabel('Frequency')

plt.subplot(2, 3, 5)
all_data['sed_eflux'].hist(bins=30, color='purple', edgecolor='black')
plt.title('sed_eflux Distribution')
plt.xlabel('sed_eflux')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 2. Pair Plot
sns.pairplot(all_data[['_RAJ2000', '_DEJ2000', 'sed_freq', 'sed_flux', 'sed_eflux']], diag_kind='kde')
plt.suptitle('Pair Plot of Features', y=1.02)
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = all_data[['_RAJ2000', '_DEJ2000', 'sed_freq', 'sed_flux', 'sed_eflux']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 4. Box Plots
plt.figure(figsize=(16, 8))
sns.boxplot(data=all_data[['_RAJ2000', '_DEJ2000', 'sed_freq', 'sed_flux', 'sed_eflux']])
plt.title('Box Plots of Features')
plt.xticks(rotation=45)
plt.show()

# 5. Scatter Plots
plt.figure(figsize=(16, 12))
sns.scatterplot(x='sed_freq', y='sed_flux', hue='sed_eflux', data=all_data, palette='viridis')
plt.title('Scatter Plot of sed_freq vs sed_flux')
plt.xlabel('sed_freq')
plt.ylabel('sed_flux')
plt.show()
