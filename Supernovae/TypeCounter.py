import pandas as pd
import os

# Define the base path for your dataset
base_path = r'C:\Users\pazol\Programms\Supernovae'

# Initialize lists to hold data and labels
data_list = []
labels = []

# Load data from the specified folder
folder_path = os.path.join(base_path, 'supernova_data')

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)

        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Check if the 'Type' column exists
            if 'Type' in df.columns:
                # Count the occurrences of each type
                type_counts = df['Type'].value_counts()

                # Display the results
                print(f"Population of the Type column in {filename}:")
                print(type_counts)
                print()  # Print a newline for better readability
            else:
                print(f"'Type' column not found in {filename}.")
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")