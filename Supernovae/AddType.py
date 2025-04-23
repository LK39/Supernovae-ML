import pandas as pd
import os

# Define the base path for your dataset
base_path = r'C:\Users\pazol\Programms\Supernovae'
output_file = os.path.join(base_path, 'filtered_specific_type_II_supernovae_with_label.csv')

# List of specific types to include for Type II
specific_types_ii = [
    "II", "IIn", "II:", "IIPec", "IIP", "IIL", 
    "IIb", "IIn:", "IIPec:", "IIb:", "II/Ic:", 
    "IIPL", "IInPec"
]

# Initialize a DataFrame to hold all filtered data
filtered_data_ii = pd.DataFrame()

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
                # Filter for the specified Type II types
                ii_types = df[df['Type'].isin(specific_types_ii)]

                # Add a new column for Type II label
                ii_types['Type II Label'] = 'Type II'

                # Append filtered data to the DataFrame
                filtered_data_ii = pd.concat([filtered_data_ii, ii_types], ignore_index=True)
            else:
                print(f"'Type' column not found in {filename}.")
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Save the filtered data with the new label to a new CSV file
if not filtered_data_ii.empty:
    filtered_data_ii.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}.")
else:
    print("No specified Type II supernova data points found.")