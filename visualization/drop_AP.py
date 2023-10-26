# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# Load the data from the uploaded Excel file
file_path = 'visualization/drop.xlsx'
df = pd.read_excel(file_path)

# Rename the first column for easier access
df.rename(columns={'Unnamed: 0': 'Method'}, inplace=True)

# Set the 'Method' column as the index
df.set_index('Method', inplace=True)

# Transpose the DataFrame for easier plotting
df_transposed = df.T

# Plot the data
plt.figure(figsize=(12, 8))

# Loop through each method to plot its data
for method in df_transposed.columns:
    plt.plot(df_transposed.index, df_transposed[method], marker='o', label=method, markersize=12)
    # Annotate each data point
    for x, y in zip(df_transposed.index, df_transposed[method]):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=14)

# Add title and labels
plt.title('Data Visualization', fontsize=20)
plt.xlabel('Token Drop Proportion', fontsize=16)
plt.ylabel('Average Precision  (AP) @[ IoU=0.50:0.95]', fontsize=16)
plt.grid(True)

# Add legend
plt.legend()
plt.tight_layout()
plt.savefig('visualization/drop_AP.png', dpi=300)