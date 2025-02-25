import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set font sizes globally
plt.rcParams.update({
    'font.size': 24,        # General font size
    'axes.titlesize': 24,   # Title font size
    'axes.labelsize': 24,   # Axis labels font size
    'xtick.labelsize': 24,  # X tick labels font size
    'ytick.labelsize': 24,  # Y tick labels font size
    'legend.fontsize': 24,  # Legend font size
    'figure.titlesize': 24  # Overall figure title font size
})

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(current_dir, "results", "figures")

# Ensure the results/figures directory exists
os.makedirs(figures_dir, exist_ok=True)

# File paths
LSTM = os.path.join(current_dir, "results", "lstm", "lstm_200_32_10.csv")
PERL = os.path.join(current_dir, "results", "perl", "perl_200_32_10.csv")

# Extract common part from filenames
lstm_name = os.path.basename(LSTM).replace("lstm_", "").replace(".csv", "")
perl_name = os.path.basename(PERL).replace("perl_", "").replace(".csv", "")

if lstm_name == perl_name:
    file_prefix = lstm_name
else:
    file_prefix = "unknown"
    


# Load data
data_1 = pd.read_csv(LSTM)
data_2 = pd.read_csv(PERL)

filtered = list(range(20, 201, 20))

# 筛选 data_1 和 data_2 中 data_size 在 allowed_values 列表内的行
data_1 = data_1[data_1['data_size'].isin(filtered)]
data_2 = data_2[data_2['data_size'].isin(filtered)]
# Filter data
#data_1 = data_1[(data_1['data_size'] < 101) & (data_1['data_size'] > 49)]
#data_2 = data_2[(data_2['data_size'] < 101) & (data_2['data_size'] > 49)]

# Group by data size and calculate statistics
experiment_counts_1 = data_1['data_size'].value_counts().sort_index()
experiment_counts_2 = data_2['data_size'].value_counts().sort_index()

grouped_1 = data_1.groupby('data_size')['mse']
means_1 = grouped_1.mean()
stds_1 = grouped_1.std()

grouped_2 = data_2.groupby('data_size')['mse']
means_2 = grouped_2.mean()
stds_2 = grouped_2.std()

# Confidence Interval
data_sizes_1 = means_1.index
mean_values_1 = means_1.values
std_values_1 = stds_1.values
confidence_interval_1 = 2.576 * (std_values_1 / np.sqrt(experiment_counts_1))  # 99% CI

data_sizes_2 = means_2.index
mean_values_2 = means_2.values
std_values_2 = stds_2.values
confidence_interval_2 = 2.576 * (std_values_2 / np.sqrt(experiment_counts_2))  # 99% CI

# Plot
plt.figure(figsize=(12, 8))
plt.plot(data_sizes_1, mean_values_1, label='LSTM Mean MSE', color='blue', linestyle='-', marker='o', markersize=10)
plt.plot(data_sizes_2, mean_values_2, label='PERL Mean MSE', color='red', linestyle='--', marker='s', markersize=10)

plt.fill_between(data_sizes_1, 
                 mean_values_1 - confidence_interval_1, 
                 mean_values_1 + confidence_interval_1, 
                 color='blue', alpha=0.2, label='LSTM 99% CI')

plt.fill_between(data_sizes_2, 
                 mean_values_2 - confidence_interval_2, 
                 mean_values_2 + confidence_interval_2, 
                 color='red', alpha=0.2, label='PERL 99% CI')

# Using log scale
plt.xscale('log')
plt.yscale('log')

plt.xlabel('Training data size')
plt.ylabel('$MSE^a_{test}$ ($\mathrm{m}^2/\mathrm{s}^4$)') 
plt.legend()
plt.grid(True)

# Save the figure using extracted prefix
save_path = os.path.join(figures_dir, f"datasize_log.png")
plt.savefig(save_path, bbox_inches='tight')

print(f"Figure saved to {save_path}")
