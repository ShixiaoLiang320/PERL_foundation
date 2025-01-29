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

# Data Input
#LSTM = "/Users/shixiaoliang/Library/Mobile Documents/com~apple~CloudDocs/Documents/PERL_foundation/code/results/lstm/200_32_30.csv"
#PERL = "/Users/shixiaoliang/Library/Mobile Documents/com~apple~CloudDocs/Documents/PERL_foundation/code/results/perl/200_32_30.csv"
#test
LSTM = "/Users/shixiaoliang/Library/Mobile Documents/com~apple~CloudDocs/Documents/PERL_foundation/code/lstm_200_128_10.csv"
PERL = "/Users/shixiaoliang/Library/Mobile Documents/com~apple~CloudDocs/Documents/PERL_foundation/code/perl_200_128_10.csv"

#LSTM = "/Users/shixiaoliang/Library/Mobile Documents/com~apple~CloudDocs/Documents/PERL_foundation/code/lstm_output.csv"
#PERL = "/Users/shixiaoliang/Library/Mobile Documents/com~apple~CloudDocs/Documents/PERL_foundation/code/perl_output.csv"

data_1 = pd.read_csv(LSTM)
data_2 = pd.read_csv(PERL)

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
#plt.xscale('log')
#plt.yscale('log')

plt.xlabel('Training data size')
plt.ylabel('$MSE^a_{test}$ ($\mathrm{m}^2/\mathrm{s}^4$)') 
#plt.title('MSE vs Data Size with Confidence Intervals')
plt.legend()
plt.grid(True)
plt.show()




