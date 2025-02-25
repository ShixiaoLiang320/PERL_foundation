import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set font sizes globally
plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'figure.titlesize': 24
})

current_dir = os.path.dirname(os.path.abspath(__file__))

LSTM = os.path.join(current_dir, "results", "lstm", "lstm_loss_curves.csv")
PERL = os.path.join(current_dir, "results", "perl", "perl_loss_curves.csv")

loss_data_lstm = pd.read_csv(LSTM)
loss_data_perl = pd.read_csv(PERL)

# 按照 epoch 计算均值和标准差
arch1_grouped = loss_data_lstm.groupby("epoch")["val_loss"]
arch1_mean = arch1_grouped.mean()
arch1_std = arch1_grouped.std()

arch2_grouped = loss_data_perl.groupby("epoch")["val_loss"]
arch2_mean = arch2_grouped.mean()
arch2_std = arch2_grouped.std()

# X 轴：所有 epoch
epochs = arch1_mean.index

# 画图
plt.figure(figsize=(12, 8))
plt.plot(epochs, arch1_mean, label="LSTM", color='blue', linestyle='-', marker='o')
plt.plot(epochs, arch2_mean, label="PERL", color='red', linestyle='--', marker='s')

# 添加置信区间 (95% 置信区间)
plt.fill_between(epochs, arch1_mean - 1.645 * (arch1_std / np.sqrt(arch1_grouped.count())),
                 arch1_mean + 1.645 * (arch1_std / np.sqrt(arch1_grouped.count())), 
                 color='blue', alpha=0.2, label='LSTM 90% CI')

plt.fill_between(epochs, arch2_mean - 1.645 * (arch2_std / np.sqrt(arch2_grouped.count())),
                 arch2_mean + 1.645 * (arch2_std / np.sqrt(arch2_grouped.count())), 
                 color='red', alpha=0.2, label='PERL 90% CI')

plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid(True)

# 保存
plt.savefig("convergence_comparison.png", bbox_inches='tight')
plt.show()

print("Plot saved as convergence_comparison.png")