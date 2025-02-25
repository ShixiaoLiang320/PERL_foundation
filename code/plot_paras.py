import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.size': 24,        # 全局字体大小
    'axes.titlesize': 24,   # 标题字体大小
    'axes.labelsize': 24,   # 坐标轴标签字体大小
    'xtick.labelsize': 24,  # X 轴刻度字体大小
    'ytick.labelsize': 24,  # Y 轴刻度字体大小
    'legend.fontsize': 24,  # 图例字体大小
    'figure.titlesize': 24  # 整体标题字体大小
})

current_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(current_dir, "results", "figures")

# LSTM 和 PERL 的 CSV 结果文件路径
LSTM_CSV = os.path.join(current_dir, "results", "lstm", "lstm_all_paras_test1.csv")
PERL_CSV = os.path.join(current_dir, "results", "perl", "perl_all_paras_test1.csv")

# 读取数据
data_lstm = pd.read_csv(LSTM_CSV)
data_perl = pd.read_csv(PERL_CSV)

# 过滤 LSTM 参数列表
#filtered_units = [16, 32, 48, 64, 80, 96, 112, 128]
filtered_units = [16, 32, 64, 96, 128]
data_lstm = data_lstm[data_lstm['lstm_units'].isin(filtered_units)]
data_perl = data_perl[data_perl['lstm_units'].isin(filtered_units)]

# 按 `lstm_units` 计算均值和标准差
grouped_lstm = data_lstm.groupby('lstm_units')['mse']
means_lstm = grouped_lstm.mean()
stds_lstm = grouped_lstm.std()

grouped_perl = data_perl.groupby('lstm_units')['mse']
means_perl = grouped_perl.mean()
stds_perl = grouped_perl.std()

# 计算 99% 置信区间 (CI)
experiment_counts_lstm = data_lstm['lstm_units'].value_counts().sort_index()
experiment_counts_perl = data_perl['lstm_units'].value_counts().sort_index()

ci_lstm = 2.576 * (stds_lstm / np.sqrt(experiment_counts_lstm))  # 99% CI
ci_perl = 2.576 * (stds_perl / np.sqrt(experiment_counts_perl))  # 99% CI

# X 轴 - LSTM 参数大小
lstm_units = means_lstm.index

# ======== 画图 ========
plt.figure(figsize=(12, 8))

# LSTM 曲线
plt.plot(lstm_units, means_lstm, label="LSTM Model", color='blue', linestyle='-', marker='o', markersize=10)
plt.fill_between(lstm_units, means_lstm - ci_lstm, means_lstm + ci_lstm, color='blue', alpha=0.2, label="LSTM 99% CI")

# PERL 曲线
plt.plot(lstm_units, means_perl, label="PERL Model", color='red', linestyle='--', marker='s', markersize=10)
plt.fill_between(lstm_units, means_perl - ci_perl, means_perl + ci_perl, color='red', alpha=0.2, label="PERL 99% CI")

# ======== 坐标轴设置 ========
plt.xlabel("The number of parameters")
plt.ylabel('$MSE^a_{test}$ ($\mathrm{m}^2/\mathrm{s}^4$)') 
plt.legend()
plt.grid(True)
#plt.xscale('log')  # 采用 log 坐标
#plt.yscale('log')  # 采用 log 坐标
#plt.title("LSTM vs. PERL: MSE vs. LSTM Parameter Size")

# 保存图片
save_path = os.path.join(figures_dir, "para_vs_mse_test4.png")
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"Figure saved to {save_path}")
