import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as stats

# 读取 CSV 文件
#file_path = "../Ultra_AV/Longitudinal Trajectory Dataset/OpenACC/step3_ASta.csv"  # 替换为实际路径

current_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(
    current_dir,
    "../Ultra_AV/Longitudinal Trajectory Dataset/Waymo/step3_motion.csv"
)

data = pd.read_csv(file_path)

# 提取 FAV_ACC 列
fav_acc = data["Acc_FAV"]

np.max(fav_acc)

mean = 1
std_dev = 9  # 标准差
num_bins = 20  # 希望的区间数量

# 生成正态分布的累积分布函数 (CDF) 的分位点
percentiles = np.linspace(0, 1, num_bins + 1)
bins = stats.norm.ppf(percentiles, loc=mean, scale=std_dev)

# 保证区间边界精度
bins = np.round(bins, 3)

bins = [ 0 ,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200]
bins = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
# 统计每个区间的数量
counts, edges = np.histogram(fav_acc, bins=bins)

# 定义人为等间隔的显示位置
positions = np.arange(len(bins) - 1)  # 等间隔的 x 轴位置

# 定义区间标签
labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

# 绘制柱状图
plt.figure(figsize=(10, 6))
plt.bar(positions, counts, width=0.6, alpha=0.7, edgecolor='black')

# 设置人为等间隔的 x 轴标签
plt.xticks(positions, labels, rotation=45, fontsize=10)

# 添加标题和轴标签
plt.title("Distribution with Equal Spacing for Intervals", fontsize=14)
plt.xlabel("FAV_ACC Range (Equal Spacing)", fontsize=12)
plt.ylabel("Count", fontsize=12)

# 显示网格
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图表
plt.show()

