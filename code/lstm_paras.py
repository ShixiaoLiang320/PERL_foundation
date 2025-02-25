import tensorflow as tf
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from lstm import build_lstm_model
from sklearn.metrics import mean_squared_error
from data_filter import DataProcessor

# 设置目录 & 输出文件
current_dir = os.path.dirname(os.path.abspath(__file__))
output_csv = os.path.join(current_dir, "results", "lstm", "lstm_all_paras_test2.csv")

# 读取数据
file_path = os.path.join(current_dir, "../data/step3_ASta.csv")
input_steps = 30
output_steps = 1
output_features = 1
batch_size = 32
max_epochs = 200
mse_threshold=0.01
n_runs = 10
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1

# 初始化数据处理
data_processor = DataProcessor(file_path, input_steps, output_steps, mse_threshold)

# 获取数据集
train_dataset = data_processor.get_tf_dataset("train", batch_size=batch_size)
val_dataset = data_processor.get_tf_dataset("val", batch_size=batch_size)
test_dataset = data_processor.get_tf_dataset("test", batch_size=batch_size)

# 设置实验参数
data_sizes = 200  # 固定数据集大小
train_size = int(data_sizes * 0.7)  
val_size = int(data_sizes * 0.15)  
test_size = data_sizes - train_size - val_size  

# 划分数据
X_train_subset = data_processor.X_train[:train_size]
y_train_subset = data_processor.y_train[:train_size]

X_val_subset = data_processor.X_val[:val_size]
y_val_subset = data_processor.y_val[:val_size]

X_test_subset = data_processor.X_test[:test_size]
y_test_subset = data_processor.y_test[:test_size]

# LSTM 参数规模
lstm_units_list = [16, 32, 48, 64, 80, 96, 112, 128]
#lstm_units_list = [16, 32, 64, 128]
all_results = []

# 训练不同 LSTM 参数的 Baseline
for lstm_units in lstm_units_list:
    for run in range(n_runs): 
        print(f"\n========== Running LSTM Baseline {run + 1}/{n_runs} with {lstm_units} Units ==========")

        lstm_model = build_lstm_model(X_train_subset.shape[1], X_train_subset.shape[2], output_features, lstm_units)

        history = lstm_model.fit(
            X_train_subset, y_train_subset[:, -1],  # 直接预测目标值
            epochs=max_epochs,  
            batch_size=32,
            validation_data=(X_val_subset, y_val_subset[:, -1]),
            verbose=1  
        )

        # 计算测试集 MSE
        y_pred = lstm_model.predict(X_test_subset)
        mse = mean_squared_error(y_test_subset, y_pred)

        # 记录实验结果
        all_results.append({
            "lstm_units": lstm_units,
            "run": run + 1,
            "mse": mse
        })

# 存储结果
with open(output_csv, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["lstm_units", "run", "mse"])
    writer.writeheader()
    writer.writerows(all_results)

print(f"Results saved to {output_csv}")
