import tensorflow as tf
import numpy as np
import os
from lstm import build_lstm_model
import matplotlib.pyplot as plt
import time
import csv
from physics_model import FVD, IDM
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_filter import DataProcessor

def load_best_params(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        if "Best Parameters" in line:
            params_str = line.split(":")[1].strip()
            params = tuple(map(float, params_str.strip("()").split(", ")))
            return params 

    raise ValueError("Best Parameters not found in the file.")

def extract_subset(dataset, total_samples):

    inputs, labels = [], []
    for batch in dataset.as_numpy_iterator():
        batch_inputs, batch_labels = batch
        inputs.append(batch_inputs)
        labels.append(batch_labels)
        
        # 检查是否已满足所需的样本数量
        if len(np.concatenate(inputs)) >= total_samples:
            break

    # 合并所有批次数据
    inputs = np.concatenate(inputs)[:total_samples]
    labels = np.concatenate(labels)[:total_samples]
    return inputs, labels

current_dir = os.path.dirname(os.path.abspath(__file__))
output_csv = os.path.join(current_dir, "results", "perl", "perl_all_paras_test1.csv")


file_path = os.path.join(
    current_dir,
    "../data/step3_ASta.csv"
)

input_steps = 30
output_steps = 1
output_features = 1
batch_size = 32
max_epochs = 200
mse_threshold = 0.01
n_runs = 10

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1

data_processor = DataProcessor(file_path, input_steps, output_steps, mse_threshold)

train_dataset = data_processor.get_tf_dataset("train", batch_size=batch_size)
val_dataset = data_processor.get_tf_dataset("val", batch_size=batch_size)
test_dataset = data_processor.get_tf_dataset("test", batch_size=batch_size)

data_sizes = 200
results = []  

train_size = int(data_sizes * 0.7)  
val_size = int(data_sizes * 0.15)  
test_size = data_sizes - train_size - val_size  

X_train_subset = data_processor.X_train[:train_size]
y_train_subset = data_processor.y_train[:train_size]

X_val_subset = data_processor.X_val[:val_size]
y_val_subset = data_processor.y_val[:val_size]

X_test_subset = data_processor.X_test[:test_size]
y_test_subset = data_processor.y_test[:test_size]

Speed_LV_train = X_train_subset[:, :, 0]
Speed_FAV_train = X_train_subset[:, :, 2]
Spatial_Gap_train = X_train_subset[:, :, 4]  
Delta_v_train = Speed_LV_train - Speed_FAV_train

Speed_LV_val = X_val_subset[:, :, 0]
Speed_FAV_val = X_val_subset[:, :, 2]
Spatial_Gap_val = X_val_subset[:, :, 4]  
Delta_v_val = Speed_LV_val - Speed_FAV_val

Speed_LV_test = X_test_subset[:, :, 0]
Speed_FAV_test = X_test_subset[:, :, 2]
Spatial_Gap_test = X_test_subset[:, :, 4] 
Delta_v_test = Speed_LV_test - Speed_FAV_test

best_params_file = "calibration_results_OpenACC_ASta.txt"

try:
    idm_params = load_best_params(best_params_file)
    print(f"Loaded IDM parameters: {idm_params}")
except Exception as e:
    print(f"Error loading IDM parameters: {e}")
    idm_params = None

ahat_idm_train = IDM(idm_params, Speed_FAV_train, Delta_v_train, Spatial_Gap_train)
ahat_idm_val = IDM(idm_params, Speed_FAV_val, Delta_v_val, Spatial_Gap_val)
ahat_idm_test = IDM(idm_params, Speed_FAV_test[:, -1], Delta_v_test[:, -1], Spatial_Gap_test[:, -1])

residual_train = y_train_subset - ahat_idm_train
residual_val = y_val_subset - ahat_idm_val


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  
    patience=5,
    min_lr=1e-5
)

lstm_units_list = [16, 32, 48, 64, 80, 96, 112, 128]  # LSTM 参数数量列表
#lstm_units_list = [16, 32, 64, 128]
all_results = []

for lstm_units in lstm_units_list:
    for run in range(n_runs):  
        print(f"\n========== Running Experiment {run + 1}/{n_runs} with PERL {lstm_units} Units ==========")

        # 训练新的 LSTM 模型
        residual_model = build_lstm_model(X_train_subset.shape[1], X_train_subset.shape[2], output_features, lstm_units)

        history = residual_model.fit(
            X_train_subset, residual_train[:, -1],
            epochs=max_epochs,  
            batch_size=32,
            validation_data=(X_val_subset, residual_val[:, -1]),
            verbose=1  
        )

        # 计算测试集 MSE
        residual_pred = residual_model.predict(X_test_subset)
        ahat_idm_test = ahat_idm_test.reshape(-1, 1) 
        ahat_perl = ahat_idm_test + residual_pred
        mse = mean_squared_error(y_test_subset, ahat_perl)

        all_results.append({
            "lstm_units": lstm_units,
            "run": run + 1,
            "mse": mse
        })

# 存储最终的训练结果
with open(output_csv, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["lstm_units", "run", "mse"])
    writer.writeheader()
    writer.writerows(all_results)

print(f"Results saved to {output_csv}")
