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
        
        if len(np.concatenate(inputs)) >= total_samples:
            break

    inputs = np.concatenate(inputs)[:total_samples]
    labels = np.concatenate(labels)[:total_samples]
    return inputs, labels

def read_existing_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            return list(reader)  
    return []  # 文件不存在时返回空列表

current_dir = os.path.dirname(os.path.abspath(__file__))
output_csv = os.path.join(current_dir, "results", "perl", "perl_convergence_test_2.csv")

file_path = os.path.join(
    current_dir,
    "../data/step3_ASta.csv"
)

input_steps = 10
output_steps = 1
output_features = 1
batch_size = 32
max_epochs = 1000
mse_threshold = 0.1
#target_mse_list = [0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01]  
#target_mse_list = [0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02]  
target_mse_list = [0.03]
epochs_to_reach_target = None 
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只使用 GPU 1

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
    min_lr=5e-4
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,  
    restore_best_weights=True,
    min_delta=1e-5 
)

for target_mse in target_mse_list:
    val_loss_window = []
    training_loss = []
    validation_loss = []
    patience = 5
    best_epochs = {}

    residual_model = build_lstm_model(X_train_subset.shape[1], X_train_subset.shape[2], output_steps, output_features)
    start_time = time.time()
    consecutive_below_target = 0  
    for epoch in range(1, max_epochs + 1):
        history = residual_model.fit(
            X_train_subset, residual_train[:, -1],
            epochs=max_epochs,
            batch_size=32,
            validation_data=(X_val_subset, residual_val[:, -1]),
            callbacks=[reduce_lr, early_stopping],
            verbose=1  
        )
    
        training_loss.append(history.history['loss'][-1])
        validation_loss.append(history.history['val_loss'][-1])

        val_loss = history.history['val_loss'][-1]
        val_loss_window.append(val_loss)
        
        if len(val_loss_window) > 5:
            val_loss_window.pop(0)

        # 计算最近 5 轮的均值，避免 val_loss 偶然下降导致误判
        if np.mean(val_loss_window) <= target_mse:
            consecutive_below_target += 1
        else:
            consecutive_below_target = 0

        if consecutive_below_target >= patience or epoch == max_epochs:
            best_epochs[target_mse] = epoch
            epochs_to_reach_target = epoch
            print(f"Model reached MSE={target_mse} at epoch {epoch}")
            break
            
    end_time = time.time()
    
    residual_pred = residual_model.predict(X_test_subset)
    ahat_idm_test = ahat_idm_test.reshape(-1, 1) 
    ahat_perl = ahat_idm_test + residual_pred
    
    mse = mean_squared_error(y_test_subset, ahat_perl)
    print(mse)
    
    results.append({
        "target_mse": target_mse,
        "epochs": best_epochs.get(target_mse, max_epochs),  # 若未收敛，默认 epochs=max_epochs
        "total_time": end_time - start_time,
        "mse": mse
    })


# 读取旧数据
existing_results = read_existing_results(output_csv)

with open(output_csv, mode='a+', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["target_mse", "epochs", "total_time", "mse"])
    
    if file.tell() == 0:
        writer.writeheader()
    
    writer.writerows(results)

print(f"Results saved to {output_csv}")

