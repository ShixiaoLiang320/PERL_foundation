import tensorflow as tf
import numpy as np
import os
from lstm import build_lstm_model
import matplotlib.pyplot as plt
import time
import csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_filter import DataProcessor

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
output_csv = os.path.join(current_dir, "results", "lstm", "lstm_convergence_test_2.csv")

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
#target_mse_list = [0.05, 0.045, 0.04, 0.035, 0.03]
target_mse_list = [0.03]

epochs_to_reach_target = None 
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只使用 GPU 1

data_processor = DataProcessor(file_path, input_steps, output_steps, mse_threshold)

train_dataset = data_processor.get_tf_dataset("train", batch_size=batch_size)
val_dataset = data_processor.get_tf_dataset("val", batch_size=batch_size)
test_dataset = data_processor.get_tf_dataset("test", batch_size=batch_size)

data_size = 200
results = []  # 存储实验结果

train_size = int(data_size * 0.7)  # training set
val_size = int(data_size * 0.15)  # validation set
test_size = data_size - train_size - val_size  # test set

# 提取子集
X_train_subset = data_processor.X_train[:train_size]
y_train_subset = data_processor.y_train[:train_size]

X_val_subset = data_processor.X_val[:val_size]
y_val_subset = data_processor.y_val[:val_size]

X_test_subset = data_processor.X_test[:test_size]
y_test_subset = data_processor.y_test[:test_size]

# training

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  
    patience=5,
    min_lr=1e-6
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
    
    lstm_model = build_lstm_model(X_train_subset.shape[1], X_train_subset.shape[2], output_steps, output_features)
    start_time = time.time()    
    consecutive_below_target = 0
    for epoch in range(1, max_epochs + 1):
        history = lstm_model.fit(
            X_train_subset, y_train_subset,
            epochs=max_epochs,
            batch_size=32,
            validation_data=(X_val_subset, y_val_subset),
            callbacks=[reduce_lr, early_stopping],
            verbose = 1
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
            save_path = "training_validation_loss.png"  
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(training_loss)), training_loss, label="Training Loss", linestyle='-', marker='o')
            plt.plot(range(len(validation_loss)), validation_loss, label="Validation Loss", linestyle='-', marker='s')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training vs Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")  # dpi=300 使得图片更清晰

            break

            
    end_time = time.time()

    ahat_pred = lstm_model.predict(X_test_subset)
    mse = mean_squared_error(y_test_subset, ahat_pred)
    print(mse)

    results.append({
        "target_mse": target_mse,
        "epochs": best_epochs.get(target_mse, max_epochs),  # 若未收敛，默认 epochs=max_epochs
        "total_time": end_time - start_time,
        "mse": mse
    })


# 读取旧数据
existing_results = read_existing_results(output_csv)

# **方法 2：追加新数据**
with open(output_csv, mode='a+', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["target_mse", "epochs", "total_time", "mse"])
    
    # **如果文件是新创建的，写入 header**
    if file.tell() == 0:
        writer.writeheader()
    
    # **追加新数据**
    writer.writerows(results)

print(f"Results saved to {output_csv}")



