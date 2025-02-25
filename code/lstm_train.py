import tensorflow as tf
import numpy as np
import os
from lstm import build_lstm_model
import matplotlib.pyplot as plt
import time
import csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_filter import DataProcessor


current_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(
    current_dir,
    "../data/step3_ASta.csv"
)

input_steps = 10
output_steps = 1
output_features = 1
batch_size = 32
mse_threshold = 0.1
max_epochs = 300

data_processor = DataProcessor(file_path, input_steps, output_steps, mse_threshold)

# 获取数据集
train_dataset = data_processor.get_tf_dataset("train", batch_size=batch_size)
val_dataset = data_processor.get_tf_dataset("val", batch_size=batch_size)
test_dataset = data_processor.get_tf_dataset("test", batch_size=batch_size)

# 测试不同数据规模
data_sizes = np.arange(40, 201, 10)
data_sizes = [200]
results = []  # 存储实验结果

for data_size in data_sizes:
    train_size = int(data_size * 0.6)  # training set
    val_size = int(data_size * 0.2)  # validation set
    test_size = data_size - train_size - val_size  # test set
    # 提取子集
    X_train_subset = data_processor.X_train[:train_size]
    y_train_subset = data_processor.y_train[:train_size]

    X_val_subset = data_processor.X_val[:val_size]
    y_val_subset = data_processor.y_val[:val_size]

    X_test_subset = data_processor.X_test[:test_size]
    y_test_subset = data_processor.y_test[:test_size]
 
    
    # training
    lstm_model = build_lstm_model(X_train_subset.shape[1], X_train_subset.shape[2], output_steps, output_features)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,  
        restore_best_weights=True,
        min_delta=1e-5 
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  
        patience=5,
        min_lr=1e-6
    )

    val_loss_window = []
    patience_increase = 10  # 连续验证损失上升的最大次数
    consecutive_increase = 0

    training_loss = []
    validation_loss = []
    
    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        history = lstm_model.fit(
            X_train_subset, y_train_subset,
            epochs=1,
            batch_size=32,
            validation_data=(X_val_subset, y_val_subset),
            callbacks=[early_stopping, reduce_lr]
        )
        
        training_loss.append(history.history['loss'][-1])
        validation_loss.append(history.history['val_loss'][-1])
    
        val_loss = history.history['val_loss'][-1]
        val_loss_window.append(val_loss)


        if len(val_loss_window) > 1 and val_loss > val_loss_window[-2]:
            consecutive_increase += 1
        else:
            consecutive_increase = 0

    
        if consecutive_increase >= patience_increase:
            print(f"Model stopped due to rising val_loss at epoch {epoch} for data size {data_size}")
            break
    
        if len(val_loss_window) > 10:
            val_loss_window.pop(0)
            if abs(val_loss_window[-1] - np.mean(val_loss_window[:-1])) < 1e-3:
                print(f"Model converged at epoch {epoch} for data size {data_size}")
                break
            
    end_time = time.time()

    ahat_pred = lstm_model.predict(X_test_subset)


    mse = mean_squared_error(y_test_subset, ahat_pred)
    print(mse)

    results.append({
        "data_size": data_size,
        "epochs": epoch,
        "total_time": end_time - start_time,
        "mse": mse
    })

# 将结果保存到 CSV 文件
output_csv = os.path.join(current_dir, "results", "lstm", "lstm_200_test.csv")

# 如果文件已存在，读取现有内容
def read_existing_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            return list(reader)
    return []

existing_results = read_existing_results(output_csv)

# 添加新结果到现有内容
existing_results.extend(results)
with open(output_csv, mode='w+', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["data_size", "epochs", "total_time", "mse"])
    writer.writeheader()
    writer.writerows(existing_results)

print(f"Results saved to {output_csv}")


