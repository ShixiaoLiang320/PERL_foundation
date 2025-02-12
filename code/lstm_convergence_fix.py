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
output_csv = os.path.join(current_dir, "results", "lstm", "lstm_convergence_test_fix.csv")
output_loss_csv = os.path.join(current_dir, "results", "lstm", "lstm_loss_curves.csv")  # 存储损失曲线

file_path = os.path.join(
    current_dir,
    "../data/step3_ASta.csv"
)

input_steps = 30
output_steps = 1
output_features = 1
batch_size = 32
max_epochs = 100
mse_threshold = 0.1


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
    min_lr=1e-5
)


all_results = []  # 记录所有实验结果
all_loss_curves = []  # 记录损失曲线数据

for run in range(10):  # 运行 10 组实验
    print(f"\n========== Running Experiment {run + 1}/10 ==========")
    
    # 训练新的 LSTM 模型
    lstm_model = build_lstm_model(X_train_subset.shape[1], X_train_subset.shape[2], output_steps, output_features)

    history = lstm_model.fit(
        X_train_subset, y_train_subset,
        epochs=max_epochs,  # 固定训练 200 轮
        batch_size=32,
        validation_data=(X_val_subset, y_val_subset),
        verbose=1  
    )

    # 记录损失曲线
    for epoch, (train_loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
        all_loss_curves.append({
            "run": run + 1,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

    # 计算测试集 MSE
    ahat_pred = lstm_model.predict(X_test_subset)
    mse = mean_squared_error(y_test_subset, ahat_pred)

    all_results.append({
        "run": run + 1,
        "epochs": max_epochs,  # 由于所有模型训练满 200 轮，所以 epoch 直接存 200
        "mse": mse
    })

print("Training complete for all target MSEs.")

# 存储最终的训练结果
with open(output_csv, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["run", "epochs", "mse"])
    writer.writeheader()
    writer.writerows(all_results)

print(f"Results saved to {output_csv}")

# 存储损失曲线
with open(output_loss_csv, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["run", "epoch", "train_loss", "val_loss"])
    writer.writeheader()
    writer.writerows(all_loss_curves)

print(f"Loss curves saved to {output_loss_csv}")


