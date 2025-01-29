import tensorflow as tf
import numpy as np
import os
from lstm import build_lstm_model
import matplotlib.pyplot as plt
import time
import csv
from physics_model import FVD, IDM
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_multi import DataProcessor

def load_best_params(file_path):
    """从文本文件中读取最优的 IDM 参数"""
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


def extract_features(data_subset):

    Speed_LV = data_subset[:, :, 0]
    Speed_FAV = data_subset[:, :, 2]
    Spatial_Gap = data_subset[:, :, 4]
    Delta_v = Speed_LV - Speed_FAV
    return Speed_LV, Speed_FAV, Spatial_Gap, Delta_v


current_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(
    current_dir,
    "../Ultra_AV/Longitudinal Trajectory Dataset/OpenACC/step3_ASta.csv"
)

input_steps = 30
output_steps = 5
output_features = 1
batch_size = 32
max_epochs = 200

data_processor = DataProcessor(file_path, input_steps, output_steps)

# 测试不同数据规模
#data_sizes = np.arange(50, 1001, 50)
data_sizes = [50, 100, 200]
results = []  # 存储实验结果

for data_size in data_sizes:
    train_size = int(data_size * 0.8)  # training set
    val_size = int(data_size * 0.1)  # validation set
    test_size = data_size - train_size - val_size  # test set

    X_train_subset = data_processor.X_train[:train_size]
    y_train_subset = data_processor.y_train[:train_size]

    X_val_subset = data_processor.X_val[:val_size]
    y_val_subset = data_processor.y_val[:val_size]

    X_test_subset = data_processor.X_test[:test_size]
    y_test_subset = data_processor.y_test[:test_size]

    Speed_LV_train, Speed_FAV_train, Spatial_Gap_train, Delta_v_train = extract_features(X_train_subset)
    Speed_LV_val, Speed_FAV_val, Spatial_Gap_val, Delta_v_val = extract_features(X_val_subset)
    Speed_LV_test, Speed_FAV_test, Spatial_Gap_test, Delta_v_test = extract_features(X_test_subset)

    Speed_LV_train_ForPhy, Speed_FAV_train_ForPhy, Spatial_Gap_train_ForPhy, Delta_v_train_ForPhy = extract_features(y_train_subset)
    Speed_LV_val_ForPhy, Speed_FAV_val_ForPhy, Spatial_Gap_val_ForPhy, Delta_v_val_ForPhy = extract_features(y_val_subset)
    Speed_LV_test_ForPhy, Speed_FAV_test_ForPhy, Spatial_Gap_test_ForPhy, Delta_v_test_ForPhy = extract_features(y_test_subset)


    best_params_file = "calibration_results_OpenACC_ASta.txt"

    # IDM results
    try:
        idm_params = load_best_params(best_params_file)
        print(f"Loaded IDM parameters: {idm_params}")
    except Exception as e:
        print(f"Error loading IDM parameters: {e}")
        idm_params = None
    

    ahat_idm_train = IDM(idm_params,Speed_FAV_train , Delta_v_train, Spatial_Gap_train)
    ahat_idm_val = IDM(idm_params, Speed_FAV_val, Delta_v_val, Spatial_Gap_val)
    ahat_idm_test = IDM(idm_params, Speed_FAV_test, Delta_v_test, Spatial_Gap_test)
    ahat_idm_train_ForPhy = IDM(idm_params, Speed_FAV_train_ForPhy ,Delta_v_train_ForPhy, Spatial_Gap_train_ForPhy)
    
    ahat_idm_val_ForPhy = IDM(idm_params, Speed_FAV_val_ForPhy, Delta_v_val_ForPhy, Spatial_Gap_val_ForPhy)
    print(y_test_subset[:, 0, 3], ahat_idm_test[:, -1])
    
    print(ahat_idm_train_ForPhy.shape)
    print(ahat_idm_val_ForPhy)
    #print(ahat_idm_train.shape, ahat_idm_val.shape, ahat_idm_test.shape)
    print(y_train_subset.shape, y_val_subset.shape, y_test_subset.shape)
    #print(X_train_subset.shape, X_val_subset.shape, X_test_subset.shape)
    
    residual_train_X = y_train_subset[:, 0, 3] - ahat_idm_train[:, -1]
    residual_train_y = y_train_subset[:, 1:, 3] - ahat_idm_train_ForPhy[:, :-1]
    residual_train = np.hstack((residual_train_X[:, np.newaxis], residual_train_y))
    print(residual_train_y - y_train_subset[:, 1:, 3])
    residual_val_X = y_val_subset[:, 0, 3] - ahat_idm_val[:, -1]
    residual_val_y = y_val_subset[:, 1:, 3] - ahat_idm_val_ForPhy[:, :-1]
    residual_val = np.hstack((residual_val_X[:, np.newaxis], residual_val_y))
    
    print("Residual train (first 10):")
    print(residual_train[:10])
    print("Y train subset (dimension 3, first 10):")
    print(y_train_subset[:10, :, 3])
    print("phy")
    print(ahat_idm_train_ForPhy[:, :-1])
    print("""""")
    print("MSE for IDM")
    mse_phy = mean_squared_error(y_test_subset[:, 0, 3], ahat_idm_test[:, -1])
    print(y_test_subset[:, 0, 3], ahat_idm_test[:, -1])
    print(mse_phy)
    print("""""")

    #print(X_train_subset.shape)
    #print(residual_train.shape)
    #print(y_train_subset[0])
    #print(ahat_idm_train[0])
    #print(residual_val.shape)
    #print(X_val_subset.shape)

'''
    # training
    residual_model = build_lstm_model(X_train_subset.shape[1], X_train_subset.shape[2], output_steps, output_features)

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
    patience_increase = 5  # 连续验证损失上升的最大次数
    consecutive_increase = 0

    training_loss = []
    validation_loss = []
    
    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        history = residual_model.fit(
            X_train_subset, residual_train,
            epochs=1,
            batch_size=32,
            validation_data=(X_val_subset, residual_val),
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
            if abs(val_loss_window[-1] - np.mean(val_loss_window[:-1])) < 1e-4:
                print(f"Model converged at epoch {epoch} for data size {data_size}")
                break
            
    end_time = time.time()

    #plt.plot(training_loss, label='Training Loss')
    #plt.plot(validation_loss, label='Validation Loss')
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.legend()
    #plt.show()

    residual_pred = residual_model.predict(X_test_subset)
    #print(ahat_idm_test.shape,residual_pred.shape)
    #ahat_idm_test = ahat_idm_test.reshape(-1, 1) 
    ahat_perl = ahat_idm_test[:, -1] + residual_pred[:, 0]

    #ahat_perl = ahat_idm_test
    #print(y_test_subset.shape)
    #print(ahat_perl.shape)
    mse_phy = mean_squared_error(y_test_subset[:, 0, 3], ahat_idm_test[:, -1])
    mse = mean_squared_error(y_test_subset[:, 0, 3], ahat_perl)
    print(mse)

    results.append({
        "data_size": data_size,
        "epochs": epoch,
        "total_time": end_time - start_time,
        "mse": mse
    })
    

# 将结果保存到 CSV 文件
output_csv = os.path.join(current_dir, f"perl_experiment_results_multi.csv")

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

'''    


