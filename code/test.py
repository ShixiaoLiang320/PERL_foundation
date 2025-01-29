import tensorflow as tf
import numpy as np
import os
from physics_model import IDM
from sklearn.metrics import mean_squared_error
from data_filter import DataProcessor

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

def extract_features(data_subset):
    """提取特征"""
    Speed_LV = data_subset[:, :, 0]
    Speed_FAV = data_subset[:, :, 2]
    Spatial_Gap = data_subset[:, :, 4]
    Delta_v = Speed_LV - Speed_FAV
    return Speed_LV, Speed_FAV, Spatial_Gap, Delta_v

# 主函数部分
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 数据文件路径
    file_path = os.path.join(
        current_dir,
        "../Ultra_AV/Longitudinal Trajectory Dataset/OpenACC/step3_ASta.csv"
    )

    input_steps = 20
    output_steps = 1
    mse_threshold = 0.001
    batch_size = 32

    # 初始化数据处理器
    data_processor = DataProcessor(file_path, input_steps, output_steps, mse_threshold=mse_threshold)

    # 获取筛选后的数据
    X_train, X_val, X_test = data_processor.X_train, data_processor.X_val, data_processor.X_test
    y_train, y_val, y_test = data_processor.y_train, data_processor.y_val, data_processor.y_test

    # 提取特征
    Speed_LV_train, Speed_FAV_train, Spatial_Gap_train, Delta_v_train = extract_features(X_train)
    Speed_LV_val, Speed_FAV_val, Spatial_Gap_val, Delta_v_val = extract_features(X_val)
    Speed_LV_test, Speed_FAV_test, Spatial_Gap_test, Delta_v_test = extract_features(X_test)

    # 加载 IDM 参数
    best_params_file = "calibration_results_OpenACC_ASta.txt"
    try:
        idm_params = load_best_params(best_params_file)
        print(f"Loaded IDM parameters: {idm_params}")
    except Exception as e:
        print(f"Error loading IDM parameters: {e}")
        idm_params = None

    # 使用 IDM 模型预测
    ahat_idm_train = IDM(idm_params, Speed_FAV_train, Delta_v_train, Spatial_Gap_train)
    ahat_idm_val = IDM(idm_params, Speed_FAV_val, Delta_v_val, Spatial_Gap_val)
    ahat_idm_test = IDM(idm_params, Speed_FAV_test, Delta_v_test, Spatial_Gap_test)

    # 输出训练集前 10 个样本的残差
    residual_train = y_train[:, 0] - ahat_idm_train[:, -1]
    print("\nFirst 10 residuals for training set:")
    print(residual_train[:10])

    # 输出训练集和物理模型预测的对比（前 10 个样本）
    print("\nFirst 10 actual vs predicted for training set:")
    print("Actual:", y_train[:10, 0])
    print("Predicted:", ahat_idm_train[:10, -1])
    
    mse_phy = mean_squared_error(y_train[:10, 0], ahat_idm_train[:10, -1])
    print(mse_phy)

