import tensorflow as tf
import numpy as np
import pandas as pd
from physics_model import IDM
import os

class DataProcessor:
    def __init__(self, file_path, input_steps, output_steps, mse_threshold=0.01, t_interval=0.1, t_fre=0.1):
        self.file_path = file_path
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.mse_threshold = mse_threshold
        self.t_interval = t_interval
        self.t_fre = t_fre

        # 加载和预处理数据
        self.data = self.load_data()
        self.data = self.filter_low_acceleration(self.data, threshold=0.1)
        self.X, self.y = self.preprocess_data()
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.split_data()

    def load_best_params(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if "Best Parameters" in line:
                params_str = line.split(":")[1].strip()
                params = tuple(map(float, params_str.strip("()").split(", ")))
                return params
        raise ValueError("Best Parameters not found in the file.")

    def load_data(self):
        data = pd.read_csv(self.file_path)
        return data.iloc[0::int(self.t_fre / self.t_interval), :]

    def filter_low_acceleration(self, data, threshold=0.1):
        return data[np.abs(data['Acc_FAV']) >= threshold]

    def preprocess_data(self):
        samples_X, samples_y = [], []
        current_trajectory = None
        tra_data = []

        idm_params = self.load_best_params("calibration_results_OpenACC_ASta.txt")

        for i, row in self.data.iterrows():
            # 检测 Trajectory ID 变化，重新开始新轨迹
            if row['Trajectory_ID'] != current_trajectory:
                current_trajectory = row['Trajectory_ID']
                tra_data = []

            # 添加当前行数据
            tra_data.append([row['Speed_LV'], row['Acc_LV'], row['Speed_FAV'], row['Acc_FAV'], row['Spatial_Gap']])

            # 如果当前轨迹数据满足 input_steps + output_steps，生成样本
            if len(tra_data) == self.input_steps + self.output_steps:
                input_data = np.array(tra_data[:self.input_steps])
                output_data = np.array(tra_data[self.input_steps:])[:, 3]  # 取 Acc_FAV

                # 筛选样本
                if self.is_sample_valid(input_data, output_data, idm_params):
                    samples_X.append(input_data)
                    samples_y.append(output_data)

                # 滑动窗口：丢弃第一个数据点
                tra_data = tra_data[1:]

        return np.array(samples_X), np.array(samples_y)

    def is_sample_valid(self, input_data, output_data, idm_params):
        Speed_LV = input_data[-1, 0]
        Speed_FAV = input_data[-1, 2]
        Spatial_Gap = input_data[-1, 4]
        Delta_v = Speed_LV - Speed_FAV

        # 使用 IDM 模型计算预测值
        ahat_idm = IDM(idm_params, np.array([Speed_FAV]), np.array([Delta_v]), np.array([Spatial_Gap]))

        # 手动计算 MSE
        mse = (output_data[0] - ahat_idm[0]) ** 2
        return mse < self.mse_threshold

    def split_data(self):
        total_samples = len(self.X)
        train_size = int(total_samples * 0.7)
        val_size = int(total_samples * 0.15)

        X_train = self.X[:train_size]
        y_train = self.y[:train_size]

        X_val = self.X[train_size:train_size + val_size]
        y_val = self.y[train_size:train_size + val_size]

        X_test = self.X[train_size + val_size:]
        y_test = self.y[train_size + val_size:]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_tf_dataset(self, dataset_type, batch_size, drop_remainder=True, shuffle_buffer_size = 200):
        """获取 TensorFlow 数据集"""
        if dataset_type == 'train':
            dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        elif dataset_type == 'val':
            dataset = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        elif dataset_type == 'test':
            dataset = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))
        else:
            raise ValueError("dataset_type must be 'train', 'val', or 'test'")
        
        if dataset_type == 'train' and shuffle_buffer_size:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)


        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        return dataset

# # 调试主程序
# if __name__ == "__main__":
#     # 参数
#     current_dir = os.path.dirname(os.path.abspath(__file__))

#     # 数据文件路径
#     file_path = os.path.join(
#         current_dir,
#         "../Ultra_AV/Longitudinal Trajectory Dataset/OpenACC/step3_ASta.csv"
#     )   
#     input_steps = 30
#     output_steps = 1
#     mse_threshold = 0.001
#     batch_size = 32

    # # 初始化数据处理器
    # processor = DataProcessor(file_path, input_steps, output_steps, mse_threshold)

    # # 输出调试信息
    # print(f"Number of training samples: {len(processor.X_train)}")
    # print(f"Number of validation samples: {len(processor.X_val)}")
    # print(f"Number of test samples: {len(processor.X_test)}")

    # # 获取 TensorFlow 数据集
    # train_dataset = processor.get_tf_dataset('train', batch_size)
    # val_dataset = processor.get_tf_dataset('val', batch_size)
    # test_dataset = processor.get_tf_dataset('test', batch_size)

    # # 打印数据集的部分内容
    # for batch_X, batch_y in train_dataset.take(1):
    #     print("Batch X shape:", batch_X.shape)
    #     print("Batch y shape:", batch_y.shape)
        