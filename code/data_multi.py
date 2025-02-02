import tensorflow as tf
import numpy as np
import pandas as pd
from physics_model import IDM

class DataProcessor:
    def __init__(self, file_path, input_steps, output_steps, train=True, t_interval=0.1, t_fre=0.1, mse_threshold=0.01):
        self.file_path = file_path
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.t_interval = t_interval
        self.t_fre = t_fre
        self.mse_threshold = mse_threshold

        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.preprocess_data()
        self._train = train
      
    def load_best_params(self, file_path):
        """从文本文件中读取最优的 IDM 参数"""
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
        data = data.iloc[0::int(self.t_fre / self.t_interval), :]
        data = self.filter_low_acceleration(data, threshold=0.1)
        self.data = data

    def filter_low_acceleration(self, data, threshold=0.1):
        return data[np.abs(data['Acc_FAV']) >= threshold]

    def preprocess_data(self):
        self.load_data()
        X, y = [], []
        X_tra, y_tra = [], []
        current_trajectory = None

        for i, row in self.data.iterrows():
            if row['Trajectory_ID'] != current_trajectory:
                if len(X_tra) > 0:
                    X.append(X_tra)
                    y.append(y_tra)
                current_trajectory = row['Trajectory_ID']
                tra_data = []
                X_tra, y_tra = [], []

            tra_data.append([row['Speed_LV'], row['Acc_LV'], row['Speed_FAV'], row['Acc_FAV'], row['Spatial_Gap']])
            if len(tra_data) == self.input_steps + self.output_steps:
                X_state = np.array(tra_data[:self.input_steps])
                y_state = np.array(tra_data[self.input_steps:])
                if self.is_sample_valid(X_state, y_state):
                    X_tra.append(X_state)
                    y_tra.append(y_state)
                tra_data = tra_data[1:]

        X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []
        for i in range(len(X)):
            train_len = int(len(X[i]) * 0.7)
            val_len = int(len(X[i]) * 0.15)

            X_train.extend(X[i][:train_len - val_len])
            y_train.extend(y[i][:train_len - val_len])

            X_val.extend(X[i][train_len - val_len:train_len])
            y_val.extend(y[i][train_len - val_len:train_len])

            X_test.extend(X[i][train_len:])
            y_test.extend(y[i][train_len:])

        return (
            np.array(X_train), np.array(X_val), np.array(X_test),
            np.array(y_train), np.array(y_val), np.array(y_test)
        )

    def is_sample_valid(self, input_data, output_data):
        Speed_LV = input_data[-1, 0]
        Speed_FAV = input_data[-1, 2]
        Spatial_Gap = input_data[-1, 4]
        Delta_v = Speed_LV - Speed_FAV

        idm_params = self.load_best_params("calibration_results_OpenACC_ASta.txt")
        ahat_idm = IDM(idm_params, np.array([Speed_FAV]), np.array([Delta_v]), np.array([Spatial_Gap]))
        mse = (output_data[0, 3] - ahat_idm[0]) ** 2
        return mse < self.mse_threshold

    def get_tf_dataset(self, dataset_type, batch_size, drop_remainder=True, shuffle=True, buffer_size=200):
        if dataset_type == 'train':
            dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        elif dataset_type == 'val':
            dataset = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        elif dataset_type == 'test':
            dataset = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))
        else:
            raise ValueError("dataset_type must be 'train', 'val', or 'test'")
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        return dataset
