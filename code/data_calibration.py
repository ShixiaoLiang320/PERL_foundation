import random
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import pandas as pd
from physics_model import IDM, FVD
import os

def monte_carlo_optimization(df, num_iterations):
    best_mse = float('inf')
    best_arg = None
    df = df.sort_values(by=['Trajectory_ID', 'Time_Index'])
    df['Speed_FAV_previous'] = df.groupby('Trajectory_ID')['Speed_FAV'].shift(1)  # 前一时间步的跟车车速

    with tqdm(total=num_iterations, desc='Iterations', postfix={'Best RMSE': float('inf')}) as pbar:
        for _ in range(num_iterations):
            # 随机生成 IDM 参数
            vf = random.uniform(20, 25)  # 自由流速度
            A = random.uniform(0, 5)  # 最大加速度
            b = random.uniform(0, 5)  # 舒适减速度
            s0 = random.uniform(0.5, 2.5 )  # 最小空间间距
            T = random.uniform(0.5, 2.5)  # 跟车时间间隔
            arg = (round(vf, 3), round(A, 3), round(b, 3), round(s0, 3), round(T, 3))

            # 计算预测的加速度
            df['a_hat'] = df.apply(
                lambda row: IDM(arg, row['Speed_FAV'], row['Speed_Diff'], row['Spatial_Gap']),
                axis=1
            )

            # 计算均方误差
            mse = mean_squared_error(df['Acc_FAV'].dropna(), df['a_hat'].dropna())

            # 验证数据集中的速度预测误差
            df_valid = df.dropna(subset=['Speed_FAV_previous']).copy()
            df_valid['V_hat'] = df_valid['Speed_FAV_previous'] + df_valid['a_hat'] * 0.1  # 使用 0.1s 时间步更新速度
            mse_v = mean_squared_error(df_valid['Speed_FAV'].dropna(), df_valid['V_hat'].dropna())

            # 更新最佳参数
            if mse < best_mse:
                best_mse = mse
                best_arg = arg

            pbar.set_postfix_str({'Best MSE': round(best_mse, 3), 'best_arg': best_arg})
            pbar.update(1)

    return best_arg, best_mse


# Load data
current_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(
    current_dir,
    "../Ultra_AV/Longitudinal Trajectory Dataset/OpenACC/step3_ASta.csv"
)

df = pd.read_csv(file_path)


#df = df[df['ID_LV'] != 0]  # 过滤无效的前车信息
#df = df.dropna(subset=['Speed_LV', 'Acc_LV', 'Speed_FAV', 'Acc_FAV'])  # 删除缺失值
#df = df[(df['Spatial_Gap'] > 2) & (df['Spatial_Gap'] < 200)]  # 过滤空间间距异常值
#df = df[(df['Speed_Diff'] > -20) & (df['Speed_Diff'] < 20)]  # 过滤速度差异常值

df = df.groupby('Trajectory_ID').filter(lambda x: len(x) >= 10)
df = df.groupby('Trajectory_ID').apply(lambda x: x.sample(len(x), random_state=1)).reset_index(drop=True)

print('After filtering and sampling len(df)=', len(df))

print("Starting Monte Carlo Optimization...")
best_arg, best_rmse = monte_carlo_optimization(df, num_iterations=5000)

print("\nOptimization Completed!")
print(f"Best Parameters (vf, A, b, s0, T): {best_arg}")
print(f"Best MSE: {best_rmse:.6f}")

output_file = "calibration_results_test.txt"
with open(output_file, "w") as file:
    file.write("Monte Carlo Optimization Results\n")
    file.write("===============================\n")
    file.write(f"Best Parameters (vf, A, b, s0, T): {best_arg}\n")
    file.write(f"Best MSE: {best_rmse:.6f}\n")

print(f"\nResults have been saved to {output_file}")
