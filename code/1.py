import pandas as pd

# 读取 CSV 文件
file_path = "/Users/shixiaoliang/Library/Mobile Documents/com~apple~CloudDocs/Documents/PERL_foundation/code/200_128_50.csv"
data = pd.read_csv(file_path)

# 初始化存储数据的列表
lstm_data = []
perl_data = []

# 初始化计数器，用于跟踪遇到 data_size = 40 的次数
datasize_40_counter = 0
collecting = []  # 用于暂存每次遇到 40 到 200 的数据

# 遍历 CSV 中的每一行
for index, row in data.iterrows():
    datasize = row['data_size']  # 假设列名是 DataSize
    
    # 如果遇到 data_size = 40，先处理前一次的数据组
    if datasize == 40:
        if datasize_40_counter > 0:  # 不是第一次遇到 40
            # 根据当前的计数器决定分配到 LSTM 或 PERL
            if datasize_40_counter % 2 == 0:
                lstm_data.extend(collecting)  # 偶数次存入 LSTM
            else:
                perl_data.extend(collecting)  # 奇数次存入 PERL
            
            collecting = []  # 清空暂存组

        datasize_40_counter += 1  # 更新计数器

    # 将当前行存入暂存组
    collecting.append(row)

# 最后一组数据（如果存在），根据计数器分配
if datasize_40_counter % 2 == 0:
    lstm_data.extend(collecting)
else:
    perl_data.extend(collecting)

# 将结果转为 DataFrame
lstm_df = pd.DataFrame(lstm_data)
perl_df = pd.DataFrame(perl_data)

# 保存结果（如果需要）
lstm_df.to_csv("lstm_output.csv", index=False)
perl_df.to_csv("perl_output.csv", index=False)

# 检查分组结果
print("LSTM Data:")
print(lstm_df.head())
print("\nPERL Data:")
print(perl_df.head())
