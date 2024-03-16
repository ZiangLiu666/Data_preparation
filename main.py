import pandas as pd
import numpy as np
from src.corruptions.generic import InjectMissingValues
from src.corruptions.SwapValues import SwapColumnValues
from src.corruptions.gaussian_noise import GaussianNoise
from src.corruptions.gaussian_noise import Scaling

# 加载数据集
df = pd.read_csv('/Users/liuziang/PycharmProjects/Data preparation/data/winequality-red.csv')

# 获取数值型列的列表
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# 随机选择列进行操作
random_column = np.random.choice(numeric_columns)

# 注入缺失值到随机选中的列
missing_value_injector = InjectMissingValues(column=random_column, fraction=0.1)
corrupted_df = missing_value_injector.transform(df)

# 在相同的随机选中列添加高斯噪声
noise_injector = GaussianNoise(column=random_column, fraction=0.1)
noisy_df = noise_injector.transform(corrupted_df)

# 对随机选中的列进行缩放
scaling_transformer = Scaling(column=random_column, fraction=0.1)
scaled_df = scaling_transformer.transform(noisy_df)

# 选择两个随机列进行交换，确保选择的是不同的列
random_columns = np.random.choice(numeric_columns, size=2, replace=False)
column_swapper = SwapColumnValues(column=random_columns[0], swap_with=random_columns[1], fraction=0.1)
final_df = column_swapper.transform(scaled_df)

# 验证操作
print(final_df[random_columns].head())
print(f"Missing values in '{random_column}':", final_df[random_column].isnull().sum())
