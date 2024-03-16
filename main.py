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

# 对于每个扰动操作，随机选择不同的列进行操作
# 注入缺失值的列
missing_columns = np.random.choice(numeric_columns, size=np.random.randint(1, len(numeric_columns)), replace=False)
for col in missing_columns:
    missing_value_injector = InjectMissingValues(column=col, fraction=0.3)
    df = missing_value_injector.transform(df)

# 添加高斯噪声的列
noise_columns = np.random.choice(numeric_columns, size=np.random.randint(1, len(numeric_columns)), replace=False)
for col in noise_columns:
    noise_injector = GaussianNoise(column=col, fraction=0.3)
    df = noise_injector.transform(df)

# 缩放操作的列
scaling_columns = np.random.choice(numeric_columns, size=np.random.randint(1, len(numeric_columns)), replace=False)
for col in scaling_columns:
    scaling_transformer = Scaling(column=col, fraction=0.6)
    df = scaling_transformer.transform(df)

# 选择两个随机列进行交换
random_columns = np.random.choice(numeric_columns, size=2, replace=False)
column_swapper = SwapColumnValues(column=random_columns[0], swap_with=random_columns[1], fraction=0.6)
final_df = column_swapper.transform(df)

# 验证操作
print(final_df[random_columns].head())
for col in missing_columns:
    print(f"Missing values in '{col}':", final_df[col].isnull().sum())
