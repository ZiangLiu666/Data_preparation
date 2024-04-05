import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.preprocess import data_cleaning_pipeline
from src.features.build_features import build_features_pipeline

# 导入所需的模型类
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from src.models.train_model import evaluate_model, rmse, train_and_evaluate

if __name__ == "__main__":
    # 加载数据集
    df = pd.read_csv('../data/raw/googleplaystore.csv')
    # 删除异常行
    df.drop(df.index[10472], inplace=True)

    # 应用数据清洗Pipeline，假设这已经包含了对Rating缺失值的处理
    df_cleaned = data_cleaning_pipeline.fit_transform(df)

    # 分离特征和目标变量
    X = df_cleaned.drop('Rating', axis=1)  # 移除Rating列作为特征
    y = df_cleaned['Rating']  # 目标变量

    # 对比不使用数据污染和修复的情况
    results_without_pollution = train_and_evaluate(X, y, pollute_and_repair=False)

    # 对比使用数据污染和修复的情况
    results_with_pollution = train_and_evaluate(X, y, pollute_and_repair=True)

    # 打印评估结果
    for model_name, scores in results_without_pollution.items():
        print(f"Results for {model_name}:")
        print(f"Train RMSE: {scores['train']['RMSE']:.2f}")
        print(f"Train MAE: {scores['train']['MAE']:.2f}")
        print(f"Train R2: {scores['train']['R2']:.2f}")
        print(f"Test RMSE: {scores['test']['RMSE']:.2f}")
        print(f"Test MAE: {scores['test']['MAE']:.2f}")
        print(f"Test R2: {scores['test']['R2']:.2f}")
        print('---------------------------------')

    for model_name, scores in results_with_pollution.items():
        print(f"Results for {model_name} with pollution and repair:")
        print(f"Train RMSE: {scores['train']['RMSE']:.2f}")
        print(f"Train MAE: {scores['train']['MAE']:.2f}")
        print(f"Train R2: {scores['train']['R2']:.2f}")
        print(f"Test RMSE: {scores['test']['RMSE']:.2f}")
        print(f"Test MAE: {scores['test']['MAE']:.2f}")
        print(f"Test R2: {scores['test']['R2']:.2f}")
        print('---------------------------------')
