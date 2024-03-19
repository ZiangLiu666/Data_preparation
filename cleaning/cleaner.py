from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class CustomDataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # 这里可以加入对数据的学习，比如模式识别、计算缺失值填充的值等
        # 例如，预计算模式（mode）
        self.mode_current_ver = X['Current Ver'].mode()[0]
        return self

    def transform(self, X, y=None):
        X = X.copy()
        
        # 使用XGBRegressor填充'Rating'列的缺失值
        impute = IterativeImputer(estimator=XGBRegressor(), max_iter=10, random_state=42)
        X['Rating'] = impute.fit_transform(X[['Rating']])

        # 填充'Current Ver'的缺失值
        X['Current Ver'].fillna(self.mode_current_ver, inplace=True)

        # 仅保留大小以'M','k'或'Varies with device'结尾的行
        size_pattern = r'(\d+M|\d+k|Varies with device)$'
        X = X[X['Size'].str.match(size_pattern)]
        
        # 转换'Size'列
        X['Size'] = X['Size'].apply(self.convert_size)
        
        # 填充'Size'列的缺失值
        X['Size'] = X['Size'].fillna(X['Size'].mean())

        # 转换'Installs'列
        X['Installs'] = X['Installs'].apply(self.convert_installs)

        # 转换'Price'列
        X['Price'] = X["Price"].apply(self.convert_price)

        # 转换'Reviews'列
        X['Reviews'] = X['Reviews'].apply(lambda x: pd.to_numeric(x.replace("'", '') if "'" in x else x))
        
        # 删除无关列
        X.drop(columns=['Last Updated', 'Current Ver', 'Android Ver', 'Content Rating'], inplace=True)
        
        return X
    
    def convert_size(self, convertings):
        if 'M' in convertings:
            return pd.to_numeric(convertings.replace('M', '')) * 1024
        elif 'k' in convertings:
            return pd.to_numeric(convertings.replace('k', ''))
        elif 'Varies with device' in convertings:
            return np.nan
        else:
            return pd.to_numeric(convertings)
        
    def convert_installs(self, convertings):
        if ',' in convertings:
            return convertings.replace(',', '').replace('+', '')
        else:
            return pd.to_numeric(convertings)
    
    def convert_price(self, convert):
        if '$' in convert:
            return pd.to_numeric(convert.replace('$', ''))
        else:
            return pd.to_numeric(convert)

# 使用自定义清洗器
cleaner = CustomDataCleaner()

# 假设df是你的原始DataFrame
# df = pd.read_csv('googleplaystore.csv')
# 清洗数据
# df_clean = cleaner.transform(df)
