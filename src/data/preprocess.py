from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


class BaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise NotImplementedError()


# Any size should >= 10k, otherwise it should be wrong data
class TransformSize(BaseTransformer):
    def transform(self, X, y=None):
        X = X.copy()
        X['Size'] = (X['Size'].str.replace('M', '000')
                     .str.replace('k', '')
                     .replace("Varies with device", np.nan)
                     .astype(float))
        X['Size'] = X.apply(lambda row: row['Size'] * 1000 if row['Size'] < 10 else row['Size'], axis=1)
        X['Size'] = X['Size'] / 1000
        return X


class CleanReviews(BaseTransformer):
    def transform(self, X, y=None):
        X = X.copy()
        X["Reviews"] = X["Reviews"].astype(int)
        return X


class CleanInstallsAndPrice(BaseTransformer):
    def transform(self, X, y=None):
        X = X.copy()
        items_to_remove = ['+', ',', '$']
        for item in items_to_remove:
            X['Installs'] = X['Installs'].str.replace(item, '')
            X['Price'] = X['Price'].str.replace(item, '')
        X['Installs'] = X['Installs'].astype(int)
        X['Price'] = X['Price'].astype(float)
        return X


class TransformLastUpdated(BaseTransformer):
    def transform(self, X, y=None):
        X = X.copy()
        X['Last Updated'] = pd.to_datetime(X['Last Updated'])
        X['Updated_Month'] = X['Last Updated'].dt.month
        X['Updated_Year'] = X['Last Updated'].dt.year
        X.drop('Last Updated', axis=1, inplace=True)
        return X


class DropUnnecessaryColumns(BaseTransformer):
    def transform(self, X, y=None):
        X = X.copy()
        columns_to_drop = ['Genres', 'Current Ver', 'Android Ver']
        X.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        return X


class RatingImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='median'):
        self.strategy = strategy

    def fit(self, X, y=None):
        if self.strategy == 'median':
            self.fill_value = X['Rating'].median()
        elif self.strategy == 'mean':
            self.fill_value = X['Rating'].mean()
        else:
            raise ValueError("Unsupported strategy")
        return self

    def transform(self, X):
        X = X.copy()
        X['Rating'] = X['Rating'].fillna(self.fill_value)
        return X


class DropMissingValues(BaseTransformer):
    """自定义转换器：删除包含缺失值的行"""

    def __init__(self, column):
        self.column = column

    def transform(self, X, y=None):
        return X.dropna(subset=[self.column])


data_cleaning_pipeline = Pipeline(steps=[
    ('drop_columns', DropUnnecessaryColumns()),
    ('clean_reviews', CleanReviews()),
    ('clean_installs_price', CleanInstallsAndPrice()),
    ('transform_size', TransformSize()),
    ('transform_last_updated', TransformLastUpdated()),
    # ('drop_missing_ratings', DropMissingValues(column='Rating')),
    ('impute_rating', RatingImputer(strategy='median')),
])
