import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import BinaryEncoder
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.preprocess import BaseTransformer
from src.corruptions.gaussian_noise import GaussianNoise
from src.corruptions.generic import InjectMissingValues
from src.corruptions.SwapValues import SwapColumnValues
from src.corruptions.text_noise import TextNoise

class CategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = None

    def fit(self, X, y=None):
        self.encoder = BinaryEncoder(cols=['Category'], return_df=True)
        self.encoder.fit(X)
        return self

    def transform(self, X):
        return self.encoder.transform(X)

class ContentRatingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.content_rating_mapping = {
            'Unrated': 0,
            'Everyone': 1,
            'Everyone 10+': 2,
            'Teen': 3,
            'Mature 17+': 4,
            'Adults only 18+': 5
        }

    def fit(self, X, y=None):
        # No fitting necessary for a simple mapping, so just return self
        return self

    def transform(self, X):
        X['Content Rating'] = X['Content Rating'].map(self.content_rating_mapping)
        return X
class AppTextTransformer(BaseTransformer):
    def __init__(self, column, max_features=500):
        self.column = column
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=self.max_features)

    def fit(self, X, y=None):
        text_data = X[self.column]
        self.vectorizer.fit(text_data)
        return self

    def transform(self, X):
        text_data = X[self.column]
        text_features = self.vectorizer.transform(text_data)

        # 将稀疏矩阵转换为DataFrame
        text_features_df = pd.DataFrame(text_features.toarray(), columns=self.vectorizer.get_feature_names_out())

        # 删除原始文本列，附加TF-IDF特征
        X = X.drop(columns=[self.column])
        X = pd.concat([X.reset_index(drop=True), text_features_df.reset_index(drop=True)], axis=1)

        return X


class DataPollutionTransformer(BaseTransformer):
    def __init__(self, gaussian_fraction=0.3, missing_fraction=0.3, swap_fraction=0.3, text_noise_fraction=0.3):
        self.gaussian_fraction = gaussian_fraction
        self.missing_fraction = missing_fraction
        self.swap_fraction = swap_fraction
        self.text_noise_fraction = text_noise_fraction

    def fit(self, X, y=None):
        # No fitting necessary for error injection
        return self

    def transform(self, X):
        # Apply Gaussian Noise
        gaussian_noise_transformer = GaussianNoise(column='Size', fraction=self.gaussian_fraction)
        X = gaussian_noise_transformer.transform(X)

        # Inject Missing Values
        missing_values_transformer = InjectMissingValues(column='Current Ver', fraction=self.missing_fraction)
        X = missing_values_transformer.transform(X)


        # Add Text Noise
        text_noise_transformer = TextNoise(column='App', fraction=self.text_noise_fraction)
        X = text_noise_transformer.transform(X)

        return X

import pandas as pd
class DataRepairTransformer(BaseTransformer):
    def transform(self, X, y=None):
        #X_repaired = X.copy()
        clf = IsolationForest(n_estimators=100, contamination='auto')
        clf.fit(X)
        predictions = clf.predict(X)
        outlier_indices = np.where(predictions == -1)[0]
        X_repaired = X.drop(index=X.index[outlier_indices]).copy()

        return X_repaired