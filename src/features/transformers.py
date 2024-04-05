from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import BinaryEncoder
import pandas as pd

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
