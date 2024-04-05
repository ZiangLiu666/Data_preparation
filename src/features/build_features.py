# build_features.py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 假设这些自定义转换器已经在transformers.py文件中定义
from src.features.transformers import CategoryEncoder, ContentRatingTransformer
from src.features.transformers import AppTextTransformer
from src.features.transformers import DataPollutionTransformer, DataRepairTransformer


def build_features_pipeline(pollute_and_repair=False):
    """pipeline for feature engineering"""
    numeric_features = ['Reviews', 'Size', 'Installs', 'Price', 'Updated_Month', 'Updated_Year']
    categorical_features = ['Type']
    category_feature = ['Category']
    text_feature = ['App']

    # numeric features pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # categorical features pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # custom transformer
    content_rating_transformer = ContentRatingTransformer()
    category_transformer = CategoryEncoder()
    text_transformer = AppTextTransformer(column='App')

    # ColumnTransformer中整合所有特征处理
    preprocessor = ColumnTransformer(
        transformers=[
            # fill missing values and scale numeric features
            ('num', numeric_transformer, numeric_features),
            # fill missing values and encode categorical features
            ('cat', categorical_transformer, categorical_features),
            # custom transformer for category feature
            ('category', category_transformer, category_feature),
            # custom transformer for content rating feature
            ('content_rating', content_rating_transformer, ['Content Rating']),
            # custom transformer for text feature
            ('text', text_transformer, text_feature),
        ],remainder='drop')

    if pollute_and_repair:
        pipeline = Pipeline(steps=[
            ('pollute', DataPollutionTransformer()),
            ('repair', DataRepairTransformer()),
            ('preprocessor', preprocessor),
        ])
    else:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
        ])

    return pipeline
