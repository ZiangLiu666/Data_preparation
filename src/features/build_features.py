# build_features.py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 假设这些自定义转换器已经在transformers.py文件中定义
from src.features.transformers import CategoryEncoder, ContentRatingTransformer


def build_features_pipeline():
    """pipeline for feature engineering"""
    numeric_features = ['Reviews', 'Size', 'Installs', 'Price', 'Updated_Month', 'Updated_Year']
    categorical_features = ['Type']
    category_feature = ['Category']

    # 数值型特征的填充与缩放
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 分类型特征的填充与编码
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 自定义特征的转换器
    content_rating_transformer = ContentRatingTransformer()
    category_transformer = CategoryEncoder()

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
            ('content_rating', content_rating_transformer, ['Content Rating'])
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    return pipeline
