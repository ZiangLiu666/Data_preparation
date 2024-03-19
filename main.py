import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

# 假设这些类导入语句有效
from src.corruptions.gaussian_noise import GaussianNoise
from src.corruptions.generic import InjectMissingValues
from src.corruptions.gaussian_noise import Scaling

# 加载数据集
# 这里使用的是示意性的导入方式，实际应用中请替换为实际的数据加载方式
from Test import reviews_with_products_and_ratings

# 创建二元目标变量
reviews_with_products_and_ratings['is_helpful'] = reviews_with_products_and_ratings['helpful_votes'] > 0

# 特征选择
features = reviews_with_products_and_ratings[['star_rating', 'category_id', 'above_median_total_votes']]
labels = reviews_with_products_and_ratings['is_helpful']

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 定义模型
model = HistGradientBoostingClassifier(random_state=42)

# 特征化
featurization = ColumnTransformer([
    ('scale', StandardScaler(), ['star_rating']),
    ('onehot', OneHotEncoder(), ['category_id']),
], remainder='passthrough')

X_train_transformed = featurization.fit_transform(X_train)
X_test_transformed = featurization.transform(X_test)

# 训练模型
model.fit(X_train_transformed, y_train)

# 计算基线准确率
baseline_accuracy = accuracy_score(y_test, model.predict(X_test_transformed))
print(f'Baseline Accuracy: {baseline_accuracy:.4f}')

# 应用数据错误并评估影响
corruptions = [
    GaussianNoise(column='star_rating', fraction=0.6),
    InjectMissingValues(column='star_rating', fraction=0.6),
    Scaling(column='star_rating', fraction=0.6)
]

for corruption in corruptions:
    # 注入错误
    X_test_corrupted = X_test.copy()
    corruption.transform(X_test_corrupted)

    # 对注入错误后的数据进行特征化
    X_test_corrupted_transformed = featurization.transform(X_test_corrupted)

    # 评估
    corrupted_accuracy = accuracy_score(y_test, model.predict(X_test_corrupted_transformed))
    print(f'Accuracy after applying {corruption.__class__.__name__}: {corrupted_accuracy:.4f}')
