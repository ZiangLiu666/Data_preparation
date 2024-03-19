import pandas
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Data Loading
reviews = pandas.read_csv("reviews/reviews.csv.gz", compression='gzip', index_col=0)
ratings = pandas.read_csv("reviews/ratings.csv", index_col=0)
products = pandas.read_csv("reviews/products.csv", index_col=0)
categories = pandas.read_csv("reviews/categories.csv", index_col=0)

# Data Merging
reviews_with_ratings = reviews.merge(ratings, on='review_id')
products_with_categories = products.merge(categories, left_on='category_id', right_on='id', suffixes=('_prod', '_cat'))
reviews_with_products_and_ratings = reviews_with_ratings.merge(products_with_categories, on='product_id')

# Create a binary target variable
reviews_with_products_and_ratings['is_helpful'] = reviews_with_products_and_ratings['helpful_votes'] > 0

# Feature Engineering
median_total_votes_per_category = reviews_with_products_and_ratings.groupby('category_id').agg(median_total_votes=('total_votes', 'median')).reset_index()
reviews_with_products_and_ratings = reviews_with_products_and_ratings.merge(median_total_votes_per_category, on='category_id')
reviews_with_products_and_ratings['above_median_total_votes'] = reviews_with_products_and_ratings['total_votes'] > reviews_with_products_and_ratings['median_total_votes']

# Selecting Features and Labels
features = reviews_with_products_and_ratings[['star_rating', 'category_id', 'above_median_total_votes']]
labels = reviews_with_products_and_ratings['is_helpful']

# Featurization
featurization = ColumnTransformer([
    ('scale', StandardScaler(), ['star_rating']),
    ('onehot', OneHotEncoder(), ['category_id']),
    ('passthrough', 'passthrough', ['above_median_total_votes'])
])

# Prepare the data
features_transformed = featurization.fit_transform(features)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_transformed, labels, test_size=0.2, random_state=42)

# Model Training
model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

# Evaluation
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print(f'Training Accuracy: {accuracy_score(y_train, y_pred_train):.4f}')
print(f'Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}')
