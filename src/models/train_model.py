# train_model.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Import the necessary models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# Import the pipeline building function
from src.features.build_features import build_features_pipeline

# Evaluation metric
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train the model and calculate evaluation metrics."""
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate evaluation metrics
    scores = {
        'train': {
            'RMSE': rmse(y_train, y_train_pred),
            'MAE': mean_absolute_error(y_train, y_train_pred),
            'R2': r2_score(y_train, y_train_pred),
        },
        'test': {
            'RMSE': rmse(y_test, y_test_pred),
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'R2': r2_score(y_test, y_test_pred),
        }
    }

    return scores


def train_and_evaluate(X, y):

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the feature engineering pipeline
    feature_pipeline = build_features_pipeline()

    # Define the models to train
    models = {
        'LinearRegression': LinearRegression(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'RandomForestRegressor': RandomForestRegressor(random_state=42)
    }

    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        # Create a pipeline that includes feature processing and the model
        full_pipeline = Pipeline([
            ('features', feature_pipeline),
            ('model', model)
        ])

        # Evaluate the model
        results[name] = evaluate_model(full_pipeline, X_train, X_test, y_train, y_test)

    return results
