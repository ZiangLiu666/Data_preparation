import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from corruptions.SwapValues import SwapColumnValues
from corruptions.gaussian_noise import GaussianNoise, Scaling
from corruptions.generic import InjectMissingValues

# Load the dataset
df = pd.read_csv('/Users/liuziang/PycharmProjects/DP/data_clean/student-scores.csv')

# Create binary target variable based on the median of average scores across subjects
df['average_score'] = df[['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']].mean(axis=1)
median_score = df['average_score'].median()
df['above_median'] = (df['average_score'] > median_score).astype(int)

# Define preprocessing
numerical_features = ['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score', 'weekly_self_study_hours']
categorical_features = ['gender', 'part_time_job', 'extracurricular_activities', 'career_aspiration']
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Split the data
features = df.drop(['id', 'above_median', 'average_score', 'first_name', 'last_name', 'email'], axis=1)  # Assume 'id', 'first_name', 'last_name', 'email' are not used for training
labels = df['above_median']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Preprocess the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Train the baseline model
baseline_model = HistGradientBoostingClassifier(random_state=42)
baseline_model.fit(X_train_preprocessed, y_train)

# Evaluate the baseline model
baseline_accuracy = accuracy_score(y_test, baseline_model.predict(X_test_preprocessed))
print(f'Baseline Accuracy: {baseline_accuracy:.4f}')

# Function to inject errors into a dataframe
def inject_errors_to_multiple_columns(df):
    # Apply Gaussian noise to multiple numerical columns
    for column in ['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score']:
        gaussian_noise = GaussianNoise(column=column, fraction=0.3)  # Increased fraction for a more pronounced effect
        df = gaussian_noise.transform(df)

    # Swap values between pairs of columns to simulate data entry errors
    swap_pairs = [('history_score', 'physics_score'), ('chemistry_score', 'biology_score')]
    for column, swap_with in swap_pairs:
        swap_column_values = SwapColumnValues(column=column, swap_with=swap_with, fraction=0.3)
        df = swap_column_values.transform(df)

    # Scale 'english_score' and 'geography_score'
    for column in ['english_score', 'geography_score']:
        scaling = Scaling(column=column, fraction=0.3)
        df = scaling.transform(df)

    # Inject missing values into 'weekly_self_study_hours' and a categorical column
    for column in ['weekly_self_study_hours', 'extracurricular_activities']:
        inject_missing_values = InjectMissingValues(column=column, fraction=0.3)
        df = inject_missing_values.transform(df)

    return df


# Inject errors into multiple columns
df_noisy_multiple = inject_errors_to_multiple_columns(df)

# Prepare data with more extensive errors for training
X_noisy_multiple = df_noisy_multiple.drop(['id', 'above_median', 'average_score', 'first_name', 'last_name', 'email'],
                                          axis=1)
y_noisy_multiple = df_noisy_multiple['above_median']
X_train_noisy_multiple, X_test_noisy_multiple, y_train_noisy_multiple, y_test_noisy_multiple = train_test_split(
    X_noisy_multiple, y_noisy_multiple, test_size=0.2, random_state=42)

X_train_noisy_multiple_preprocessed = preprocessor.fit_transform(X_train_noisy_multiple)
X_test_noisy_multiple_preprocessed = preprocessor.transform(X_test_noisy_multiple)

# Train and evaluate the model on data with more extensive errors
model_noisy_multiple = HistGradientBoostingClassifier(random_state=42)
model_noisy_multiple.fit(X_train_noisy_multiple_preprocessed, y_train_noisy_multiple)
noisy_multiple_data_accuracy = accuracy_score(y_test_noisy_multiple,
                                              model_noisy_multiple.predict(X_test_noisy_multiple_preprocessed))
print(f'Accuracy After Introducing Errors to Multiple Columns: {noisy_multiple_data_accuracy:.4f}')
