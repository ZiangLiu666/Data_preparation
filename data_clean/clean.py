import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

def fill_missing_values(df):
    """
    Fill missing values with the median of each column.
    """
    imputer = SimpleImputer(strategy='median')
    df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_filled

def reverse_column_swaps(df):
    """
    Reverse column swaps. This example assumes 'math_score' and 'history_score' might have been swapped.
    """
    # Logic to identify and reverse the swap.
    # This is just an example and would need actual logic to detect swaps.
    df['math_score'], df['history_score'] = df['history_score'], df['math_score']
    return df

def reduce_noise(df):
    """
    Apply a RobustScaler to reduce the impact of outliers (which could be a result of Gaussian noise).
    """
    scaler = RobustScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def clean_data(df):
    """
    Apply all cleaning steps to the dataframe.
    """
    df_cleaned = fill_missing_values(df)
    df_cleaned = reverse_column_swaps(df_cleaned)
    df_cleaned = reduce_noise(df_cleaned)
    return df_cleaned


