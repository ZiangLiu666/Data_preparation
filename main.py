import pandas as pd
from src.corruptions.generic import InjectMissingValues
from src.corruptions.SwapValues import SwapColumnValues
from src.corruptions.gaussian_noise import GaussianNoise
from src.corruptions.gaussian_noise import Scaling
# Load the dataset
df = pd.read_csv('/Users/liuziang/PycharmProjects/Data preparation/data/winequality-red.csv')

# Inject missing values
missing_value_injector = InjectMissingValues(column='fixed acidity', fraction=0.1)
corrupted_df = missing_value_injector.transform(df)

# Add Gaussian noise to 'fixed acidity'
noise_injector = GaussianNoise(column='fixed acidity', fraction=0.1)
noisy_df = noise_injector.transform(corrupted_df)

# 对'fixed acidity'列进行缩放
scaling_transformer = Scaling(column='fixed acidity', fraction=0.1)
scaled_df = scaling_transformer.transform(noisy_df)

# Swap values between 'fixed acidity' and 'volatile acidity'
column_swapper = SwapColumnValues(column='fixed acidity', swap_with='volatile acidity', fraction=0.1)
final_df = column_swapper.transform(scaled_df)

# Verify the operations
print(final_df[['fixed acidity', 'volatile acidity']].head())
print("Missing values in 'fixed acidity':", final_df['fixed acidity'].isnull().sum())

