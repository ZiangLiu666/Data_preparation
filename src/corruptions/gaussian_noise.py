# numerical.py
import numpy as np
import pandas as pd
import random
from src.basis import TabularCorruption

class GaussianNoise(TabularCorruption):
    def __init__(self, column, fraction, sampling='CAR', na_value=np.nan):
        super().__init__(column, fraction, sampling)
        self.na_value = na_value

    def transform(self, data):
        df = data.copy(deep=True)
        stddev = np.std(df[self.column])
        scale = random.uniform(1, 5)

        if self.fraction > 0:
            rows = self.sample_rows(data)
            noise = np.random.normal(0, scale * stddev, size=len(rows))
            df.loc[rows, self.column] += noise
        return df

class Scaling(TabularCorruption):
    def __init__(self, column, fraction, sampling='CAR'):
        super().__init__(column, fraction, sampling)

    def transform(self, data):
        df = data.copy(deep=True)
        scale_factor = np.random.choice([10, 100, 1000])

        if self.fraction > 0:
            rows = self.sample_rows(data)
            df.loc[rows, self.column] *= scale_factor
        return df
