from src.basis import TabularCorruption
import numpy as np
import pandas as pd

class InjectMissingValues(TabularCorruption):
    def __init__(self, column, fraction, sampling='CAR', na_value=np.nan):
        super().__init__(column, fraction, sampling)
        self.na_value = na_value

    def transform(self, data):
        corrupted_data = data.copy(deep=True)
        rows_to_corrupt = self.sample_rows(corrupted_data)
        corrupted_data.loc[rows_to_corrupt, self.column] = self.na_value
        return corrupted_data
