from src.basis import TabularCorruption
import numpy as np
import pandas as pd

class SwapColumnValues(TabularCorruption):
    def __init__(self, column, swap_with, fraction, sampling='CAR'):
        super().__init__(column, fraction, sampling)
        self.swap_with = swap_with

    def transform(self, data):
        corrupted_data = data.copy(deep=True)
        rows_to_swap = self.sample_rows(corrupted_data)
        temp_values = corrupted_data.loc[rows_to_swap, self.column].copy()
        corrupted_data.loc[rows_to_swap, self.column] = corrupted_data.loc[rows_to_swap, self.swap_with]
        corrupted_data.loc[rows_to_swap, self.swap_with] = temp_values
        return corrupted_data
