from src.basis import TabularCorruption

class SwapColumnValues(TabularCorruption):
    def __init__(self, column, swap_with, fraction, sampling='CAR'):
        super().__init__(column, fraction, sampling)
        self.swap_with = swap_with

    def transform(self, data):
        corrupted_data = data.copy(deep=True)
        rows_to_swap = self.sample_rows(corrupted_data)

        # 保存原始数据类型
        original_dtype_column = corrupted_data[self.column].dtype
        original_dtype_swap_with = corrupted_data[self.swap_with].dtype

        # 执行交换操作前，将数据转换为对方列的数据类型
        temp_values = corrupted_data.loc[rows_to_swap, self.column].astype(original_dtype_swap_with).copy()
        corrupted_data.loc[rows_to_swap, self.column] = corrupted_data.loc[rows_to_swap, self.swap_with].astype(original_dtype_column)
        corrupted_data.loc[rows_to_swap, self.swap_with] = temp_values

        # 如果需要，可以在这里将列的数据类型转换回原始数据类型
        # corrupted_data[self.column] = corrupted_data[self.column].astype(original_dtype_column)
        # corrupted_data[self.swap_with] = corrupted_data[self.swap_with].astype(original_dtype_swap_with)

        return corrupted_data
