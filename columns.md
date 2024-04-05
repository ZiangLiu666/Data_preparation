There are 10841 rows and 13 columns in the dataset.

There are differen data types.

The columns in the dataset are:

- `['App', 'Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type', Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver','Android Ver']`

There are missing values in the dataset which we read by info's and also deals with them later on in this notebook.

There are some columns which are object instead of they must be in numeric data type.

- `['Reviews', 'Size', 'Installs', 'Price']`

## 

- Rating column

  ```python
  
  ```

- Size column

  ### 数据预处理：

  1. **转换大小单位**：首先，定义了一个函数 `convert_size`，它的目的是将应用大小的字符串表示（如'10M', '20k'）转换成数值格式，单位为千字节(KB)。这个函数处理了不同的情况：
     - 如果大小以'M'结尾（代表兆字节），则移除'M'并将剩余部分转换为数值，然后乘以1024转换为KB。
     - 如果大小以'k'结尾（代表千字节），则直接移除'k'并转换为数值。
     - 如果值是'NaN'或者字符串'Varies with device'，则返回'NaN'（代表缺失值）。
  2. **应用转换函数**：使用 `.apply()` 方法将 `convert_size` 函数应用于 DataFrame 的 'Size' 列，来转换所有大小的表示。
  3. **重命名列**：将 'Size' 列重命名为 'Size in Kbs' 以反映其单位和内容的变化。

- Installs column

  1. **定义转换函数**：
     - 名为 `convert` 的函数接受一个名为 `convertings` 的字符串参数，这个参数代表安装量的字符串表示（例如，'1,000,000+', '10,000+' 等）。
     - 函数内部，首先检查字符串是否包含逗号 ','。如果包含，则使用 `.replace()` 方法移除逗号。这是因为逗号在这里用作数字的千位分隔符，需要被去除以便正确转换成数值。
     - 接下来，检查字符串是否包含加号 '+'。如果包含，则同样用 `.replace()` 方法移除加号。这里的加号通常表示实际安装量可能超过显示的数量，但在转换为数值时应去除。
     - 最后，如果字符串不包含这些特殊字符，直接使用 `pd.to_numeric()` 方法将字符串转换为数值。
  2. **应用转换函数**：
     - 使用 `.apply()` 方法将 `convert` 函数应用于 DataFrame `data` 的 'Installs' 列。这个过程会遍历 'Installs' 列的每个条目，将其转换成数值，并将结果存回相同的列中。

- Content Rating

- Price column

  1. **转换数据类型**：定义了一个名为 `convertings` 的函数，它接受一个名为 `convert` 的字符串参数，这个参数代表应用的价格。这个函数的目的是将价格的字符串表示转换成数值格式。
  2. **移除非数字值**：“Price”列中可能包含一些非数字值，比如“Everyone”。这些值对于价格信息是无效的，需要将它们移除或替换。
  3. 如果 `convert` 的值是 '0'，表示应用是免费的，函数直接返回数值 0。这里使用 `pd.to_numeric()` 是多余的，因为 '0' 已经是数值 0 的正确表示，但这不会影响结果。
  4. 如果 `convert` 包含美元符号 ('$')，说明表示的是应用的实际价格，此时会移除 '$' 并将剩余的部分转换为数值，任何无法转换的值（由于格式错误或其他原因）将被设置为NaN。
  5. **应用转换函数**：使用 `.apply()` 方法将 `convertings` 函数应用于 DataFrame `data` 的 'Price' 列，将所有价格从字符串转换为数值格式。这样可以使得后续的数据分析和处理更加方便。
  6. **统计和去重**：使用`.value_counts()`统计每个不同价格值出现的次数，使用`.unique()`获取所有独特的价格值。这有助于检查数据清洗后的结果。
  7. **移除逗号和加号**：如果存在，应从数据中移除逗号和加号，因为它们可能干扰将字符串转换为数值。
  8. **数据清洗后的验证**：在处理结束后，重要的是要验证数据的清洗是否成功，确保“Price”列现在完全由数值组成，没有任何格式错误或非法值。

- Reviews column

  ```python
  # converting reviews column to numeric type
  data['Reviews'] = data['Reviews'].apply(lambda x: pd.to_numeric(x.replace("'", '') if "'" in x else x))
  ```

  Observations


- We have 1696 missing values in the `Size_in_bytes` and `Size_in_Mb` columns, which is 15.6% of the total values in the column.
- We have 1474 missing values in the `Rating` column, which is 13.6% of the total values in the column.
- We have 8 missing value in the `Current Ver` column, which is 0.07% of the total values in the column.
- We have 3 missing values in the `Android Ver` column, which is 0.01% of the total values in the column.
- We have only 1 missing value in `Reviews`, `Type`, `Price`, and `Content Rating` columns which is 0.009% of the total values in the column.

### Deals with Missing Values

heatmap

- unnecessary column

  ```python
  data.drop(columns=['Last Updated', 'Current Ver', 'Android Ver']，inplace=True)
  ```

- ### 缺失值预测：

  1. **分离数据**：将原始数据集分成两个部分：一个包含缺失 'Size in Kbs' 值的数据（`df2_missing`），另一个包含非缺失值的数据（`df2_not_missing`）。
  2. **设置训练数据**：从 `df2_not_missing` 中提取特征矩阵 `X_train`（除了 'Size in Kbs' 的其它列）和目标向量 `y_train`（'Size in Kbs' 列）。
  3. **选择模型和参数**：选择 `RandomForestRegressor` 作为回归模型，并定义了一系列可能的超参数，用于之后的网格搜索。
  4. **网格搜索和模型训练**：
     - 使用 `GridSearchCV` 在定义的参数空间上运行交叉验证，以找到最佳的模型参数。
     - 使用找到的最佳参数初始化新的 `RandomForestRegressor` 模型，并用非缺失数据 (`X_train`, `y_train`) 训练这个模型。
  5. **预测缺失值**：使用训练好的模型预测 `df2_missing` 中缺失的 'Size in Kbs' 值。
  6. **模型评估**：使用 R²（决定系数）评估模型在非缺失数据上的性能。R² 是衡量模型解释数据变异度的一个指标。
  7. **填充缺失值**：将预测出的缺失 'Size in Kbs' 值填充回原始的 `df` DataFrame。
  8. **输出**：打印出模型的 R² 分数和通过网格搜索找到的最佳参数。