# 可以用來進行選股實驗

import pandas as pd
import numpy as np
from Major.DataProvider import DataProvider


# 
DataProvider().get_symbols_history_data(symbol_type='FUTURES', time_type='1d')




# 读取股票数据（假设数据包含多个股票的历史价格）
# 数据应包含以下列：['Date', 'Stock_ID', 'Close']
data = pd.read_csv('stock_data.csv')

# 将日期列转换为datetime格式
data['Date'] = pd.to_datetime(data['Date'])

# 按日期和股票ID排序
data.sort_values(['Stock_ID', 'Date'], inplace=True)

# 计算每只股票的每日涨跌幅
data['Price_Change'] = data.groupby('Stock_ID')['Close'].diff()

# 删除NaN值（由于diff导致的）
data.dropna(inplace=True)

# 特征标准化（可选，根据需要）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data['Close_Scaled'] = scaler.fit_transform(data[['Close']])

# 查看预处理后的数据
print(data.head())
