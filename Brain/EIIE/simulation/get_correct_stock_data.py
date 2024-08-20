import pandas as pd
from datetime import datetime
import time
import numpy as np

def get_weekday_name(date_str):
    """
    根據給定的日期字符串返回星期名稱。

    參數:
    date_str (str): 日期字符串，格式應為 "YYYY-MM-DD"。

    返回:
    str: 給定日期的星期名稱。
    """
    # 將字符串轉換為 datetime 對象
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    # 獲取星期的索引
    weekday_index = date_obj.weekday()
    return weekday_index


def process_stock_data(file_path, stock_ids, cutoff_year):
    """
    處理股票數據，選擇特定股票並重命名列。

    參數:
    file_path (str): CSV 文件的路徑。
    stock_ids (list): 要選擇的股票代碼列表。
    cutoff_year (str): 過濾數據的截止年份。

    返回:
    DataFrame: 處理後的股票數據。
    """
    # 讀取數據
    df = pd.read_csv(file_path)

    # 選擇特定股票
    df = df[df['stock_id'].isin(stock_ids)]

    # 重命名列
    df.rename(columns={
        "stock_id": "tic",
        "開盤價": "open",
        "最高價": "high",
        "最低價": "low",
        "收盤價": "close",
        "成交股數": "volume",
    }, inplace=True)

    # 選擇列並添加 .TW
    df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'tic']]
    df['tic'] = df['tic'] + '.TW'

    # 過濾數據
    df = df[df['date'] < cutoff_year]

    # 添加星期列
    df['day'] = df['date'].apply(get_weekday_name)

    return df




def fix_data_different_len_and_na(df:pd.DataFrame):
    # 找出最長的歷史數據長度
    max_length = df.groupby('tic').count().max()

    # 取得所有時間
    all_times = set(df['date'])

    
    new_df = pd.DataFrame()
    # 對每個 tic 進行處理
    for tic in df['tic'].unique():
        # 找出當前 tic 的數據
        tic_data = df[df['tic'] == tic]
        
        # 取得當下資料的行數
        diff_times = all_times - set(tic_data['date'])
        dif_len = len(diff_times)
        
        # 如果需要填充
        if dif_len > 0:         
            fill_data = pd.DataFrame({
                'date': list(diff_times),
                'tic': tic,
                'open': np.nan,
                'high': np.nan,
                'low': np.nan,
                'close': np.nan,
                'volume': np.nan,
                'day': 0
            })
            # 將填充用的 Series 添加到原始 DataFrame
            tic_data = pd.concat([tic_data, fill_data])

        # 補上虛擬資料
        tic_data = tic_data.sort_values(by=['tic', 'date'])
        tic_data = tic_data.ffill(axis=0)
        tic_data = tic_data.bfill(axis=0)

        # 更新星期列
        tic_data['day'] = tic_data['date'].apply(get_weekday_name)    
        new_df = pd.concat([new_df,tic_data])
    

    
    # 重新排序
    new_df = new_df.sort_values(by=['tic', 'date'])    
    return new_df






if __name__ =="__main__":
    # 使用函數示例
    file_path = 'simulation/TaiwanStockHistoryDailyData.csv'
    stock_ids = ['2330', '2317','3008','1301']
    cutoff_year = '2020'
    processed_df = process_stock_data(file_path, stock_ids, cutoff_year)
    df = fix_data_different_len_and_na(processed_df)
    df.to_csv('train_data.csv')