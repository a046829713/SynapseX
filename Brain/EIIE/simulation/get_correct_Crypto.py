import pandas as pd
import numpy as np


def fix_data_different_len_and_na(df: pd.DataFrame):
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
            })
            # 將填充用的 Series 添加到原始 DataFrame
            tic_data = pd.concat([tic_data, fill_data])

        # 補上虛擬資料
        tic_data = tic_data.sort_values(by=['tic', 'date'])
        tic_data = tic_data.ffill(axis=0)
        tic_data = tic_data.bfill(axis=0)

        new_df = pd.concat([new_df, tic_data])

    # 重新排序
    new_df = new_df.sort_values(by=['tic', 'date'])
    return new_df


def generate_data(begin_time, end_time,tag:str = None):
    new_df = pd.DataFrame()

    #  ['INJUSDT','BNBUSDT','ETCUSDT','LTCUSDT','TRBUSDT',"ENSUSDT","SOLUSDT",'ETHUSDT','BCHUSDT',"AVAXUSDT","BTCUSDT"]
    for each_symbol in ['BNBUSDT','TRBUSDT',"SOLUSDT",'ETHUSDT',"BTCUSDT"]:
        df = pd.read_csv(f'EIIE\simulation\data\{each_symbol}-F-30-Min.csv')
        df['tic'] = each_symbol
        df.rename(columns={"Datetime": 'date',
                           "Close": "close",
                           "High": "high",
                           "Low": "low",
                           'Open': 'open',
                           'Volume': 'volume'
                           }, inplace=True)

        df = df[(df['date'] > begin_time) & (df['date'] < end_time)]
        new_df = pd.concat([new_df, df])

    new_df = fix_data_different_len_and_na(new_df)

    if tag == 'train':
        new_df.to_csv(f'EIIE\simulation\{begin_time}-{end_time}-train_data.csv')
    elif tag == 'test':
        new_df.to_csv(r'EIIE\simulation\test_data.csv')


if __name__ == '__main__':
    generate_data(begin_time='2023-01-01', end_time='2023-12-31',tag='train')
