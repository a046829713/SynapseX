import pandas as pd
import numpy as np
import os
import datetime
from datetime import timedelta
import time

def fix_data_different_len_and_na(df: pd.DataFrame):
    # 取得所有時間
    all_times = set(df['Datetime'])

    new_df = pd.DataFrame()
    # 對每個 tic 進行處理
    for tic in df['tic'].unique():
        # 找出當前 tic 的數據
        tic_data = df[df['tic'] == tic]

        # 取得當下資料的行數
        diff_times = all_times - set(tic_data['Datetime'])
        dif_len = len(diff_times)

        # 如果需要填充
        if dif_len > 0:
            fill_data = pd.DataFrame({
                'Datetime': list(diff_times),
                'tic': tic,
                'Open': np.nan,
                'High': np.nan,
                'Low': np.nan,
                'Close': np.nan,
                'Volume': np.nan,
                'Fake_tag':1
            })
            # 將填充用的 Series 添加到原始 DataFrame
            tic_data = pd.concat([tic_data, fill_data])

        # 補上虛擬資料
        tic_data = tic_data.sort_values(by=['tic', 'Datetime'])
        tic_data = tic_data.fillna(0.0)
        new_df = pd.concat([new_df, tic_data])

    # 重新排序
    new_df = new_df.sort_values(by=['tic', 'Datetime'])
    return new_df

def generate_data(tag:str = None):
    """
        1.我希望只是將原始數據集合起來, 讓Env 自己去判斷.
        2.保留近3個月的資料做為測試集
    """
    new_df = pd.DataFrame()


    for each_symbol in ['ETHUSDT', 'SUIUSDT', 'ALGOUSDT', 'NEARUSDT', 'POPCATUSDT', 'XRPUSDT', 'TRXUSDT', 'JUPUSDT', 'HBARUSDT', 'BTCUSDT', 'GALAUSDT', 'ZENUSDT', 'WLDUSDT', 'SOLUSDT', 'TIAUSDT', 'CRVUSDT', 'BNBUSDT', 'ONDOUSDT', 'DOGEUSDT', 'ENAUSDT', 'AVAXUSDT', 'AGLDUSDT', 'LINKUSDT', 'WIFUSDT', 'FILUSDT', 'AAVEUSDT', 'ADAUSDT', 'DOTUSDT', 'LTCUSDT', 'XLMUSDT']:
        file_path = os.path.join("Brain","simulation","data",f"{each_symbol}-F-30-Min.csv")
        df = pd.read_csv(file_path)
        df['tic'] = each_symbol
        df['Fake_tag'] = 0
        new_df = pd.concat([new_df, df])
    
    new_df = fix_data_different_len_and_na(new_df)
    last_datetime = new_df[new_df['tic'] =='BTCUSDT'].iloc[-1]['Datetime']
    end_day = datetime.datetime.strptime(last_datetime,"%Y-%m-%d %H:%M:%S")
    last_train_day = str(end_day + timedelta(days= -90))
    new_df = new_df[(new_df['Datetime'] <= last_train_day)] 
    new_df.set_index('Datetime',inplace=True)

    if tag == 'train':
        output_file_path = os.path.join("Brain","PPO","simulation","train_data.csv")
        new_df.to_csv(output_file_path)
        print("資料寫入完成")
        
    elif tag == 'test':
        new_df.to_csv(r'Brain\PPO\simulation\test_data.csv')


if __name__ == '__main__':
    generate_data(tag='train')
