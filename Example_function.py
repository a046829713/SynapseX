from Major.DataProvider import DataProvider
from Major.Datatransformer import Datatransformer
# from EIIE.lib import Train_neural_networks
import matplotlib.pyplot as plt
from pathlib import Path

from binance.client import Client
from Major.UserManager import UserManager
import json
import time
# from EIIE.lib.simple_evaluate import evaluate_train_test_performance
import re
import pandas as pd
from Database.SQL_operate import SqlSentense
 




"""
    1.取得日線資料
    2.整理日線資料
    3.保存日線資料to local.

"""


# def fix_data_different_len_and_na(df: pd.DataFrame):
#     # 找出最長的歷史數據長度
#     max_length = df.groupby('tic').count().max()

#     # 取得所有時間
#     all_times = set(df['date'])

#     new_df = pd.DataFrame()
#     # 對每個 tic 進行處理
#     for tic in df['tic'].unique():
#         # 找出當前 tic 的數據
#         tic_data = df[df['tic'] == tic]

#         # 取得當下資料的行數
#         diff_times = all_times - set(tic_data['date'])
#         dif_len = len(diff_times)

#         # 如果需要填充
#         if dif_len > 0:
#             fill_data = pd.DataFrame({
#                 'date': list(diff_times),
#                 'tic': tic,
#                 'open': np.nan,
#                 'high': np.nan,
#                 'low': np.nan,
#                 'close': np.nan,
#                 'volume': np.nan,
#             })
#             # 將填充用的 Series 添加到原始 DataFrame
#             tic_data = pd.concat([tic_data, fill_data])

#         # 補上虛擬資料
#         tic_data = tic_data.sort_values(by=['tic', 'date'])
#         tic_data = tic_data.ffill(axis=0)
#         tic_data = tic_data.bfill(axis=0)

#         new_df = pd.concat([new_df, tic_data])

#     # 重新排序
#     new_df = new_df.sort_values(by=['tic', 'date'])
#     return new_df


# def generate_data(begin_time, end_time,tag:str = None):


#     #  ['INJUSDT','BNBUSDT','ETCUSDT','LTCUSDT','TRBUSDT',"ENSUSDT","SOLUSDT",'ETHUSDT','BCHUSDT',"AVAXUSDT","BTCUSDT"]
#     for each_symbol in ['BNBUSDT','TRBUSDT',"SOLUSDT",'ETHUSDT',"BTCUSDT"]:
#         df = pd.read_csv(f'EIIE\simulation\data\{each_symbol}-F-30-Min.csv')
#         df['tic'] = each_symbol
#         df.rename(columns={"Datetime": 'date',
#                            "Close": "close",
#                            "High": "high",
#                            "Low": "low",
#                            'Open': 'open',
#                            'Volume': 'volume'
#                            }, inplace=True)

#         df = df[(df['date'] > begin_time) & (df['date'] < end_time)]
#         new_df = pd.concat([new_df, df])

#     new_df = fix_data_different_len_and_na(new_df)

#     if tag == 'train':
#         new_df.to_csv(f'EIIE\simulation\{begin_time}-{end_time}-train_data.csv')
#     elif tag == 'test':
#         new_df.to_csv(r'EIIE\simulation\test_data.csv')





def getAllDailyData():

    
    # get alive symbol
    all_symbols_dataframes = DataProvider().get_symbols_history_data(
        symbol_type='FUTURES', time_type='1d')

    new_df = pd.DataFrame()

    for tb_symbol_name,df in all_symbols_dataframes:        
        new_df = pd.concat([new_df, df])


    print(new_df)










def checksymbol(symbol:str):
    account, passwd = UserManager.GetAccount_Passwd('author')
    client = Client(account, passwd)
    info = client.get_exchange_info()
    for s in info['symbols']:
        if s['symbol']  == symbol:
            print(s)
    






def get_futures_position_information():
    account, passwd = UserManager.GetAccount_Passwd('author')
    client = Client(account, passwd)
    data = client.futures_position_information(symbol="BTCUSDT")
    print(float(data[0]['positionAmt']) != 0)
    # for i in data:
    #     print(i['symbol'],i['positionAmt'])
    #     print('*'*120)


def example_get_symboldata():
    """
        introduction:
            this function is for download history data to experiment.

    """
    

    symbols = list(
        set(['KOMAUSDT', 'VIRTUALUSDT', 'SPXUSDT', 'MEUSDT', 'AVAUSDT', 'DEGOUSDT', 'VELODROMEUSDT', 'MOCAUSDT', 'VANAUSDT', 'PENGUUSDT', 'LUMIAUSDT', 'USUALUSDT', 'AIXBTUSDT', 'FARTCOINUSDT', 'KMNOUSDT', 'CGPTUSDT', 'HIVEUSDT', 'DEXEUSDT', 'PHAUSDT', 'DFUSDT', 'GRIFFAINUSDT', 'ZEREBROUSDT', 'BIOUSDT', 'COOKIEUSDT', 'ALCHUSDT', 'SWARMSUSDT', 'SONICUSDT', 'DUSDT', 'PROMUSDT', 'SUSDT', 'SOLVUSDT', 'ARCUSDT', 'AVAAIUSDT', 'TRUMPUSDT', 'MELANIAUSDT', 'ANIMEUSDT', 'VINEUSDT', 'PIPPINUSDT', 'VVVUSDT', 'BERAUSDT', 'TSTUSDT', 'LAYERUSDT', 'HEIUSDT', 'IPUSDT', 'GPSUSDT', 'SHELLUSDT', 'KAITOUSDT', 'REDUSDT', 'VICUSDT', 'EPICUSDT', 'BMTUSDT', 'MUBARAKUSDT', 'FORMUSDT', 'BIDUSDT', 'TUTUSDT', 'SIRENUSDT', 'BRUSDT', 'PLUMEUSDT', 'NILUSDT', 'PARTIUSDT', 'JELLYJELLYUSDT', 'MAVIAUSDT', 'PAXGUSDT', 'WALUSDT', 'MLNUSDT', 'GUNUSDT', 'ATHUSDT', 'BABYUSDT', 'FORTHUSDT', 'PROMPTUSDT', 'XCNUSDT', 'STOUSDT', 'FHEUSDT', 'KERNELUSDT', 'WCTUSDT', 'INITUSDT', 'AERGOUSDT', 'BANKUSDT', 'DEEPUSDT', 'HYPERUSDT', 'FISUSDT', 'JSTUSDT', 'SIGNUSDT', 'PUNDIXUSDT', 'CTKUSDT', 'AIOTUSDT', 'DOLOUSDT', 'HAEDALUSDT', 'SXTUSDT', 'ASRUSDT', 'ALPINEUSDT', 'MILKUSDT', 'SYRUPUSDT', 'OBOLUSDT', 'OGUSDT', 'ZKJUSDT', 'SKYAIUSDT', 'NXPCUSDT', 'CVCUSDT', 'AWEUSDT']))



    for _each_symbol_name in symbols:
        DataProvider().Downloader(symbol_name=_each_symbol_name, save=True, freq=30)


def example_get_targetsymobls():
    """
        取得有效名稱
    """
    print(DataProvider().get_both_type_targetsymbols())


def example_get_target_symbol(filter_type: str):
    """
        introduction:
            this function is for filter 
    """
    all_symbols = DataProvider().get_symbols_history_data(
        symbol_type='FUTURES', time_type='1d')

    if filter_type == 'MTM':
        example = Datatransformer().get_mtm_filter_symbol(all_symbols,max_symbols=30)
    elif filter_type == 'VOLUME':
        example = Datatransformer().get_volume_top_filter_symobl(all_symbols, max_symbols=30)
    elif filter_type == 'NEW':
        example = Datatransformer().get_newthink_symbol(all_symbols)
    else:
        raise ValueError("please, this filter_type undefine")
    
    print(example)
    print('*'*120)


def example_reload_all_data(symbol_type:str, time_type: str):
    """
    Args:
        time_type (str): "1m","1d"
        symbol_type(str) :"SPOT","FUTURES"
    """
    DataProvider().reload_all_data(time_type=time_type,symbol_type=symbol_type)


def example_Train_neural_networks():
    Train_neural_networks.train(Train_data_path='EIIE/simulation/train_data.csv',
                                Meta_path="EIIE\Meta\policy_EIIE.pt",
                                Train_path="EIIE\Train\policy_EIIE.pt",
                                episodes=100000,
                                save=True,
                                pre_train=False,
                                )  # True


def example_simple_evaluate():
    evaluate_train_test_performance(Train_data_path=r'EIIE\simulation\train_data.csv',
                                    Test_data_path=r'EIIE\simulation\test_data.csv',
                                    Meta_path=r'EIIE\Meta\policy_EIIE.pt')


# from Database.BackUp import BasePreparator


# BasePreparator().import_all_tables()

# getAllDailyData()
example_reload_all_data(symbol_type="FUTURES",time_type = '1m')
# example_get_symboldata()
# checksymbol(symbol='TUSDUSDT')
example_get_target_symbol(filter_type='MTM')