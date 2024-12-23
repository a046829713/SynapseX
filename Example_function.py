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
    # symbols = ['BTCUSDT','BCHUSDT','BTCDOMUSDT','BNBUSDT','BTCUSDT','ETHUSDT','SOLUSDT','SSVUSDT']
    # symbols = ['KSMUSDT','ENSUSDT','LPTUSDT','GMXUSDT','TRBUSDT','ARUSDT','XMRUSDT','ETHUSDT', 'AAVEUSDT',  'ZECUSDT', 'SOLUSDT', 'DEFIUSDT', 'BTCUSDT',  'ETCUSDT',  'BNBUSDT', 'LTCUSDT', 'BCHUSDT']

    symbols = list(
        set(['COMPUSDT', 'ZENUSDT', 'BCHUSDT', 'AVAXUSDT', 'KSMUSDT', 'AAVEUSDT', 'ETCUSDT', 'ZECUSDT', 'LINKUSDT', 'DEFIUSDT']))

    for _each_symbol_name in symbols:
        DataProvider().Downloader(symbol_name=_each_symbol_name, save=True, freq=30)


def ex():
    all_symbols = DataProvider().get_symbols_history_data(
        symbol_type='FUTURES', time_type='1d')
    


    # 1. 獲取 24 小時價格變動統計數據
    tickers = DataProvider().Binanceapp.futures_ticker()

    # 2. 過濾掉非 USDT 本位合約
    filter_tickers = []
    for ticker in tickers:
        print(f"本次樣本:{ticker['symbol']}")
        if ticker['symbol'].endswith('USDT'):
            # 目前怪怪的合約都拋棄不要（有數字的大概論都有特殊意義）
            clear_name = re.findall(
                '[A-Z]+USDT', ticker['symbol'])
            
            if clear_name:
                if ticker['symbol'] == clear_name[0]:
                    # 還要過濾出時間太短的10天暫時是神經網絡的極限
                    for each_data in all_symbols:
                        symbolname = each_data[0]                        
                        
                        if symbolname.upper().split('-F-D')[0] == ticker['symbol']:
                            print(symbolname.upper().split('-F-D')[0]) # btcusdt-f-d
                            length = len(each_data[1])
                            print(length)
                            if length >10:
                                filter_tickers.append(ticker)


    print(filter_tickers)
    

    # # 3. 根據交易量（quoteVolume）排序，並取前 10 名
    sorted_tickers = sorted(
        filter_tickers, key=lambda x: float(x['quoteVolume']), reverse=True
    )
    top_10_tickers = sorted_tickers[:10]

    # 4. 提取幣種名稱
    top_10_coins = [ticker['symbol'] for ticker in top_10_tickers]

    print(top_10_coins)
    return top_10_coins

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
        example = Datatransformer().get_mtm_filter_symbol(all_symbols)
    elif filter_type == 'VOLUME':
        example = Datatransformer().get_volume_top_filter_symobl(all_symbols, max_symbols=10)
    elif filter_type == 'NEW':
        example = Datatransformer().get_newthink_symbol(all_symbols)
    else:
        raise ValueError("please, this filter_type undefine")
    print(example)
    print('*'*120)


def example_reload_all_data(time_type: str):
    """
    Args:
        time_type (str): '1m','1d'
    """
    DataProvider().reload_all_data(time_type=time_type,
                                   symbol_type='FUTURES')


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


example_get_symboldata()