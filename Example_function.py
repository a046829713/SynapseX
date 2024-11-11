from Major.DataProvider import DataProvider
from Major.Datatransformer import Datatransformer
# from EIIE.lib import Train_neural_networks
import matplotlib.pyplot as plt
from pathlib import Path
# from EIIE.lib.simple_evaluate import evaluate_train_test_performance


def example_get_symboldata():
    """
        introduction:
            this function is for download history data to experiment.

    """
    # symbols = ['BTCUSDT','BCHUSDT','BTCDOMUSDT','BNBUSDT','ARUSDT','BTCUSDT','ETHUSDT','SOLUSDT','SSVUSDT']
    # symbols = ['KSMUSDT','ENSUSDT','LPTUSDT','GMXUSDT','TRBUSDT','ARUSDT','XMRUSDT','ETHUSDT', 'AAVEUSDT',  'ZECUSDT', 'SOLUSDT', 'DEFIUSDT', 'BTCUSDT',  'ETCUSDT',  'BNBUSDT', 'LTCUSDT', 'BCHUSDT']

    symbols = ['BTCUSDT']
    for _each_symbol_name in symbols:
        DataProvider().Downloader(symbol_name=_each_symbol_name, save=True, freq=30)


def example_get_targetsymobls():
    """
        取得有效名稱
    """
    print(DataProvider().get_both_type_targetsymbols())


def example_get_target_symbol(filter_type:str):
    """
        introduction:
            this function is for filter 
    """
    all_symbols = DataProvider().get_symbols_history_data(
        symbol_type='FUTURES', time_type='1d')

    if filter_type =='MTM':
        example = Datatransformer().get_mtm_filter_symbol(all_symbols)
    elif filter_type == 'VOLUME':
        example = Datatransformer().get_volume_top_filter_symobl(all_symbols, max_symbols=11)
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


example_reload_all_data(time_type='1m')