

from DQN.lib.Backtest import Strategy, RL_evaluate, Backtest
from utils.AppSetting import AppSetting
import re
import time
import os



def creaet_strategy(model_path:str,symbol:str):
    setting = AppSetting.RL_test_setting()
    # DQN\Meta\ORDIUSDT-300B-30K.pt
    # DQN\Meta\Meta.pt
    info,feature,data = model_path.split('-')
    feature_len = re.findall('\d+',feature)[0]
    data_len = re.findall('\d+',data)[0]
    strategytype,_,_ = info.split('\\')    
    strategy = Strategy(strategytype=strategytype,
                    symbol_name=symbol,
                    freq_time=int(data_len),
                    model_feature_len = int(feature_len),
                    fee=setting['BACKTEST_DEFAULT_COMMISSION_PERC'],
                    slippage=setting['DEFAULT_SLIPPAGE'],
                    model_count_path=model_path)

    strategy.load_data(local_data_path=f'DQN\simulation\data\{symbol}-F-{data_len}-Min.csv')
    return strategy

# KSMUSDT is very special
strategy = creaet_strategy('DQN\Meta\Meta-300B-30K.pt', symbol='GMXUSDT') # KSMUSDT TRBUSDT MKRUSDT


# def creaet_strategy(model_path:str):
#     info,feature,data = model_path.split('-')
#     feature_len = re.findall('\d+',feature)[0]
#     data_len = re.findall('\d+',data)[0]    
#     strategytype,_,symbol = info.split(os.sep)
    
#     strategy =  Strategy(strategytype=strategytype,
#                     symbol_name=symbol,
#                     freq_time=int(data_len),
#                     model_feature_len = int(feature_len),
#                     fee=setting['BACKTEST_DEFAULT_COMMISSION_PERC'],
#                     slippage=setting['DEFAULT_SLIPPAGE'],
#                     model_count_path=model_path
#                     )
    
#     strategy.load_data(local_data_path=f'DQN\simulation\data\{symbol}-F-{data_len}-Min.csv')
#     return strategy


# strategy = creaet_strategy('DQN\Meta\Meta-300B-30K.pt')


re_evaluate = RL_evaluate(strategy) 


info = Backtest(re_evaluate, strategy).order_becktest(
    re_evaluate.record_orders, ifplot=True)


for i in range(0, len(info['marketpostion_array']), 100):
    print(info['marketpostion_array'][i:i+100])
    print('*' * 120)