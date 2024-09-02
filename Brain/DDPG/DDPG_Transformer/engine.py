from DQN.lib.Backtest import Strategy, RL_evaluate, Backtest
from utils.AppSetting import AppSetting
import pandas as pd
import re
import os
import time

class EngineBase():
    def __init__(self) -> None:        
        self.setting = AppSetting.get_DQN_setting()        

    def get_if_order_map(self, df: pd.DataFrame) -> dict:
        if_order_map = {}
        for each_strategy in self.strategys:
            # 載入所需要的資料
            each_strategy.load_Real_time_data(
                df[df['tic'] == each_strategy.symbol_name])

            re_evaluate = RL_evaluate(each_strategy)
            info = Backtest(re_evaluate, each_strategy).order_becktest(
                re_evaluate.record_orders, ifplot=False)

            if_order_map[each_strategy.symbol_name] = info['marketpostion_array'][-1]
        return if_order_map
    
    def creaet_strategy(self,model_path:str,symbol:str):
        info,feature,data = model_path.split('-')
        feature_len = re.findall('\d+',feature)[0]
        data_len = re.findall('\d+',data)[0]    
        strategytype,_,_ = info.split(os.sep)

        return Strategy(strategytype=strategytype,
                        symbol_name=symbol,
                        freq_time=int(data_len),
                        model_feature_len = int(feature_len),
                        fee=self.setting['BACKTEST_DEFAULT_COMMISSION_PERC'],
                        slippage=self.setting['DEFAULT_SLIPPAGE'],
                        model_count_path=model_path,
                        formal=True)
        
    def strategy_prepare(self, targetsymbols):
        """
            change multiple model path  to  singl model path
        """
        Meta_model_path = os.path.join('DQN','Meta','Meta-300B-30K.pt')        

        self.strategys = []
        for symbol in targetsymbols:                     
            self.strategys.append(self.creaet_strategy(Meta_model_path, symbol=symbol))