import re
from Brain.DQN.lib.Backtest import Strategy, RL_evaluate, Backtest
import os
import time
from Brain.Common.engine import EngineBase

from Major.DataProvider import DataProvider


# 範例使用方式
if __name__ == "__main__":

    # first_date_map = DataProvider().get_symbol_first_day(
    #     symbol_type="FUTURES", time_type="1m"
    # )
    
    first_date_map ={
        "BTCUSDT": None
    }
    engine = EngineBase("ONE_TO_MANY", first_date_map, formal=False)
    test_symbols = os.listdir(
        os.path.join(os.getcwd(), "Brain", "simulation", "test_data")
    )
    test_symbols = ['BTCUSDT-F-30-Min.csv']
    engine.strategy_prepare(test_symbols)
    engine.analyze_result(ifplot=True)
