import re
from Brain.DQN.lib.Backtest import Strategy, RL_evaluate, Backtest
import os
import time
from utils.AppSetting import RLConfig


class StrategyBuilder:
    """
    StrategyBuilder 負責從模型路徑與交易對(symbol)中解析出所需的參數，
    並建立對應的 Strategy 實例。
    """

    def __init__(self):
        # 如果未提供設定，則從 AppSetting 載入預設設定
        self.config = RLConfig()
        
    def create_strategy(self, model_path: str, symbol_file_name: str) -> Strategy:
        # 以下程式碼為解析 model_path 的邏輯
        # 假設路徑格式: 'Brain\DQN\Meta\Meta-300B-30K.pt'
        # 以 '-' 進行分割，取得資訊
        info, feature, data = model_path.split('-')
        feature_len = re.findall(r'\d+', feature)[0]
        data_len = re.findall(r'\d+', data)[0]

        # 將 info 分解 (例如: 'Brain\DQN\Meta\Meta')
        parts = info.split(os.sep)

        # parts: ['Brain', 'DQN', 'Meta', 'Meta']
        strategytype = parts[1]

        # 建立 Strategy 實例
        strategy = Strategy(
            strategytype=strategytype,
            symbol_name=symbol_file_name.split('-')[0],
            freq_time=int(data_len),
            model_feature_len=int(feature_len),
            fee=self.config.MODEL_DEFAULT_COMMISSION_PERC_test,
            slippage=self.config.DEFAULT_SLIPPAGE,
            model_count_path=model_path
        )
        strategy.symbol_file_name = symbol_file_name
        # 載入資料
        data_path = os.path.join("Brain","simulation","test_data",symbol_file_name)
        strategy.load_data(local_data_path=data_path)


        return strategy


class BacktestRunner:
    """
    BacktestRunner 負責對傳入的 Strategy 進行評估與回測，
    並可擴充以加入更多評估或分析功能。
    """

    def __init__(self, strategy: Strategy):
        self.strategy = strategy

    def run(self, ifplot: bool = True):
        # 使用 RL_evaluate 對策略進行評估
        re_evaluate = RL_evaluate(self.strategy)
        
        # 使用 Backtest 對策略執行回測
        backtest_info = Backtest(
            re_evaluate, self.strategy).order_becktest(ifplot=ifplot)

        return backtest_info


# 範例使用方式
if __name__ == "__main__":
    builder = StrategyBuilder()
    test_symbols = os.listdir(os.path.join(os.getcwd() , "Brain","simulation","test_data"))


    for test_symbol in test_symbols:
        if test_symbol != 'BTCUSDT-F-30-Min.csv':continue
        
        # 建立策略
        strategy = builder.create_strategy(
            os.path.join("Brain","DQN","Meta","Meta-300B-30K.pt"), symbol_file_name=test_symbol)

        # 執行回測
        runner = BacktestRunner(strategy)
        info = runner.run(ifplot=True)

        # 分批列印結果
        for i in range(0, len(info['marketpostion_array']), 100):
            print(info['marketpostion_array'][i:i+100])
            print('*' * 120)
