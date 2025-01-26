import re
from Brain.DQN.lib.Backtest import Strategy, RL_evaluate, Backtest
from utils.AppSetting import AppSetting


class StrategyBuilder:
    """
    StrategyBuilder 負責從模型路徑與交易對(symbol)中解析出所需的參數，
    並建立對應的 Strategy 實例。
    """

    def __init__(self, settings=None):
        # 如果未提供設定，則從 AppSetting 載入預設設定
        if settings is None:
            settings = AppSetting.RL_test_setting()
        self.settings = settings

    def create_strategy(self, model_path: str, symbol: str) -> Strategy:
        # 以下程式碼為解析 model_path 的邏輯
        # 假設路徑格式: 'Brain\DQN\Meta\Meta-300B-30K.pt'
        # 以 '-' 進行分割，取得資訊
        info, feature, data = model_path.split('-')
        feature_len = re.findall(r'\d+', feature)[0]
        data_len = re.findall(r'\d+', data)[0]

        # 將 info 分解 (例如: 'Brain\DQN\Meta\Meta')
        parts = info.split('\\')
        # parts: ['Brain', 'DQN', 'Meta', 'Meta']
        # 根據原有程式碼邏輯取出 strategytype:
        # _, strategytype, _, _ = parts -> strategytype = 'DQN'
        strategytype = parts[1]

        # 建立 Strategy 實例
        strategy = Strategy(
            strategytype=strategytype,
            symbol_name=symbol,
            freq_time=int(data_len),
            model_feature_len=int(feature_len),
            fee=self.settings['BACKTEST_DEFAULT_COMMISSION_PERC'],
            slippage=self.settings['DEFAULT_SLIPPAGE'],
            model_count_path=model_path
        )

        # 載入資料
        data_path = f'Brain\\simulation\\data\\{symbol}-F-{data_len}-Min.csv'
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
    # test_symbols = ['BTCUSDT','ETHUSDT','SUIUSDT','SOLUSDT']
    test_symbols = ['ARUSDT', 'BCHUSDT', 'COMPUSDT', 'DASHUSDT', 'DEFIUSDT', 'EGLDUSDT', 'ENSUSDT', 'ETCUSDT', 'GMXUSDT', 'ILVUSDT', 'INJUSDT', 'KSMUSDT', 'MKRUSDT', 'MOVEUSDT', 'QNTUSDT', 'SSVUSDT', 'TRBUSDT', 'XMRUSDT', 'YFIUSDT', 'ZECUSDT']

    for test_symbol in test_symbols:
        # 建立策略
        strategy = builder.create_strategy(
            'Brain\\DQN\\Meta\\Meta-300B-30K.pt', symbol=test_symbol)

        # 執行回測
        runner = BacktestRunner(strategy)
        info = runner.run(ifplot=True)

        # 分批列印結果
        for i in range(0, len(info['marketpostion_array']), 100):
            print(info['marketpostion_array'][i:i+100])
            print('*' * 120)
