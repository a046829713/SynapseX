import pandas as pd
from Brain.Common.DataFeature import OriginalDataFeature
import time


class Strategy(object):
    """
        神經網絡的模型基底
    """

    def __init__(
        self,
        strategytype: str,
        symbol_name: str,
        freq_time: int,
        model_feature_len: int,
        fee: float,
        slippage: float,
        model_count_path: str,
        init_cash: float = 10000.0,
        symobl_type: str = "Futures",
        lookback_date: str = None,
        symbol_first_trade_date=None,
        formal: bool = False,
    ) -> None:
        self.strategytype = strategytype
        self.symbol_name = symbol_name  # 商品名稱
        self.freq_time = freq_time  # 商品週期
        self.model_feature_len = model_feature_len  # 商品週期
        self.fee = fee  # 手續費
        self.slippage = slippage  # 滑價
        self.model_count_path = model_count_path  # 模型路徑
        self.init_cash = init_cash  # 起始資金
        self.symobl_type = symobl_type  # 每個策略會有一個商品別(期貨現貨別)
        self.lookback_date = lookback_date  # 策略回測日期
        self.symbol_first_trade_date = symbol_first_trade_date
        self.formal = formal  # 策略是否於正式交易環境
        self.strategyDataManger = StrategyDataManger(self)

    def load_data(self, local_data_path: str = None, df: pd.DataFrame = None):
        """
        通用資料載入進入點
        非正式模式 (`formal=False`)：從 CSV 本地檔案載入
        正式模式 (`formal=True`)：傳入即時 DataFrame
        """
        if not self.formal:
            if local_data_path is None:
                raise ValueError(
                    "In backtest mode (formal=False), local_data_path must be provided."
                )
            self.strategyDataManger.load_data_from_csv(local_data_path)
        else:
            if df is not None:
                self.strategyDataManger.load_realtime_data(df)
            else:
                raise ValueError(
                    "inRealtime doesn't get data,please check"
                )

    def _strategy_name(self):
        return f"{self.strategytype}-{self.symbol_name}-{self.freq_time}"

    @property
    def datafeature(self):
        return self.strategyDataManger.datafeature

    @property
    def df(self):
        return self.strategyDataManger.df


class StrategyDataManger(object):
    def __init__(self, strategy: Strategy) -> None:
        self.strategy = strategy
        self.df = None
        self.datafeature = None
        self.originalDataFeature = OriginalDataFeature()

    def dataRelod(self):
        pass

    def dataFeatureChange(self):
        """
        when model needs data change we also need to change df,
        want to keep both at the same length.

        """
        assert self.df is not None, "no data please check."

        self.datafeature = self.originalDataFeature.get_train_net_work_data_by_pd(
            symbol=self.strategy.symbol_name,
            df=self.df,
            first_date=self.strategy.symbol_first_trade_date,
        )

    def dataChange(self):
        """
        we need to keep both at the same length.
        i use self.originalDataFeature.df replace self.df.
        """
        self.df = self.originalDataFeature.df


    def load_data_from_csv(self, local_data_path: str):
        """
        如果非正式交易的的時候，可以啟用
        """
        self.df = pd.read_csv(local_data_path)
        self.df.set_index("Datetime", inplace=True)
        self.dataFeatureChange()
        self.dataChange()

    def load_realtime_data(self, df: pd.DataFrame):
        self.df = df[
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_av",
                "trades",
                "tb_base_av",
                "tb_quote_av",
            ]
        ].copy()
        self.df.rename(
            columns={
                "date": "Datetime",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )

        self.df.set_index("Datetime", inplace=True)
        self.dataFeatureChange()
        self.dataChange()