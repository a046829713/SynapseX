from Brain.DQN.lib.Backtest import Strategy, RL_evaluate, Backtest
from utils.AppSetting import AppSetting, RLConfig
import pandas as pd
import re
import os
import time
from typing import Tuple, Optional
from pathlib import Path


class EngineBase:
    def __init__(
        self,
        strategy_keyword: str,
        first_date_map: dict,
        formal=True,
    ) -> None:
        """
            負責用來協調Backtest

        Args:
            strategy_keyword (str): ONE_TO_MANY
            symbols (list, optional): _description_. Defaults to [].
            formal (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: _description_
        """
        self.strategy_keyword = strategy_keyword
        self.first_date_map = first_date_map

        self.setting = AppSetting.Trading_setting()
        self.formal = formal

        if not self.formal:
            self.config = RLConfig()

    def get_if_order_map(self, df: pd.DataFrame) -> dict:
        """
            用來生成正式交易的訂單

        Args:
            df (pd.DataFrame): _description_

        Returns:
            dict: _description_
        """
        if_order_map = {}
        for each_strategy in self.strategys:
            # 載入所需要的資料
            each_strategy.load_Real_time_data(
                df[df["tic"] == each_strategy.symbol_name]
            )

            re_evaluate = RL_evaluate(each_strategy, formal=self.formal)
            info = Backtest(re_evaluate, each_strategy).order_becktest(ifplot=False)

            if_order_map[each_strategy.symbol_name] = info["marketpostion_array"][-1]

        return if_order_map

    def create_strategy(self, model_path: str, symbol: str) -> Strategy:
        info, feature_len, data_len, strategytype = self._parse_model_path(model_path)

        return Strategy(
            strategytype=strategytype,
            symbol_name=symbol,
            freq_time=int(data_len),
            model_feature_len=int(feature_len),
            fee=self.setting["BACKTEST_DEFAULT_COMMISSION_PERC"],
            slippage=self.setting["DEFAULT_SLIPPAGE"],
            model_count_path=model_path,
            symbol_first_trade_date=self.first_date_map[symbol],
            formal=True,
        )

    def strategy_prepare(self, targetsymbols):
        if self.strategy_keyword == "ONE_TO_MANY":
            if self.formal:
                Meta_model_path = os.path.join(
                    "Brain", "DQN", "Meta", "Meta-300B-30K.pt"
                )

                self.strategys = []
                for symbol in targetsymbols:
                    self.strategys.append(
                        self.create_strategy(Meta_model_path, symbol=symbol)
                    )

            else:
                Meta_model_path = os.path.join(
                    "Brain", "DQN", "Meta", "Meta-300B-30K.pt"
                )

                self.strategys = []
                for symbol_file_name in targetsymbols:
                    self.strategys.append(
                        self.create_strategy_from_csv(
                            Meta_model_path, symbol_file_name=symbol_file_name
                        )
                    )
        else:
            raise ValueError("STRATEGY_KEYWORD didn't match,please check")

    def create_strategy_from_csv(
        self, model_path: str, symbol_file_name: str
    ) -> Strategy:
        """
            To use in backtest, not in formal environment

        Args:
            model_path (str): Brain/DQN/Meta/Meta-300B-30K.pt
            symbol_file_name (str): KAITOUSDT-F-30-Min.csv

        Returns:
            Strategy: _description_
        """
        info, feature_len, data_len, strategytype = self._parse_model_path(model_path)
        symbol = symbol_file_name.split("-")[0]

        # 建立 Strategy 實例
        strategy = Strategy(
            strategytype=strategytype,
            symbol_name=symbol,
            freq_time=int(data_len),
            model_feature_len=int(feature_len),
            fee=self.config.MODEL_DEFAULT_COMMISSION_PERC_test,
            slippage=self.config.DEFAULT_SLIPPAGE,
            model_count_path=model_path,
            symbol_first_trade_date=self.first_date_map[symbol],
            formal=False,
        )

        strategy.symbol_file_name = symbol_file_name

        # 載入資料
        data_path = os.path.join("Brain", "simulation", "test_data", symbol_file_name)
        strategy.load_data(local_data_path=data_path)
        return strategy

    def analyze_result(self, ifplot: bool = True):
        for each_strategy in self.strategys:
            # 使用 RL_evaluate 對策略進行評估
            re_evaluate = RL_evaluate(each_strategy, formal=False)

            # 使用 Backtest 對策略執行回測
            backtest_info = Backtest(re_evaluate, each_strategy).order_becktest(
                ifplot=ifplot
            )

            for i in range(0, len(backtest_info["marketpostion_array"]), 100):
                print(backtest_info["marketpostion_array"][i : i + 100])
                print("*" * 120)

    def _parse_model_path(self, model_path: str) -> Tuple[str, int, int, str]:
        """
        從模型路徑解析出所需資訊
        model_path (str): Brain/DQN/Meta/Meta-300B-30K.pt

        """
        path = Path(model_path)
        # Ex: Meta-300B-30K.pt -> ('Meta', '300B', '30K.pt')
        info, feature, data_part = path.stem.split("-")

        feature_len = int(re.findall(r"\d+", feature)[0])
        data_len = int(re.findall(r"\d+", data_part)[0])

        # Ex: Brain/DQN/Meta -> ['Brain', 'DQN', 'Meta'] -> 'DQN'
        strategy_type = path.parent.parts[1]

        return info, feature_len, data_len, strategy_type
