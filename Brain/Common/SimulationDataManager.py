from pathlib import Path
from Major.DataProvider import DataProvider
import random
import pandas as pd
import numpy as np
import os


class SimulationDataManager:
    def __init__(self, base_dir="Brain/simulation"):
        self.train_path = Path(base_dir) / "train_data"
        self.test_path = Path(base_dir) / "test_data"
        self.train_path.mkdir(parents=True, exist_ok=True)
        self.test_path.mkdir(parents=True, exist_ok=True)
        self.dataProvider = DataProvider()

    def random_choose_path(self):
        return random.choice([self.train_path, self.test_path])

    def clip(self, x):
        if x > 1:
            return 1

        if x < 0:
            return 0

        return x

    def Downloader(self, symbol_type: str = "FUTURES", time_type="1m"):
        """
            We need to reload_all_data in first, because reload_all_data can save data into database first.

        Args:
            symbol_type (str, optional): _description_. Defaults to "FUTURES".
        """

        self.dataProvider.reload_all_data(time_type=time_type, symbol_type=symbol_type)

        for symbol_name in self.dataProvider.Binanceapp.get_targetsymobls(
            symbol_type=symbol_type
        ):

            self.dataProvider.Downloader(
                symbol_name=symbol_name,
                save=True,
                freq=30,
                symbol_type="FUTURES",
                save_path=self.random_choose_path(),
            )

    def get_smybols_mockuse(self, flag="train"):
        if flag == "train":
            file_path = self.train_path

        for symbol in os.listdir(file_path):
            self.genrateDate(symbol.split("-")[0], repetTimes=5)

    def genrateDate(self, symbol: str, repetTimes: int):
        # 讀入原始資料
        df = pd.read_csv(
            rf"Brain/simulation/train_data/{symbol}-F-30-Min.csv"
        )

        for i in range(repetTimes):
            # 計算前一個時間步的收盤價
            df["prev_close"] = df["Close"].shift(1)
            # 加上噪音後的 log return
            df["noisy_log_return"] = np.log(
                df["Close"] / df["prev_close"]
            ) + np.random.normal(loc=0.0, scale=0.002, size=len(df))
            # 利用 noisy log return 反推新的合成收盤價
            df["new_close"] = df["prev_close"] * np.exp(df["noisy_log_return"])
            df.dropna(inplace=True)

            # ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'quote_av','trades', 'tb_base_av', 'tb_quote_av']
            # rebuild Open
            df["new_open"] = df["new_close"] + df["Open"] - df["Close"]
            df["new_high"] = df["new_close"] + df["High"] - df["Close"]
            df["new_low"] = df["new_close"] + df["Low"] - df["Close"]

            noise = np.random.normal(loc=0.0, scale=0.1, size=len(df))
            safe_volume = np.where(df["Volume"] > 0, df["Volume"], 1.0)
            df["new_vloume"] = np.where(
                df["Volume"] > 0,
                np.exp(np.log(safe_volume) + noise),
                0.0,
            )
            df["new_quote_av"] = df["new_vloume"] * df["new_close"]
            df["new_trades"] = df.apply(
                lambda x: x["Volume"] / x["trades"] if x["trades"] > 0 else 0,
                axis=1,
            )
            df["new_trades"] = df.apply(
                lambda x: (
                    x["new_vloume"] / x["new_trades"] if x["new_trades"] > 0 else 0
                ),
                axis=1,
            )
            df["new_trades"] = df["new_trades"].apply(lambda x: float(int(x)))
            df["new_tb_base_av"] = df.apply(
                lambda x: x["tb_base_av"] / x["Volume"] if x["Volume"] > 0 else 0,
                axis=1,
            )
            df["new_tb_base_av"] = df["new_tb_base_av"] + np.random.normal(
                loc=0.0, scale=0.1, size=len(df)
            )
            df["new_tb_base_av"] = df["new_tb_base_av"].apply(self.clip)
            df["new_tb_base_av"] = df["new_vloume"] * df["new_tb_base_av"]
            df["new_tb_quote_av"] = df["new_tb_base_av"] * df["Close"]

            new_df = df[
                [
                    "Datetime",
                    "new_open",
                    "new_high",
                    "new_low",
                    "new_close",
                    "new_vloume",
                    "new_quote_av",
                    "new_trades",
                    "new_tb_base_av",
                    "new_tb_quote_av",
                ]
            ].copy()

            new_df.rename(
                columns={
                    "new_open": "Open",
                    "new_high": "High",
                    "new_low": "Low",
                    "new_close": "Close",
                    "new_vloume": "Volume",
                    "new_quote_av": "quote_av",
                    "new_trades": "trades",
                    "new_tb_base_av": "tb_base_av",
                    "new_tb_quote_av": "tb_quote_av",
                },
                inplace=True,
            )

            new_df.set_index("Datetime", inplace=True)
            new_df.to_csv(
                rf"Brain/simulation/train_data/{symbol}-F-30-Min-Fake{i}.csv"
            )


