import pandas as pd
import collections
import time
import numpy as np


# class DataFeature():
#     """
#         I got the faile to change all data to the tensor version,
#         it made all process to be slow.

#     """

#     def __init__(self, formal: bool = False) -> None:
#         self.formal = formal
#         self.PricesObject = collections.namedtuple('Prices', field_names=[
#             "open", "high", "low", "close", "volume", "volume2",
#             "quote_av", "quote_av2", "trades", "trades2",
#             "tb_base_av", "tb_base_av2", "tb_quote_av", "tb_quote_av2",
#             "open_c_change", "open_p_change",
#             "high_c_change", "high_p_change",
#             "low_c_change", "low_p_change",
#             "close_c_change", "close_p_change"
#         ])
#         # self.PricesObject = collections.namedtuple('Prices', field_names=[
#         #     "open", "high", "low", "close", "volume", "volume2",
#         #     "quote_av", "quote_av2", "trades", "trades2",
#         #     "tb_base_av", "tb_base_av2", "tb_quote_av", "tb_quote_av2"
#         # ])

#     def get_train_net_work_data_by_pd(self, symbol: str, df: pd.DataFrame) -> dict:
#         """
#             用來取得類神經網絡所需要的資料,正式交易的時候
#         """
#         out_dict = {}
#         self.df = df
#         out_dict.update({symbol: self.load_relative()})
#         return out_dict

#     def get_train_net_work_data_by_path(self, symbols: list) -> dict:
#         """
#             用來取得類神經網絡所需要的資料
#         """
#         out_dict = {}
#         for symbol in symbols:
#             df = pd.read_csv(
#                 f'Brain/simulation/data/{symbol}-F-30-Min.csv')
#             df.set_index('Datetime', inplace=True)
#             self.df = df
#             # 使用 PyTorch Tensor 的方法
#             out_dict.update({symbol: self.load_relative()})
#         return out_dict

#     def calculate_previous_change(self, values):
#         """
#             Calculate relative data change based on the previous value.
#             This method calculates the difference between the current value and
#             the previous value, then divides by the previous value.

#             Returns:
#                 A numpy array representing the relative change in percentage.
#                 This approach emphasizes how much the current value has changed
#                 compared to the previous step, relative to the previous value.
#         """
#         shift_data = np.roll(values, shift=1, axis=0)
#         shift_data[0] = 0
#         diff_data = values - shift_data
#         return np.divide(diff_data, shift_data, out=np.zeros_like(diff_data), where=shift_data != 0)

#     def calculate_current_change(self, values):
#         """
#             Calculate relative data change based on the current value.
#             This method calculates the difference between the current value and
#             the previous value, then divides by the current value.

#             Returns:
#                 A numpy array representing the relative change in percentage.
#                 This approach emphasizes how much the current value has changed
#                 compared to the previous step, relative to the current value.
#         """
#         shift_data = np.roll(values, shift=1, axis=0)
#         shift_data[0] = 0
#         diff_data = values - shift_data
#         return np.divide(diff_data, values, out=np.zeros_like(diff_data), where=values != 0)

#     def load_relative(self):
#         """
#             CSV最後排序為:

#             Open, High, Low, Close, Volume, quote_av, trades, tb_base_av, tb_quote_av
#         """
#         np_data = np.array(self.df.values, dtype=np.float32)

#         open = np_data[:, 0]
#         high = np_data[:, 1]
#         low = np_data[:, 2]
#         close = np_data[:, 3]

#         volume = self.calculate_current_change(np_data[:, 4])
#         volume2 = self.calculate_previous_change(np_data[:, 4])

#         quote_av = self.calculate_current_change(np_data[:, 5])
#         quote_av2 = self.calculate_previous_change(np_data[:, 5])

#         trades = self.calculate_current_change(np_data[:, 6])
#         trades2 = self.calculate_previous_change(np_data[:, 6])

#         tb_base_av = self.calculate_current_change(np_data[:, 7])
#         tb_base_av2 = self.calculate_previous_change(np_data[:, 7])

#         tb_quote_av = self.calculate_current_change(np_data[:, 8])
#         tb_quote_av2 = self.calculate_previous_change(np_data[:, 8])

#         rh = (high - open) / open
#         rl = (low - open) / open
#         rc = (close - open) / open

#         open_c_change = self.calculate_current_change(open)
#         open_p_change = self.calculate_previous_change(open)

#         high_c_change = self.calculate_current_change(high)
#         high_p_change = self.calculate_previous_change(high)

#         low_c_change = self.calculate_current_change(low)
#         low_p_change = self.calculate_previous_change(low)

#         close_c_change = self.calculate_current_change(close)
#         close_p_change = self.calculate_previous_change(close)

#         return self.PricesObject(
#             open=open,
#             high=rh,
#             low=rl,
#             close=rc,
#             volume=volume,
#             volume2=volume2,
#             quote_av=quote_av,
#             quote_av2=quote_av2,
#             trades=trades,
#             trades2=trades2,
#             tb_base_av=tb_base_av,
#             tb_base_av2=tb_base_av2,
#             tb_quote_av=tb_quote_av,
#             tb_quote_av2=tb_quote_av2,
#             open_c_change=open_c_change,
#             open_p_change=open_p_change,
#             high_c_change=high_c_change,
#             high_p_change=high_p_change,
#             low_c_change=low_c_change,
#             low_p_change=low_p_change,
#             close_c_change=close_c_change,
#             close_p_change=close_p_change
#         )


class OriginalDataFrature():
    def __init__(self, formal: bool = False) -> None:
        self.formal = formal
        self.PricesObject = collections.namedtuple('Prices', field_names=[
            'open',
            'high',
            'low',
            'close',
            'log_open',
            'log_high',
            'log_low',
            'log_close',
            'log_volume',
            'log_quote_av',
            'log_trades',
            'log_tb_base_av',
            'log_tb_quote_av',
        ])

    def cleanData(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan)  # 将 Inf 替换为 NaN
        df = df.ffill(axis=0)
        df = df.bfill(axis=0)
        return df

    def get_train_net_work_data_by_pd(self, symbol: str, df: pd.DataFrame) -> dict:
        """
            用來取得類神經網絡所需要的資料,正式交易的時候
        """
        out_dict = {}
        self.df = df
        out_dict.update({symbol: self.load_relative()})
        return out_dict

    def get_train_net_work_data_by_path(self, symbolNames: list) -> dict:
        """
            用來取得類神經網絡所需要的資料
        """
        out_dict = {}
        for symbolName in symbolNames:
            df = pd.read_csv(
                f'Brain/simulation/data/{symbolName}.csv')
            df.set_index('Datetime', inplace=True)
            self.df = self.cleanData(df)
            # 使用 PyTorch Tensor 的方法
            out_dict.update({symbolName: self.load_relative()})

        return out_dict

    def load_relative(self, if_log=True):
        """
            CSV最後排序為:

            Open, High, Low, Close, Volume, quote_av, trades, tb_base_av, tb_quote_av
        """
        np_data = np.array(self.df.values, dtype=np.float32)

        if if_log:
            # 經過我的評估認為log 已經可以極大化避免極端值
            # clamp_min, clamp_max = 0.01, 100000.0
            # np_data = np.clip(np_data, clamp_min, clamp_max)

            # 2) 再進行 log(x+1) 變換 (可改用 np.log1p)

            log_np_data = np.log(np_data + 1.0)

            return self.PricesObject(
                open=np_data[:, 0],
                high=np_data[:, 1],
                low=np_data[:, 2],
                close=np_data[:, 3],
                log_open=log_np_data[:, 0],
                log_high=log_np_data[:, 1],
                log_low=log_np_data[:, 2],
                log_close=log_np_data[:, 3],
                log_volume=log_np_data[:, 4],
                log_quote_av=log_np_data[:, 5],
                log_trades=log_np_data[:, 6],
                log_tb_base_av=log_np_data[:, 7],
                log_tb_quote_av=log_np_data[:, 8],
            )
