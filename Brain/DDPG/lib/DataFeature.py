import pandas as pd
import collections
import torch
import time


class DataFeature():
    """
        I specialize in using tensors to accelerate computations
        because I believe that converting between GPU and CPU consumes a significant amount of computational resources.
    """

    def __init__(self, device: torch.device, formal: bool = False) -> None:
        self.formal = formal
        self.device = device
        self.PricesObject = collections.namedtuple('Prices', field_names=[
            "open", "high", "low", "close", "volume", "volume2",
            "quote_av", "quote_av2", "trades", "trades2",
            "tb_base_av", "tb_base_av2", "tb_quote_av", "tb_quote_av2"
        ])

    def load_relative(self):
        """
            CSV最後排序為:

            Open, High, Low, Close, Volume, quote_av, trades, tb_base_av, tb_quote_av
        """
        tensor_data = torch.tensor(
            self.df.values, dtype=torch.float32, device=self.device)

        open = tensor_data[:, 0]
        high = tensor_data[:, 1]
        low = tensor_data[:, 2]
        close = tensor_data[:, 3]

        volume = self.calculate_current_change(tensor_data[:, 4])
        volume2 = self.calculate_previous_change(tensor_data[:, 4])

        quote_av = self.calculate_current_change(tensor_data[:, 5])
        quote_av2 = self.calculate_previous_change(tensor_data[:, 5])

        trades = self.calculate_current_change(tensor_data[:, 6])
        trades2 = self.calculate_previous_change(tensor_data[:, 6])

        tb_base_av = self.calculate_current_change(tensor_data[:, 7])
        tb_base_av2 = self.calculate_previous_change(tensor_data[:, 7])

        tb_quote_av = self.calculate_current_change(tensor_data[:, 8])
        tb_quote_av2 = self.calculate_previous_change(tensor_data[:, 8])

        rh = (high - open) / open
        rl = (low - open) / open
        rc = (close - open) / open

        return self.PricesObject(
            open=open,
            high=rh,
            low=rl,
            close=rc,
            volume=volume,
            volume2=volume2,
            quote_av=quote_av,
            quote_av2=quote_av2,
            trades=trades,
            trades2=trades2,
            tb_base_av=tb_base_av,
            tb_base_av2=tb_base_av2,
            tb_quote_av=tb_quote_av,
            tb_quote_av2=tb_quote_av2
        )

    def calculate_current_change(self, values):
        """
            Calculate relative data change based on the current value.
            This method calculates the difference between the current value and 
            the previous value, then divides by the current value.

            Returns:
                A tensor representing the relative change in percentage.
                This approach emphasizes how much the current value has changed 
                compared to the previous step, relative to the current value.
        """
        shift_data = torch.roll(values, shifts=1, dims=0)
        shift_data[0] = 0
        diff_data = values - shift_data
        return torch.where(
            values != 0, diff_data / values, torch.zeros_like(diff_data))

    def calculate_previous_change(self, values):
        """
            Calculate relative data change based on the previous value.
            This method calculates the difference between the current value and 
            the previous value, then divides by the previous value.

            Returns:
                A tensor representing the relative change in percentage.
                This approach emphasizes how much the current value has changed 
                compared to the previous step, relative to the previous value.
        """
        shift_data = torch.roll(values, shifts=1, dims=0)
        shift_data[0] = 0
        diff_data = values - shift_data
        return torch.where(
            shift_data != 0, diff_data / shift_data, torch.zeros_like(diff_data))

    def get_train_net_work_data_by_path(self, symbols: list) -> dict:
        """
            用來取得類神經網絡所需要的資料
        """
        out_dict = {}
        for symbol in symbols:
            df = pd.read_csv(
                f'Brain/DDPG/simulation/data/{symbol}-F-30-Min.csv')
            df.set_index('Datetime', inplace=True)
            self.df = df
            # 使用 PyTorch Tensor 的方法
            out_dict.update({symbol: self.load_relative()})
        return out_dict

    def get_train_net_work_data_by_pd(self, symbol: str, df: pd.DataFrame) -> dict:
        """
            用來取得類神經網絡所需要的資料,正式交易的時候
        """
        out_dict = {}
        self.df = df
        # 使用 PyTorch Tensor 的方法
        out_dict.update({symbol: self.load_relative()})
        return out_dict
