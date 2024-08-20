# 用來轉換成 類神經網絡可以使用的資料特徵
import pandas as pd
import collections
import numpy as np
from typing import Optional

PricesObject = collections.namedtuple(
    'Prices', field_names=['open', 'high', 'low', 'close', 'volume'])


class DataFeature():
    """
        用來產生資料特徵,不過實際產生game狀態的在 environ裡面
    """

    def __init__(self, formal: bool = False) -> None:
        self.formal = formal

    def load_relative(self):
        array_data = self.df.values
        volume_change = self.calculate_volume_change(array_data[:, 4])
        return self.prices_to_relative(PricesObject(open=array_data[:, 0],
                                                    high=array_data[:, 1],
                                                    low=array_data[:, 2],
                                                    close=array_data[:, 3],
                                                    volume=volume_change,
                                                    ))

    def calculate_volume_change(self, volumes):
        """
        Calculate relative volume change
        """
        shift_data = np.roll(volumes, 1)
        shift_data[0] = 0
        diff_data = volumes - shift_data
        # 如果除數為0會返回0
        volume_change = np.divide(
            diff_data, volumes, out=np.zeros_like(diff_data), where=volumes != 0)
        return volume_change

    def prices_to_relative(self, prices):
        """
        # 原始作者不知道為甚麼,使用原始的volume, 我打算使用前一根量的變化來餵給神經網絡
        Convert prices to relative in respect to open price
        :param ochl: tuple with open, close, high, low
        :return: tuple with open, rel_close, rel_high, rel_low
        """
        assert isinstance(prices, PricesObject)
        rh = (prices.high - prices.open) / prices.open
        rl = (prices.low - prices.open) / prices.open
        rc = (prices.close - prices.open) / prices.open
        return PricesObject(open=prices.open, high=rh, low=rl, close=rc, volume=prices.volume)

    def get_train_net_work_data_by_path(self, symbols:list) -> dict:
        """
            用來取得類神經網絡所需要的資料
        """
        out_dict = {}
        for symbol in symbols:
            df = pd.read_csv(f'DQN\simulation\data\{symbol}-F-30-Min.csv')
            df.set_index('Datetime', inplace=True)
            self.df = df
            out_dict.update({symbol: self.load_relative()})        
        return out_dict
    
    def get_train_net_work_data_by_pd(self, symbol:str, df: pd.DataFrame) -> dict:
        """
            用來取得類神經網絡所需要的資料,正式交易的時候
        """
        out_dict = {}
        self.df = df
        out_dict.update({symbol: self.load_relative()})        
        return out_dict

    def get_test_net_work_data(self, symbol: str, symbol_data: Optional[pd.DataFrame] = None):
        """

            單一回測的資料
        Args:
            symbol (str): _description_

        Returns:
            _type_: _description_
        """

        out_dict = {}
        assert isinstance(
            symbol_data, pd.DataFrame), "formal model is on,symbol data can't be None"
        self.df = symbol_data
        out_dict.update({symbol: self.load_relative()})
        return out_dict
