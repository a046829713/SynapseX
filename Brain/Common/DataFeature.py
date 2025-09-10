import pandas as pd
import collections
import time
import numpy as np
from Brain.Common.DataFeature_tool import VolatilityCalculator
from utils.Debug_tool import debug
from Brain.Common.Error import InvalidModeError



Prices = collections.namedtuple('Prices', field_names=[
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




class OriginalDataFrature():
    def __init__(self,) -> None:
        pass
    
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
    
    
    def get_train_net_work_data_by_path(self, symbolNames: list, typeName = 'train_data') -> dict:
        """
            用來取得類神經網絡所需要的資料
        """
        out_dict = {}
        for symbolName in symbolNames:
            df = pd.read_csv(
                f'Brain/simulation/{typeName}/{symbolName}.csv')
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

            return Prices(
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


class Resume():
    def __init__(self,symmbol_data:pd.DataFrame):
        # vol_df = VolatilityCalculator(annualization_factor= 365).calculate(symmbol_data,price_col="Close")
        
        # print(vol_df)
        # time.sleep(100) 
      
        # 波動性特徵 (Volatility): 商品有多活躍？

        # volatility_90d: 近90天的年化波動率。

        # 趨勢/均值回歸特徵 (Trend / Mean-Reversion): 商品是傾向於一路上漲/下跌，還是喜歡來回震盪？

        # hurst_exponent: 赫斯特指數。 > 0.5 表示趨勢性，< 0.5 表示均值回歸性，≈ 0.5 表示隨機遊走。

        # 市場相關性特徵 (Market Correlation): 商品與大盤的關係如何？

        # beta_90d: 相對於某個基準（例如比特幣指數或S&P 500）的Beta值，衡量其系統性風險。

        # 流動性特徵 (Liquidity): 商品的交易熱度如何？

        # avg_dollar_volume_90d: 近90天的平均每日成交額。

        # 類別特徵 (Categorical): 商品屬於哪個大類？

        # asset_class: 例如，加密貨幣=0, 股票=1, 外匯=2...
        pass



# 去極值 中性化 標準化