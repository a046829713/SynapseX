import pandas as pd
import collections
import time
import numpy as np
from Brain.Common.DataFeature_tool import VolatilityCalculator
from utils.Debug_tool import debug
from Brain.Common.Error import InvalidModeError
from datetime import timedelta



Prices = collections.namedtuple(
    "Prices",
    field_names=[
        "open",
        "high",
        "low",
        "close",
        "log_open",
        "log_high",
        "log_low",
        "log_close",
        "log_volume",
        "log_quote_av",
        "log_trades",
        "log_tb_base_av",
        "log_tb_quote_av",
        # 新增均線特徵
        "log_ma_30",
        "log_ma_60",
        "log_ma_120",
        "log_ma_240",
        "log_ma_360",


        "age_log_minutes",
        "age_years",
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "week_sin",
        "week_cos",
        "atr_Volatility"
    ],
)


class OriginalDataFrature:
    def __init__(
        self,
    ) -> None:
        pass

    def cleanData(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan)  # 将 Inf 替换为 NaN
        df = df.ffill(axis=0)
        df = df.bfill(axis=0)
        return df

    def get_train_net_work_data_by_pd(
        self, symbol: str, df: pd.DataFrame, first_date
    ) -> dict:
        """
        用來取得類神經網絡所需要的資料,正式交易的時候
        """
        out_dict = {}
        self.df = self.add_time_feature(df, first_date)
        self.df = self.add_ATR(self.df)
        out_dict.update({symbol: self.load_relative()})


        
        return out_dict

    def get_train_net_work_data_by_path(
        self, symbolNames: list, typeName="train_data"
    ) -> dict:
        """
        用來取得類神經網絡所需要的資料
        """
        out_dict = {}
        for symbolName in symbolNames:
            df = pd.read_csv(f"Brain/simulation/{typeName}/{symbolName}.csv")
            df.set_index("Datetime", inplace=True)
            self.df = self.cleanData(df)
            self.df = self.add_average_metric(self.df)
            self.df = self.add_time_feature(self.df)
            self.df = self.add_ATR(self.df)

            

            # 使用 PyTorch Tensor 的方法
            out_dict.update({symbolName: self.load_relative()})


        return out_dict
    
    def add_average_metric(self, df: pd.DataFrame, periods=[30, 60, 120, 240, 360]):
        """
            計算均線並生成相對特徵：log(Close / MA)
            這能告訴模型目前價格相對於均線的偏離程度。
            # 均線邏輯：不需要 shift
        """
        for p in periods:
            ma_col_name = f'MA_{p}'
            feature_col_name = f'log_ma_{p}'
            
            # 計算簡單移動平均 (SMA)
            df[ma_col_name] = df['Close'].rolling(window=p).mean()
            df[feature_col_name] = np.log(df['Close'] / df[ma_col_name])            
            df.drop(columns=[ma_col_name], inplace=True)

        # 因為 rolling 會產生 NaN (例如 MA_360 前 359 筆是空的)，這裡需要清除
        df = df.dropna()
        return df

    def add_ATR(self, df: pd.DataFrame, period: int = 14):
        """
            becasue i want use in feature, so i need to let agent know the state,not be a strategy trigger.

        Args:
            df (pd.DataFrame): _description_
            period (int, optional): _description_. Defaults to 5.

        Returns:
            _type_: _description_
        """
        prev_close = df['Close'].shift(1)

        # 計算 True Range (TR) 的三個組成部分
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - prev_close).abs()
        tr3 = (df['Low'] - prev_close).abs()

        # 取得真實範圍 (True Range, TR) - 取三者最大值
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        
        # 計算 ATR (使用 Wilder's Smoothing)
        atr_col_name = f'ATR_{period}'
        
        # ewm(alpha=1/period, adjust=False) 是 pandas 中
        # 實現 Wilder's Smoothing (RMA) 的標準方法。
        df[atr_col_name] = tr.ewm(alpha=1/period, adjust=False).mean()
        df[atr_col_name] = df[atr_col_name].shift(-1)
        df[atr_col_name] = np.log1p(df[atr_col_name])
        df = df.dropna()

        return df

    def add_time_feature(self, df: pd.DataFrame, first_date=None):
        datetime_index = pd.to_datetime(df.index)

        if first_date is None:
            first_date = datetime_index[0]
        else:
            first_date = first_date - timedelta(minutes=1)

        # 1. 資產年齡 (分鐘計，對數尺度)
        # 這本身就是一個從0開始的相對特徵，且log使其增長緩慢
        minutes_since_start = (datetime_index - first_date).total_seconds() / 86400
        df["age_log_minutes"] = np.log1p(minutes_since_start)

        # 2. 資產年齡 (年計)
        # 這也是一個從0開始的相對特徵，增長非常慢
        df["age_years"] = (datetime_index.year - first_date.year).astype(float)
        # (可選) 對年份做一個簡單的約束，防止其變得過大，例如除以一個常數
        df["age_years"] = df["age_years"] / 100.0

        # --- 週期性特徵 (天然標準化在 [-1, 1]) ---
        df["month_sin"] = np.sin(2 * np.pi * datetime_index.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * datetime_index.month / 12)

        df["day_sin"] = np.sin(2 * np.pi * datetime_index.day / 31)
        df["day_cos"] = np.cos(2 * np.pi * datetime_index.day / 31)

        df["hour_sin"] = np.sin(2 * np.pi * datetime_index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * datetime_index.hour / 24)

        df["minute_sin"] = np.sin(2 * np.pi * datetime_index.minute / 60)
        df["minute_cos"] = np.cos(2 * np.pi * datetime_index.minute / 60)

        df["dayofweek_sin"] = np.sin(2 * np.pi * datetime_index.dayofweek / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * datetime_index.dayofweek / 7)

        df["week_sin"] = np.sin(
            2 * np.pi * datetime_index.isocalendar().week.astype(float) / 52
        )
        df["week_cos"] = np.cos(
            2 * np.pi * datetime_index.isocalendar().week.astype(float) / 52
        )

        return df

    def load_relative(self, if_log=True):
        """
        最後排序為:
        [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "quote_av",
                "trades",
                "tb_base_av",
                "tb_quote_av",
                "age_log_minutes",
                "age_years",
                "month_sin",
                "month_cos",
                "day_sin",
                "day_cos",
                "hour_sin",
                "hour_cos",
                "minute_sin",
                "minute_cos",
                "dayofweek_sin",
                "dayofweek_cos",
                "week_sin",
                "week_cos",
                "atr_Volatility",
            ]

        """
        if if_log:
            
            return Prices(
                open=self.df["Open"].values,
                high=self.df["High"].values,
                low=self.df["Low"].values,
                close=self.df["Close"].values,
                log_open=np.log1p(self.df["Open"].values),
                log_high=np.log1p(self.df["High"].values),
                log_low=np.log1p(self.df["Low"].values),
                log_close=np.log1p(self.df["Close"].values),
                log_volume=np.log1p(self.df["Volume"].values),
                log_quote_av=np.log1p(self.df["quote_av"].values),
                log_trades=np.log1p(self.df["trades"].values),
                log_tb_base_av=np.log1p(self.df["tb_base_av"].values),
                log_tb_quote_av=np.log1p(self.df["tb_quote_av"].values),
                log_ma_30 = self.df["log_ma_30"].values,
                log_ma_60= self.df["log_ma_60"].values,
                log_ma_120=self.df["log_ma_120"].values ,
                log_ma_240=self.df["log_ma_240"].values ,
                log_ma_360=self.df["log_ma_360"].values ,
                age_log_minutes=self.df["age_log_minutes"].values,
                age_years=self.df["age_years"].values,
                month_sin=self.df["month_sin"].values,
                month_cos=self.df["month_cos"].values,
                day_sin=self.df["day_sin"].values,
                day_cos=self.df["day_cos"].values,
                hour_sin=self.df["hour_sin"].values,
                hour_cos=self.df["hour_cos"].values,
                minute_sin=self.df["minute_sin"].values,
                minute_cos=self.df["minute_cos"].values,
                dayofweek_sin=self.df["dayofweek_sin"].values,
                dayofweek_cos=self.df["dayofweek_cos"].values,
                week_sin=self.df["week_sin"].values,
                week_cos=self.df["week_cos"].values,
                atr_Volatility=self.df["ATR_14"].values,

            )


class Resume:
    def __init__(self, symmbol_data: pd.DataFrame):
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

