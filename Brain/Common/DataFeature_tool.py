import pandas as pd
import numpy as np
from typing import List, Optional
import time

class VolatilityCalculator:
    """
        一個使用 pandas 計算指定時間窗口滾動年化波動率的類別。
    """
    def __init__(
        self, 
        windows: Optional[List[int]] = None, 
        annualization_factor: int =252
    ):
        """
            Args:
                windows (Optional[List[int]]): 
                    一個包含所需計算時間窗口的整數列表。
                    如果為 None，預設為 [5, 10, 20, 60]。
                
                annualization_factor (int): 
                    年化因子。預設為 252，適用於股票市場（一年的交易日數）。
                    對於加密貨幣或外匯市場，您可能會使用 365。
        """
        if windows is None:
            self.windows = [5, 10, 20, 60]
        else:
            self.windows = windows
            
        # 預先計算根號，提升效率
        self.annualization_sqrt = np.sqrt(annualization_factor)
        print(f"VolatilityCalculator initialized with windows: {self.windows} and annualization factor: {annualization_factor}")

    def calculate(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        計算波動率並將其作為新欄位新增到 DataFrame 中。

        Args:
            df (pd.DataFrame): 
                包含價格數據的 DataFrame，需要有一個時間序列索引。
            price_col (str): 
                用於計算波動率的價格欄位名稱。預設為 'close'。

        Returns:
            pd.DataFrame: 
                一個新的 DataFrame，其中包含了計算出的波動率欄位。
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("輸入的 'df' 必須是 pandas DataFrame。")
        
        if price_col not in df.columns:
            raise ValueError(f"在 DataFrame 中找不到指定的價格欄位 '{price_col}'。")
            
        # 建立一個副本以避免修改原始 DataFrame
        df_out = df.copy()
        
        # 1. 計算對數回報率
        log_returns = np.log(df_out[price_col] / df_out[price_col].shift(1))
        
        # 2. 遍歷所有指定的時間窗口
        for window in self.windows:
            col_name = f'vol{window}'
            # 3. 計算滾動標準差
            rolling_std = log_returns.rolling(window=window, min_periods=window).std()
            # 4. 進行年化並新增為新欄位
            df_out[col_name] = rolling_std * self.annualization_sqrt
            
        return df_out
    


class RSICalculator:
    """
    一個使用 pandas 計算相對強弱指數 (RSI) 的類別。
    """
    def __init__(self, windows: Optional[List[int]] = None):
        """
        Args:
            windows (Optional[List[int]]): 
                一個包含所需計算時間窗口的整數列表。
                如果為 None，預設為 [6, 12, 24]。
        """
        if windows is None:
            self.windows = [6, 12, 24]
        else:
            self.windows = windows
        print(f"RSICalculator initialized with windows: {self.windows}")

    def calculate(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        計算 RSI 並將其作為新欄位新增到 DataFrame 中。

        Args:
            df (pd.DataFrame): 
                包含價格數據的 DataFrame。
            price_col (str): 
                用於計算的價格欄位名稱。預設為 'close'。

        Returns:
            pd.DataFrame: 
                一個新的 DataFrame，其中包含了計算出的 RSI 欄位。
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("輸入的 'df' 必須是 pandas DataFrame。")
        
        if price_col not in df.columns:
            raise ValueError(f"在 DataFrame 中找不到指定的價格欄位 '{price_col}'。")
            
        df_out = df.copy()
        
        # 1. 計算價格差異
        delta = df_out[price_col].diff()
        
        for window in self.windows:
            col_name = f'rsi{window}'
            
            # 2. 分別取得上漲和下跌的變化量
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss) # 將損失轉為正數以便計算

            # 3. 計算平均上漲和平均下跌 (使用指數移動平均)
            # 使用 adjust=False 以匹配常見的技術分析庫行為
            avg_gain = gain.ewm(com=window - 1, min_periods=window, adjust=False).mean()
            avg_loss = loss.ewm(com=window - 1, min_periods=window, adjust=False).mean()
            
            # 4. 計算相對強度 (RS)
            rs = avg_gain / avg_loss
            
            # 5. 計算 RSI
            rsi = 100 - (100 / (1 + rs))
            
            df_out[col_name] = rsi
            
        return df_out


class MTMCalculator:
    """
    一個使用 pandas 計算動量指標 (MTM) 的類別。
    """
    def __init__(self, windows: Optional[List[int]] = None):
        """
        Args:
            windows (Optional[List[int]]): 
                一個包含所需計算時間窗口的整數列表。
                如果為 None，預設為 [10, 20, 60]。
        """
        if windows is None:
            self.windows = [10, 20, 60]
        else:
            self.windows = windows
        print(f"MTMCalculator initialized with windows: {self.windows}")

    def calculate(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        計算 MTM 並將其作為新欄位新增到 DataFrame 中。

        Args:
            df (pd.DataFrame): 
                包含價格數據的 DataFrame。
            price_col (str): 
                用於計算的價格欄位名稱。預設為 'close'。

        Returns:
            pd.DataFrame: 
                一個新的 DataFrame，其中包含了計算出的 MTM 欄位。
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("輸入的 'df' 必須是 pandas DataFrame。")
        
        if price_col not in df.columns:
            raise ValueError(f"在 DataFrame 中找不到指定的價格欄位 '{price_col}'。")
            
        df_out = df.copy()
        
        for window in self.windows:
            col_name = f'mtm{window}'
            # 1. 計算當前價格與 N 天前價格的差值
            df_out[col_name] = df_out[price_col].diff(window)
            
        return df_out
    
class DIFFCalculator:
    """
    一個使用 pandas 計算差離值 (DIFF, MACD 中的快線) 的類別。
    """
    def __init__(
        self, 
        fast_periods: Optional[List[int]] = None
    ):
        """
        Args:
            fast_periods (Optional[List[int]]): 
                一個包含快線 EMA 週期的整數列表。慢線週期將自動設為約 2.17 倍。
                如果為 None，預設為 [12]。經典組合為 (12, 26)。
        """
        if fast_periods is None:
            self.fast_periods = [12]
        else:
            self.fast_periods = fast_periods
        print(f"DIFFCalculator initialized with fast_periods: {self.fast_periods}")

    def calculate(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        計算 DIFF 並將其作為新欄位新增到 DataFrame 中。

        Args:
            df (pd.DataFrame): 
                包含價格數據的 DataFrame。
            price_col (str): 
                用於計算的價格欄位名稱。預設為 'close'。

        Returns:
            pd.DataFrame: 
                一個新的 DataFrame，其中包含了計算出的 DIFF 欄位。
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("輸入的 'df' 必須是 pandas DataFrame。")
        
        if price_col not in df.columns:
            raise ValueError(f"在 DataFrame 中找不到指定的價格欄位 '{price_col}'。")
            
        df_out = df.copy()
        
        for fast_period in self.fast_periods:
            # 根據常見的 12/26 比例計算慢線週期
            slow_period = int(np.round(fast_period * (26 / 12.0)))
            col_name = f'diff_{fast_period}_{slow_period}'
            
            # 1. 計算快線和慢線的指數移動平均 (EMA)
            ema_fast = df_out[price_col].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df_out[price_col].ewm(span=slow_period, adjust=False).mean()
            
            # 2. 計算差離值
            df_out[col_name] = ema_fast - ema_slow
            
        return df_out
    
class PSYCalculator:
    """
    一個使用 pandas 計算心理線指標 (PSY) 的類別。
    """
    def __init__(self, windows: Optional[List[int]] = None):
        """
        Args:
            windows (Optional[List[int]]): 
                一個包含所需計算時間窗口的整數列表。
                如果為 None，預設為 [12, 24]。
        """
        if windows is None:
            self.windows = [12, 24]
        else:
            self.windows = windows
        print(f"PSYCalculator initialized with windows: {self.windows}")

    def calculate(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        計算 PSY 並將其作為新欄位新增到 DataFrame 中。

        Args:
            df (pd.DataFrame): 
                包含價格數據的 DataFrame。
            price_col (str): 
                用於計算的價格欄位名稱。預設為 'close'。

        Returns:
            pd.DataFrame: 
                一個新的 DataFrame，其中包含了計算出的 PSY 欄位。
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("輸入的 'df' 必須是 pandas DataFrame。")
        
        if price_col not in df.columns:
            raise ValueError(f"在 DataFrame 中找不到指定的價格欄位 '{price_col}'。")
            
        df_out = df.copy()
        
        # 1. 判斷每日是否上漲 (1 代表上漲, 0 代表下跌或持平)
        is_gain = (df_out[price_col].diff() > 0).astype(int)
        
        for window in self.windows:
            col_name = f'psy{window}'
            # 2. 計算在滾動窗口內上漲的天數
            gain_days_in_window = is_gain.rolling(window=window, min_periods=window).sum()
            
            # 3. 計算 PSY
            df_out[col_name] = (gain_days_in_window / window) * 100
            
        return df_out