import gymnasium as gym
import numpy as np
import pandas as pd
import time

class Crypto_PortfolioOptimztionEnv(gym.Env):
    def __init__(self,
                 df:pd.DataFrame,
                 time_window=300,
                 initial_amount = 25000,
                 time_column ="Datetime",
                 tic_column="tic",
                 order_df=True,
                 ):
        """
            想要訓練PPO,讓智能體可以評估0-1之間的連續動作,
            假設商品為10,每個商品的投資資金為1/10,

            example:
                action :[0.5,0.4,0.3,0.2,0.1,0.6,0.7,0.8,0.9,1.0]

                實際下單的時候 會乘上各資金上限 確保不會超過總體資金1.
                [0.05,0.04,0.03,0.02,0.01,0.06,0.07,0.08,0.09,0.10]


            Args:
                #   initial_amount: Initial amount of cash available to be invested.
        """
        self._df = df
        self._tic_column = tic_column

        self._assets_dim = 10

        # every symbols length is equal.
        self._assets = self._df[self._tic_column].unique()
        self._length = len(self._df[self._df[self._tic_column] == self._assets[0]])
        
        
        self._time_window =time_window
        self._initial_amount = initial_amount
    
        self._time_column =time_column
        # sort datetimes and define episode length
        self._sorted_times = sorted(set(self._df[self._time_column]))

        # preprocess data        
        self._preprocess_data(order_df)
    
    def reset(self):
        """
            everytime need to pickup different symbols.
        """
        self._state = None
        self.rand_pick_symbos(symbols_number=self._assets_dim)
        
        
        # time_index must start a little bit in the future to implement lookback
        # self._time_index = self._time_window - 1
        # self._reset_memory()
        # self._state, self._info = self._get_state_and_info_from_time_index(
        #     self._time_index)

        self._portfolio_value = self._initial_amount
        # self._terminal = False
        
        return self._state
    
    def rand_pick_symbos(self, symbols_number:int):    
        """
            symbols_number(int) : every time need to choose ten symbols.
        """
        self._targetsymbol = np.random.choice(self._tic_column, size=symbols_number)
        self._time_index = np.random.choice(self._length - self._time_window -1) + self._time_window
        self._traget_df = self._df[self._df['tic'].isin(self._targetsymbol)]

    def step(self,actions):
        """
            我的定義為 Agent 所產生的actions,即為調整完成的權重,
            我必須要在兩次權重變化之間, 取得reward        
        """
        print(actions)

        end_time = self._sorted_times[self._time_index]
        print(end_time)

        self._price_variation = self._traget_df

        # print(self._traget_df['Datetime'] ==)

        # self._price_variation = self._df_price_variation[
        #     self._df_price_variation[self._time_column] == end_time
        # ][self._valuation_feature].to_numpy()


    def _preprocess_data(self, order):
        """
            Orders and cleans the environment's dataframe.

        Args:
            order: If true, the dataframe will be ordered by ticker list
                and datetime.
        """
        if order:
            self._df = self._df.sort_values(
                by=[self._tic_column, self._time_column])


        self._df[self._time_column] = pd.to_datetime(self._df[self._time_column])
        
        # 
        self._df.astype("float32")

        time.sleep(100)
        numeric_cols = self._df.select_dtypes(include=[np.number]).columns

        self._df[self._features] = self._df[self._features].astype("float32")
        self._df_price_variation[self._features] = self._df_price_variation[self._features].astype("float32")
        
        self._df_normalization = self._normalization_df()
        print(self._df_price_variation)
        time.sleep(100)

    def _normalization_df(self):
        """
            只對 self._df 中的數值型欄位進行 log normalization，
            並將結果與非數值欄位合併後返回。

            
        """
        print(self._df.info())
        time.sleep(100)
        # 選取數值型欄位
        numeric_cols = self._df.select_dtypes(include=[np.number]).columns
        df_numeric = self._df[numeric_cols].copy()
        # 進行 log normalization
        df_numeric = np.log(df_numeric + 1.0)
        
        # 複製原始 DataFrame 並更新數值型欄位
        df_norm = self._df.copy()
        df_norm[numeric_cols] = df_numeric
        

        print(df_norm)
        print(self._df)
        return df_norm
