"""From FinRL https://github.com/AI4Finance-LLC/FinRL/tree/master/finrl/env"""
import gymnasium as gym
import math
import matplotlib
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
from pathlib import Path
import quantstats as qs
import time
from utils.Debug_tool import debug

matplotlib.use("Agg")


class PortfolioOptimizationEnv(gym.Env):


    def rand_pick_symbos(self, symbols_number:int):    
        """
            symbols_number(int) : every time need to choose ten symbols.
        """
        self._targetsymbol = np.random.choice(self._tic_column, size=symbols_number)
        self._offset = np.random.choice(self._length-self._time_window*10) + self._time_window

    def _reset_memory(self):
        self._actions_memory = [np.array([0] * self._assets_dim, dtype=np.float32)]
        self._final_weights = [np.array([0] * self._assets_dim, dtype=np.float32)]
        self._asset_memory = {
            "initial": [self._initial_amount],
            "final": [self._initial_amount]
        }



    def step(self, actions):
        """
        """
        print("本次動作為：",actions)
        weights = np.array(actions, dtype=np.float32) * (1 / self._assets_dim)
        self._actions_memory.append(weights)
        last_weights = self._final_weights[-1]


        time.sleep(100)
        last_mu = 1
        # author to create mu is different paper
        mu = 1 - 2 * self._comission_fee_pct + self._comission_fee_pct ** 2

        print("計算前---","last_mu:",last_mu,"mu:",mu)
        # 但是他這裡的算法是完全比照paper的
        while abs(mu - last_mu) > 1e-10:
            last_mu = mu
            mu = (1 - self._comission_fee_pct * weights[0] -
                    (2 * self._comission_fee_pct - self._comission_fee_pct ** 2) *
                    np.sum(
                        np.maximum(last_weights[1:] - mu * weights[1:], 0))
                    ) / (1 - self._comission_fee_pct * weights[0])

        self._info["trf_mu"] = mu
        self._portfolio_value = mu * self._portfolio_value
        print("計算後---","last_mu:",last_mu,"mu:",mu)
        print('*'*120)
        time.sleep(5)


        # 儲存這一時間步的初始投資組合價值
        self._asset_memory["initial"].append(self._portfolio_value)

        # # 時間過去，時間變化改變了投資組合的分布
        portfolio = self._portfolio_value * \
            (weights * self._price_variation)

        # 計算新的投資組合價值和權重
        self._portfolio_value = np.sum(portfolio)
        weights = portfolio / self._portfolio_value

        # 儲存這一時間步的最終投資組合價值和權重
        self._asset_memory["final"].append(self._portfolio_value)
        self._final_weights.append(weights)

        # 儲存日期記憶
        self._date_memory.append(self._info["end_time"])

        # 定義投資組合收益率
        rate_of_return = self._asset_memory["final"][-1] / \
            self._asset_memory["final"][-2]

        portfolio_return = rate_of_return - 1
        portfolio_reward = np.log(rate_of_return)

        # 儲存投資組合收益記憶
        self._portfolio_return_memory.append(portfolio_return)
        self._portfolio_reward_memory.append(portfolio_reward)

        # 定義投資組合收益
        self._reward = portfolio_reward
        self._reward = self._reward * self._reward_scaling

        self._offset += 1
        done |= self._offset >= self.length - 1
        return self._state, self._reward, self._terminal, False, self._info


    def _get_state_and_info_from_time_index(self, time_index):
        """
        Gets state and information given a time index. It also updates "data"
        attribute with information about the current simulation step.

        Lewis :
            For neural networks, this function does not perform regularization (according to paper 3.2 Price Tensor)."


            paper shape:
                f,n,m
                f = feature number
                n = 50 (1 day and an hour if T = 30 minutes)
                m = assets of finance market

                but this program use CNN neural nework,Does shape has to be transpose?  
        Args:
            time_index: An integer that represents the index of a specific datetime.
                The initial datetime of the dataframe is given by 0.

        Note:
            If the environment was created with "return_last_action" set to
            True, the returned state will be a Dict. If it's set to False,
            the returned state will be a Box. You can check the observation
            state through the attribute "observation_space".

        Returns:
            A tuple with the following form: (state, info).

            state: The state of the current time index. It can be a Box or a Dict.
            info: A dictionary with some informations about the current simulation
                step. The dict has the following keys::

                {
                "tics": List of ticker symbols,
                "start_time": Start time of current time window,
                "start_time_index": Index of start time of current time window,
                "end_time": End time of current time window,
                "end_time_index": Index of end time of current time window,
                "data": Data related to the current time window,
                "price_variation": Price variation of current time step
                }
        """
        # returns state in form (channels, tics, timesteps)
        end_time = self._sorted_times[time_index]
        start_time = self._sorted_times[time_index - (self._time_window - 1)]

        # define data to be used in this time step
        self._data = self._df[
            (self._df[self._time_column] >= start_time) &
            (self._df[self._time_column] <= end_time)
        ][[self._time_column, self._tic_column] + self._features]

        # define price variation of this time_step ( 用來取得收盤價的變化 )
        self._price_variation = self._df_price_variation[
            self._df_price_variation[self._time_column] == end_time
        ][self._valuation_feature].to_numpy()

        # 將第一項的價格變化量插入
        self._price_variation = np.insert(self._price_variation, 0, 1)

        # define state to be returned
        state = None
        for tic in self._tic_list:
            tic_data = self._data[self._data[self._tic_column] == tic]
            vt = tic_data['close'].values[-1]
            tic_data = tic_data[self._features].to_numpy().T
            tic_data = tic_data / vt
            tic_data = tic_data[..., np.newaxis]

            state = tic_data if state is None else np.append(
                state, tic_data, axis=2)

        state = state.transpose((0, 2, 1))

        info = {
            "tics": self._tic_list,
            "start_time": start_time,
            "start_time_index": time_index - (self._time_window - 1),
            "end_time": end_time,
            "end_time_index": time_index,
            "data": self._data,
            "price_variation": self._price_variation
        }
        return self._standardize_state(state), info

    def _standardize_state(self, state):
        """
        Standardize the state given the observation space. If "return_last_action"
        is set to False, a three-dimensional box is returned. If it's set to True, a
        dictionary is returned. The dictionary follows the standard below::

            {
            "state": Three-dimensional box representing the current state,
            "last_action": One-dimensional box representing the last action
            }

        """
        last_action = self._actions_memory[-1]
        if self._return_last_action:
            return {"state": state, "last_action": last_action}
        else:
            return state





# class PortfolioOptimizationEnv(gym.Env):
#     """A portfolio allocantion environment for OpenAI gym.

#     This environment simulates the interactions between an agent and the financial market
#     based on data provided by a dataframe. The dataframe contains the time series of
#     features defined by the user (such as closing, high and low prices) and must have
#     a time and a tic column with a list of datetimes and ticker symbols respectively.
#     An example of dataframe is shown below::

#             date        high            low             close           tic
#         0   2020-12-23  0.157414        0.127420        0.136394        ADA-USD
#         1   2020-12-23  34.381519       30.074295       31.097898       BNB-USD
#         2   2020-12-23  24024.490234    22802.646484    23241.345703    BTC-USD
#         3   2020-12-23  0.004735        0.003640        0.003768        DOGE-USD
#         4   2020-12-23  637.122803      560.364258      583.714600      ETH-USD
#         ... ...         ...             ...             ...             ...

#     Based on this dataframe, the environment will create an observation space that can
#     be a Dict or a Box. The Box observation space is a three-dimensional array of shape
#     (f, n, t), where f is the number of features, n is the number of stocks in the
#     portfolio and t is the user-defined time window. If the environment is created with
#     the parameter return_last_action set to True, the observation space is a Dict with
#     the following keys::

#         {
#         "state": three-dimensional Box (f, n, t) representing the time series,
#         "last_action": one-dimensional Box (n+1,) representing the portfolio weights
#         }

#     Note that the action space of this environment is an one-dimensional Box with size
#     n + 1 because the portfolio weights must contains the weights related to all the
#     stocks in the portfolio and to the remaining cash.

#     Attributes:
#         action_space: Action space.
#         observation_space: Observation space.
#         episode_length: Number of timesteps of an episode.
#     """

#     # 在這裡應該意義不大，只有回傳當前狀態而已，沒有可視化
#     metadata = {
#         "render.modes": ["human"]
#     }

#     def __init__(
#         self,
#         df,

#         order_df=True,
#         return_last_action=False,
#         normalize_df: str = None,
#         reward_scaling=1,
#         comission_fee_pct=0,
#         features=["close", "high", "low"],
#         valuation_feature="close",
#         time_column="date",
#         time_format="%Y-%m-%d",
#         tic_column="tic",

#         cwd="./",
#     ):
#         """Initializes environment's instance.

#         Args:
#             df: Dataframe with market information over a period of time.

#             order_df: If True input dataframe is ordered by time.
#             return_last_action: If True, observations also return the last performed
#                 action. Note that, in that case, the observation space is a Dict.

#             normalize_df: Defines the normalization method applied to input dataframe.
#                 Possible values are "by_previous_time", "by_fist_time_window_value",
#                 "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
#                 name) and a custom function. If None no normalization is done.
#                 "custom_normalize" is lewis to chang custom function            


#             reward_scaling: A scaling factor to multiply the reward function. This
#                 factor can help training.

#             comission_fee_pct: Percentage to be used in comission fee. It must be a value
#                 between 0 and 1.

#             features: List of features to be considered in the observation space. The
#                 items of the list must be names of columns of the input dataframe.
#             valuation_feature: Feature to be considered in the portfolio value calculation.
#             time_column: Name of the dataframe's column that contain the datetimes that
#                 index the dataframe.
#             time_format: Formatting string of time column.
#             tic_name: Name of the dataframe's column that contain ticker symbols.
#             time_window: Size of time window.
#             cwd: Local repository in which resulting graphs will be saved.
#         """

#         self._time_window = time_window
#         self._time_index = time_window - 1
#         self._time_column = time_column  # 標示時間的那一個類別
#         self._time_format = time_format

#         self._df = df  # 輸入資料時應保證資料完整性及正確性



#         self._return_last_action = return_last_action
#         self._reward_scaling = reward_scaling
#         self._comission_fee_pct = comission_fee_pct
#         self._features = features
#         self._valuation_feature = valuation_feature
#         self._cwd = Path(cwd)


#         # results file
#         self._results_file = self._cwd / "results" / "rl"
#         self._results_file.mkdir(parents=True, exist_ok=True)

#         # price variation
#         self._df_price_variation = None

#         # preprocess data
#         self._preprocess_data(order_df, normalize_df)

#         # dims and spaces
#         self._tic_list = self._df[self._tic_column].unique()
#         self._stock_dim = len(self._tic_list)
#         action_space = 1 + self._stock_dim

#         # sort datetimes and define episode length
#         self._sorted_times = sorted(set(self._df[self._time_column]))

#         self.episode_length = len(self._sorted_times) - time_window + 1

#         # define action space
#         self.action_space = spaces.Box(low=0, high=1, shape=(action_space,))

#         # define observation state # 我懷疑這裡寫錯
#         if self._return_last_action:
#             # if  last action must be returned, a dict observation
#             # is defined
#             self.observation_space = spaces.Dict({
#                 "state": spaces.Box(
#                     low=-np.inf,
#                     high=np.inf,
#                     shape=(
#                         len(self._features),
#                         self._stock_dim,
#                         self._time_window
#                     )
#                 ),
#                 "last_action": spaces.Box(
#                     low=0, high=1, shape=(action_space,)
#                 )
#             })
#         else:
#             # if information about last action is not relevant,
#             # a 3D observation space is defined
#             self.observation_space = spaces.Box(
#                 low=-np.inf,
#                 high=np.inf,
#                 shape=(
#                     len(self._features),
#                     self._stock_dim,
#                     self._time_window
#                 ),
#             )

#         self._reset_memory()
#         self._terminal = False








#     def _softmax_normalization(self, actions):
#         """Normalizes the action vector using softmax function.

#         Returns:
#             Normalized action vector (portfolio vector).
#         """
#         numerator = np.exp(actions)
#         denominator = np.sum(np.exp(actions))
#         softmax_output = numerator / denominator
#         return softmax_output

#     def enumerate_portfolio(self):
#         """Enumerates the current porfolio by showing the ticker symbols
#         of all the investments considered in the portfolio.
#         """
#         print("Index: 0. Tic: Cash")
#         for index, tic in enumerate(self._tic_list):
#             print("Index: {}. Tic: {}".format(index + 1, tic))

#     def _preprocess_data(self, order, normalize):
#         """
#         Orders and normalizes the environment's dataframe.

#         Args:
#             order: If true, the dataframe will be ordered by ticker list
#                 and datetime.

#             normalize: Defines the normalization method applied to the dataframe.
#                 Possible values are "by_previous_time", "by_fist_time_window_value",
#                 "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
#                 name) and a custom function. If None no normalization is done.
#         """

#         # order time dataframe by tic and time
#         if order:
#             self._df = self._df.sort_values(
#                 by=[self._tic_column, self._time_column])

#         # defining price variation after ordering dataframe
#         self._df_price_variation = self._temporal_variation_df()

#         # apply normalization
#         # 目前還舞法確定改了 正則化會發生甚麼?
#         if normalize:
#             self._normalize_dataframe(normalize)

#         # transform str to datetime
#         self._df[self._time_column] = pd.to_datetime(
#             self._df[self._time_column])
#         self._df_price_variation[self._time_column] = pd.to_datetime(
#             self._df_price_variation[self._time_column])

#         # transform numeric variables to float32 (compatibility with pytorch)
#         self._df[self._features] = self._df[self._features].astype("float32")
#         self._df_price_variation[self._features] = self._df_price_variation[self._features].astype(
#             "float32")

#     def _reset_memory(self):
#         """Resets the environment's memory."""
#         date_time = self._sorted_times[self._time_index]
#         # memorize portfolio value each step

#         # memorize portfolio return and reward each step
#         self._portfolio_return_memory = [0]
#         self._portfolio_reward_memory = [0]
#         # initial action: all money is allocated in cash

#         # memorize portfolio weights at the ending of time step

#         # memorize datetimes
#         self._date_memory = [date_time]

#     def _normalize_dataframe(self, normalize):
#         """"Normalizes the environment's dataframe.

#         Args:
#             normalize: Defines the normalization method applied to the dataframe.
#                 Possible values are "by_previous_time", "by_fist_time_window_value",
#                 "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
#                 name) and a custom function. If None no normalization is done.

#         Note:
#             If a custom function is used in the normalization, it must have an
#             argument representing the environment's dataframe.
#         """
#         if type(normalize) == str:
#             if normalize == "by_fist_time_window_value":
#                 print("Normalizing {} by first time window value...".format(
#                     self._features))
#                 self._df = self._temporal_variation_df(self._time_window - 1)
#             elif normalize == "by_previous_time":
#                 print("Normalizing {} by previous time...".format(self._features))
#                 #     date      tic     close      high       low
#                 # 0     2011-01-03  2317.TW  1.000000  1.000000  1.000000
#                 # 1     2011-01-04  2317.TW  1.000000  1.008475  1.004274
#                 # 2     2011-01-05  2317.TW  0.974468  0.991597  0.970213
#                 # 3     2011-01-06  2317.TW  1.000000  0.974576  1.000000
#                 self._df = self._temporal_variation_df()
#             elif normalize.startswith("by_"):
#                 normalizer_column = normalize[3:]
#                 print("Normalizing {} by {}".format(
#                     self._features, normalizer_column))
#                 for column in self._features:
#                     self._df[column] = self._df[column] / \
#                         self._df[normalizer_column]

#         elif callable(normalize):
#             print("Applying custom normalization function...")
#             self._df = normalize(self._df)

#         else:
#             print("No normalization was performed.")

#     def _temporal_variation_df(self, periods=1):
#         """
#         Calculates the temporal variation dataframe. For each feature, this
#         dataframe contains the rate of the current feature's value and the last
#         feature's value given a period. It's used to normalize the dataframe.

#         parper to use three kind prices
#             Features for asset i on Period t are its closing, highest, and lowest prices

#         Args:
#             periods: Periods (in time indexes) to calculate temporal variation.

#         Returns:
#             Temporal variation dataframe.
#         """
#         df_temporal_variation = self._df.copy()
#         prev_columns = []

#         for column in self._features:
#             prev_column = "prev_{}".format(column)
#             prev_columns.append(prev_column)
#             df_temporal_variation[prev_column] = df_temporal_variation.groupby(
#                 self._tic_column)[column].shift(periods=periods)
#             df_temporal_variation[column] = df_temporal_variation[column] / \
#                 df_temporal_variation[prev_column]

#         df_temporal_variation = df_temporal_variation.drop(
#             columns=prev_columns).fillna(1).reset_index(drop=True)

#         return df_temporal_variation

#     def _seed(self, seed=None):
#         """Seeds the sources of randomness of this environment to guarantee
#         reproducibility.

#         Args:
#             seed: Seed value to be applied.

#         Returns:
#             Seed value applied.
#         """
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]

#     def render(self, mode="human"):
#         """Renders the environment.

#         Returns:
#             Observation of current simulation step.
#         """
#         return self._state

#     def _plot(self, data, title: str, xlabel: str, ylabel, save_path, file_name, specified_color: str = None, special=False):
#         """
#             绘制并保存图表。

#             参数:
#             - data: 要绘制的数据。
#             - title: 图表标题。
#             - xlabel: x轴标签。
#             - ylabel: y轴标签。
#             - save_path: 保存图表的路径。
#             - file_name: 保存的文件名。
#             - special:是否需要特殊處理
#         """
#         plt.figure()
#         if special:
#             # 转置data，以便每次循环处理一个资产的数据
#             for i, asset_data in enumerate(np.array(data).T):
#                 local_tic = ['Cash_asset']
#                 local_tic.extend(self._tic_list)
#                 plt.plot(asset_data, label=local_tic[i])
#         else:
#             if specified_color is not None:
#                 plt.plot(data, specified_color)
#             else:
#                 plt.plot(data)

#         plt.title(title)
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#         plt.legend()  # 显示图例
#         plt.savefig(Path(save_path) / file_name)
#         plt.close()
