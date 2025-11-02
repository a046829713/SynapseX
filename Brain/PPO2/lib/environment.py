import numpy as np
import enum
import time
import gymnasium as gym
import numpy as np
import collections
import gymnasium as gym
from gymnasium import spaces
from Brain.Common.env_components import State_time_step_template
from abc import ABC, abstractmethod
from Brain.Common.DataFeature import OriginalDataFrature


class Reward:
    def __init__(self):
        self.tradeReturn_weight = 0.4
        self.closeReturn_weight = 1 - self.tradeReturn_weight
        self.wrongTrade_weight = 1
        self.trendTrade_weight = 0.5
        self.drawdown_penalty_weight = 0.001

    def tradeReturn(self, last_value: float, previous_value: float) -> float:
        """
        主要用於淨值 = 起始資金 + 手續費(累積) +  已平倉損益(累積) + 未平倉損益(單次)


        Return = last_value - previous_value.
        """
        return self.tradeReturn_weight * (last_value - previous_value)


class Actions(enum.Enum):
    Hold = 0
    Buy = 1
    Sell = 2


class State_time_step(State_time_step_template):
    def __init__(
        self,
        bars_count,
        commission_perc,
        model_train,
        default_slippage,
        N_steps,
        win_payoff_weight,
    ):
        super().__init__(
            bars_count=bars_count,
            commission_perc=commission_perc,
            model_train=model_train,
            default_slippage=default_slippage,
            N_steps = N_steps,            
        )

        self.win_payoff_weight = win_payoff_weight

    def reset(self, prices, offset):
        assert offset >= self.bars_count - 1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    def step(self, marketpostion: int, percentage: float):
        """
            marketpostion (0,1,2)
            percentage : 0 -1 continous.
        """
        reward = 0.0
        done = False
        close = self._prices.close[self._offset]

        return reward, done


class BaseTradingEnv(gym.Env):
    def __init__(self, state:State_time_step):
        self._state = state
        # 0 Hold 1 Buy 2 Sell
        N_DISCRETE_ACTIONS = 3

        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(N_DISCRETE_ACTIONS),
                spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            )
        )
        # 為了讓資料分流但是同時保持gym 的標準設計
        self.observation_space = spaces.Dict({
            "states": spaces.Box(
                low=-np.inf, high=np.inf, shape=self._state.getStateShape(), dtype=np.float32
            ),
            "time_states": spaces.Box(
                low=-np.inf, high=np.inf, shape=self._state.getTimeShape(), dtype=np.float32
            )
        })




class TrainingEnv(BaseTradingEnv):
    """
        用於模型訓練的環境。
        它會動態地從數據源加載數據，並在每次 reset 時隨機化起始位置。
    """

    def __init__(self, config):
        """
        Args:
            config: 包含所有配置的物件。
            is_test_mode (bool): 如果為 True，則使用測試數據和固定的起始點（用於驗證）。
                               如果為 False，則使用訓練數據和隨機的起始點。
        """
        self.config = config
        self.unique_symbols = self.config.UNIQUE_SYMBOLS
        
        
        # 根據模式決定數據類型和佣金
        self.data_type_name = "train_data"
        random_symbol = np.random.choice(self.unique_symbols)
        self.all_data = self._load_data_for_instrument(random_symbol)

        state_params = {
            "bars_count": self.config.BARS_COUNT,
            "commission_perc": self.config.MODEL_DEFAULT_COMMISSION_PERC_TRAING,
            "model_train": True,
            "default_slippage": self.config.DEFAULT_SLIPPAGE,
            "N_steps": self.config.N_STEPS,
            "win_payoff_weight": self.config.WIN_PAYOFF_WEIGHT,
        }

        super().__init__(state=State_time_step(**state_params))

    def _load_data_for_instrument(self, instrument: str):
        """一個輔助方法，專門用來載入特定商品的數據。"""
        return OriginalDataFrature().get_train_net_work_data_by_path(
            [instrument], typeName=self.data_type_name
        )

    def reset(self, symbol: str = None):
        if symbol is None:
            self._instrument = np.random.choice(self.unique_symbols)
        else:
            self._instrument = symbol

        all_prices = self._load_data_for_instrument(self._instrument)
        prices = all_prices[self._instrument]

        offset = (
            np.random.choice(prices.high.shape[0] - self._state.bars_count * 10)
            + self._state.bars_count
        )

        print(
            f"[{self.data_type_name}] Actor resetting env with symbol: {self._instrument} at offset: {offset}"
        )

        self._state.reset(prices, offset)
        return self._state.encode()
    
    def step(self, marketpostion: int, percentage: float):        
        reward, done = self._state.step(marketpostion, percentage)  # 這邊會更新步數
        obs = self._state.encode()  # 呼叫這裡的時候就會取得新的狀態

        info = {
            "instrument": self._instrument,
            "offset": self._state._offset,
            "postion": float(self._state.have_position),
        }

        return obs, reward, done, info