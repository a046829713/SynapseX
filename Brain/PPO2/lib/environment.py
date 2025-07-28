import numpy as np
import enum
import time
import gymnasium as gym
import numpy as np
import collections
import gymnasium as gym
from gymnasium import spaces




class Reward():
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


class State:
    def __init__(self, init_prices: collections.namedtuple, bars_count, commission_perc, model_train, default_slippage):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0

        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.N_steps = 1000  # 這遊戲目前使用多少步學習
        self.model_train = model_train
        self.default_slippage = default_slippage
        self.win_payoff_weight = 1
        self.build_fileds(init_prices)
        self.reward_function = Reward()

        
    def build_fileds(self, init_prices):
        self.field_names = list(init_prices._fields)
        for i in ['open', 'high', 'low', 'close']:
            if i in self.field_names:
                self.field_names.remove(i)
        


    def reset(self, prices, offset):
        assert offset >= self.bars_count-1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset
        self.game_steps = 0  # 這次遊戲進行了多久

        self.cost_sum = 0.0
        self.closecash = 0.0
        self.canusecash = 1.0
        self.equity_peak = None  # 可以用 1.0 或其它初始值
        self.trade_bar = 0 # 用來紀錄持倉多久
        self.bar_dont_change_count = 0  # 計算K棒之間轉換過了多久 感覺下一次實驗也可以將這個部份加入
        
        # 新增：初始化交易統計資料
        self.total_trades = 0
        self.win_trades = 0
        self.total_win = 0.0
        self.total_loss = 0.0
        
        self.beforeBar = 50

    def step(self,marketpostion:int,percentage:float):
        """
            marketpostion (0,1,2)
            percentage : 0 -1 continous.
        """        
        reward = 0.0
        done = False
        close = self._prices.close[self._offset]
        



        return reward, done


class State_time_step(State):
    """
        專門用於transformer的時序函數。
    """
    @property
    def shape(self):
        return (self.bars_count, len(self.field_names) + 3)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)

        ofs = self.bars_count
        for bar_idx in range(self.bars_count):
            for idx, field in enumerate(self.field_names):  # 編碼所有字段
                res[bar_idx][idx] = getattr(self._prices, field)[
                    self._offset - ofs + bar_idx]

        if self.have_position:
            res[:, len(self.field_names)] = 1.0
            res[:, len(self.field_names) + 1 ] = (self._prices.close[self._offset] - self.open_price) / \
                self.open_price
            res[:, len(self.field_names) + 2 ] = self.trade_bar
        
        # print(res) 
        # # 加入傅立葉變換 res.shape # (300, 29)
        # fourier_features = self.fourier_transform(res[:, :len(self.field_names)])
        # print(fourier_features)
        # # 可以選擇如何將傅立葉特徵與原有特徵結合，例如拼接或者替換
        # # 這裡選擇拼接
        # res = np.concatenate((res, fourier_features), axis=1)
        return res
    
    def fourier_transform(self, data):
        """
        對每一列數據進行傅立葉變換，並提取頻率和幅度信息

        Args:
            data: (np.ndarray) shape 為 (bars_count, num_features) 的數據

        Returns:
            fourier_features: (np.ndarray) shape 為 (bars_count, num_features * 2) 的傅立葉特徵
                                         每一列數據對應兩列傅立葉特徵：頻率的實部和虛部
        """
        fourier_features = np.zeros((data.shape[0], data.shape[1] * 2), dtype=np.float32)
        for i in range(data.shape[1]):
            # 使用 rfft 進行實數傅立葉變換，只計算正頻率部分
            fft_values = np.fft.rfft(data[:, i])
            # 提取頻率的實部和虛部
            fourier_features[:, i * 2] = np.real(fft_values)
            fourier_features[:, i * 2 + 1] = np.imag(fft_values)
        return fourier_features


class Env(gym.Env):
    def __init__(self, prices, state, random_ofs_on_reset):
        self._prices = prices
        self._state = state
        
        
        # 0 Hold 1 Buy 2 Sell
        N_DISCRETE_ACTIONS = 3 
        self.action_space = spaces.Tuple((
            spaces.Discrete(N_DISCRETE_ACTIONS),
            spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        ))


        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._state.shape,
            dtype=np.float32
        )

        self.random_ofs_on_reset = random_ofs_on_reset

    def reset(self):
        self._instrument = np.random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]

        if self.random_ofs_on_reset:
            offset = np.random.choice(prices.high.shape[0]-self._state.bars_count*10) + self._state.bars_count
        else:
            offset = self._state.bars_count

        print("目前步數:", offset)

        self._state.reset(prices, offset)

        return self._state.encode()

    def step(self, marketpostion:int, percentage:float):
        reward, done = self._state.step(marketpostion,percentage)  # 這邊會更新步數
        obs = self._state.encode()  # 呼叫這裡的時候就會取得新的狀態

        info = {
            "instrument": self._instrument,
            "offset": self._state._offset,
            "postion": float(self._state.have_position),
        }

        return obs, reward, done, info