import enum
import time
import gymnasium as gym
import numpy as np
import collections


class Actions(enum.Enum):
    Hold = 0
    Buy = 1
    Sell = 2


class State:
    def __init__(self, init_prices: collections.namedtuple, bars_count, commission_perc, model_train, default_slippage):
        assert isinstance(bars_count, int) and bars_count > 0
        assert isinstance(commission_perc, float) and commission_perc >= 0.0

        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.N_steps = 1000  # 學習步數
        self.model_train = model_train
        self.default_slippage = default_slippage

        # 以下參數可根據需求調整
        self.holding_reward_factor = 0.001    # 獲利持有時，每步獎勵
        self.holding_penalty_factor = 0.001   # 虧損持有時，按持有時間乘以虧損比例進行懲罰
        self.minimal_hold_period = 10         # 交易間至少需持有10步，否則交易會有額外懲罰
        self.frequent_trade_penalty = 0.005   # 過於頻繁的買賣懲罰

        self.build_fileds(init_prices)

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
        self.game_steps = 0
        self.cost_sum = 0.0
        self.closecash = 0.0
        self.canusecash = 1.0
        self.last_trade_step = 0  # 用來記錄上次交易的步數

    def step(self, action):
        """
        新的獎勵函數設計：
          - 買入：開倉時扣除手續費，若已持倉則輕罰
          - 平倉：若有持倉則計算平倉獲利，並根據獲利回撤狀況給予額外獎勵，
                    虧損平倉則實現損失（鼓勵及時止損）
          - 持有：當持有獲利時，給予小額正獎勵；若處於虧損狀態則根據持有時間和虧損幅度加懲罰
          - 同時對過於頻繁的買賣進行額外扣分
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._prices.close[self._offset]
        cost = 0.0

        # 計算自上次交易以來的持有時間
        time_since_trade = self.game_steps - self.last_trade_step

        if action == Actions.Buy:
            if not self.have_position:
                # 開倉動作
                self.have_position = True
                self.open_price = close * (1 + self.default_slippage)
                cost = -self.commission_perc
                reward += cost
                self.last_trade_step = self.game_steps
            else:
                # 重複買入
                reward = -0.01

        elif action == Actions.Sell:
            if self.have_position:
                cost = -self.commission_perc
                # 計算平倉獲利率
                realized_profit = (close * (1 - self.default_slippage) - self.open_price) / self.open_price
                reward += realized_profit + cost
                self.have_position = False
                self.open_price = 0.0
                self.last_trade_step = self.game_steps
            else:
                # 空倉賣出
                reward = -0.01

        elif action == Actions.Hold:
            if self.have_position:
                # 計算未平倉損益
                unrealized_profit = (close - self.open_price) / self.open_price

                if unrealized_profit >= 0:
                    # 獲利持倉，每步給予微小獎勵
                    reward += self.holding_reward_factor
                else:
                    # 虧損持倉，根據持有時間與虧損幅度施以懲罰
                    reward -= self.holding_penalty_factor * time_since_trade * abs(unrealized_profit)
            else:
                reward += 0.0  # 空倉持有無變化

        # 若過於頻繁的買賣，額外扣分
        if time_since_trade < self.minimal_hold_period and action in [Actions.Buy, Actions.Sell]:
            reward -= self.frequent_trade_penalty

        # 以未平倉損益（若有）更新資產淨值，並將淨值變化納入獎勵
        if self.have_position:
            opencash_diff = (close - self.open_price) / self.open_price
        else:
            opencash_diff = 0.0
        
        last_equity = 1.0 + self.cost_sum + self.closecash
        self.canusecash = 1.0 + self.cost_sum + self.closecash + opencash_diff
        reward += self.canusecash - last_equity

        self.cost_sum += cost

        self._offset += 1
        self.game_steps += 1

        # 判斷是否達到結束條件
        done = self._offset >= self._prices.close.shape[0] - 1 or \
               (self.game_steps == self.N_steps and self.model_train)
        
        return reward, done


class State_time_step(State):
    """
        專門用於transformer的時序函數。
    """
    @property
    def shape(self):
        return (self.bars_count, len(self.field_names) + 2)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)





        ofs = self.bars_count
        for bar_idx in range(self.bars_count):
            for idx, field in enumerate(self.field_names):  # 編碼所有字段
                res[bar_idx][idx] = getattr(self._prices, field)[
                    self._offset - ofs + bar_idx]

        if self.have_position:
            res[:, len(self.field_names)] = 1.0
            res[:, len(self.field_names) + 1] = (self._prices.close[self._offset] - self.open_price) / \
                self.open_price
        
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
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.random_ofs_on_reset = random_ofs_on_reset

    def reset(self):
        self._instrument = np.random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]

        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = np.random.choice(prices.high.shape[0]-bars*10) + bars
        else:
            offset = bars

        print("目前步數:", offset)

        self._state.reset(prices, offset)

        return self._state.encode()

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)  # 這邊會更新步數
        obs = self._state.encode()  # 呼叫這裡的時候就會取得新的狀態

        info = {
            "instrument": self._instrument,
            "offset": self._state._offset,
            "postion": float(self._state.have_position),
        }

        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def engine_info(self):
        if self._state.__class__ == State_time_step:
            return {
                "input_size": self._state.shape[1],
            }
