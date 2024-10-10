import enum
import time
import gymnasium as gym
import numpy as np


class Actions(enum.Enum):
    Close = 0
    Buy = 1
    Sell = 2


class State:
    def __init__(self, bars_count, commission_perc, model_train, default_slippage):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0

        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.N_steps = 1000  # 這遊戲目前使用多少步學習
        self.model_train = model_train
        self.default_slippage = default_slippage

    def reset(self, prices, offset):
        assert offset >= self.bars_count-1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset        
        self.game_steps = 0  # 這次遊戲進行了多久
        self.bar_dont_change_count = 0 # 計算K棒之間轉換過了多久

    def _cur_close(self):
        """
        Calculate real close price for the current bar

        # 為甚麼會這樣寫的原因是因為 透過rel_close 紀錄的和open price 的差距(百分比)來取得真實的收盤價
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)
    
    def count_postion_change_reward(self, action):
        # 更新 self.bar_dont_change_count
        if ((action == Actions.Buy and self.have_position) or
            (action == Actions.Sell and not self.have_position) or
            (action == Actions.Close)):
            self.bar_dont_change_count += 1
        else:
            self.bar_dont_change_count = 0

        half_steps = self.N_steps / 2
        max_reward = 0.01  # 最大奖励值，可根据需要调整
        max_penalty = -0.01  # 最大惩罚值，可根据需要调整

        if self.bar_dont_change_count == 0:
            reward = 0
        else:
            if self.bar_dont_change_count <= half_steps:
                reward = (half_steps - self.bar_dont_change_count) / half_steps * max_reward
            else:
                reward = (self.bar_dont_change_count - half_steps) / half_steps * max_penalty

        # print("改變前部位獎勵:",reward)
        return reward
    
    def trend_reward(self, window_size: int):
        if self._offset >= window_size:
            open_prices = self._prices.open[self._offset - window_size:self._offset]
            rel_close = self._prices.close[self._offset - window_size:self._offset]
            original_prices = open_prices * (1.0 + rel_close)
            # 使用线性回归斜率作为趋势
            x = np.arange(window_size)
            y = original_prices
            slope, _ = np.polyfit(x, y, 1)
            trend = slope
        else:
            trend = 0.0

        reward = 0.0
        threshold = 0.0  # 趋势判断的阈值
        base_reward = 0.0005
        base_penalty = 0.0005

        if trend > threshold and self.have_position:
            reward += base_reward * trend  # 奖励与趋势强度相关
        elif trend < -threshold and not self.have_position:
            reward += base_reward * (-trend)
        else:
            reward -= base_penalty

        return reward

    def step(self, action):        
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close()
        profit = 0.0
        position_change_reward = self.count_postion_change_reward(action)  # 调用更新后的奖励函数

        # 判断当前是否持仓
        if self.have_position:
            # 還在考慮是否要吐給神經網絡
            profit = (close - self.open_price) / self.open_price

        # 根据动作更新状态和计算奖励
        if action == Actions.Buy:
            if not self.have_position:
                # 开仓买入
                self.have_position = True
                self.open_price = close * (1 + self.default_slippage)                
                reward = -self.commission_perc  # 扣除交易成本
            else:
                # 已持有仓位，重复买入（可能是错误行为）
                reward = -0.01  # 给予小的惩罚，鼓励合理交易
        elif action == Actions.Sell:
            if self.have_position:
                # 平仓卖出
                self.have_position = False
                sell_price = close * (1 - self.default_slippage)
                profit = (sell_price - self.open_price) / self.open_price                
                reward = profit - self.commission_perc  # 盈利减去交易成本
                self.open_price = 0.0
            else:
                # 空仓卖出（可能是错误行为）
                reward = -0.01  # 给予小的惩罚
        else:
            # 持仓或空仓时的持有动作
            reward = 0.0  # 可根据需要加入持仓成本


        # 可选：惩罚持仓过久的行为
        if self.have_position:
            holding_penalty = -0.0001  # 每个时间步的小惩罚
            reward += holding_penalty

        # 添加仓位变动奖励
        reward += position_change_reward

        # 添加趋势奖励
        reward += self.trend_reward(window_size=self.bars_count)


        # 更新状态
        self._offset += 1
        self.game_steps += 1
        done |= self._offset >= self._prices.close.shape[0] - 1
        if self.game_steps == self.N_steps and self.model_train:
            done = True

        return reward, done



class State_time_step(State):
    """
        專門用於transformer的時序函數。
    """
    @property
    def shape(self):
        return (self.bars_count, 16)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)

        ofs = self.bars_count
        for bar_idx in range(self.bars_count):
            res[bar_idx][0] = self._prices.high[self._offset - ofs + bar_idx]
            res[bar_idx][1] = self._prices.low[self._offset - ofs + bar_idx]
            res[bar_idx][2] = self._prices.close[self._offset - ofs + bar_idx]
            res[bar_idx][3] = self._prices.volume[self._offset - ofs + bar_idx]
            res[bar_idx][4] = self._prices.volume2[self._offset - ofs + bar_idx]
            res[bar_idx][5] = self._prices.quote_av[self._offset - ofs + bar_idx]
            res[bar_idx][6] = self._prices.quote_av2[self._offset - ofs + bar_idx]
            res[bar_idx][7] = self._prices.trades[self._offset - ofs + bar_idx]
            res[bar_idx][8] = self._prices.trades2[self._offset - ofs + bar_idx]
            res[bar_idx][9] = self._prices.tb_base_av[self._offset - ofs + bar_idx]
            res[bar_idx][10] = self._prices.tb_base_av2[self._offset - ofs + bar_idx]
            res[bar_idx][11] = self._prices.tb_quote_av[self._offset - ofs + bar_idx]
            res[bar_idx][12] = self._prices.tb_quote_av2[self._offset - ofs + bar_idx]

        if self.have_position:
            res[:, 13] = 1.0
            res[:, 14] = (self._cur_close() - self.open_price) / \
                self.open_price

        res[:,15] = self.bar_dont_change_count
        return res


class State1D(State):
    """
        用於處理 1D 數據，如時間序列或序列數據。輸入數據的形狀通常是 (N, C, L)，其中 N 是批次大小，C 是通道數，L 是序列長度。
        典型應用：自然語言處理、時間序列分析（如語音識別、文本分類等）。
    """
    @property
    def shape(self):
        return (6, self.bars_count)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count-1
        res[0] = self._prices.high[self._offset-ofs:self._offset+1]
        res[1] = self._prices.low[self._offset-ofs:self._offset+1]
        res[2] = self._prices.close[self._offset-ofs:self._offset+1]
        res[3] = self._prices.volume[self._offset-ofs:self._offset+1]
        dst = 4
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = (self._cur_close() - self.open_price) / \
                self.open_price

        return res


class State2D(State):
    """
        用於處理 2D 數據，如圖像。輸入數據的形狀通常是 (N, C, H, W)，其中 N 是批次大小，C 是通道數，H 是高度，W 是寬度。
        典型應用：圖像處理、計算機視覺任務（如圖像分類、物體檢測等）。
    """
    @property
    def shape(self):
        return (6, self.bars_count)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count-1
        res[0] = self._prices.high[self._offset-ofs:self._offset+1]
        res[1] = self._prices.low[self._offset-ofs:self._offset+1]
        res[2] = self._prices.close[self._offset-ofs:self._offset+1]
        res[3] = self._prices.volume[self._offset-ofs:self._offset+1]
        dst = 4
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = (self._cur_close() - self.open_price) / \
                self.open_price

        res = np.expand_dims(res, 0)
        return res


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
