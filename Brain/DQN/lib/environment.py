import enum
import time
import gymnasium as gym
import numpy as np
import collections


class Actions(enum.Enum):
    Close = 0
    Buy = 1
    Sell = 2


class State:
    def __init__(self,init_prices:collections.namedtuple, bars_count, commission_perc, model_train, default_slippage):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0

        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.N_steps = 1000  # 這遊戲目前使用多少步學習
        self.model_train = model_train
        self.default_slippage = default_slippage

        self.build_fileds(init_prices)

    def build_fileds(self,init_prices):
        # "open"要記得拿掉
        self.field_names =list(init_prices._fields)
        self.field_names.remove("open")

    def reset(self, prices, offset):
        assert offset >= self.bars_count-1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset        
        self.game_steps = 0  # 這次遊戲進行了多久
        self.bar_dont_change_count = 0 # 計算K棒之間轉換過了多久

        self.cost_sum = 0.0
        self.closecash = 0.0
        self.canusecash = 1.0

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
        """
            重新設計
            最佳動作空間探索的獎勵函數

            "找尋和之前所累積的獎勵之差距"

        Args:
            action (_type_): _description_
        """
        assert isinstance(action, Actions)

        reward = 0.0
        done = False
        close = self._cur_close()
        # 以平倉損益每局從新歸零
        closecash_diff = 0.0
        # 未平倉損益
        opencash_diff = 0.0
        # 手續費
        cost = 0.0

        # 第一根買的時候不計算未平倉損益
        if self.have_position:
            opencash_diff = (close - self.open_price) / self.open_price

        if action == Actions.Buy:
            if not self.have_position:
                self.have_position = True
                # 記錄開盤價
                self.open_price = close * (1 + self.default_slippage)
                cost = -self.commission_perc
                reward += 0.001  # 可以考虑动态调整或基于条件的奖励
            else:
                # 已持有仓位，重复买入（可能是错误行为）
                reward = -0.01  # 给予小的惩罚，鼓励合理交易
                
        elif action == Actions.Sell:
            if self.have_position:
                cost = -self.commission_perc
                self.have_position = False
                # 計算出賣掉的資產變化率,並且累加起來
                closecash_diff = (
                    close * (1 - self.default_slippage) - self.open_price) / self.open_price
                self.open_price = 0.0
                opencash_diff = 0.0
                reward += 0.001  # 可以考虑动态调整或基于条件的奖励
            else:
                # 空仓卖出（可能是错误行为）
                reward = -0.01  # 给予小的惩罚
                
        self.cost_sum += cost
        self.closecash += closecash_diff
        last_canusecash = self.canusecash
        # 累積的概念為? 淨值 = 起始資金 + 手續費 +  已平倉損益 + 未平倉損益
        self.canusecash = 1.0 + self.cost_sum + self.closecash + opencash_diff
        reward += self.canusecash - last_canusecash
        
        # 新獎勵設計
        # print("目前部位",self.have_position,"單次手續費:",cost,"單次已平倉損益:",closecash_diff,"單次未平倉損益:", opencash_diff)
        # print("目前動作:",action,"總資金:",self.canusecash,"手續費用累積:",self.cost_sum,"累積已平倉損益:",self.closecash,"獎勵差:",reward)
        # print('*'*120)

        # 上一個時步的狀態 ================================

        self._offset += 1
        self.game_steps += 1  # 本次遊戲次數
        # 判斷遊戲是否結束
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
        return (self.bars_count, len(self.field_names) + 2 )

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)

        ofs = self.bars_count
        for bar_idx in range(self.bars_count):
            for idx, field in enumerate(self.field_names):  # 編碼所有字段
                res[bar_idx][idx] = getattr(self._prices, field)[self._offset - ofs + bar_idx]

        if self.have_position:
            res[:, len(self.field_names) ] = 1.0
            res[:, len(self.field_names) + 1 ] = (self._cur_close() - self.open_price) / \
                self.open_price
            
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
