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
    def __init__(self, init_prices: collections.namedtuple, bars_count, commission_perc, default_slippage):
        """
            A2C 演算法的N-step 會寫在外面控制.
        """
        assert isinstance(bars_count, int) and bars_count > 0
        assert isinstance(commission_perc, float) and commission_perc >= 0.0

        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.default_slippage = default_slippage

        # 以下參數可根據需求調整
        self.pullback_threshold = 0.02        # 當資產淨值回落超過2%時，認定為獲利拉回
        self.bonus_factor = 0.01              # 在拉回時賣出可獲得的額外獎勵比例
        self.holding_reward_factor = 0.001    # 獲利持有時，每步獎勵
        self.holding_penalty_factor = 0.001   # 虧損持有時，按持有時間乘以虧損比例進行懲罰
        self.minimal_hold_period = 10         # 交易間至少需持有10步，否則交易會有額外懲罰
        self.frequent_trade_penalty = 0.005   # 過於頻繁的買賣懲罰

        self.build_fileds(init_prices)

    def build_fileds(self, init_prices):
        self.field_names = list(init_prices._fields)

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
        self.equity_peak = None  # 用來追蹤持倉期間的資產淨值最高點
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
                bonus = 0.0
                # 若獲利且存在獲利拉回，額外獎勵
                if realized_profit > 0 and self.equity_peak is not None:
                    # 此處計算平倉時的近似當前淨值（未平倉獲利納入考慮）
                    current_equity = 1.0 + self.cost_sum + self.closecash + (close - self.open_price) / self.open_price
                    pullback = (self.equity_peak - current_equity) / self.equity_peak
                    if pullback > self.pullback_threshold:
                        bonus = self.bonus_factor * pullback
                
                reward += realized_profit + cost + bonus
                self.have_position = False
                self.open_price = 0.0
                self.equity_peak = None
                self.last_trade_step = self.game_steps
            else:
                # 空倉賣出
                reward = -0.01

        elif action == Actions.Hold:
            if self.have_position:
                # 計算未平倉損益
                unrealized_profit = (close - self.open_price) / self.open_price
                # 更新持倉期間的淨值峰值
                current_equity = 1.0 + self.cost_sum + self.closecash + unrealized_profit
                if self.equity_peak is None:
                    self.equity_peak = current_equity
                else:
                    self.equity_peak = max(self.equity_peak, current_equity)
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
        done = self._offset >= self._prices.close.shape[0] - 1 
        
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

        print("目前步數:", offset,"目前商品:",self._instrument)

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
