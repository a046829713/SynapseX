import enum
import time
import gymnasium as gym
import numpy as np
import collections



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
        self.build_fileds(init_prices)

    def build_fileds(self, init_prices):
        self.field_names = list(init_prices._fields)
        for i in ['open', 'high', 'low', 'close']:
            if i in self.field_names:
                self.field_names.remove(i)
        
    def reset(self, prices, offset):
        assert offset >= self.bars_count-1
        self._prices = prices
        self._offset = offset
        self.game_steps = 0  # 這次遊戲進行了多久

        self._information = {
            "action":0.0,
            "current_cash":10000.0,
            "current_postion":0.0,
            "current_postion_value":0.0,
            "current_asset":10000.0
        }
    def step(self, action):
        """
            action 是一個 [0, 1] 的浮點數，表示目標持倉比例。

            # 什麼可以用來很好評價智能體的表現？
            # 1.每次賣出所獲得獎勵
            # (暫時不實現)2.開倉以來MDD的懲罰
            # (暫時不實現)3.每1000個step 資產所獲得的最大提升
        """
        assert 0 <= action <= 1, "Action must be between 0 and 1"
        self._information['action'] = action
        reward = 0.0
        done = False
        close = self._prices.close[self._offset]
        self._information['close'] = close
        last_asset = self._information['current_asset']

        
        if action > 0:
            if self._information['current_postion'] == 0.0 :
                # 將手續費加入 
                cost = (self._information['current_cash'] * action)  # 投入的現金
                self._information['current_postion'] = cost / (close*(1 + self.commission_perc))
                # 計算部位淨值
                self._information['current_postion_value'] = self._information['current_postion'] * close 
                # 計算即時資金 
                actual_cost = self._information['current_postion'] * close * (1 + self.commission_perc)  # 包含手續費的總成本
                self._information['current_cash'] -= actual_cost
                
                # 計算剩餘總資產
                self._information['current_asset'] = self._information['current_postion_value'] + self._information['current_cash']







            elif self._information['current_postion'] > 0:
                # 計算部位變化 # 本次模型需要買到的部位
                newpostion = (self._information['current_asset'] * action) / (close*(1 + self.commission_perc))

                
                # 計算即時資金
                if newpostion > self._information['current_postion']:
                    more_cost = (newpostion - self._information['current_postion']) * (close *(1+self.commission_perc))
                    self._information['current_cash'] = self._information['current_cash'] - more_cost
                elif newpostion == self._information['current_postion']:
                    pass
                elif newpostion < self._information['current_postion']:
                    sell_money = (self._information['current_postion'] - newpostion) * (close * (1 - self.commission_perc))
                    self._information['current_cash'] = self._information['current_cash'] + sell_money

                self._information['current_postion'] = newpostion
                # 計算部位淨值
                self._information['current_postion_value'] = self._information['current_postion'] * close 
                # 計算剩餘總資產
                self._information['current_asset'] = self._information['current_postion_value'] + self._information['current_cash']
        
        elif action == 0:
            if self._information['current_postion'] == 0.0:
                pass
            elif self._information['current_postion'] > 0.0:
                # 計算即時資金
                sell_money = (self._information['current_postion']) * (close * (1 - self.commission_perc))
                self._information['current_cash'] = self._information['current_cash'] + sell_money 
                
                # 計算剩餘總資產
                self._information['current_postion'] = 0.0
                
                # 計算部位淨值
                self._information['current_postion_value'] = 0.0
                self._information['current_asset'] = self._information['current_postion_value'] + self._information['current_cash']


        self._offset += 1
        self.game_steps += 1  # 本次遊戲次數
        # 判斷遊戲是否結束
        done |= self._offset >= self._prices.close.shape[0] - 1
        
        if self.game_steps == self.N_steps and self.model_train:
            done = True



        reward = (self._information['current_asset'] - last_asset) / last_asset
        print(self._information)
        print(reward)
        print('*'*120)
        return reward, done

class State_time_step(State):
    """
        時序特徵處理。
    """
    @property
    def shape(self):
        return (self.bars_count, len(self.field_names) + 1)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)

        ofs = self.bars_count
        for bar_idx in range(self.bars_count):
            for idx, field in enumerate(self.field_names):  # 編碼所有字段
                res[bar_idx][idx] = getattr(self._prices, field)[
                    self._offset - ofs + bar_idx]


        res[:, len(self.field_names)] = self._information['current_postion']
        res[:, len(self.field_names)] = self._information['current_postion']
        res[:, len(self.field_names)] = self._information['current_postion']
        return res



class Env(gym.Env):
    def __init__(self, prices, state, random_ofs_on_reset):
        self._prices = prices
        self._state = state
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
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

    def step(self, action):
        reward, done = self._state.step(action)  # 這邊會更新步數
        obs = self._state.encode()  # 呼叫這裡的時候就會取得新的狀態

        info = {
            "instrument": self._instrument,
            "offset": self._state._offset,
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

