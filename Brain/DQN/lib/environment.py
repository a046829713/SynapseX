import gym
import gym.spaces
import enum
import numpy as np
import time


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
        self.cost_sum = 0.0
        self.closecash = 0.0
        self.canusecash = 1.0
        self.game_steps = 0  # 這次遊戲進行了多久

        # 用來記錄各個時間的
        self.diff_percent = 0.0

    @property
    def shape(self):
        return (3*self.bars_count + 1 + 1, )

    def encode(self):
        """
        Convert current state into numpy array.

        用來製作state 一維狀態的函數

        return res:
            [ 0.01220753 -0.00508647 -0.00508647  0.00204918 -0.0204918  -0.0204918
            0.01781971 -0.00419287 -0.00419287  0.         -0.0168421  -0.00736842
            0.01359833 -0.0041841   0.00732218  0.00314795 -0.00629591 -0.00314795
            0.00634249 -0.00422833 -0.00317125  0.01800847  0.          0.01800847
            0.01155462 -0.00315126  0.00945378  0.0096463  -0.00214362  0.0096463
            0.          0.        ]

            # 倒數第二個0 為部位
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)

        shift = 0

        # 我認為這邊有一些問題,為甚麼要從1開始,而不從0開始呢?
        # 1-10
        for bar_idx in range(-self.bars_count+1, 1):
            res[shift] = self._prices.high[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.low[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.close[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.volume[self._offset + bar_idx]
            shift += 1

        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = (self._cur_close() - self.open_price) / \
                self.open_price

        return res

    def _cur_close(self):
        """
        Calculate real close price for the current bar

        # 為甚麼會這樣寫的原因是因為 透過rel_close 紀錄的和open price 的差距(百分比)來取得真實的收盤價
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)

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

        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            # 記錄開盤價
            self.open_price = close * (1 + self.default_slippage)
            cost = -self.commission_perc
            reward += 0.001  # 可以考虑动态调整或基于条件的奖励

        elif action == Actions.Sell and self.have_position:
            cost = -self.commission_perc
            self.have_position = False
            # 計算出賣掉的資產變化率,並且累加起來
            closecash_diff = (
                close * (1 - self.default_slippage) - self.open_price) / self.open_price
            self.open_price = 0.0
            opencash_diff = 0.0
            reward += 0.001  # 可以考虑动态调整或基于条件的奖励

        # 原始獎勵設計
        # reward += cost + closecash_diff + opencash_diff
        self.cost_sum += cost
        self.closecash += closecash_diff
        last_canusecash = self.canusecash
        # 累積的概念為? 淨值 = 起始資金 + 手續費 +  已平倉損益 + 未平倉損益
        self.canusecash = 1.0 + self.cost_sum + self.closecash + opencash_diff
        reward += self.canusecash - last_canusecash

        # 未平倉價格距離百分比
        self.diff_percent = 0.0 if self.open_price == 0.0 else (
            self._cur_close() - self.open_price) / self.open_price
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

        return reward, done,


class State_time_step(State):
    """
    """
    @property
    def shape(self):
        return (self.bars_count, 6)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count
        for bar_idx in range(self.bars_count):
            res[bar_idx][0] = self._prices.high[self._offset - ofs + bar_idx]
            res[bar_idx][1] = self._prices.low[self._offset - ofs + bar_idx]
            res[bar_idx][2] = self._prices.close[self._offset - ofs + bar_idx]
            res[bar_idx][3] = self._prices.volume[self._offset - ofs + bar_idx]

        if self.have_position:
            res[:, 4] = 1.0
            res[:, 5] = (self._cur_close() - self.open_price) / \
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
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, state, random_ofs_on_reset):

        self._prices = prices
        self._state = state
        self.action_space = gym.spaces.Discrete(n=len(Actions))

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)

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
            "diff_percent": self._state.diff_percent
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
