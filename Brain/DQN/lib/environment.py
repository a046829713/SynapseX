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
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0

        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.N_steps = 1000  # 這遊戲目前使用多少步學習
        self.model_train = model_train
        self.default_slippage = default_slippage
        self.win_payoff_weight = 0.1
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

    def step(self, action):
        """
            重新設計
            最佳動作空間探索的獎勵函數

            "找尋和之前所累積的獎勵之差距"

            '我認為可以將持倉後的K棒數量也讓智能體知道'

        Args:
            action (_type_): _description_
        """
        assert isinstance(action, Actions)

        reward = 0.0
        done = False
        close = self._prices.close[self._offset]
        # 以平倉損益每局從新歸零
        closecash_diff = 0.0
        # 未平倉損益
        opencash_diff = 0.0
        # 手續費
        cost = 0.0

        # 第一根買的時候不計算未平倉損益
        if self.have_position:
            opencash_diff = (close - self.open_price) / self.open_price
            self.trade_bar +=1

        if action == Actions.Buy:
            if not self.have_position:
                self.have_position = True
                # 記錄開盤價
                self.open_price = close * (1 + self.default_slippage)
                cost = -self.commission_perc
                self.trade_bar = 1
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
                
                
                # 新增：記錄每筆交易績效
                self.total_trades += 1

                really_closecash_diff = closecash_diff - 2 * self.commission_perc
                if really_closecash_diff > 0:
                    self.win_trades += 1
                    self.total_win += really_closecash_diff
                else:
                    self.total_loss += really_closecash_diff  # closecash_diff 為負數代表虧損
                
                self.open_price = 0.0
                opencash_diff = 0.0
                self.equity_peak = None
                self.trade_bar = 0 
            else:
                # 空仓卖出（可能是错误行为）
                reward = -0.01  # 给予小的惩罚

        self.cost_sum += cost
        self.closecash += closecash_diff
        last_canusecash = self.canusecash
        # 累積的概念為? 淨值 = 起始資金 + 手續費 +  已平倉損益 + 未平倉損益
        self.canusecash = 1.0 + self.cost_sum + self.closecash + opencash_diff
        reward += self.canusecash - last_canusecash

        #  不要強制智體去交易
        if self.have_position:
            if self.equity_peak is None:
                self.equity_peak = self.canusecash
            else:
                self.equity_peak = max(self.equity_peak, self.canusecash)

            # 计算标准drawdown：峰值与当前净值之间的下降比例
            current_drawdown = (self.equity_peak - self.canusecash) / self.equity_peak
            drawdown_penalty = 0.001 * current_drawdown

            reward -= drawdown_penalty
        
        # 新獎勵設計
        # print("起始損益：",self.equity_peak,"總資金:",self.canusecash)
        # print("目前部位",float(self.have_position),"單次手續費:",cost,"單次已平倉損益:",closecash_diff,"單次未平倉損益:", opencash_diff)
        # print("目前動作:",action,"總資金:",self.canusecash,"手續費用累積:",self.cost_sum,"累積已平倉損益:",self.closecash,"獎勵差:",reward)
        # print('*'*120)
        # time.sleep(10)
        # 上一個時步的狀態 ================================

        self._offset += 1
        self.game_steps += 1  # 本次遊戲次數
        # 判斷遊戲是否結束
        done |= self._offset >= self._prices.close.shape[0] - 1
        
        if self.game_steps == self.N_steps and self.model_train:
            done = True

        # 若 episode 結束，加入根據勝率與賠率計算的額外獎勵
        if done:
            if self.total_trades > 0:
                win_rate = self.win_trades / self.total_trades
                avg_win = self.total_win / self.win_trades if self.win_trades > 0 else 0.0
                num_losses = self.total_trades - self.win_trades
                avg_loss = abs(self.total_loss) / num_losses if num_losses > 0 else 0.0
                

                print("獲勝勝率：",win_rate )
                print("平均獲利：",avg_win )
                print("獲勝期望值：",(win_rate * avg_win))
                print("虧損率：",1-win_rate) 
                print("平均虧損：",avg_loss)
                print("虧損期望值：",((1-win_rate) * avg_loss))
                print("總交易次數：",self.total_trades)
                
                extra_reward = self.win_payoff_weight * self.total_trades * ((win_rate * avg_win) - ((1-win_rate) * avg_loss))
            else:
                extra_reward = 0.0

            print("特殊獎勵：",extra_reward)
            print("*"*120)
            reward += extra_reward
        

        
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
