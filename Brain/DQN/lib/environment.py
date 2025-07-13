import enum
import time
import gymnasium as gym
import numpy as np
import collections
from typing import Optional


class Actions(enum.Enum):
    Hold = 0
    Buy = 1
    Sell = 2


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

    def closeReturn(self, CloseCash: float, cost: float, havePostion: bool, action: Actions):
        """
            To caculate return when close the postion. 

        """
        _reward = 0
        if havePostion and action == Actions.Sell:
            _reward = CloseCash - 2 * cost

        return self.closeReturn_weight * _reward, _reward

    def drawdown_penalty(self,drawdown:float) -> float:
        """
            hope Agent can learn sell quick.
        """
        _reward = - drawdown
        return self.drawdown_penalty_weight * _reward


    def wrongTrade(self, havePostion: bool, action: Actions):
        """
            When already holding a position, making an additional buy incurs a small penalty to encourage prudent trading.
        """
        _reward = 0
        if havePostion and action == Actions.Buy:
            _reward = 0.01

        elif not (havePostion) and action == Actions.Sell:
            _reward = 0.01

        return self.wrongTrade_weight * _reward * -1

    def trendTrade(self, havePostion: bool, action: Actions, slope: float):
        _reward = 0
        if not (havePostion) and action == Actions.Buy:
            _reward = slope

        return self.trendTrade_weight * _reward


class RewardHelp():
    def __init__(self):
        pass

    def CaculateCost(self, havePostion: bool, action: Actions, cost: float) -> float:
        _cost = 0.0
        if havePostion and action == Actions.Sell:
            _cost = cost
        if not (havePostion) and action == Actions.Buy:
            _cost = cost

        return -_cost

    def CaculateGameDoneInfo(self, total_trades: int, win_trades:int,total_win:float,total_loss:float, havePostion: bool, action: Actions,really_closecash_diff:float) -> int:
        if havePostion and action == Actions.Sell:
            total_trades += 1
            if really_closecash_diff > 0:
                win_trades += 1
                total_win += really_closecash_diff
            else:
                total_loss += really_closecash_diff


        # print("目前交易次數:",total_trades)
        # print("目前勝利次數:",win_trades)
        # print("目前總贏比例:",total_win)
        # print("目前總輸比例:",total_loss)
        # print('*'*120)
        # time.sleep(1)
        return total_trades, win_trades, total_win, total_loss

    def Caculatetrade_bar(self, trade_bar: int, havePostion: bool, action: Actions) -> int:
        if havePostion and (action == Actions.Buy or action == Actions.Hold):
            trade_bar += 1
        if havePostion and action == Actions.Sell:
            trade_bar = 0
        if not (havePostion) and Actions.Buy:
            trade_bar = 1

        return trade_bar

    def CaculatePostion(self, havePostion: bool, action: Actions) -> bool:
        if havePostion and action == Actions.Sell:
            havePostion = False

        if not (havePostion) and action == Actions.Buy:
            havePostion = True

        return havePostion

    def CaculateCloseProfit(self, havePostion: bool, action: Actions, openPrice: float, default_slippage: float, closePrcie: float) -> float:
        closecash_diff = 0.0
        if havePostion and action == Actions.Sell:
            closecash_diff = (
                closePrcie * (1 - default_slippage) - openPrice) / openPrice
    
        return closecash_diff

    def CaculateOpenProfit(self, havePostion: bool, action: Actions, closePrice: float, OpenPrice: float) -> float:
        opencash_diff = 0.0

        if havePostion and (action == Actions.Buy or Actions == Actions.Hold):
            opencash_diff = (closePrice - OpenPrice) / OpenPrice

        return opencash_diff

    def CaculateOpenPrcie(self, openPrice: float, havePostion: bool, action: Actions, default_slippage: float, closePrcie: float) -> float:
        if not havePostion and action == Actions.Buy:
            openPrice = closePrcie * (1 + default_slippage)

        if havePostion and action == Actions.Sell:
            openPrice = 0.0

        return openPrice

    def clip(self, inputslope: float):
        if inputslope > 0.01:
            return 0.01

        if inputslope < -0.01:
            return -0.01

        return inputslope

    def CaculateEquity_peak_before(self, equity_peak: Optional[float], havePostion: bool, action: Actions) -> Optional[float]:
        if havePostion and action == Actions.Sell:
            equity_peak = None

        return equity_peak

    def CaculateEquity_peak_after(self, equity_peak: Optional[float], havePostion: bool, canUseCash: float):
        _current_drawdown = 0.0
        if havePostion:
            equity_peak = canUseCash if equity_peak is None else max(
                equity_peak, canUseCash)

            if equity_peak > 0:
                # if have_position then caculate drawdown：峰值与当前净值之间的下降比例
                _current_drawdown = (equity_peak - canUseCash) / equity_peak

        return equity_peak, _current_drawdown


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
        self.reward_help = RewardHelp()

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
        self.trade_bar = 0  # 用來紀錄持倉多久
        self.bar_dont_change_count = 0  # 計算K棒之間轉換過了多久 感覺下一次實驗也可以將這個部份加入

        # 新增：初始化交易統計資料
        self.total_trades = 0
        self.win_trades = 0
        self.total_win = 0.0
        self.total_loss = 0.0

        self.beforeBar = 50

    def step(self, action: Actions):
        """
            重新設計
            最佳動作空間探索的獎勵函數

        Args:
            action (Actions): the action of Actions

            1. 每次交易要立刻將手續費損失傳出
            2. 持有部位的期間, 要不斷對於回徹做出懲罰
        """
        assert isinstance(action, Actions)

        reward = 0.0
        done = False
        close = self._prices.close[self._offset]

        wrongTrade_reward = self.reward_function.wrongTrade(
            self.have_position, action=action)
        
        # print("交易錯誤懲罰:",wrongTrade_reward)
        reward += wrongTrade_reward
        

        trendTrade_reward = self.reward_function.trendTrade(self.have_position, action=action, slope=self.reward_help.clip(
            (close - self._prices.close[self._offset - self.beforeBar]) / self.beforeBar))
        reward +=trendTrade_reward
        # print("趨勢交易獎勵值:",trendTrade_reward)


        # closecash_diff should be first update .
        closecash_diff = self.reward_help.CaculateCloseProfit(self.have_position, action=action, openPrice=self.open_price, default_slippage=self.default_slippage, closePrcie=close)
        self.open_price = self.reward_help.CaculateOpenPrcie(self.open_price, self.have_position, action=action, default_slippage=self.default_slippage, closePrcie=close)
        
        cost = self.reward_help.CaculateCost(havePostion=self.have_position, action=action, cost=self.commission_perc)
        

        # really_closecash_diff need isloate caculate because we need to know the finally game reward.
        close_reward, really_closecash_diff = self.reward_function.closeReturn(closecash_diff, cost=self.commission_perc, havePostion=self.have_position, action=action)
        # print("平倉交易獎勵值:",close_reward)
        reward += close_reward

        opencash_diff = self.reward_help.CaculateOpenProfit(self.have_position, action=action, closePrice=close, OpenPrice=self.open_price)
        self.total_trades, self.win_trades, self.total_win, self.total_loss = self.reward_help.CaculateGameDoneInfo(self.total_trades, self.win_trades, self.total_win, self.total_loss, self.have_position, action, really_closecash_diff)
        self.trade_bar = self.reward_help.Caculatetrade_bar(self.trade_bar, self.have_position, action=action)
        self.equity_peak = self.reward_help.CaculateEquity_peak_before(self.equity_peak, self.have_position, action=action)

        # last change make sure everything is caculate done.
        self.have_position = self.reward_help.CaculatePostion(self.have_position, action=action)

        self.cost_sum += cost
        self.closecash += closecash_diff
        last_can_use_cash = self.canusecash
        self.canusecash = 1.0 + self.cost_sum + self.closecash + opencash_diff
        # print("整體權益數:",self.canusecash)

        tradeReturn_reward =  self.reward_function.tradeReturn(last_value=self.canusecash, previous_value=last_can_use_cash)
        # print("整體交易獎勵值:",tradeReturn_reward)
        reward += tradeReturn_reward
        self.equity_peak, current_drawdown = self.reward_help.CaculateEquity_peak_after(self.equity_peak, self.have_position, canUseCash=self.canusecash)
        
        drawdown_penalty_reward =  self.reward_function.drawdown_penalty(current_drawdown)
        # print("DD 獎勵值:",drawdown_penalty_reward)
        reward += drawdown_penalty_reward

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
                avg_loss = abs(self.total_loss) / \
                    num_losses if num_losses > 0 else 0.0
                print("獲勝勝率：",win_rate )
                print("平均獲利：",avg_win )
                print("獲勝期望值：",(win_rate * avg_win))
                print("虧損率：",1-win_rate)
                print("平均虧損：",avg_loss)
                print("虧損期望值：",((1-win_rate) * avg_loss))
                print("總交易次數：",self.total_trades)
                extra_reward = self.win_payoff_weight * self.total_trades * \
                    ((win_rate * avg_win) - ((1-win_rate) * avg_loss))
                print("總期望值:",extra_reward)
                print('*'*120)
            else:
                extra_reward = 0.0

            reward += extra_reward

        # 新獎勵設計
        # print("本次總獎勵：",reward)
        # print('*'*120)
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
            res[:, len(self.field_names) + 1] = (self._prices.close[self._offset] - self.open_price) / \
                self.open_price
            res[:, len(self.field_names) + 2] = self.trade_bar

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
        fourier_features = np.zeros(
            (data.shape[0], data.shape[1] * 2), dtype=np.float32)
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

        if self.random_ofs_on_reset:
            offset = np.random.choice(
                prices.high.shape[0]-self._state.bars_count*10) + self._state.bars_count
        else:
            offset = self._state.bars_count

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
