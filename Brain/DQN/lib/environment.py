import enum
import time
import gymnasium as gym
import numpy as np
import collections
from typing import Optional
from Brain.Common.DataFeature import OriginalDataFrature
import pandas as pd
from abc import ABC, abstractmethod
from Brain.Common.env_components import State_time_step_template


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

        self.reward_function = Reward()
        self.reward_help = RewardHelp()
        self.win_payoff_weight = win_payoff_weight

    def reset(self, prices, offset):
        assert offset >= self.bars_count - 1
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
        self.bar_dont_change_count = (
            0  # 計算K棒之間轉換過了多久 感覺下一次實驗也可以將這個部份加入
        )

        # 新增：初始化交易統計資料
        self.total_trades = 0
        self.win_trades = 0
        self.total_win = 0.0
        self.total_loss = 0.0

        self.beforeBar = 50
        self.max_profit_this_trade = 0.0

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
            self.have_position, action=action
        )

        # print("交易錯誤懲罰:",wrongTrade_reward)
        reward += wrongTrade_reward

        trendTrade_reward = self.reward_function.trendTrade(
            self.have_position,
            action=action,
            slope=self.reward_help.clip(
                (close - self._prices.close[self._offset - self.beforeBar])
                / self.beforeBar
            ),
        )
        reward += trendTrade_reward
        # print("趨勢交易獎勵值:",trendTrade_reward)

        # closecash_diff should be first update .
        closecash_diff = self.reward_help.CaculateCloseProfit(
            self.have_position,
            action=action,
            openPrice=self.open_price,
            default_slippage=self.default_slippage,
            closePrcie=close,
        )
        self.open_price = self.reward_help.CaculateOpenPrcie(
            self.open_price,
            self.have_position,
            action=action,
            default_slippage=self.default_slippage,
            closePrcie=close,
        )

        cost = self.reward_help.CaculateCost(
            havePostion=self.have_position, action=action, cost=self.commission_perc
        )

        # really_closecash_diff need isloate caculate because we need to know the finally game reward.
        close_reward, really_closecash_diff = self.reward_function.closeReturn(
            closecash_diff,
            cost=self.commission_perc,
            havePostion=self.have_position,
            action=action,
        )
        # print("平倉交易獎勵值:",close_reward)
        reward += close_reward

        opencash_diff = self.reward_help.CaculateOpenProfit(
            self.have_position,
            action=action,
            closePrice=close,
            OpenPrice=self.open_price,
        )
        self.total_trades, self.win_trades, self.total_win, self.total_loss = (
            self.reward_help.CaculateGameDoneInfo(
                self.total_trades,
                self.win_trades,
                self.total_win,
                self.total_loss,
                self.have_position,
                action,
                really_closecash_diff,
            )
        )
        self.trade_bar = self.reward_help.Caculatetrade_bar(
            self.trade_bar, self.have_position, action=action
        )
        self.equity_peak = self.reward_help.CaculateEquity_peak_before(
            self.equity_peak, self.have_position, action=action
        )

        # if markpostion change we need to change
        self.max_profit_this_trade = self.reward_help.Caculate_max_profit_this_trade(
            self.max_profit_this_trade, self.have_position, action=action
        )

        # last change make sure everything is caculate done.
        self.have_position = self.reward_help.CaculatePostion(
            self.have_position, action=action
        )

        self.cost_sum += cost
        self.closecash += closecash_diff
        last_can_use_cash = self.canusecash
        self.canusecash = 1.0 + self.cost_sum + self.closecash + opencash_diff
        # print("整體權益數:",self.canusecash)

        tradeReturn_reward = self.reward_function.tradeReturn(
            last_value=self.canusecash, previous_value=last_can_use_cash
        )
        # print("整體交易獎勵值:",tradeReturn_reward)
        reward += tradeReturn_reward

        self.equity_peak, current_drawdown = self.reward_help.CaculateEquity_peak_after(
            self.equity_peak, self.have_position, canUseCash=self.canusecash
        )

        drawdown_penalty_reward = self.reward_function.drawdown_penalty(
            current_drawdown
        )
        # print("DD 獎勵值:",drawdown_penalty_reward)
        reward += drawdown_penalty_reward

        # ===== 新增獎勵/懲罰機制 =====
        holding_shaping_reward = 0.0
        holding_shaping_weight = 0.01  # 可調參數

        if self.have_position:
            if opencash_diff > self.max_profit_this_trade:
                # 只有在利潤創下新高時才給予獎勵
                new_profit_reward = (
                    opencash_diff - self.max_profit_this_trade
                ) * holding_shaping_weight
                reward += new_profit_reward
                # 更新本次交易的最高利潤紀錄
                self.max_profit_this_trade = opencash_diff
            else:
                # 2. 【懲罰持有虧損】: 如果有浮虧，懲罰會隨持有時間加劇
                # opencash_diff 是負數, trade_bar 是正數, 所以結果是負的懲罰
                time_penalty_weight = 0.001  # 可調參數
                holding_shaping_reward = (
                    opencash_diff * self.trade_bar * time_penalty_weight
                )

        reward += holding_shaping_reward
        # =============================

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
                avg_win = (
                    self.total_win / self.win_trades if self.win_trades > 0 else 0.0
                )
                num_losses = self.total_trades - self.win_trades
                avg_loss = abs(self.total_loss) / num_losses if num_losses > 0 else 0.0
                # print("獲勝勝率：",win_rate )
                # print("平均獲利：",avg_win )
                # print("獲勝期望值：",(win_rate * avg_win))
                # print("虧損率：",1-win_rate)
                # print("平均虧損：",avg_loss)
                # print("虧損期望值：",((1-win_rate) * avg_loss))
                # print("總交易次數：",self.total_trades)
                extra_reward = (
                    self.win_payoff_weight
                    * self.total_trades
                    * ((win_rate * avg_win) - ((1 - win_rate) * avg_loss))
                )
                # print("總期望值:",extra_reward)
                # print('*'*120)
            else:
                extra_reward = 0.0

            reward += extra_reward

        # 新獎勵設計
        # print("本次總獎勵：",reward)
        # print('*'*120)
        return reward, done


class Reward:
    def __init__(self):
        self.tradeReturn_weight = 0.4
        self.closeReturn_weight = 1 - self.tradeReturn_weight
        self.wrongTrade_weight = 1
        self.trendTrade_weight = 0.5
        self.drawdown_penalty_weight = 0.01

    def tradeReturn(self, last_value: float, previous_value: float) -> float:
        """
        主要用於淨值 = 起始資金 + 手續費(累積) +  已平倉損益(累積) + 未平倉損益(單次)


        Return = last_value - previous_value.
        """
        return self.tradeReturn_weight * (last_value - previous_value)

    def closeReturn(
        self, CloseCash: float, cost: float, havePostion: bool, action: Actions
    ):
        """
        To caculate return when close the postion.

        """
        _reward = 0
        if havePostion and action == Actions.Sell:
            _reward = CloseCash - 2 * cost

        return self.closeReturn_weight * _reward, _reward

    def drawdown_penalty(self, drawdown: float) -> float:
        """
        hope Agent can learn sell quick.
        """
        _reward = -drawdown
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


class RewardHelp:
    def __init__(self):
        pass

    def CaculateCost(self, havePostion: bool, action: Actions, cost: float) -> float:
        _cost = 0.0
        if havePostion and action == Actions.Sell:
            _cost = cost
        if not (havePostion) and action == Actions.Buy:
            _cost = cost

        return -_cost

    def CaculateGameDoneInfo(
        self,
        total_trades: int,
        win_trades: int,
        total_win: float,
        total_loss: float,
        havePostion: bool,
        action: Actions,
        really_closecash_diff: float,
    ) -> int:
        if havePostion and action == Actions.Sell:
            total_trades += 1
            if really_closecash_diff > 0:
                win_trades += 1
                total_win += really_closecash_diff
            else:
                total_loss += really_closecash_diff

        return total_trades, win_trades, total_win, total_loss

    def Caculatetrade_bar(
        self, trade_bar: int, havePostion: bool, action: Actions
    ) -> int:
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

    def CaculateCloseProfit(
        self,
        havePostion: bool,
        action: Actions,
        openPrice: float,
        default_slippage: float,
        closePrcie: float,
    ) -> float:
        closecash_diff = 0.0
        if havePostion and action == Actions.Sell:
            closecash_diff = (
                closePrcie * (1 - default_slippage) - openPrice
            ) / openPrice

        return closecash_diff

    def CaculateOpenProfit(
        self, havePostion: bool, action: Actions, closePrice: float, OpenPrice: float
    ) -> float:
        opencash_diff = 0.0

        if havePostion and (action == Actions.Buy or Actions == Actions.Hold):
            opencash_diff = (closePrice - OpenPrice) / OpenPrice

        return opencash_diff

    def CaculateOpenPrcie(
        self,
        openPrice: float,
        havePostion: bool,
        action: Actions,
        default_slippage: float,
        closePrcie: float,
    ) -> float:
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

    def CaculateEquity_peak_before(
        self, equity_peak: Optional[float], havePostion: bool, action: Actions
    ) -> Optional[float]:
        if havePostion and action == Actions.Sell:
            equity_peak = None

        return equity_peak

    def CaculateEquity_peak_after(
        self, equity_peak: Optional[float], havePostion: bool, canUseCash: float
    ):
        _current_drawdown = 0.0
        if havePostion:
            equity_peak = (
                canUseCash if equity_peak is None else max(equity_peak, canUseCash)
            )

            if equity_peak > 0:
                # if have_position then caculate drawdown：峰值与当前净值之间的下降比例
                _current_drawdown = (equity_peak - canUseCash) / equity_peak

        return equity_peak, _current_drawdown

    def Caculate_max_profit_this_trade(
        self, max_profit_this_trade: float, havePostion: bool, action: Actions
    ):
        if not havePostion and action == Actions.Buy:
            return 0.0
        return max_profit_this_trade


class BaseTradingEnv(gym.Env, ABC):
    """
    交易環境的抽象基礎類別。
    包含了生產和訓練環境共享的核心邏輯，特別是 step 方法。
    """

    def __init__(self, state: State_time_step):
        """
        基礎類別的初始化。

        Args:
            state (State_time_step): 狀態管理物件。
        """
        super().__init__()
        self._state = state
        self._instrument = None  # 將由子類的 reset 方法設置
        self.action_space = gym.spaces.Discrete(n=len(Actions))

    @abstractmethod
    def reset(self):
        """
        重置環境。這是一個抽象方法，必須在子類中被實現。
        子類需要在此方法中：
        1. 選擇一個交易商品 (instrument)。
        2. 準備好該商品的價格數據 (prices)。
        3. 設置起始點 (offset)。
        4. 調用 self._state.reset()。
        5. 返回初始觀察值 (observation)。
        """
        raise NotImplementedError("子類必須實現 reset 方法")

    def step(self, action_idx: int):
        """
        執行一個時間步。這個邏輯在所有環境中都是相同的。
        """
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()

        info = {
            "instrument": self._instrument,
            "offset": self._state._offset,
            "postion": float(self._state.have_position),
        }

        return obs, reward, done, info

    def engine_info(self):
        if isinstance(self._state, State_time_step):
            return {
                "data_input_size": self._state.getStateShape()[1],
                "time_input_size": self._state.getTimeShape()[1],
                "action_space_n": self.action_space.n,
            }
        return {}


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
            "commission_perc": self.config.MODEL_DEFAULT_COMMISSION_PERC_traing,
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


class ProductionEnv(BaseTradingEnv):
    """
    用於生產（或推論）的環境。
    它在初始化時接收所有必要的、預先載入的數據。
    """

    def __init__(self, prices_data: dict, state: State_time_step):
        """
        Args:
            prices_data (dict): 一個字典，key 是商品名稱，value 是對應的價格數據。
        """
        super().__init__(state=state)
        self._prices = prices_data
        self._instrument = list(self._prices.keys())[0]

    def reset(self):
        prices = self._prices[self._instrument]
        offset = self._state.bars_count

        print(
            f"[Production] Resetting env with symbol: {self._instrument} at offset: {offset}"
        )

        self._state.reset(prices, offset)
        return self._state.encode()
