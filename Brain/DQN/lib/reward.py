import numpy as np
import enum
from typing import Optional

class Actions(enum.Enum):
    Hold = 0
    Buy = 1
    Sell = 2

class DSR_Calculator:
    """
    計算微分夏普比率 (Differential Sharpe Ratio, DSR)
    
    基於 John Moody & Matthew Saffell (2001) 的論文。
    使用指數移動平均 (EMA) 來即時更新報酬的統計數據。
    """
    def __init__(self, window: int, risk_free_rate: float = 0.0, epsilon: float = 1e-9):
        """
        初始化計算器
        
        Args:
            window (int): 用於計算 EMA 的窗口大小 (例如 100, 250)。
                         這相當於論文中的學習率 eta = 1 / window。
            risk_free_rate (float): 無風險利率 (例如，日化的美國國債利率)。
                                    對於日內交易或加密貨幣，通常設為 0。
            epsilon (float): 一個極小值，用於防止除以零。
        """
        self.eta = 1.0 / window  # EMA 的學習率
        self.risk_free_rate = risk_free_rate
        self.epsilon = epsilon
        
        # A_t: 報酬率 r_t 的指數移動平均 (EMA)
        self.A = 0.0
        # B_t: 報酬率平方 r_t^2 的指數移動平均 (EMA)
        self.B = 0.0
        
        self.t = 0 # 記錄時間步，用於 "warm-up"
        self.window = window

    def reset(self):
        """重置內部狀態"""
        self.A = 0.0
        self.B = 0.0
        self.t = 0
        
    def _update_moments(self, r_t: float):
        """
        使用當前的報酬率 r_t 來更新 A 和 B
            A_t = A_{t-1} + eta * (r_t - A_{t-1})
            B_t = B_{t-1} + eta * (r_t^2 - B_{t-1})
        """
        self.A += self.eta * (r_t - self.A)
        self.B += self.eta * (r_t**2 - self.B)

    def step(self, r_t: float) -> float:
        """
        接收一個時間步的報酬率 r_t，返回 DSR 獎勵。
        
        Args:
            r_t (float): 單一步驟的報酬率 (Portfolio Return)，
                         即 (current_value / previous_value) - 1。
                         
        Returns:
            float: DSR 獎勵值。

            
        DSR math formula:
            $$
            D_t = \frac{(B_{t-1} \cdot \Delta A_t) - (\frac{1}{2} A_{t-1} \cdot \Delta B_t)}{(B_{t-1} - A_{t-1}^2)^{3/2}}
            $$

        We implement Numerically Stable Version.
        """
        self.t += 1
        
        # 處理無風險利率 (r_t' = r_t - R_f)
        r_t_adj = r_t - self.risk_free_rate
        
        # --- Warm-up 處理 ---
        # 在 DSR 的計算初期 (例如前 100 步)，A 和 B 的估計非常不準確，
        # DSR 獎勵可能會是極大的 NaN 或 Inf，導致網路爆炸。
        # 我們需要在 warm-up 期間回傳一個穩定的獎勵值。
        if self.t < self.window:
             # 在 warm-up 期間，返回調整後的報酬率作為獎勵
             # 這樣 Agent 至少知道要往正報酬的方向走

            # 2. 更新 A 和 B (為 t 時刻做準備)
            # 注意：我們使用 r_t_adj 來更新，這樣 A 和 B 會變成超額報酬的統計量
            self._update_moments(r_t_adj)
            return r_t_adj
        
        # 1. 計算 DSR 獎勵 (使用 t-1 時刻的統計數據 A 和 B)
        # 論文中的公式 (式 13)： R_D(t) = (B_{t-1} * d/dA - A_{t-1}/2 * d/dB) * dr_t
        # 簡化後的實作公式：
        # reward = ( (A_{t-1} - r_t_adj) * B_{t-1} - 0.5 * A_{t-1} * (A_{t-1}^2 - r_t_adj^2) ) / ( (B_{t-1} - A_{t-1}^2)**1.5 + epsilon )
        
        # 為了數值穩定性，我們使用另一種等價且更穩定的形式
        std_dev = np.sqrt(self.B - self.A**2)
        
        if std_dev < self.epsilon:
            # 如果標準差接近於 0 (例如剛開始或回報恆定)，我們無法計算 DSR
            # 在此 "warm-up" 期間，可以給予 0 或簡單的 r_t 作為獎勵
            reward = 0.0
        else:
            sharpe_t_minus_1 = self.A / (std_dev + self.epsilon)
            
            # DSR 的核心公式
            term_1 = (r_t_adj - self.A) / (std_dev + self.epsilon)
            term_2 = (sharpe_t_minus_1 * (r_t_adj**2 - self.B)) / (2 * (std_dev**2 + self.epsilon))
            
            reward = (self.eta / (1.0 - self.eta)) * (term_1 - term_2) # 這裡的 eta  scaling 是為了匹配 EMA 的特性
            
        # ★修改點 2: 防躺平機制 (Anti-Slacker Logic) ★
        # 如果歷史績效是負的 (A < 0)，且當前這一步幾乎沒賺錢 (r_t 接近 0)
        # 數學公式會給出正獎勵 (0 - 負數 = 正數)，我們要手動修正這個 Bug。
        if self.A < -1e-5 and abs(r_t_adj) < 1e-6:
            # 強制將獎勵歸零，甚至給予極小的懲罰
            # 告訴 Agent: "雖然你沒賠錢，但因為你之前賠太慘了，現在不動是不給分的"
            reward = 0.0 
            
            # 或者更激進一點，給予回撤期間的時間懲罰：
            # reward = -0.0001

        # 更新統計量 (為下一步做準備)
        self._update_moments(r_t_adj)

        return np.clip(reward, -1.0, 1.0)
        
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
        if not (havePostion) and action == Actions.Buy:
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

        if havePostion and (action == Actions.Buy or action == Actions.Hold):
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