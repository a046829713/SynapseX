import time
import gymnasium as gym
import numpy as np
from Brain.Common.DataFeature import OriginalDataFrature, Prices
import pandas as pd
from abc import ABC, abstractmethod
from Brain.Common.env_components import State_time_step_template
from Brain.DQN.lib.reward import Actions, Reward, RewardHelp, DSR_Calculator , RelativeDSR_Calculator


class State_time_step(State_time_step_template):
    def __init__(
        self,
        bars_count,
        commission_perc,
        model_train,
        default_slippage,
        N_steps,
        win_payoff_weight = None,
        dsr_window=100,  # DSR 窗口
        dsr_weight=1
    ):
        super().__init__(
            bars_count=bars_count,
            commission_perc=commission_perc,
            model_train=model_train,
            default_slippage=default_slippage,
            N_steps = N_steps,          
        )
        
        self.reward_help = RewardHelp()
        self.reward_function = Reward()
        
        # ★ 修改點 1: 初始化 Relative DSR 計算器 (移除 risk_free_rate)
        self.dsr_calc = RelativeDSR_Calculator(window=dsr_window)
        
        self.current_step = 0
        self.annealing_steps = 500000
        self.max_commission = commission_perc
        self.max_default_slippage = default_slippage
        self.beforeBar = 50
        self.dsr_weight = dsr_weight

    def _get_current_commission(self):
        if self.current_step >= self.annealing_steps:
            return self.max_commission
        return self.max_commission * (self.current_step / self.annealing_steps)

    def _get_current_default_slippage(self):
        if self.current_step >= self.annealing_steps:
            return self.max_default_slippage
        return self.max_default_slippage * (self.current_step / self.annealing_steps)
    
    def reset(self, prices:Prices, offset):
        assert offset >= self.bars_count - 1

        self.canusecash = 1.0
        self.dsr_calc.reset()
        self.open_price = 0.0
        self.trade_bar = 0 
        self.have_position = False
        self._prices = prices
        self._offset = offset
        self.game_steps = 0 
        self.cost_sum = 0.0
        self.closecash = 0.0
        self.bar_dont_change_count = 0
        self.pre_open_diff = 0.0

    def step(self, action: Actions):
        assert isinstance(action, Actions)
        self.current_step += 1
        reward = 0.0 
        done = False        
        time_cost = 0.0

        # 獲取當前價格 (P_t) 和 上一根價格 (P_{t-1})
        close = self._prices.close[self._offset]

        # 用於計算 Benchmark Return
        prev_close = self._prices.close[self._offset - 1] 
        

        
        
        # 獲取上一步的總淨值
        previous_equity = self.canusecash 
        

        # 1. 計算規則懲罰
        wrongTrade_reward = self.reward_function.wrongTrade(
             self.have_position, action=action
        )
        # print("錯誤交易獎勵值:",wrongTrade_reward)
        reward += wrongTrade_reward 



        # 3. 計算平倉損益
        closecash_diff = self.reward_help.CaculateCloseProfit(
            self.have_position,
            action=action,
            openPrice=self.open_price,
            default_slippage=self.max_default_slippage,
            closePrcie=close,
        )



        # 4. 更新開倉價格
        self.open_price = self.reward_help.CaculateOpenPrcie(
            self.open_price,
            self.have_position,
            action=action,
            default_slippage=self.max_default_slippage,
            closePrcie=close,
        )


        # 5. 預先計算「動作後」的持倉狀態
        next_have_position = self.reward_help.CaculatePostion(
            self.have_position, action=action
        )



        # 6. 計算交易成本
        current_step_cost = self.reward_help.CaculateCost(
            havePostion=self.have_position, action=action, cost=self.max_commission
        )

        self.cost_sum += current_step_cost
        self.closecash += closecash_diff


        # 7. 計算浮動損益 (基於 next_have_position)
        opencash_diff = self.reward_help.CaculateOpenProfit(
            next_have_position, 
            action=action,
            closePrice=close,
            OpenPrice=self.open_price,
        )


        



        # 8. 更新統計數據
        self.trade_bar = self.reward_help.Caculatetrade_bar(
            self.trade_bar, self.have_position, action=action
        )

        # 9. 正式更新持倉狀態
        self.have_position = next_have_position


        # 10. 計算當前的總淨值 (Equity)
        current_equity = 1.0 - self.cost_sum + self.closecash + opencash_diff
        self.canusecash = current_equity 

        
        portfolio_return_rt = current_equity - previous_equity            
        benchmark_return_rt = (close / prev_close) - 1.0

        # C. 傳入兩個回報率給計算器
        dsr_reward = self.dsr_calc.step(portfolio_return_rt, benchmark_return_rt)
        
        reward = reward + (
            #  current_equity - previous_equity
            + self.dsr_weight * dsr_reward 
        )


        # --- 11. 更新步數與結束判斷 ---
        self._offset += 1
        self.game_steps += 1
        done |= self._offset >= self._prices.close.shape[0] - 1

        if self.game_steps == self.N_steps and self.model_train:
            done = True       

        return reward, done
    
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
        self.unique_symbols = self.config.unique_symbols
        
        # 根據模式決定數據類型和佣金
        self.data_type_name = "train_data"


        state_params = {
            "bars_count": self.config.bars_count,
            "commission_perc": self.config.model_default_commission_perc_traing,
            "model_train": True,
            "default_slippage": self.config.default_slippage,
            "N_steps": self.config.n_steps,
            "win_payoff_weight": self.config.win_payoff_weight,
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

    def getModelBase_feature(self):
        atr_Volatility = self._state._prices.atr_Volatility[self._state._offset]
        return atr_Volatility


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
