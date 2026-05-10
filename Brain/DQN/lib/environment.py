import time
import gymnasium as gym
import numpy as np
from Brain.Common.DataFeature import OriginalDataFrature, Prices
import pandas as pd
from abc import ABC, abstractmethod
from Brain.Common.env_components import State_time_step_template
from Brain.DQN.lib.reward import Actions, Reward, RewardHelp, DSR_Calculator , RelativeDSR_Calculator, RelativeSortino_Calculator,Window_RelativeSortino_Calculator
from collections import deque





class State_time_step(State_time_step_template):
    def __init__(
        self,
        bars_count,
        commission_perc,
        model_train,
        default_slippage,
        N_steps,
        
    ):
        """
            base on Risk-Aware Reinforcement Learning Reward for Financial Trading

        Args:
            bars_count (_type_): _description_
            commission_perc (_type_): _description_
            model_train (_type_): _description_
            default_slippage (_type_): _description_
            N_steps (_type_): _description_
        """

        super().__init__(
            bars_count=bars_count,
            commission_perc=commission_perc,
            model_train=model_train,
            default_slippage=default_slippage,
            N_steps = N_steps,          
        )

        self.max_default_slippage = default_slippage
        self.max_commission = commission_perc
        self.reward_function = Reward()
        self.reward_help = RewardHelp()

        self.lookbackDays = 30

        self.weights = {
            'w1_ann_return': 0.4,  # 年化報酬權重
            'w2_down_risk': 0.1,   # 下行風險權重 (懲罰項)
            'w3_diff_return': 0.35, # 差異報酬權重
            'w4_treynor': 0.15      # 崔諾指標權重
        }
        self.risk_free_rate = 0.0

        self.portfolio_history_returns = deque(maxlen=self.lookbackDays)
        self.benchmark_returns = deque(maxlen=self.lookbackDays)

    def calculate_geometric_annualized_return(self, daily_returns):
        """
        計算精確的幾何年化報酬率（考慮複利）
        :param daily_returns: 一維的 numpy array 或 list，包含這段期間每天的報酬率 (例如 0.01 代表 1%)
        """
        if len(daily_returns) == 0:
            return 0.0
        
        t_days = len(daily_returns)
        # 將每天的報酬率加上 1，然後全部乘起來 (計算累積淨值)
        cumulative_return = np.prod(1 + np.array(daily_returns))
        
        # 開次方並年化
        ann_return = (cumulative_return ** (252 / t_days)) - 1
        
        return ann_return



    def calculate_downside_risk_numpy(self,returns):
        """
        計算投資組合的下行風險 (Downside Risk)
        
        參數:
        returns (np.ndarray): 包含一段時間內報酬率的陣列，例如 [0.05, -0.02, -0.05, 0.01]
        
        回傳:
        float: 下行風險值
        """
        # 步驟 1: 實作 max(0, -R_{p,t})
        # np.maximum 會逐一比較陣列元素與 0，只保留正數（也就是虧損的部分，因為前面加了負號）
        downside_diff = np.maximum(0, - np.array(returns))


        # 步驟 2: 將取出的虧損數值進行平方
        squared_downside = downside_diff ** 2
        
        # 步驟 3: 加總平均後開根號 (Root Mean Square)
        downside_risk = np.sqrt(np.mean(squared_downside))
        
        return downside_risk


    def calculate_step_differential_return(self, current_p_return, current_b_return, min_beta=0.3):
        """
            在 step() 裡面呼叫這個函數，傳入「當下這一步」的報酬
            計算簡化差異報酬 (Simplified Differential Return)
        """
        # 2. 防呆機制：如果歷史資料少於 2 筆，無法計算 Beta
        if len(self.portfolio_history_returns) < 2:
            # 資料不足時，退化成最簡單的「絕對超額報酬」，不除以 Beta
            return current_p_return - current_b_return, 1
            
        # 3. 資料足夠，轉成 NumPy 陣列準備計算
        p_returns = np.array(self.portfolio_history_returns)
        b_returns = np.array(self.benchmark_returns)
        
        # 4. 防呆機制：檢查大盤是否完全沒有波動 (例如連續幾天假日或停牌)
        benchmark_variance = np.var(b_returns, ddof=1)
        if benchmark_variance < 1e-8:
            # 大盤沒波動，Beta 預設為 1
            beta_p = 1.0
        else:
            # 計算共變異數矩陣
            cov_matrix = np.cov(p_returns, b_returns)
            covariance = cov_matrix[0, 1]
            beta_p = covariance / benchmark_variance
            
        # 5. 限制 Beta 範圍，防止除以極小值導致獎勵爆炸
        beta_p = np.clip(beta_p, a_min=min_beta, a_max=3.0)
        
        # 6. 計算這一段窗格內的平均報酬
        mu_p = np.mean(p_returns)
        mu_b = np.mean(b_returns)
        
        # 7. 回傳計算好的差異報酬
        differential_return = (mu_p - mu_b) / beta_p
        
        

        
        return differential_return, beta_p

    def reset(self, prices:Prices, offset):
        assert offset >= self.bars_count - 1
        self._prices = prices
        self.have_position = False
        self.portfolio_history_returns.clear()
        self.benchmark_returns.clear()
        self.canusecash = 1.0
        self._offset = offset 
        self.open_price = 0.0
        self.game_steps = 0 
        self.closecash = 0.0
        self.cost_sum = 0.0
        self.trade_bar = 0
        self.TotalPortfolioPercent = 1.0
        #         self.bar_dont_change_count = (
        #             0  # 計算K棒之間轉換過了多久 感覺下一次實驗也可以將這個部份加入
        #         )

    def step(self, action: Actions):
        """
            step return: we need the every step return


            起始資金 (100 %) + 已平倉損益 + 未平倉損益 - 手許費用

            Rate of Return
        """
        assert isinstance(action, Actions)
        
        reward = 0.0 
        done = False        


        _close_price = self._prices.close[self._offset]
        prev_close = self._prices.close[self._offset - 1]

        self.benchmark_returns.append((_close_price - prev_close) / prev_close)


        # # 獲取上一步的總淨值
        previous_PortfolioPercent = self.TotalPortfolioPercent 


        # 3. 計算平倉損益 （不包含交易稅）
        closecash_diff = self.reward_help.CaculateCloseProfit(
            self.have_position,
            action=action,
            openPrice=self.open_price,
            default_slippage=self.max_default_slippage,
            closePrcie=_close_price,
        )

        # 4. 更新開倉價格
        self.open_price = self.reward_help.CaculateOpenPrcie(
            self.open_price,
            self.have_position,
            action=action,
            default_slippage=self.max_default_slippage,
            closePrcie=_close_price,
        )

        # 5. 預先計算「動作後」的持倉狀態
        next_have_position = self.reward_help.CaculatePostion(
            self.have_position, action=action
        )


        # # 6. 計算交易成本
        current_step_cost = self.reward_help.CaculateCost(
            havePostion=self.have_position, action=action, cost=self.max_commission
        )

        self.cost_sum += current_step_cost
        self.closecash += closecash_diff


        # 7. 計算浮動損益 (基於 next_have_position)
        opencash_diff = self.reward_help.CaculateOpenProfit(
            next_have_position, 
            action=action,
            closePrice=_close_price,
            OpenPrice=self.open_price,
        )

        # 8. 更新統計數據
        self.trade_bar = self.reward_help.Caculatetrade_bar(
            self.trade_bar, self.have_position, action=action
        )

        # 9. 正式更新持倉狀態
        self.have_position = next_have_position


        # 10. 計算當前的總淨值 (Equity)
        self.TotalPortfolioPercent = 1.0 - self.cost_sum + self.closecash + opencash_diff 


        self.portfolio_history_returns.append(self.TotalPortfolioPercent - previous_PortfolioPercent)



        # 獎勵計算=================================================================
        # 1. 計算規則懲罰
        wrongTrade_reward = self.reward_function.wrongTrade(
             self.have_position, action=action
        )
        # print("錯誤交易獎勵值:",wrongTrade_reward)
        reward += wrongTrade_reward 





        # Annualized Return
        
        AnnualizedReturn = self.calculate_geometric_annualized_return(self.portfolio_history_returns)
        # print("年化報仇率獎勵：",AnnualizedReturn)


        # Downside Risk
        DownsideRisk = self.calculate_downside_risk_numpy(self.portfolio_history_returns)
        # print("下行風險獎勵：",DownsideRisk)


        # Differential Return
        DifferentialReturn, beta_p = self.calculate_step_differential_return(current_p_return=self.TotalPortfolioPercent - previous_PortfolioPercent, current_b_return=(_close_price - prev_close) / prev_close)
        # print("大盤差異性獎勵：",DifferentialReturn)



        # 指標 4: 崔諾指標 (Treynor Ratio)
        treynor = (AnnualizedReturn - self.risk_free_rate) / beta_p

        # print("Treynor 指標獎勵：",treynor)





        # --- 11. 更新步數與結束判斷 ---
        self._offset += 1
        self.game_steps += 1
        done |= self._offset >= self._prices.close.shape[0] - 1

        if self.game_steps == self.N_steps and self.model_train:
            done = True     



        # ==========================================
        # 1. 內部數值對齊 (Internal Scaling)
        # 把所有指標強制拉平到相似的量級，避免單一指標綁架神經網路
        # 這些係數是根據你跑出來的資料分佈設定的固定值
        # ==========================================
        scale_ann_return = 1.0
        scale_down_risk  = 100.0   # 放大 100 倍
        scale_diff_return = 100.0  # 放大 100 倍
        scale_treynor    = 0.3     # 縮小為 30%

        norm_ann_return = AnnualizedReturn * scale_ann_return
        norm_down_risk  = DownsideRisk * scale_down_risk
        norm_diff_return = DifferentialReturn * scale_diff_return
        norm_treynor    = treynor * scale_treynor

        # 實務上建議你可以把對齊後的數字印出來看一下，確認它們是不是都在 0.1 到 1.0 的區間：
        # print(f"對齊後 -> 年化:{norm_ann_return:.3f} | 風險:{norm_down_risk:.3f} | 差異:{norm_diff_return:.3f} | 崔諾:{norm_treynor:.3f}")

        # ==========================================
        # 2. 組合最終獎勵函數 (乘上你設定的固定權重)
        # ==========================================
        composite_reward = (
            self.weights['w1_ann_return'] * norm_ann_return
            - self.weights['w2_down_risk'] * norm_down_risk
            + self.weights['w3_diff_return'] * norm_diff_return
            + self.weights['w4_treynor'] * norm_treynor
        )

        # ==========================================
        # 3. 總獎勵壓縮 (Reward Clipping / Global Scaling)
        # 防止 Episode 走 1000 步累積幾千分，導致模型 Loss 爆炸
        # 將每一步的綜合獎勵壓縮到較小的區間
        # ==========================================
        global_reward_scaling = 0.01  # 你可以依據實驗結果微調這個數字 (例如 0.05 或 0.001)

        final_step_reward = composite_reward * global_reward_scaling
        # print("最後獎勵值：",final_step_reward)
        # print("*"*120)
        reward += final_step_reward

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
        self.unique_symbols = self.config.UNIQUE_SYMBOLS
        # 根據模式決定數據類型和佣金
        self.data_type_name = "train_data"
        random_symbol = np.random.choice(self.unique_symbols)

        state_params = {
            "bars_count": self.config.BARS_COUNT,
            "commission_perc": self.config.MODEL_DEFAULT_COMMISSION_PERC_TRAING,
            "model_train": True,
            "default_slippage": self.config.DEFAULT_SLIPPAGE,
            "N_steps": self.config.N_STEPS,
            
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
