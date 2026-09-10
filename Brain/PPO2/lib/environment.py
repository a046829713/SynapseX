from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod
from Brain.Common.env_components import State_time_step_template
from Brain.Common.DataFeature import Prices, OriginalDataFeature


class State_time_step(State_time_step_template):
    def __init__(
        self,
        bars_count: int,
        commission_perc: float,
        model_train: bool,
        default_slippage: float,
        N_steps: int,
        deadzone_threshold: float = 0.01,
        downside_window_size: int = 60,
        w1_step_return: float = 1.0,
        w2_downside_ratio: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            bars_count=bars_count,
            commission_perc=commission_perc,
            model_train=model_train,
            default_slippage=default_slippage,
            N_steps=N_steps,
        )

        self.deadzone_threshold = deadzone_threshold
        self.downside_window_size = downside_window_size
        self.prev_downside_risk = 0.0

        self.weights = {
            "w1_step_return": w1_step_return,  # 單步報酬權重
            "w2_downside_ratio": w2_downside_ratio,  # 下行風險增量懲罰權重
        }

        # 狀態與記帳變數初始化
        self.position = 0.0
        self.open_price = 0.0
        self.trade_bar = 0
        self.TotalPortfolioPercent = 1.0
        self.cost_sum = 0.0
        self.closecash = 0.0
        self.game_steps = 0
        # 確保佇列長度至少能容納下行風險視窗大小
        self.return_history = deque(maxlen=max(N_steps, self.downside_window_size))
        self._prev_close = 0.0

    @property
    def have_position(self) -> bool:
        return abs(self.position) > 0.0

    def reset(self, prices: Prices, offset: int):
        assert offset >= self.bars_count - 1
        self._prices = prices
        self._offset = offset
        self.position = 0.0
        self.open_price = 0.0
        self.trade_bar = 0
        self.TotalPortfolioPercent = 1.0
        self.cost_sum = 0.0
        self.closecash = 0.0
        self.game_steps = 0
        self.return_history.clear()
        self.prev_downside_risk = 0.0
        self._prev_close = float(self._prices.close[self._offset])

    def calculate_downside_risk_numpy(self, returns) -> float:
        """
        計算投資組合的下行風險 (Downside Risk / Semi-Deviation)
        """
        if len(returns) == 0:
            return 0.0
        downside_diff = np.maximum(0.0, -np.asarray(returns, dtype=np.float64))
        squared_downside = downside_diff**2
        return float(np.sqrt(np.mean(squared_downside)))

    def calculate_step_downside_penalty(self) -> float:
        """
        計算單步滑動視窗下行風險增量懲罰 (Rolling Window Downside Risk Penalty)
        """
        if len(self.return_history) < 2:
            return 0.0

        rolling_slice = list(self.return_history)[-self.downside_window_size :]
        current_downside_risk = self.calculate_downside_risk_numpy(rolling_slice)
        downside_delta = max(0.0, current_downside_risk - self.prev_downside_risk)
        self.prev_downside_risk = current_downside_risk

        return float(self.weights["w2_downside_ratio"] * downside_delta)

    def step(self, action: float):
        """
        執行單步連續動作撮合、記帳、淨值與獎勵計算
        """
        # 1. 動作死區過濾
        if abs(action) <= self.deadzone_threshold:
            target_pos = 0.0
        else:
            target_pos = float(action)

        # 剛結束時段的資產收盤價
        close_price = float(self._prices.close[self._offset])
        prev_pos = self.position
        safe_open_price = max(self.open_price, 1e-8)

        # 2. 換倉量與交易摩擦成本 (手續費 + 滑價)
        delta_p = target_pos - prev_pos
        friction_rate = self.commission_perc + self.default_slippage
        cost_t = abs(delta_p) * friction_rate
        self.cost_sum += cost_t

        # 3. 部位轉移與開倉均價維護、已實現損益結算
        # Case A: 兩者皆為 0 (無持倉狀態)
        if prev_pos == 0.0 and target_pos == 0.0:
            self.trade_bar = 0
            self.open_price = 0.0

        # Case B: 由空倉建立新部位 (做多或放空)
        elif prev_pos == 0.0 and target_pos != 0.0:
            self.open_price = close_price
            self.trade_bar = 1

        # Case C: 反手翻倉 (Flip: 多翻空 或 空翻多)
        elif prev_pos * target_pos < 0.0:
            # 結算上一期全部持倉已實現損益
            closed_size = abs(prev_pos)
            if prev_pos > 0:
                realized_pnl = (
                    closed_size
                    * (close_price - safe_open_price)
                    / safe_open_price
                )
            else:
                realized_pnl = (
                    closed_size
                    * (safe_open_price - close_price)
                    / safe_open_price
                )
            self.closecash += realized_pnl

            # 以當前價格建立新方向部位
            self.open_price = close_price
            self.trade_bar = 1

        # Case D: 同向持倉調整 (加倉、減倉、完全平倉或維持不變)
        else:
            prev_size = abs(prev_pos)
            curr_size = abs(target_pos)

            if curr_size > prev_size:
                # 同向加倉：加權平均成本
                added_size = curr_size - prev_size
                self.open_price = (
                    prev_size * self.open_price + added_size * close_price
                ) / curr_size
                self.trade_bar += 1
            elif curr_size < prev_size:
                # 同向減倉 或 完全平倉：開倉均價不變，結算減倉比例損益
                reduced_size = prev_size - curr_size
                if prev_pos > 0:
                    realized_pnl = (
                        reduced_size
                        * (close_price - safe_open_price)
                        / safe_open_price
                    )
                else:
                    realized_pnl = (
                        reduced_size
                        * (safe_open_price - close_price)
                        / safe_open_price
                    )
                self.closecash += realized_pnl

                if target_pos == 0.0:
                    # 完全平倉
                    self.open_price = 0.0
                    self.trade_bar = 0
                else:
                    self.trade_bar += 1
            else:
                # 部位比例完全相同，繼續持有
                self.trade_bar += 1

        # 更新持倉比例
        self.position = target_pos

        # 4. 單步盯市報酬 (Mark-to-Market Return) 與總淨值演進
        # 報酬由持有部位經歷的價格變動減去交易摩擦成本組成，杜絕加倉時均價分母跳變失真
        safe_prev_close = max(self._prev_close, 1e-8)
        step_price_return = (close_price - safe_prev_close) / safe_prev_close
        current_p_return = prev_pos * step_price_return - cost_t

        self.TotalPortfolioPercent += current_p_return
        self.return_history.append(current_p_return)
        self._prev_close = close_price

        # 5. 下行風險懲罰與總獎勵
        downside_penalty = self.calculate_step_downside_penalty()
        reward = float(
            self.weights["w1_step_return"] * current_p_return - downside_penalty
        )

        # 6. 推進時間步與判斷終止
        self._offset += 1
        self.game_steps += 1
        done = bool(self._offset >= self._prices.close.shape[0] - 1)
        if self.game_steps >= self.N_steps and self.model_train:
            done = True

        return reward, done

    def encode(self):
        """
        觀測空間編碼覆寫：回傳 Dict 格式 {"states": data_res, "time_states": time_res}
        採用 NumPy 切片賦值向量化加速，嚴格杜絕 Look-Ahead Bias 未來價格洩漏！
        """
        data_res = np.zeros(shape=self.getStateShape(), dtype=np.float32)
        time_res = np.zeros(shape=self.getTimeShape(), dtype=np.float32)

        start = self._offset - self.bars_count
        end = self._offset

        # 向量化特徵賦值
        for idx, field in enumerate(self.info_list):
            data_res[:, idx] = getattr(self._prices, field)[start:end]

        # 特徵矩陣部位狀態與浮動盈虧編碼
        data_res[:, len(self.info_list)] = self.position

        if abs(self.position) > 0 and self.open_price > 0:
            prev_close = float(self._prices.close[self._offset - 1])
            unrealized_return = (
                self.position
                * ((prev_close - self.open_price) / max(self.open_price, 1e-8))
            )
            data_res[:, len(self.info_list) + 1] = unrealized_return
            data_res[:, len(self.info_list) + 2] = float(self.trade_bar)
        else:
            data_res[:, len(self.info_list) + 1] = 0.0
            data_res[:, len(self.info_list) + 2] = 0.0

        for idx, field in enumerate(self.timelist):
            time_res[:, idx] = getattr(self._prices, field)[start:end]

        return {"states": data_res, "time_states": time_res}


class BaseTradingEnv(gym.Env, ABC):
    """
    交易環境的抽象基礎類別。
    """

    def __init__(self, state: State_time_step):
        super().__init__()
        self._state = state
        self._instrument = None
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "states": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self._state.getStateShape(),
                    dtype=np.float32,
                ),
                "time_states": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self._state.getTimeShape(),
                    dtype=np.float32,
                ),
            }
        )

    @abstractmethod
    def reset(self):
        raise NotImplementedError("子類必須實現 reset 方法")

    def step(self, action):
        """
        執行一個時間步。
        支援 scalar, 1D/2D ndarray, tensor, list 輸入，並裁剪至 [-1.0, 1.0]。
        """
        action_val = float(np.asarray(action).squeeze())
        action_val = float(np.clip(action_val, -1.0, 1.0))
        reward, done = self._state.step(action_val)
        obs = self._state.encode()

        info = {
            "instrument": self._instrument,
            "offset": self._state._offset,
            "position": float(self._state.position),
            "postion": float(self._state.position),
        }

        return obs, reward, done, info

    def engine_info(self):
        if isinstance(self._state, State_time_step):
            return {
                "data_input_size": self._state.getStateShape()[1],
                "time_input_size": self._state.getTimeShape()[1],
                "action_space_dim": self.action_space.shape[0],
            }
        return {}


class TrainingEnv(BaseTradingEnv):
    """
    用於模型訓練的環境。
    動態加載數據，支援記憶體快取以避免重複磁碟 I/O，並在每次 reset 時隨機化起始位置（邊界安全）。
    """

    def __init__(self, config):
        self.config = config
        cfg_training = getattr(self.config, "training", self.config)
        self.unique_symbols = cfg_training.UNIQUE_SYMBOLS
        self.data_type_name = "train_data"
        self._data_cache = {}

        state_params = {
            "bars_count": cfg_training.BARS_COUNT,
            "commission_perc": getattr(
                cfg_training, "MODEL_DEFAULT_COMMISSION_PERC_TRAING", 0.0003
            ),
            "model_train": True,
            "default_slippage": getattr(cfg_training, "DEFAULT_SLIPPAGE", 0.0001),
            "N_steps": getattr(cfg_training, "N_STEPS", 250),
            "deadzone_threshold": getattr(cfg_training, "DEADZONE_THRESHOLD", 0.01),
            "downside_window_size": getattr(cfg_training, "DOWNSIDE_WINDOW_SIZE", 60),
            "w1_step_return": getattr(cfg_training, "W1_STEP_RETURN", 1.0),
            "w2_downside_ratio": getattr(cfg_training, "W2_DOWNSIDE_RATIO", 1.0),
        }

        super().__init__(state=State_time_step(**state_params))

    def _load_data_for_instrument(self, instrument: str):
        if instrument not in self._data_cache:
            self._data_cache[instrument] = (
                OriginalDataFeature().get_train_net_work_data_by_path(
                    [instrument], typeName=self.data_type_name
                )
            )
        return self._data_cache[instrument]

    def reset(self, symbol: str = None):
        if symbol is None:
            self._instrument = np.random.choice(self.unique_symbols)
        else:
            self._instrument = symbol

        all_prices = self._load_data_for_instrument(self._instrument)
        prices = all_prices[self._instrument]

        max_offset = prices.high.shape[0] - self._state.N_steps - 1
        min_offset = self._state.bars_count
        assert (
            max_offset > min_offset
        ), f"Dataset length {prices.high.shape[0]} too short for N_steps={self._state.N_steps}"
        offset = np.random.randint(min_offset, max_offset)

        print(
            f"[{self.data_type_name}] Actor resetting env with symbol: {self._instrument} at offset: {offset}"
        )

        self._state.reset(prices, offset)
        return self._state.encode()


class ProductionEnv(BaseTradingEnv):
    """
    用於生產（或推論）的環境。
    在初始化時接收預先載入的數據字典，支援按商品 reset。
    """

    def __init__(self, prices_data: dict, state: State_time_step):
        super().__init__(state=state)
        self._prices = prices_data
        self._instrument = list(self._prices.keys())[0]

    def reset(self, symbol: str = None):
        if symbol is not None:
            assert symbol in self._prices, f"Symbol {symbol} not found in preloaded prices"
            self._instrument = symbol
        prices = self._prices[self._instrument]
        offset = self._state.bars_count

        print(
            f"[Production] Resetting env with symbol: {self._instrument} at offset: {offset}"
        )

        self._state.reset(prices, offset)
        return self._state.encode()