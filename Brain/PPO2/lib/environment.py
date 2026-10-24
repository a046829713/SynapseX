import multiprocessing as mp
from typing import Optional, List, Dict, Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod
from Brain.Common.env_components import State_time_step_template
from Brain.Common.DataFeature import Prices, OriginalDataFeature
from Brain.PPO2.lib.reward import DownsidePenaltyReward


class State_time_step(State_time_step_template):
    def __init__(
        self,
        bars_count: int,
        commission_perc: float,
        model_train: bool,
        default_slippage: float,
        N_steps: int,
        deadzone_threshold: float = 0.01,
        rebalance_threshold: float = 0.03,
        w1_step_return: float = 1.0,
        vol_window: int = 96,
        risk_lambda: float = 0.5,
        reward_clip: float = 5.0,
        min_sigma: float = 1e-3,
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
        self.rebalance_threshold = rebalance_threshold

        self.weights = {
            "w1_step_return": w1_step_return,  # 單步報酬權重
        }

        # 狀態變數初始化
        self.position = 0.0
        self.open_price = 0.0
        self.trade_bar = 0
        self.game_steps = 0
        self._prev_close = 0.0

        # Per-step diagnostic accounting attributes
        self.last_step_return = 0.0
        self.last_gross_return = 0.0
        self.last_cost = 0.0
        self.last_turnover = 0.0
        self.last_traded = False

        assert min_sigma > 0.0, "min_sigma must be positive"
        self.vol_window = int(vol_window)
        self.min_sigma = float(min_sigma)
        self.reward_fn = DownsidePenaltyReward(
            risk_lambda=risk_lambda,
            clip=reward_clip,
            min_sigma=self.min_sigma,
        )

    def reset(self, prices: Prices, offset: int):
        assert offset >= self.bars_count - 1
        self._prices = prices
        self._offset = offset
        self.position = 0.0
        self.open_price = 0.0
        self.trade_bar = 0
        self.game_steps = 0
        self._prev_close = float(self._prices.close[self._offset - 1])
        self.last_step_return = 0.0
        self.last_gross_return = 0.0
        self.last_cost = 0.0
        self.last_turnover = 0.0
        self.last_traded = False

    def calculate_asset_sigma(self) -> float:
        """
        Computes ex-ante asset volatility using only completed bars prior to self._offset
        (close[offset - vol_window - 1 : offset]), strictly preventing look-ahead bias.
        """
        start = max(0, self._offset - self.vol_window - 1)
        closes = np.maximum(
            np.asarray(self._prices.close[start : self._offset], dtype=np.float64),
            1e-8,
        )

        if closes.shape[0] < 3:
            return max(1e-2, self.min_sigma)

        sigma = float(np.std(np.diff(np.log(closes))))
        return max(sigma, self.min_sigma)
    
    def step(self, action: float):
        """
        執行單步連續動作撮合、記帳、淨值與獎勵計算（Market-Open 撮合與對數報酬 Option A）
        """
        # 1. 動作死區過濾
        if abs(action) <= self.deadzone_threshold:
            candidate_pos = 0.0
        else:
            candidate_pos = float(action)

        open_price = float(self._prices.open[self._offset])
        close_price = float(self._prices.close[self._offset])
        prev_pos = self.position

        # 2. 再平衡調倉死區過濾 (抑制微幅抖動所致之手續費磨損)
        # 規則：
        # - 若為由空倉建立新部位 (prev == 0.0)、平倉退場 (candidate == 0.0) 或多空反轉翻倉 (candidate * prev < 0.0)，立即執行調倉
        # - 若為同向持倉調整且部位變化小於 rebalance_threshold，維持前一期部位 (Δp = 0, 0 手續費)
        if prev_pos == 0.0 or candidate_pos == 0.0 or (candidate_pos * prev_pos < 0.0):
            target_pos = candidate_pos
        elif abs(candidate_pos - prev_pos) < self.rebalance_threshold:
            target_pos = prev_pos
        else:
            target_pos = candidate_pos

        # 3. 換倉量與交易摩擦成本 (手續費 + 滑價，在開盤成交時扣除)
        delta_p = target_pos - prev_pos
        friction_rate = self.commission_perc + self.default_slippage
        cost_t = abs(delta_p) * friction_rate

        # 4. 部位轉移與開倉均價維護 (以 open_price 成交)
        # Case A: 兩者皆為 0 (無持倉狀態維持)
        if prev_pos == 0.0 and target_pos == 0.0:
            self.trade_bar = 0
            self.open_price = 0.0

        # Case B: 由空倉建立新部位 (做多或放空)
        elif prev_pos == 0.0 and target_pos != 0.0:
            self.open_price = open_price
            self.trade_bar = 1

        # Case C: 反手翻倉 (Flip: 多翻空 或 空翻多)
        elif prev_pos * target_pos < 0.0:
            # 以開盤價格建立新方向部位
            self.open_price = open_price
            self.trade_bar = 1

        # Case D: 同向持倉調整 (加倉、減倉、完全平倉或維持不變)
        else:
            prev_size = abs(prev_pos)
            curr_size = abs(target_pos)

            if curr_size > prev_size:
                # 同向加倉：加權平均開盤成本
                added_size = curr_size - prev_size
                self.open_price = (
                    prev_size * self.open_price + added_size * open_price
                ) / curr_size
                self.trade_bar += 1
            elif curr_size < prev_size:
                # 同向減倉 或 完全平倉：開倉均價不變
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

        # 5. 單步對數報酬分解 (Mark-to-Market Log Return - Option A)
        safe_prev_close = max(self._prev_close, 1e-8)
        safe_curr_open = max(open_price, 1e-8)
        safe_curr_close = max(close_price, 1e-8)

        gap_log_return = float(np.log(safe_curr_open / safe_prev_close))
        bar_log_return = float(np.log(safe_curr_close / safe_curr_open))

        gross_p_return = float(
            prev_pos * gap_log_return + target_pos * bar_log_return
        )
        current_p_return = float(gross_p_return - cost_t)

        self.last_step_return = current_p_return
        self.last_gross_return = gross_p_return
        self.last_cost = float(cost_t)
        self.last_turnover = float(abs(delta_p))
        self.last_traded = bool(abs(delta_p) > 1e-8)

        self._prev_close = close_price

        # 6. 下行風險懲罰與總獎勵
        reward = float(
            self.weights["w1_step_return"]
            * self.reward_fn.step(
                step_return=current_p_return,
                asset_sigma=self.calculate_asset_sigma(),
            )
        )

        # 7. 推進時間步與判斷終止
        self._offset += 1
        self.game_steps += 1
        terminated = bool(self._offset >= self._prices.close.shape[0] - 1)
        truncated = bool(self.game_steps >= self.N_steps and self.model_train)

        return reward, terminated, truncated

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
            unrealized_return = float(
                self.position
                * (
                    np.log(max(prev_close, 1e-8))
                    - np.log(max(self.open_price, 1e-8))
                )
            )
            data_res[:, len(self.info_list) + 1] = unrealized_return
            # 特徵 3: 持倉時間累計 (以日為基準標準化: 48 根 30m K 棒為 1 天)
            data_res[:, len(self.info_list) + 2] = float(self.trade_bar) / 48.0
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
        reward, terminated, truncated = self._state.step(action_val)
        obs = self._state.encode()

        info = {
            "instrument": self._instrument,
            "offset": self._state._offset,
            "position": float(self._state.position),
            "step_return": float(self._state.last_step_return),
            "gross_return": float(self._state.last_gross_return),
            "cost": float(self._state.last_cost),
            "turnover": float(self._state.last_turnover),
            "traded": bool(self._state.last_traded),
        }

        return obs, reward, terminated, truncated, info

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

    def __init__(self, config, preloaded_data: Optional[Dict[str, Prices]] = None):
        self.config = config
        cfg_training = self.config.training
        self.unique_symbols = cfg_training.UNIQUE_SYMBOLS
        self.data_type_name = "train_data"
        self._data_cache = preloaded_data if preloaded_data is not None else {}

        state_params = {
            "bars_count": cfg_training.BARS_COUNT,
            "commission_perc": cfg_training.MODEL_DEFAULT_COMMISSION_PERC_TRAING,
            "model_train": True,
            "default_slippage": cfg_training.DEFAULT_SLIPPAGE,
            "N_steps": cfg_training.EPISODE_STEPS,
            "deadzone_threshold": cfg_training.DEADZONE_THRESHOLD,
            "rebalance_threshold": getattr(cfg_training, "REBALANCE_THRESHOLD", 0.03),
            "w1_step_return": cfg_training.W1_STEP_RETURN,
            "vol_window": getattr(cfg_training, "VOL_WINDOW", 96),
            "risk_lambda": getattr(cfg_training, "RISK_LAMBDA", 0.5),
            "reward_clip": getattr(cfg_training, "REWARD_CLIP", 5.0),
            "min_sigma": getattr(cfg_training, "MIN_ASSET_SIGMA", 1e-3),
        }

        super().__init__(state=State_time_step(**state_params))

    def _load_data_for_instrument(self, instrument: str):
        if instrument not in self._data_cache:
            loaded = OriginalDataFeature().get_train_net_work_data_by_path(
                [instrument], typeName=self.data_type_name
            )
            self._data_cache.update(loaded)
        data = self._data_cache[instrument]
        if isinstance(data, dict) and instrument in data:
            return data[instrument]
        return data

    def reset(self, symbol: str = None):
        if symbol is None:
            self._instrument = np.random.choice(self.unique_symbols)
        else:
            self._instrument = symbol

        prices = self._load_data_for_instrument(self._instrument)

        max_offset = prices.high.shape[0] - self._state.N_steps - 1
        min_offset = self._state.bars_count
        assert (
            max_offset > min_offset
        ), f"Dataset length {prices.high.shape[0]} too short for N_steps={self._state.N_steps}"
        offset = np.random.randint(min_offset, max_offset)

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


def _subproc_worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                remote.send((obs, reward, terminated, truncated, info))
            elif cmd == "reset":
                obs = env.reset(symbol=data)
                remote.send(obs)
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise NotImplementedError(
                    f"SubprocVecEnv worker received unknown command: {cmd}"
                )
    except Exception as e:
        import traceback

        traceback.print_exc()
        remote.send(("error", e))
    finally:
        if hasattr(env, "close"):
            env.close()


class SubprocVecEnv:
    """
    自建輕量級多進程向量化交易環境封裝 (基於 multiprocessing.Pipe)。
    支援 Dict 觀測空間與多商品重置，專為高吞吐量並行強化學習設計。
    """

    def __init__(
        self,
        env_fns: List[Callable[[], BaseTradingEnv]],
        context: str = "fork",
    ):
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.num_envs)]
        )
        self.ps = [
            ctx.Process(
                target=_subproc_worker,
                args=(work_remote, remote, env_fn),
                daemon=True,
            )
            for work_remote, remote, env_fn in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]
        for p in self.ps:
            p.start()
        for remote in self.work_remotes:
            remote.close()

        # 獲取 spaces
        self.remotes[0].send(("get_spaces", None))
        res = self.remotes[0].recv()
        if isinstance(res, tuple) and len(res) == 2 and res[0] == "error":
            raise RuntimeError(f"Failed to initialize worker 0: {res[1]}")
        self.observation_space, self.action_space = res

    def _stack_obs(self, obs_list):
        return {
            "states": np.stack([o["states"] for o in obs_list], axis=0),
            "time_states": np.stack([o["time_states"] for o in obs_list], axis=0),
        }

    def reset(self, symbols: Optional[List[Optional[str]]] = None):
        if symbols is None:
            symbols = [None] * self.num_envs
        for remote, sym in zip(self.remotes, symbols):
            remote.send(("reset", sym))
        obs_list = []
        for remote in self.remotes:
            res = remote.recv()
            if isinstance(res, tuple) and len(res) == 2 and res[0] == "error":
                raise RuntimeError(f"Error resetting env: {res[1]}")
            obs_list.append(res)
        return self._stack_obs(obs_list)

    def reset_at(self, env_idx: int, symbol: Optional[str] = None):
        self.remotes[env_idx].send(("reset", symbol))
        res = self.remotes[env_idx].recv()
        if isinstance(res, tuple) and len(res) == 2 and res[0] == "error":
            raise RuntimeError(f"Error resetting env {env_idx}: {res[1]}")
        return res

    def step(self, actions: np.ndarray):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = []
        for remote in self.remotes:
            res = remote.recv()
            if isinstance(res, tuple) and len(res) == 2 and res[0] == "error":
                raise RuntimeError(f"Error stepping env: {res[1]}")
            results.append(res)
        obs_list, rewards, terminateds, truncateds, infos = zip(*results)
        return (
            self._stack_obs(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=bool),
            np.array(truncateds, dtype=bool),
            list(infos),
        )

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self.ps:
            p.join(timeout=1.0)
        self.closed = True

    def __len__(self):
        return self.num_envs

