import multiprocessing as mp
from typing import Optional, List, Dict, Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod
from Brain.Common.env_components import State_time_step_template
from Brain.Common.DataFeature import Prices, OriginalDataFeature

STATIONARY_TIME_FEATURES = [
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "dayofweek_sin",
    "dayofweek_cos",
]


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
        **kwargs,
    ):
        super().__init__(
            bars_count=bars_count,
            commission_perc=commission_perc,
            model_train=model_train,
            default_slippage=default_slippage,
            N_steps=N_steps,
        )
        self.timelist = list(STATIONARY_TIME_FEATURES)

        self.deadzone_threshold = deadzone_threshold
        self.rebalance_threshold = rebalance_threshold

        self.weights = {
            "w1_step_return": float(w1_step_return),
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

    def step(self, action: float):
        """
        Executes a single continuous Long-Only [0.0, 1.0] market-open rebalance step
        with exact simple-to-log portfolio equity compounding.
        """
        clipped_action = float(np.clip(action, 0.0, 1.0))
        if clipped_action <= self.deadzone_threshold:
            candidate_pos = 0.0
        else:
            candidate_pos = clipped_action

        open_price = max(float(self._prices.open[self._offset]), 1e-8)
        close_price = max(float(self._prices.close[self._offset]), 1e-8)
        safe_prev_close = max(float(self._prev_close), 1e-8)
        prev_pos = float(self.position)

        # Rebalance deadzone filter for Long-Only [0.0, 1.0]
        if prev_pos == 0.0 or candidate_pos == 0.0:
            target_pos = candidate_pos
        elif abs(candidate_pos - prev_pos) < self.rebalance_threshold:
            target_pos = prev_pos
        else:
            target_pos = candidate_pos

        delta_p = target_pos - prev_pos
        friction_rate = self.commission_perc + self.default_slippage

        # Long-Only position & weighted-average open_price maintenance
        if target_pos == 0.0:
            self.open_price = 0.0
            self.trade_bar = 0
        elif prev_pos == 0.0:
            self.open_price = open_price
            self.trade_bar = 1
        else:
            if target_pos > prev_pos:
                added_size = target_pos - prev_pos
                self.open_price = (
                    prev_pos * self.open_price + added_size * open_price
                ) / target_pos
            self.trade_bar += 1

        self.position = target_pos

        # Exact simple-to-log portfolio equity compounding
        gap_simple_ret = (open_price - safe_prev_close) / safe_prev_close
        bar_simple_ret = (close_price - open_price) / open_price

        m_gap = max(1.0 + prev_pos * gap_simple_ret, 1e-8)
        m_gross = max(m_gap * (1.0 + target_pos * bar_simple_ret), 1e-8)
        m_net = max(
            m_gap * (1.0 - abs(delta_p) * friction_rate) * (1.0 + target_pos * bar_simple_ret),
            1e-8,
        )

        gross_p_return = float(np.log(m_gross))
        current_p_return = float(np.log(m_net))
        effective_cost = float(max(0.0, gross_p_return - current_p_return))

        self.last_step_return = current_p_return
        self.last_gross_return = gross_p_return
        self.last_cost = effective_cost
        self.last_turnover = float(abs(delta_p))
        self.last_traded = bool(abs(delta_p) > 1e-8)

        self._prev_close = close_price

        # Unbiased single-step net portfolio log-return
        reward = float(self.weights["w1_step_return"] * current_p_return)

        # Advance time offset and check termination/truncation
        self._offset += 1
        self.game_steps += 1
        terminated = bool(self._offset >= self._prices.close.shape[0] - 1)
        truncated = bool(self.game_steps >= self.N_steps and self.model_train)

        return reward, terminated, truncated

    def encode(self):
        """
        Encodes observation dict {"states": data_res, "time_states": time_res}
        using vectorized NumPy slicing without look-ahead bias.
        """
        data_res = np.zeros(shape=self.getStateShape(), dtype=np.float32)
        time_res = np.zeros(shape=self.getTimeShape(), dtype=np.float32)

        start = self._offset - self.bars_count
        end = self._offset

        for idx, field in enumerate(self.info_list):
            data_res[:, idx] = getattr(self._prices, field)[start:end]

        data_res[:, len(self.info_list)] = self.position

        if self.position > 0.0 and self.open_price > 0.0:
            prev_close = max(float(self._prices.close[self._offset - 1]), 1e-8)
            safe_open = max(float(self.open_price), 1e-8)
            raw_equity_ratio = 1.0 + self.position * ((prev_close - safe_open) / safe_open)
            data_res[:, len(self.info_list) + 1] = float(np.log(max(raw_equity_ratio, 1e-8)))
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
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
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
        Executes a single environment step, clamping continuous actions to [0.0, 1.0].
        """
        action_val = float(np.asarray(action).squeeze())
        action_val = float(np.clip(action_val, 0.0, 1.0))
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
        self.train_end_date = getattr(cfg_training, "TRAIN_END_DATE", None)
        self.min_bars = (
            int(cfg_training.BARS_COUNT) + int(cfg_training.EPISODE_STEPS) + 2
        )
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
        }

        super().__init__(state=State_time_step(**state_params))

    def _load_data_for_instrument(self, instrument: str):
        if instrument not in self._data_cache:
            loaded = OriginalDataFeature().get_train_net_work_data_by_path(
                [instrument],
                typeName=self.data_type_name,
                end_date=self.train_end_date,
                min_bars=self.min_bars,
            )
            if instrument not in loaded:
                raise ValueError(
                    f"Instrument '{instrument}' has insufficient pre-cutoff data "
                    f"(end_date={self.train_end_date}, min_bars={self.min_bars})."
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

    def reset(self, symbol: str = None, start_offset: Optional[int] = None):
        if symbol is not None:
            assert symbol in self._prices, f"Symbol {symbol} not found in preloaded prices"
            self._instrument = symbol
        prices = self._prices[self._instrument]
        if start_offset is None:
            offset = self._state.bars_count
        else:
            offset = max(self._state.bars_count, int(start_offset))
        assert offset < prices.close.shape[0], (
            f"start_offset {offset} out of bounds for symbol {self._instrument} "
            f"with length {prices.close.shape[0]}"
        )

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

