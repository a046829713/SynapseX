import os
import gc
import re
import glob
import json
import math
import time
import zipfile
from itertools import combinations
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf, DictConfig
from gymnasium import spaces

from Brain.Common.DataFeature import OriginalDataFeature, Prices
from Brain.PPO2.lib.Agent import PPO2Agent
from Brain.PPO2.lib.environment import State_time_step, STATIONARY_TIME_FEATURES
from Brain.PPO2.lib.Evaluator import PPO2Evaluator


def _natural_sort_key(s: str):
    """
    Helper key for natural numerical sorting of checkpoint files (e.g. iter_10 before iter_100).
    """
    parts = re.split(r"(\d+)", s)
    return [int(text) if text.isdigit() else text.lower() for text in parts]


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    if not np.isfinite(x):
        return 0.5
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """
    Rational approximation for the standard normal inverse CDF (Acklam's algorithm).
    """
    p_clipped = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1.0 - plow
    if p_clipped < plow:
        q = math.sqrt(-2.0 * math.log(p_clipped))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if phigh < p_clipped:
        q = math.sqrt(-2.0 * math.log(1.0 - p_clipped))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p_clipped - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def _compute_iqm(values: Union[List[float], np.ndarray]) -> float:
    """
    Computes the Interquartile Mean (25% trimmed mean across [Q25, Q75]),
    following the Google Brain / DeepMind `rliable` standard.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return 0.0
    if n < 4:
        return float(np.median(arr))
    sorted_arr = np.sort(arr)
    low_idx = int(n // 4)
    high_idx = int(n - low_idx)
    if low_idx >= high_idx:
        return float(np.median(sorted_arr))
    return float(np.mean(sorted_arr[low_idx:high_idx]))


def _compute_deflated_sharpe(
    daily_returns_matrix: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Computes Deflated Sharpe Ratio probabilities (Bailey & López de Prado, 2014)
    for a D x N matrix of daily portfolio log-returns across D days and N checkpoints.
    Returns (dsr_probs_array_of_len_N, effective_trials_N_eff).
    """
    if daily_returns_matrix.ndim != 2:
        return np.zeros(0, dtype=np.float64), 1.0
    d_days, n_models = daily_returns_matrix.shape
    if n_models == 0:
        return np.zeros(0, dtype=np.float64), 1.0
    if d_days < 3:
        return np.full(n_models, 0.5, dtype=np.float64), 1.0

    means = np.mean(daily_returns_matrix, axis=0)
    stds = np.std(daily_returns_matrix, axis=0, ddof=1)
    valid_std_mask = stds > 1e-8

    sr_daily = np.zeros(n_models, dtype=np.float64)
    sr_daily[valid_std_mask] = means[valid_std_mask] / stds[valid_std_mask]

    # Estimate effective number of independent trials N_eff via eigenvalue participation ratio
    n_eff = 1.0
    if n_models > 1 and np.sum(valid_std_mask) > 1:
        active_mat = daily_returns_matrix[:, valid_std_mask]
        centered = active_mat - np.mean(active_mat, axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normalized = centered / norms
        corr = normalized.T @ normalized
        eigvals = np.linalg.eigvalsh(corr)
        eigvals = np.maximum(eigvals, 0.0)
        denom_eig = float(np.sum(eigvals ** 2))
        if denom_eig > 1e-12:
            pr = float((np.sum(eigvals) ** 2) / denom_eig)
            n_eff = float(np.clip(pr, 1.0, float(n_models)))

    # Expected maximum daily Sharpe under the null hypothesis (False Strategy Theorem)
    sr_std_cross = float(np.std(sr_daily, ddof=1)) if n_models > 1 else 0.0
    if n_eff > 1.05 and sr_std_cross > 1e-8:
        euler_gamma = 0.5772156649015329
        z1 = _norm_ppf(1.0 - 1.0 / n_eff)
        z2 = _norm_ppf(1.0 - 1.0 / (n_eff * math.e))
        sr_star = sr_std_cross * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)
    else:
        sr_star = 0.0

    dsr_probs = np.zeros(n_models, dtype=np.float64)
    for k in range(n_models):
        if not valid_std_mask[k]:
            dsr_probs[k] = 0.0
            continue
        series = daily_returns_matrix[:, k]
        z = (series - means[k]) / stds[k]
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4))
        sr_k = float(sr_daily[k])
        var_term = 1.0 - skew * sr_k + ((kurt - 1.0) / 4.0) * (sr_k ** 2)
        denom = math.sqrt(max(var_term, 1e-8))
        stat = ((sr_k - sr_star) * math.sqrt(max(d_days - 1, 1))) / denom
        dsr_probs[k] = round(_norm_cdf(stat), 4)

    return dsr_probs, round(n_eff, 2)


def _compute_cscv_pbo(
    daily_returns_matrix: np.ndarray, n_partitions: int = 8
) -> Optional[float]:
    """
    Computes Probability of Backtest Overfitting (PBO) using Combinatorially Symmetric
    Cross-Validation (CSCV, Bailey et al., 2017) on a D x N daily return matrix.
    """
    if daily_returns_matrix.ndim != 2:
        return None
    d_days, n_models = daily_returns_matrix.shape
    if n_models < 2 or d_days < 8:
        return None

    s_parts = min(int(n_partitions), (d_days // 2) * 2)
    if s_parts % 2 != 0:
        s_parts -= 1
    if s_parts < 4:
        return None

    blocks = np.array_split(daily_returns_matrix, s_parts, axis=0)
    half = s_parts // 2
    all_indices = set(range(s_parts))

    logits: List[float] = []
    for is_idx_tuple in combinations(range(s_parts), half):
        is_set = set(is_idx_tuple)
        oos_tuple = tuple(sorted(all_indices - is_set))

        is_mat = np.concatenate([blocks[i] for i in is_idx_tuple], axis=0)
        oos_mat = np.concatenate([blocks[i] for i in oos_tuple], axis=0)

        is_std = np.maximum(np.std(is_mat, axis=0, ddof=1), 1e-8)
        oos_std = np.maximum(np.std(oos_mat, axis=0, ddof=1), 1e-8)

        is_sr = np.mean(is_mat, axis=0) / is_std
        oos_sr = np.mean(oos_mat, axis=0) / oos_std

        best_is_idx = int(np.argmax(is_sr))
        target_oos_sr = oos_sr[best_is_idx]

        # Relative rank in (0, 1) using mid-rank for ties
        less_count = float(np.sum(oos_sr < target_oos_sr))
        equal_count = float(np.sum(np.isclose(oos_sr, target_oos_sr, atol=1e-12)))
        rank = less_count + 0.5 * (equal_count + 1.0)
        omega = float(np.clip(rank / (n_models + 1.0), 1e-6, 1.0 - 1e-6))
        logits.append(math.log(omega / (1.0 - omega)))

    if not logits:
        return None
    pbo = float(np.mean(np.asarray(logits) <= 0.0))
    return round(pbo, 4)



class PPO2Engine:
    """
    PPO2 Evaluation and Order Export Engine.
    Coordinates multi-model and multi-symbol streaming evaluation, memory deallocation,
    manifest aggregation, and generates live order maps for formal execution.
    """

    def __init__(
        self,
        formal: bool = False,
        config: Optional[Union[DictConfig, str]] = None,
        device: Optional[torch.device] = None,
        base_data_dir: str = "Brain/simulation/test_data",
        test_start_date: Optional[str] = None,
        test_end_date: Optional[str] = None,
    ):
        self.formal = formal
        self.base_data_dir = base_data_dir
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Resolve configuration strictly without silent fallbacks
        if config is None:
            config_path = "configs/training/ppo_v2.yaml"
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            raw_cfg = OmegaConf.load(config_path)
            self.cfg = raw_cfg.training if hasattr(raw_cfg, "training") else raw_cfg
            self.config_path = config_path
        elif isinstance(config, str):
            if not os.path.exists(config):
                raise FileNotFoundError(f"Configuration file not found: {config}")
            raw_cfg = OmegaConf.load(config)
            self.cfg = raw_cfg.training if hasattr(raw_cfg, "training") else raw_cfg
            self.config_path = config
        elif isinstance(config, DictConfig):
            self.cfg = config.training if hasattr(config, "training") else config
            self.config_path = getattr(config, "config_path", "DictConfig")
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        # Strict fail-fast binding of all model and environment hyperparameters
        self.bars_count = int(self.cfg.BARS_COUNT)
        self.deadzone_threshold = float(self.cfg.DEADZONE_THRESHOLD)
        self.rebalance_threshold = float(self.cfg.REBALANCE_THRESHOLD)
        self.commission_perc = float(self.cfg.MODEL_DEFAULT_COMMISSION_PERC_TRAING)
        self.default_slippage = float(self.cfg.DEFAULT_SLIPPAGE)
        self.hidden_size = int(self.cfg.HIDDEN_SIZE)
        self.nlayers = int(self.cfg.NLAYERS)
        self.time_features_out = int(self.cfg.TIME_FEATURES_OUT)
        self.portfolio_dim = int(self.cfg.PORTFOLIO_DIM)
        self.w1_step_return = float(self.cfg.W1_STEP_RETURN)
        self.episode_steps = int(self.cfg.EPISODE_STEPS)
        self.init_log_std = float(getattr(self.cfg, "INIT_LOG_STD", 0.0))
        self.log_std_min = float(getattr(self.cfg, "LOG_STD_MIN", -5.0))
        self.log_std_max = float(getattr(self.cfg, "LOG_STD_MAX", 1.0))

        cfg_test_start = getattr(self.cfg, "TEST_START_DATE", None)
        cfg_test_end = getattr(self.cfg, "TEST_END_DATE", None)
        self.test_start_date = (
            test_start_date if test_start_date is not None else cfg_test_start
        )
        self.test_end_date = (
            test_end_date if test_end_date is not None else cfg_test_end
        )

        # Leaderboard & model selection configuration with backward-compatible defaults
        self.leaderboard_top_k = int(getattr(self.cfg, "LEADERBOARD_TOP_K", 5))
        self.plateau_window = int(getattr(self.cfg, "PLATEAU_WINDOW", 2))
        self.max_mdd_gate_pct = float(getattr(self.cfg, "MAX_MDD_GATE_PCT", 35.0))
        self.min_beat_bh_ratio = float(getattr(self.cfg, "MIN_BEAT_BH_RATIO", 0.50))

        self.model_paths: List[str] = []
        self.target_symbols: List[str] = []
        self.symbol_file_map: Dict[str, str] = {}
        self.agent: Optional[PPO2Agent] = None
        self.evaluator: Optional[PPO2Evaluator] = None
        self.last_leaderboard_df: Optional[pd.DataFrame] = None
        self.last_selection_summary: Optional[Dict[str, Any]] = None

    @staticmethod
    def _resolve_watch_model_dir(model_dir: Optional[str] = None) -> Optional[str]:
        """
        Resolves the directory to monitor in --watch mode.
        If model_dir is not provided, automatically locates the latest run folder under saves/.
        """
        if model_dir is not None and str(model_dir).strip() != "":
            return model_dir
        saves_root = Path("saves")
        if saves_root.exists() and saves_root.is_dir():
            subdirs = [d for d in saves_root.iterdir() if d.is_dir()]
            if subdirs:
                latest_dir = max(subdirs, key=lambda d: (d.stat().st_mtime, d.name))
                return str(latest_dir)
        return None

    def _discover_models(self, model_dir: Optional[str] = None) -> List[str]:
        """
        Discovers model checkpoints.
        Priority:
        1. Explicit model_dir if passed
        2. Formal mode: Brain/PPO2/Meta/policy_PPO2.pt or latest saves/*.pt
        3. Test mode: All .pt files under saves/ or Brain/PPO2/Meta/
        """
        if model_dir is not None:
            target_path = Path(model_dir)
            if target_path.is_file() and target_path.suffix == ".pt":
                return [str(target_path)]
            if not target_path.exists():
                return []
            pt_files = sorted(list(target_path.glob("**/*.pt")), key=lambda p: _natural_sort_key(p.name))
            return [str(p) for p in pt_files]

        if self.formal:
            meta_path = os.path.join("Brain", "PPO2", "Meta", "policy_PPO2.pt")
            if os.path.exists(meta_path):
                return [meta_path]

        # Scan saves/ directory recursively for all .pt checkpoints
        saves_pt = glob.glob(os.path.join("saves", "**", "*.pt"), recursive=True)
        meta_pt = glob.glob(os.path.join("Brain", "PPO2", "Meta", "*.pt"), recursive=True)
        all_pt = saves_pt + meta_pt
        all_pt = sorted(list(set(all_pt)), key=_natural_sort_key)
        return all_pt

    def _build_symbol_file_map(self) -> Dict[str, str]:
        """
        Builds a map from base symbol (e.g. BTCUSDT) to its CSV filename.
        """
        mapping = {}
        if not os.path.exists(self.base_data_dir):
            return mapping

        for fname in os.listdir(self.base_data_dir):
            if fname.endswith(".csv"):
                base_symbol = fname.split("-")[0].replace(".csv", "")
                mapping[base_symbol] = fname
                mapping[fname] = fname
        return mapping

    def prepare(
        self,
        target_symbols: Optional[List[str]] = None,
        model_paths: Optional[List[str]] = None,
        model_dir: Optional[str] = None,
        stride: int = 1,
    ):
        """
        Prepares symbols and model checkpoint paths, supporting optional stride sampling.
        """
        self.symbol_file_map = self._build_symbol_file_map()

        # 1. Resolve symbols
        if target_symbols is not None and len(target_symbols) > 0:
            resolved_symbols = []
            for s in target_symbols:
                base_s = s.split("-")[0].replace(".csv", "")
                if base_s in self.symbol_file_map:
                    resolved_symbols.append(base_s)
                elif s in self.symbol_file_map:
                    resolved_symbols.append(base_s)
                else:
                    # Keep as-is, attempting direct filename resolution later
                    resolved_symbols.append(base_s)
                    self.symbol_file_map[base_s] = s if s.endswith(".csv") else f"{s}.csv"
            self.target_symbols = sorted(list(set(resolved_symbols)))
        else:
            # Default to all available symbols in test_data
            unique_bases = set()
            for fname in self.symbol_file_map.values():
                base_s = fname.split("-")[0].replace(".csv", "")
                unique_bases.add(base_s)
            self.target_symbols = sorted(list(unique_bases))

        # 2. Resolve model paths
        if model_paths is not None and len(model_paths) > 0:
            discovered = sorted(model_paths, key=_natural_sort_key)
        else:
            discovered = self._discover_models(model_dir=model_dir)

        stride_val = max(1, int(stride))
        if stride_val > 1 and len(discovered) > 1:
            sampled = discovered[::stride_val]
            if discovered[-1] not in sampled:
                sampled.append(discovered[-1])
            self.model_paths = sampled
        else:
            self.model_paths = discovered

        print(
            f"[PPO2Engine] Prepared: {len(self.target_symbols)} symbols, "
            f"{len(self.model_paths)} models (stride={stride_val})."
        )

    def build_agent(self) -> PPO2Agent:
        """
        Dynamically constructs a PPO2Agent matching the training environment space.
        """
        dummy_state = State_time_step(
            bars_count=self.bars_count,
            commission_perc=self.commission_perc,
            model_train=False,
            default_slippage=self.default_slippage,
            N_steps=self.episode_steps,
            deadzone_threshold=self.deadzone_threshold,
            rebalance_threshold=self.rebalance_threshold,
            w1_step_return=self.w1_step_return,
        )
        ob_space = spaces.Dict({
            "states": spaces.Box(
                low=-np.inf, high=np.inf, shape=dummy_state.getStateShape(), dtype=np.float32
            ),
            "time_states": spaces.Box(
                low=-np.inf, high=np.inf, shape=dummy_state.getTimeShape(), dtype=np.float32
            ),
        })
        ac_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        ssm_cfg = None
        if hasattr(self.cfg, "SSM_CFG") and self.cfg.SSM_CFG is not None:
            if isinstance(self.cfg.SSM_CFG, (dict, DictConfig)):
                ssm_cfg = OmegaConf.to_container(self.cfg.SSM_CFG, resolve=True)
            else:
                ssm_cfg = self.cfg.SSM_CFG

        agent = PPO2Agent(
            ob_space=ob_space,
            ac_space=ac_space,
            device=self.device,
            seq_dim=self.bars_count,
            hidden_size=self.hidden_size,
            nlayers=self.nlayers,
            time_features_out=self.time_features_out,
            init_log_std=self.init_log_std,
            portfolio_dim=self.portfolio_dim,
            ssm_cfg=ssm_cfg,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
        )
        return agent

    def load_checkpoint(self, agent: PPO2Agent, checkpoint_path: str):
        """
        Loads weights from a .pt file into the given PPO2Agent.
        Supports both wrapped checkpoint dictionaries and raw state dictionaries.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            agent.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            agent.model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Unrecognized checkpoint format at {checkpoint_path}")
        agent.model.eval()

    def _extract_latest_obs(
        self,
        prices: Prices,
        position: float = 0.0,
        open_price: float = 0.0,
        portfolio_state: Optional[Dict[str, Any]] = None,
        current_position: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extracts the latest observation window of length self.bars_count for formal live mode,
        matching State_time_step.encode() bit-for-bit across market, portfolio, and 6-dim time features.
        """
        if portfolio_state is not None:
            position = float(portfolio_state.get("position", position))
            open_price = float(portfolio_state.get("open_price", open_price))
        elif current_position is not None:
            position = float(current_position)

        pos = float(np.clip(position, 0.0, 1.0))

        N = len(prices.close)
        if N < self.bars_count:
            raise ValueError(
                f"Data length {N} is shorter than required window {self.bars_count}"
            )

        start = N - self.bars_count
        end = N

        info_list = [
            "log_open", "log_high", "log_low", "log_close", "log_volume",
            "log_quote_av", "log_trades", "log_tb_base_av", "log_tb_quote_av",
            "log_ma_30", "log_ma_60", "log_ma_120", "log_ma_240", "log_ma_360",
        ]
        timelist = list(STATIONARY_TIME_FEATURES)

        data_res = np.zeros(shape=(self.bars_count, len(info_list) + 2), dtype=np.float32)
        time_res = np.zeros(shape=(self.bars_count, len(timelist)), dtype=np.float32)

        for idx, field in enumerate(info_list):
            data_res[:, idx] = getattr(prices, field)[start:end]

        data_res[:, len(info_list)] = pos

        if pos > 0.0 and open_price > 0.0:
            prev_close = max(float(prices.close[end - 1]), 1e-8)
            safe_open = max(float(open_price), 1e-8)
            raw_equity_ratio = 1.0 + pos * ((prev_close - safe_open) / safe_open)
            data_res[:, len(info_list) + 1] = float(np.log(max(raw_equity_ratio, 1e-8)))
        else:
            data_res[:, len(info_list) + 1] = 0.0

        for idx, field in enumerate(timelist):
            time_res[:, idx] = getattr(prices, field)[start:end]

        return {"states": data_res, "time_states": time_res}

    def _resolve_symbol_portfolio_state(
        self,
        symbol: str,
        portfolio_state: Optional[Dict[str, Any]],
    ) -> Tuple[float, float]:
        """
        Extracts (position, open_price) for a symbol from portfolio_state.
        """
        if not portfolio_state:
            return 0.0, 0.0

        entry = None
        if symbol in portfolio_state:
            entry = portfolio_state[symbol]
        elif any(k in portfolio_state for k in ("position", "open_price")):
            entry = portfolio_state

        if entry is None:
            return 0.0, 0.0
        if isinstance(entry, dict):
            pos = float(np.clip(float(entry.get("position", 0.0)), 0.0, 1.0))
            open_p = float(entry.get("open_price", 0.0))
            return pos, open_p
        if isinstance(entry, (tuple, list)):
            pos = float(np.clip(float(entry[0]) if len(entry) > 0 else 0.0, 0.0, 1.0))
            open_p = float(entry[1]) if len(entry) > 1 else 0.0
            return pos, open_p
        if isinstance(entry, (int, float, np.floating, np.integer)):
            pos = float(np.clip(float(entry), 0.0, 1.0))
            return pos, 0.0
        return 0.0, 0.0

    @staticmethod
    def _atomic_write_json(path: str, payload: Dict[str, Any]):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)

    @staticmethod
    def _atomic_write_csv(df: pd.DataFrame, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)

    def _is_valid_manifest_cache(
        self,
        manifest_path: str,
        effective_start: Optional[str],
        effective_end: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not os.path.exists(manifest_path):
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or data.get("cache_version") != 2:
                return None
            if data.get("test_start_date", "") != str(effective_start or ""):
                return None
            if data.get("test_end_date", "") != str(effective_end or ""):
                return None
            if data.get("requested_symbols") != list(self.target_symbols):
                return None
            if "symbols" not in data or "portfolio_daily_returns" not in data:
                return None
            return data
        except Exception:
            return None

    def _evaluate_hard_gate(
        self,
        evaluated_symbols: int,
        mean_exposure: float,
        mean_daily_turnover: float,
        median_mdd_pct: float,
        beat_bh_ratio: float,
        iqm_ir_2x: float,
    ) -> Tuple[bool, str]:
        if evaluated_symbols <= 0:
            return False, "FAIL:no_symbols"
        if mean_exposure < 0.01:
            return False, "FAIL:flat_collapse"
        if mean_daily_turnover < 0.005 and mean_exposure > 0.95:
            return False, "FAIL:passive_bh_collapse"
        if median_mdd_pct > self.max_mdd_gate_pct:
            return False, "FAIL:excessive_mdd"
        if beat_bh_ratio < self.min_beat_bh_ratio:
            return False, "FAIL:low_breadth"
        if iqm_ir_2x <= 0.0:
            return False, "FAIL:negative_ir_2x"
        return True, "PASS"

    def _build_and_save_leaderboard(
        self,
        model_records: List[Dict[str, Any]],
        output_dir: str,
        top_k: int = 5,
    ) -> pd.DataFrame:
        leaderboard_columns = [
            "rank",
            "model_name",
            "model_path",
            "gate_pass",
            "gate_reason",
            "ensemble_selected",
            "plateau_score",
            "iqm_ir_2x",
            "iqm_ir_1x",
            "iqm_sharpe",
            "median_sharpe",
            "iqm_sortino",
            "median_mdd_pct",
            "p90_mdd_pct",
            "iqm_cum_return_pct",
            "beat_bh_ratio",
            "positive_return_ratio",
            "mean_exposure",
            "mean_daily_turnover",
            "dsr_prob",
            "evaluated_symbols",
        ]
        if not model_records:
            empty_lb = pd.DataFrame(columns=leaderboard_columns)
            self._atomic_write_csv(empty_lb, os.path.join(output_dir, "leaderboard.csv"))
            self.last_leaderboard_df = empty_lb
            self.last_selection_summary = {
                "total_models_evaluated": 0,
                "passed_gate_count": 0,
                "effective_trials_n_eff": 1.0,
                "pbo_cscv": None,
                "selected_ensemble_models": [],
                "selected_ensemble_paths": [],
            }
            self._atomic_write_json(
                os.path.join(output_dir, "selection_summary.json"),
                self.last_selection_summary,
            )
            return empty_lb

        # Ensure chronological iteration order before neighborhood plateau smoothing
        ordered_records = sorted(
            model_records, key=lambda r: _natural_sort_key(str(r["model_name"]))
        )
        n_models = len(ordered_records)
        raw_scores = np.array(
            [float(r["iqm_ir_2x"]) for r in ordered_records], dtype=np.float64
        )
        w = max(0, int(self.plateau_window))

        for k in range(n_models):
            lo = max(0, k - w)
            hi = min(n_models, k + w + 1)
            win_scores = raw_scores[lo:hi]
            if len(win_scores) <= 1:
                p_score = float(win_scores[0])
            else:
                med = float(np.median(win_scores))
                mad = float(np.median(np.abs(win_scores - med)))
                p_score = med - 0.5 * mad
            ordered_records[k]["plateau_score"] = round(p_score, 4)

        # Build D x N daily portfolio return matrix across all evaluated checkpoints
        all_days_set = set()
        for r in ordered_records:
            all_days_set.update(r.get("portfolio_daily_returns", {}).keys())
        sorted_days = sorted(list(all_days_set))

        if len(sorted_days) >= 2:
            daily_mat = np.zeros((len(sorted_days), n_models), dtype=np.float64)
            for k, r in enumerate(ordered_records):
                day_map = r.get("portfolio_daily_returns", {})
                for d_idx, day_str in enumerate(sorted_days):
                    daily_mat[d_idx, k] = float(day_map.get(day_str, 0.0))
            dsr_probs, n_eff = _compute_deflated_sharpe(daily_mat)
            pbo_val = _compute_cscv_pbo(daily_mat, n_partitions=8)
        else:
            dsr_probs = np.full(n_models, 0.5, dtype=np.float64)
            n_eff = 1.0
            pbo_val = None

        for k in range(n_models):
            ordered_records[k]["dsr_prob"] = float(dsr_probs[k])

        # Sort by (gate_pass, plateau_score, iqm_ir_2x) descending
        ranked_records = sorted(
            ordered_records,
            key=lambda r: (
                bool(r["gate_pass"]),
                float(r["plateau_score"]),
                float(r["iqm_ir_2x"]),
                float(r["iqm_sharpe"]),
            ),
            reverse=True,
        )

        effective_top_k = max(1, int(top_k))
        selected_count = 0
        rows_for_df: List[Dict[str, Any]] = []
        selected_names: List[str] = []
        selected_paths: List[str] = []

        for idx, rec in enumerate(ranked_records):
            is_selected = False
            if rec["gate_pass"] and selected_count < effective_top_k:
                is_selected = True
                selected_count += 1
                selected_names.append(str(rec["model_name"]))
                selected_paths.append(str(rec["model_path"]))

            rows_for_df.append({
                "rank": idx + 1,
                "model_name": rec["model_name"],
                "model_path": rec["model_path"],
                "gate_pass": bool(rec["gate_pass"]),
                "gate_reason": rec["gate_reason"],
                "ensemble_selected": is_selected,
                "plateau_score": rec["plateau_score"],
                "iqm_ir_2x": rec["iqm_ir_2x"],
                "iqm_ir_1x": rec["iqm_ir_1x"],
                "iqm_sharpe": rec["iqm_sharpe"],
                "median_sharpe": rec["median_sharpe"],
                "iqm_sortino": rec["iqm_sortino"],
                "median_mdd_pct": rec["median_mdd_pct"],
                "p90_mdd_pct": rec["p90_mdd_pct"],
                "iqm_cum_return_pct": rec["iqm_cum_return_pct"],
                "beat_bh_ratio": rec["beat_bh_ratio"],
                "positive_return_ratio": rec["positive_return_ratio"],
                "mean_exposure": rec["mean_exposure"],
                "mean_daily_turnover": rec["mean_daily_turnover"],
                "dsr_prob": rec["dsr_prob"],
                "evaluated_symbols": rec["evaluated_symbols"],
            })

        # Fallback recommendation if no checkpoint passed strict hard gates
        fallback_paths: List[str] = []
        if not selected_paths and ranked_records:
            fallback_paths = [
                str(r["model_path"]) for r in ranked_records[: min(effective_top_k, len(ranked_records))]
            ]

        leaderboard_df = pd.DataFrame(rows_for_df, columns=leaderboard_columns)
        leaderboard_path = os.path.join(output_dir, "leaderboard.csv")
        self._atomic_write_csv(leaderboard_df, leaderboard_path)

        passed_count = int(sum(1 for r in ranked_records if r["gate_pass"]))
        summary_payload = {
            "total_models_evaluated": n_models,
            "passed_gate_count": passed_count,
            "effective_trials_n_eff": n_eff,
            "pbo_cscv": pbo_val,
            "top_k": effective_top_k,
            "selected_ensemble_models": selected_names,
            "selected_ensemble_paths": selected_paths,
            "fallback_top_paths": fallback_paths,
            "leaderboard_csv": leaderboard_path,
        }
        summary_path = os.path.join(output_dir, "selection_summary.json")
        self._atomic_write_json(summary_path, summary_payload)

        self.last_leaderboard_df = leaderboard_df
        self.last_selection_summary = summary_payload

        print(f"[PPO2Engine] Leaderboard saved to: {leaderboard_path} ({n_models} models ranked)")
        if pbo_val is not None:
            print(
                f"[PPO2Engine] Overfitting Diagnostics: PBO (CSCV) = {pbo_val:.2%}, "
                f"Effective Trials N_eff = {n_eff:.2f}/{n_models}"
            )
        return leaderboard_df

    def _summarize_model_manifest(
        self,
        model_name: str,
        model_path: str,
        sym_dict: Dict[str, Dict[str, Any]],
        portfolio_daily_returns: Dict[str, float],
    ) -> Dict[str, Any]:
        sym_list = list(sym_dict.values())
        n_syms = len(sym_list)
        if n_syms == 0:
            gate_pass, gate_reason = self._evaluate_hard_gate(0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return {
                "model_name": model_name,
                "model_path": model_path,
                "evaluated_symbols": 0,
                "iqm_ir_2x": 0.0,
                "iqm_ir_1x": 0.0,
                "iqm_sharpe": 0.0,
                "median_sharpe": 0.0,
                "iqm_sortino": 0.0,
                "median_mdd_pct": 0.0,
                "p90_mdd_pct": 0.0,
                "iqm_cum_return_pct": 0.0,
                "beat_bh_ratio": 0.0,
                "positive_return_ratio": 0.0,
                "mean_exposure": 0.0,
                "mean_daily_turnover": 0.0,
                "gate_pass": gate_pass,
                "gate_reason": gate_reason,
                "portfolio_daily_returns": portfolio_daily_returns,
            }

        ir_2x_vals = [float(s.get("ir_2x", 0.0)) for s in sym_list]
        ir_1x_vals = [float(s.get("ir_1x", 0.0)) for s in sym_list]
        sharpe_vals = [float(s.get("sharpe", 0.0)) for s in sym_list]
        sortino_vals = [float(s.get("sortino", 0.0)) for s in sym_list]
        mdd_vals = [float(s.get("max_drawdown_pct", 0.0)) for s in sym_list]
        cum_ret_vals = [float(s.get("cum_return_pct", 0.0)) for s in sym_list]
        exp_vals = [
            float(s.get("mean_exposure", s.get("mean_action", 0.0))) for s in sym_list
        ]
        turn_vals = [float(s.get("daily_turnover", 0.0)) for s in sym_list]

        iqm_ir_2x = round(_compute_iqm(ir_2x_vals), 4)
        iqm_ir_1x = round(_compute_iqm(ir_1x_vals), 4)
        iqm_sharpe = round(_compute_iqm(sharpe_vals), 4)
        median_sharpe = round(float(np.median(sharpe_vals)), 4)
        iqm_sortino = round(_compute_iqm(sortino_vals), 4)
        median_mdd_pct = round(float(np.median(mdd_vals)), 4)
        p90_mdd_pct = round(float(np.percentile(mdd_vals, 90)), 4)
        iqm_cum_return_pct = round(_compute_iqm(cum_ret_vals), 4)
        beat_bh_ratio = round(
            float(np.mean([1.0 if v > 0.0 else 0.0 for v in ir_1x_vals])), 4
        )
        positive_return_ratio = round(
            float(np.mean([1.0 if v > 0.0 else 0.0 for v in cum_ret_vals])), 4
        )
        mean_exposure = round(float(np.mean(exp_vals)), 4)
        mean_daily_turnover = round(float(np.mean(turn_vals)), 4)

        gate_pass, gate_reason = self._evaluate_hard_gate(
            evaluated_symbols=n_syms,
            mean_exposure=mean_exposure,
            mean_daily_turnover=mean_daily_turnover,
            median_mdd_pct=median_mdd_pct,
            beat_bh_ratio=beat_bh_ratio,
            iqm_ir_2x=iqm_ir_2x,
        )

        return {
            "model_name": model_name,
            "model_path": model_path,
            "evaluated_symbols": n_syms,
            "iqm_ir_2x": iqm_ir_2x,
            "iqm_ir_1x": iqm_ir_1x,
            "iqm_sharpe": iqm_sharpe,
            "median_sharpe": median_sharpe,
            "iqm_sortino": iqm_sortino,
            "median_mdd_pct": median_mdd_pct,
            "p90_mdd_pct": p90_mdd_pct,
            "iqm_cum_return_pct": iqm_cum_return_pct,
            "beat_bh_ratio": beat_bh_ratio,
            "positive_return_ratio": positive_return_ratio,
            "mean_exposure": mean_exposure,
            "mean_daily_turnover": mean_daily_turnover,
            "gate_pass": gate_pass,
            "gate_reason": gate_reason,
            "portfolio_daily_returns": portfolio_daily_returns,
        }

    def run_test(
        self,
        output_dir: str = "results/ppo2_orders",
        test_start_date: Optional[str] = None,
        test_end_date: Optional[str] = None,
        top_k: Optional[int] = None,
        use_cache: bool = True,
        min_file_age_sec: float = 0.0,
    ) -> pd.DataFrame:
        """
        Executes streaming batch evaluation across models and symbols with O(1) memory footprint
        and incremental manifest caching. Exports per-symbol CSV orders, per-model manifest.json,
        consolidated multi_model_test_manifest.csv, leaderboard.csv, and selection_summary.json.
        """
        manifest_columns = [
            "model_name",
            "symbol",
            "total_bars",
            "mean_action",
            "std_action",
            "long_ratio",
            "short_ratio",
            "flat_ratio",
            "order_file",
            "sharpe",
            "sortino",
            "ir_1x",
            "ir_2x",
            "max_drawdown_pct",
            "cum_return_pct",
            "bh_return_pct",
            "daily_turnover",
        ]
        if not self.model_paths:
            print("[PPO2Engine] Warning: No models to test. Please provide model_paths or model_dir.")
            return pd.DataFrame(columns=manifest_columns)

        effective_start = (
            test_start_date if test_start_date is not None else self.test_start_date
        )
        effective_end = (
            test_end_date if test_end_date is not None else self.test_end_date
        )
        effective_top_k = int(top_k) if top_k is not None else self.leaderboard_top_k

        os.makedirs(output_dir, exist_ok=True)
        manifest_rows: List[Dict[str, Any]] = []
        model_records: List[Dict[str, Any]] = []
        feature_loader = OriginalDataFeature()

        for model_idx, model_path in enumerate(self.model_paths):
            model_name = Path(model_path).stem
            model_output_dir = os.path.join(output_dir, model_name)
            manifest_json_path = os.path.join(model_output_dir, "manifest.json")

            # 1. Check incremental cache first
            if use_cache:
                cached = self._is_valid_manifest_cache(
                    manifest_json_path, effective_start, effective_end
                )
                if cached is not None:
                    print(
                        f"[{model_idx + 1}/{len(self.model_paths)}] Cached Model: "
                        f"{model_name} (loaded from manifest.json)"
                    )
                    cached_symbols = cached.get("symbols", {})
                    for sym_key, sym_stats in cached_symbols.items():
                        manifest_rows.append({
                            "model_name": model_name,
                            "symbol": sym_key,
                            "total_bars": sym_stats.get("total_bars", 0),
                            "mean_action": sym_stats.get("mean_action", 0.0),
                            "std_action": sym_stats.get("std_action", 0.0),
                            "long_ratio": sym_stats.get("long_ratio", 0.0),
                            "short_ratio": sym_stats.get("short_ratio", 0.0),
                            "flat_ratio": sym_stats.get("flat_ratio", 0.0),
                            "order_file": sym_stats.get("order_file", ""),
                            "sharpe": sym_stats.get("sharpe", 0.0),
                            "sortino": sym_stats.get("sortino", 0.0),
                            "ir_1x": sym_stats.get("ir_1x", 0.0),
                            "ir_2x": sym_stats.get("ir_2x", 0.0),
                            "max_drawdown_pct": sym_stats.get("max_drawdown_pct", 0.0),
                            "cum_return_pct": sym_stats.get("cum_return_pct", 0.0),
                            "bh_return_pct": sym_stats.get("bh_return_pct", 0.0),
                            "daily_turnover": sym_stats.get("daily_turnover", 0.0),
                        })
                    model_records.append(
                        self._summarize_model_manifest(
                            model_name=model_name,
                            model_path=str(model_path),
                            sym_dict=cached_symbols,
                            portfolio_daily_returns=cached.get(
                                "portfolio_daily_returns", {}
                            ),
                        )
                    )
                    continue

            # 2. Half-written checkpoint safety guard for live --watch mode
            if not os.path.exists(model_path):
                continue
            if min_file_age_sec > 0.0:
                try:
                    file_age = time.time() - os.path.getmtime(model_path)
                    if file_age < min_file_age_sec:
                        print(
                            f"[PPO2Engine] Deferring recently modified checkpoint "
                            f"'{model_name}' (age={file_age:.2f}s < {min_file_age_sec:.2f}s) to next cycle."
                        )
                        continue
                except OSError:
                    continue

            print(
                f"\n[{model_idx + 1}/{len(self.model_paths)}] Testing Model: "
                f"{model_name} ({model_path})"
            )

            agent = self.build_agent()
            try:
                self.load_checkpoint(agent, model_path)
            except (EOFError, zipfile.BadZipFile, OSError) as exc:
                print(
                    f"[PPO2Engine] Warning: Transient read error on checkpoint '{model_path}': {exc}"
                )
                del agent
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except (RuntimeError, ValueError, KeyError) as exc:
                msg = str(exc)
                if min_file_age_sec > 0.0 and (
                    "unexpected EOF" in msg or "PytorchStreamReader failed" in msg
                ):
                    print(
                        f"[PPO2Engine] Warning: Checkpoint '{model_path}' still being written ({exc}); deferring."
                    )
                else:
                    print(
                        f"[PPO2Engine] Warning: Skipping incompatible checkpoint '{model_path}': {exc}"
                    )
                del agent
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            evaluator = PPO2Evaluator(
                agent=agent,
                bars_count=self.bars_count,
                device=self.device,
                deadzone_threshold=self.deadzone_threshold,
                rebalance_threshold=self.rebalance_threshold,
                commission_perc=self.commission_perc,
                default_slippage=self.default_slippage,
                w1_step_return=self.w1_step_return,
                episode_steps=self.episode_steps,
                test_start_date=effective_start,
                test_end_date=effective_end,
            )

            os.makedirs(model_output_dir, exist_ok=True)
            model_symbols_dict: Dict[str, Dict[str, Any]] = {}
            daily_sum_by_date: Dict[str, float] = {}
            daily_cnt_by_date: Dict[str, int] = {}

            for sym_idx, symbol in enumerate(self.target_symbols):
                symbol_fname = self.symbol_file_map.get(symbol, f"{symbol}.csv")
                symbol_path = os.path.join(self.base_data_dir, symbol_fname)

                if not os.path.exists(symbol_path):
                    continue

                df = pd.read_csv(symbol_path)
                if len(df) <= 360:
                    continue

                prices = self._process_df_to_prices(feature_loader, df)
                raw_df = feature_loader.df

                out_csv = os.path.join(model_output_dir, f"{symbol}_orders.csv")
                order_df = evaluator.evaluate_symbol(
                    symbol=symbol,
                    raw_df=raw_df,
                    prices=prices,
                    output_csv_path=out_csv,
                    test_start_date=effective_start,
                    test_end_date=effective_end,
                )

                if order_df.empty:
                    del prices, raw_df, order_df
                    gc.collect()
                    continue

                metrics = PPO2Evaluator.compute_symbol_metrics(
                    order_df=order_df,
                    commission_perc=self.commission_perc,
                    default_slippage=self.default_slippage,
                )
                sym_daily_map = metrics.pop("daily_returns", {})
                for day_str, day_ret in sym_daily_map.items():
                    daily_sum_by_date[day_str] = daily_sum_by_date.get(day_str, 0.0) + float(day_ret)
                    daily_cnt_by_date[day_str] = daily_cnt_by_date.get(day_str, 0) + 1

                sym_stats = {
                    "total_bars": metrics["total_bars"],
                    "eval_days": metrics["eval_days"],
                    "mean_action": metrics["mean_action"],
                    "std_action": metrics["std_action"],
                    "long_ratio": metrics["long_ratio"],
                    "short_ratio": metrics["short_ratio"],
                    "flat_ratio": metrics["flat_ratio"],
                    "mean_exposure": metrics["mean_exposure"],
                    "cum_return_pct": metrics["cum_return_pct"],
                    "bh_return_pct": metrics["bh_return_pct"],
                    "sharpe": metrics["sharpe"],
                    "sortino": metrics["sortino"],
                    "ir_1x": metrics["ir_1x"],
                    "ir_2x": metrics["ir_2x"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "calmar": metrics["calmar"],
                    "daily_turnover": metrics["daily_turnover"],
                    "trade_count": metrics["trade_count"],
                    "order_file": out_csv,
                }
                model_symbols_dict[symbol] = sym_stats

                manifest_rows.append({
                    "model_name": model_name,
                    "symbol": symbol,
                    "total_bars": sym_stats["total_bars"],
                    "mean_action": sym_stats["mean_action"],
                    "std_action": sym_stats["std_action"],
                    "long_ratio": sym_stats["long_ratio"],
                    "short_ratio": sym_stats["short_ratio"],
                    "flat_ratio": sym_stats["flat_ratio"],
                    "order_file": out_csv,
                    "sharpe": sym_stats["sharpe"],
                    "sortino": sym_stats["sortino"],
                    "ir_1x": sym_stats["ir_1x"],
                    "ir_2x": sym_stats["ir_2x"],
                    "max_drawdown_pct": sym_stats["max_drawdown_pct"],
                    "cum_return_pct": sym_stats["cum_return_pct"],
                    "bh_return_pct": sym_stats["bh_return_pct"],
                    "daily_turnover": sym_stats["daily_turnover"],
                })

                # Explicit streaming memory deallocation
                del prices, raw_df, order_df
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            portfolio_daily_returns = {
                day_str: round(daily_sum_by_date[day_str] / max(daily_cnt_by_date[day_str], 1), 6)
                for day_str in sorted(daily_sum_by_date.keys())
            }

            model_manifest: Dict[str, Any] = {
                "cache_version": 2,
                "model_name": model_name,
                "model_path": str(model_path),
                "test_start_date": str(effective_start or ""),
                "test_end_date": str(effective_end or ""),
                "commission_perc": float(self.commission_perc),
                "default_slippage": float(self.default_slippage),
                "requested_symbols": list(self.target_symbols),
                "total_symbols_evaluated": len(model_symbols_dict),
                "portfolio_daily_returns": portfolio_daily_returns,
                "symbols": model_symbols_dict,
            }
            self._atomic_write_json(manifest_json_path, model_manifest)
            print(
                f"--> Saved model manifest: {manifest_json_path} "
                f"({model_manifest['total_symbols_evaluated']} symbols)"
            )

            model_records.append(
                self._summarize_model_manifest(
                    model_name=model_name,
                    model_path=str(model_path),
                    sym_dict=model_symbols_dict,
                    portfolio_daily_returns=portfolio_daily_returns,
                )
            )

            # Explicit deallocation of model and evaluator
            del agent, evaluator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        manifest_df = (
            pd.DataFrame(manifest_rows, columns=manifest_columns)
            if manifest_rows
            else pd.DataFrame(columns=manifest_columns)
        )
        consolidated_manifest_path = os.path.join(output_dir, "multi_model_test_manifest.csv")
        self._atomic_write_csv(manifest_df, consolidated_manifest_path)
        print(f"\n[PPO2Engine] Consolidated manifest saved to: {consolidated_manifest_path}")

        self._build_and_save_leaderboard(
            model_records=model_records,
            output_dir=output_dir,
            top_k=effective_top_k,
        )

        return manifest_df

    def watch_and_evaluate(
        self,
        output_dir: str = "results/ppo2_orders",
        model_dir: Optional[str] = None,
        target_symbols: Optional[List[str]] = None,
        poll_interval: float = 30.0,
        stride: int = 1,
        top_k: Optional[int] = None,
        min_file_age_sec: float = 2.0,
        max_idle_polls: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Sidecar Watcher Mode: Periodically scans saves/ (or model_dir) for newly created
        checkpoint .pt files, skips already cached checkpoints via manifest.json, evaluates
        only new checkpoints, and continuously accumulates leaderboard.csv.
        """
        effective_top_k = int(top_k) if top_k is not None else self.leaderboard_top_k
        poll_sec = max(0.5, float(poll_interval))
        idle_polls = 0
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 70)
        print(f"[PPO2Engine] Starting Sidecar Watcher Mode (poll_interval={poll_sec}s, stride={stride})")
        print(f"[PPO2Engine] Output Directory: {output_dir}")
        print("[PPO2Engine] Press Ctrl+C at any time to stop watching cleanly.")
        print("=" * 70)

        try:
            while True:
                active_model_dir = self._resolve_watch_model_dir(model_dir)
                self.prepare(
                    target_symbols=target_symbols,
                    model_dir=active_model_dir,
                    stride=stride,
                )

                pending_models: List[str] = []
                now_ts = time.time()
                for m_path in self.model_paths:
                    if not os.path.exists(m_path):
                        continue
                    if min_file_age_sec > 0.0:
                        try:
                            if (now_ts - os.path.getmtime(m_path)) < min_file_age_sec:
                                continue
                        except OSError:
                            continue
                    m_name = Path(m_path).stem
                    m_manifest = os.path.join(output_dir, m_name, "manifest.json")
                    if (
                        self._is_valid_manifest_cache(
                            m_manifest, self.test_start_date, self.test_end_date
                        )
                        is None
                    ):
                        pending_models.append(m_path)

                leaderboard_file = os.path.join(output_dir, "leaderboard.csv")
                if pending_models or (self.model_paths and not os.path.exists(leaderboard_file)):
                    print(
                        f"\n[Watcher] Detected {len(pending_models)} new checkpoint(s) "
                        f"in '{active_model_dir or 'default'}'. Updating leaderboard..."
                    )
                    self.run_test(
                        output_dir=output_dir,
                        top_k=effective_top_k,
                        use_cache=True,
                        min_file_age_sec=min_file_age_sec,
                    )
                    idle_polls = 0
                else:
                    idle_polls += 1
                    print(
                        f"[Watcher] No new checkpoints (total tracked: {len(self.model_paths)}). "
                        f"Idle poll #{idle_polls}..."
                    )

                if max_idle_polls is not None and idle_polls >= int(max_idle_polls):
                    print(f"[Watcher] Reached max_idle_polls={max_idle_polls}. Exiting watcher.")
                    break

                time.sleep(poll_sec)
        except KeyboardInterrupt:
            print("\n[Watcher] Interrupted by user (Ctrl+C). Leaderboard state is safely saved.")

        if self.last_leaderboard_df is not None:
            return self.last_leaderboard_df
        leaderboard_file = os.path.join(output_dir, "leaderboard.csv")
        if os.path.exists(leaderboard_file):
            return pd.read_csv(leaderboard_file)
        return pd.DataFrame()

    def _process_df_to_prices(
        self, feature_loader: OriginalDataFeature, df: pd.DataFrame
    ) -> Prices:
        df_copy = df.copy()
        if "Datetime" not in df_copy.columns:
            for dt_col in ["date", "Date", "datetime", "timestamp", "Timestamp"]:
                if dt_col in df_copy.columns:
                    df_copy.rename(columns={dt_col: "Datetime"}, inplace=True)
                    break
        if "Datetime" in df_copy.columns:
            df_copy.set_index("Datetime", inplace=True)

        ohlcv_rename = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        for low_col, cap_col in ohlcv_rename.items():
            if cap_col not in df_copy.columns and low_col in df_copy.columns:
                df_copy[cap_col] = df_copy[low_col]

        cleaned = feature_loader.cleanData(df_copy)

        # Defensive handling for missing volume auxiliary columns
        if "quote_av" not in cleaned.columns:
            cleaned["quote_av"] = cleaned.get("Volume", 1.0) * cleaned.get("Close", 1.0)
        if "trades" not in cleaned.columns:
            cleaned["trades"] = cleaned.get("Volume", 1.0) * 0.1
        if "tb_base_av" not in cleaned.columns:
            cleaned["tb_base_av"] = cleaned.get("Volume", 1.0) * 0.5
        if "tb_quote_av" not in cleaned.columns:
            cleaned["tb_quote_av"] = cleaned["quote_av"] * 0.5

        if "log_ma_360" not in cleaned.columns:
            cleaned = feature_loader.add_average_metric(cleaned)
        for p in [30, 60, 120, 240, 360]:
            if f"log_ma_{p}" not in cleaned.columns:
                cleaned[f"log_ma_{p}"] = 0.0

        if "age_log_minutes" not in cleaned.columns and len(cleaned) > 0:
            cleaned = feature_loader.add_time_feature(cleaned)
        for t_col in [
            "age_log_minutes", "age_years", "month_sin", "month_cos",
            "day_sin", "day_cos", "hour_sin", "hour_cos", "minute_sin",
            "minute_cos", "dayofweek_sin", "dayofweek_cos", "week_sin", "week_cos",
        ]:
            if t_col not in cleaned.columns:
                cleaned[t_col] = 0.0

        feature_loader.df = cleaned
        return feature_loader.load_relative()

    def _apply_deadzone_and_rebalance(
        self, raw_action: float, prev_position: float
    ) -> float:
        clipped_action = float(np.clip(raw_action, 0.0, 1.0))
        if clipped_action <= self.deadzone_threshold:
            candidate_pos = 0.0
        else:
            candidate_pos = clipped_action

        prev_pos = float(np.clip(prev_position, 0.0, 1.0))
        if prev_pos == 0.0 or candidate_pos == 0.0:
            target_pos = candidate_pos
        elif abs(candidate_pos - prev_pos) < self.rebalance_threshold:
            target_pos = prev_pos
        else:
            target_pos = candidate_pos
        return float(target_pos)

    def get_if_order_map(
        self,
        data_source: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
        output_path: str = "results/ppo2_orders/formal_if_order_map.json",
        portfolio_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Formal Live Mode: Generates target order map {symbol: action} in [0.0, 1.0]
        for immediate live execution by TradingSysteam.py, preserving stateful portfolio features
        and rebalance_threshold parity with ProductionEnv. Supports single checkpoint or
        multi-checkpoint Action Ensembling when multiple model_paths are configured.
        """
        feature_loader = OriginalDataFeature()
        symbol_obs_map: Dict[str, Tuple[Dict[str, np.ndarray], float]] = {}

        if data_source is None:
            for symbol in self.target_symbols:
                symbol_fname = self.symbol_file_map.get(symbol, f"{symbol}.csv")
                symbol_path = os.path.join(self.base_data_dir, symbol_fname)
                if not os.path.exists(symbol_path):
                    continue
                df = pd.read_csv(symbol_path)
                prices = self._process_df_to_prices(feature_loader, df)
                pos, open_p = self._resolve_symbol_portfolio_state(
                    symbol, portfolio_state
                )
                obs = self._extract_latest_obs(
                    prices, position=pos, open_price=open_p
                )
                symbol_obs_map[symbol] = (obs, pos)

        elif isinstance(data_source, pd.DataFrame):
            sym_col = None
            for col in ["tic", "Symbol", "symbol"]:
                if col in data_source.columns:
                    sym_col = col
                    break

            if sym_col is not None:
                symbols = data_source[sym_col].unique()
                for sym in symbols:
                    sym_str = str(sym)
                    sym_df = data_source[data_source[sym_col] == sym]
                    prices = self._process_df_to_prices(feature_loader, sym_df)
                    pos, open_p = self._resolve_symbol_portfolio_state(
                        sym_str, portfolio_state
                    )
                    obs = self._extract_latest_obs(
                        prices, position=pos, open_price=open_p
                    )
                    symbol_obs_map[sym_str] = (obs, pos)
            else:
                sym_name = self.target_symbols[0] if self.target_symbols else "DEFAULT"
                prices = self._process_df_to_prices(feature_loader, data_source)
                pos, open_p = self._resolve_symbol_portfolio_state(
                    sym_name, portfolio_state
                )
                obs = self._extract_latest_obs(
                    prices, position=pos, open_price=open_p
                )
                symbol_obs_map[sym_name] = (obs, pos)

        elif isinstance(data_source, dict):
            for sym, sym_df in data_source.items():
                sym_str = str(sym)
                prices = self._process_df_to_prices(feature_loader, sym_df)
                pos, open_p = self._resolve_symbol_portfolio_state(
                    sym_str, portfolio_state
                )
                obs = self._extract_latest_obs(
                    prices, position=pos, open_price=open_p
                )
                symbol_obs_map[sym_str] = (obs, pos)

        if_order_map: Dict[str, float] = {}

        if len(self.model_paths) > 1:
            # Multi-checkpoint Action Ensembling mode
            print(
                f"[PPO2Engine] Running Formal Action Ensemble across "
                f"{len(self.model_paths)} checkpoints..."
            )
            accum_actions: Dict[str, List[float]] = {sym: [] for sym in symbol_obs_map}
            ensemble_agent = self.build_agent()

            for m_path in self.model_paths:
                try:
                    self.load_checkpoint(ensemble_agent, m_path)
                except (RuntimeError, ValueError, KeyError, OSError) as exc:
                    print(
                        f"[PPO2Engine] Warning: Skipping checkpoint '{m_path}' in ensemble: {exc}"
                    )
                    continue
                evaluator = PPO2Evaluator(
                    agent=ensemble_agent,
                    bars_count=self.bars_count,
                    device=self.device,
                    deadzone_threshold=self.deadzone_threshold,
                    rebalance_threshold=self.rebalance_threshold,
                    commission_perc=self.commission_perc,
                    default_slippage=self.default_slippage,
                    w1_step_return=self.w1_step_return,
                    episode_steps=self.episode_steps,
                )
                for sym, (obs, pos) in symbol_obs_map.items():
                    raw_act = evaluator.evaluate_single_step(
                        obs, apply_deadzone=False, prev_position=pos
                    )
                    accum_actions[sym].append(float(raw_act))

            del ensemble_agent
            gc.collect()

            for sym, (obs, pos) in symbol_obs_map.items():
                preds = accum_actions.get(sym, [])
                mean_raw = float(np.mean(preds)) if preds else 0.0
                final_act = self._apply_deadzone_and_rebalance(mean_raw, pos)
                if_order_map[sym] = round(final_act, 4)
        else:
            # Single checkpoint mode
            if self.agent is None:
                self.agent = self.build_agent()
                if self.model_paths:
                    self.load_checkpoint(self.agent, self.model_paths[0])
                else:
                    formal_model_path = os.path.join("Brain", "PPO2", "Meta", "policy_PPO2.pt")
                    if os.path.exists(formal_model_path):
                        self.load_checkpoint(self.agent, formal_model_path)
                    else:
                        print(
                            "[PPO2Engine] Warning: No model checkpoint found for formal mode. "
                            "Running with initialized weights."
                        )

            evaluator = PPO2Evaluator(
                agent=self.agent,
                bars_count=self.bars_count,
                device=self.device,
                deadzone_threshold=self.deadzone_threshold,
                rebalance_threshold=self.rebalance_threshold,
                commission_perc=self.commission_perc,
                default_slippage=self.default_slippage,
                w1_step_return=self.w1_step_return,
                episode_steps=self.episode_steps,
            )

            for sym, (obs, pos) in symbol_obs_map.items():
                action_val = evaluator.evaluate_single_step(
                    obs, apply_deadzone=True, prev_position=pos
                )
                if_order_map[sym] = round(action_val, 4)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(if_order_map, f, indent=4)

        print(f"[PPO2Engine] Saved formal live order map to: {output_path} ({len(if_order_map)} symbols)")
        return if_order_map
