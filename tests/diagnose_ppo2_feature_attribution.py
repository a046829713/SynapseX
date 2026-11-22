"""
Offline PPO2 input feature attribution & policy behavior diagnostic tool.

Spec: spec/feature/ppo2_feature_attribution_diagnostics.md

Analyses (100% read-only w.r.t. the checkpoint on disk, no argparse):
  A1  Four-metric feature attribution for states (market + portfolio) and time_states:
        - Raw Gradient Saliency       : mean(|dY / dx_i|)
        - Scale-Normalized Saliency   : std(x_i) * mean(|dY / dx_i|)
        - Input x Gradient            : mean(|x_i * dY / dx_i|)
        - Permutation Importance      : mean(|Y(perm_i) - Y(orig)|) over PERM_REPEATS shuffles
        - First-Layer Weight L2 Norm  : ||W[:, i]||_2 of market_embedding / time_embedding
  A2  Temporal saliency profile across 300-bar lookback buckets + 50%/80% effective horizon.
  A3  Policy behavior summary (log_std, mu/action quantiles, deadzone collapse rate,
      Buy vs. Cash conditional feature means, and automated diagnostic verdicts).

Usage:
  1. Edit the UPPERCASE configuration constants below directly.
  2. Run:
     /home/b0457812963/Mamba3RL/bin/python tests/diagnose_ppo2_feature_attribution.py
"""

from datetime import datetime
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Brain.Common.DataFeature import OriginalDataFeature  # noqa: E402
from Brain.Common.env_components import BasicState  # noqa: E402
from Brain.PPO2.lib.environment import State_time_step  # noqa: E402
from Brain.PPO2.lib.model import MambaContinuousActorCritic  # noqa: E402

# =============================================================================
# 診斷設定常數（直接修改此區變數即可，無須透過命令列傳入參數）
# =============================================================================
CHECKPOINT_PATH: str = "Brain/PPO2/Meta/checkpoint_iter_9000.pt"
CONFIG_PATH: str = "configs/training/ppo_v2.yaml"
SPLIT: str = "train"                  # "train" | "test" | "all"
SYMBOLS: Optional[List[str]] = None   # None 代表依 SEED 隨機抽取 NUM_SYMBOLS 檔；或手動填如 ["ETHUSDT-F-30-Min", "SOLUSDT-F-30-Min"]
NUM_SYMBOLS: int = 8
SAMPLES_PER_SYMBOL: int = 20
BATCH_SIZE: int = 32                  # RTX 2060 6GB 安全批次大小
PERM_REPEATS: int = 3                 # 單通道置換擾動重複次數
POSITION_MODE: str = "binary"         # "binary" (0/1 交替) | "uniform" ([0,1]) | "zero" (全 0)
BUY_THRESHOLD: float = 0.1            # A3 進場 vs 空手條件對照門檻
DEVICE: str = "auto"                  # "auto" | "cuda" | "cpu"
SEED: int = 0
OUTPUT_PATH: Optional[str] = None     # None 代表自動輸出至 results/feature_attribution/<timestamp>.json

# Automated verdict thresholds
DEADZONE_COLLAPSE_FRAC_THRESHOLD: float = 0.80
DEADZONE_COLLAPSE_MEDIAN_MU_THRESHOLD: float = -4.0
CRITIC_DOMINANCE_RATIO_THRESHOLD: float = 1.8
DEAD_CHANNEL_PERM_DMU_THRESHOLD: float = 0.05
DEAD_CHANNEL_PERM_DV_THRESHOLD: float = 0.02

TEMPORAL_BUCKETS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("last_16", -16, None),
    ("bars_17_64", -64, -16),
    ("bars_65_150", -150, -64),
    ("bars_151_300", 0, -150),
)


# =============================================================================
# Core Helpers
# =============================================================================
def resolve_path(path: str) -> str:
    """Resolves a relative path against REPO_ROOT while preserving absolute paths."""
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def resolve_device(device_spec: str) -> torch.device:
    """Resolves device string ('auto', 'cuda', 'cpu', or torch device string)."""
    spec = str(device_spec).strip().lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def get_state_feature_names(portfolio_dim: int) -> List[str]:
    """Builds ordered feature names for states tensor (market + portfolio channels)."""
    assert portfolio_dim >= 0, f"portfolio_dim must be >= 0, got {portfolio_dim}"
    market_names = list(BasicState.MARKET_FEATURES)
    if portfolio_dim == 0:
        return market_names
    if portfolio_dim == 1:
        return market_names + ["position"]
    return market_names + ["position"] + [f"portfolio_{i}" for i in range(1, portfolio_dim)]


def load_model_from_checkpoint(
    checkpoint_path: str,
    cfg: DictConfig,
    device: torch.device,
) -> Tuple[MambaContinuousActorCritic, Dict[str, Any]]:
    """
    Loads a PPO2 MambaContinuousActorCritic checkpoint in strict read-only eval mode
    and validates input feature dimensions against config and BasicState.
    """
    full_ckpt_path = resolve_path(checkpoint_path)
    assert os.path.isfile(full_ckpt_path), f"Checkpoint not found: {full_ckpt_path}"

    raw_ckpt = torch.load(full_ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
        state_dict = raw_ckpt["model_state_dict"]
        ckpt_meta = {
            "iteration": int(raw_ckpt["iteration"]) if "iteration" in raw_ckpt else None,
            "total_steps": int(raw_ckpt["total_steps"]) if "total_steps" in raw_ckpt else None,
        }
    elif isinstance(raw_ckpt, dict):
        state_dict = raw_ckpt
        ckpt_meta = {"iteration": None, "total_steps": None}
    else:
        raise ValueError(f"Unrecognized checkpoint format at {full_ckpt_path}")

    portfolio_dim = int(getattr(cfg, "PORTFOLIO_DIM", 1))
    d_model = len(BasicState.MARKET_FEATURES) + portfolio_dim
    time_features_in = len(BasicState.STATIONARY_TIME_FEATURES)
    bars_count = int(cfg.BARS_COUNT)

    assert "market_embedding.weight" in state_dict, (
        "Checkpoint state_dict missing 'market_embedding.weight'"
    )
    assert "time_embedding.w" in state_dict, (
        "Checkpoint state_dict missing 'time_embedding.w'"
    )
    ckpt_d_model = int(state_dict["market_embedding.weight"].shape[1])
    ckpt_time_in = int(state_dict["time_embedding.w"].shape[0])
    assert ckpt_d_model == d_model, (
        f"Checkpoint market_embedding input dim ({ckpt_d_model}) != expected d_model ({d_model}). "
        f"Check BasicState.MARKET_FEATURES ({len(BasicState.MARKET_FEATURES)}) and PORTFOLIO_DIM ({portfolio_dim})."
    )
    assert ckpt_time_in == time_features_in, (
        f"Checkpoint time_embedding input dim ({ckpt_time_in}) != expected time_features_in ({time_features_in})."
    )

    ssm_cfg = OmegaConf.to_container(cfg.SSM_CFG, resolve=True)
    model = MambaContinuousActorCritic(
        d_model=d_model,
        time_features_in=time_features_in,
        action_dim=1,
        seq_dim=bars_count,
        hidden_size=int(cfg.HIDDEN_SIZE),
        nlayers=int(cfg.NLAYERS),
        time_features_out=int(cfg.TIME_FEATURES_OUT),
        init_log_std=float(getattr(cfg, "INIT_LOG_STD", 0.0)),
        portfolio_dim=portfolio_dim,
        ssm_cfg=ssm_cfg,
        log_std_min=float(getattr(cfg, "LOG_STD_MIN", -5.0)),
        log_std_max=float(getattr(cfg, "LOG_STD_MAX", 1.0)),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    ckpt_meta.update(
        {
            "checkpoint_path": full_ckpt_path,
            "d_model": d_model,
            "market_dim": len(BasicState.MARKET_FEATURES),
            "portfolio_dim": portfolio_dim,
            "time_features_in": time_features_in,
            "bars_count": bars_count,
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
        }
    )
    return model, ckpt_meta


def sample_observation_windows(
    cfg: DictConfig,
    split: str = "train",
    symbols: Optional[List[str]] = None,
    num_symbols: int = 8,
    samples_per_symbol: int = 20,
    position_mode: str = "binary",
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Samples real market observation windows using OriginalDataFeature + State_time_step.encode()
    to guarantee exact train/inference feature parity.

    Returns:
      states_np      : float32 array of shape (N, bars_count, 14 + portfolio_dim)
      time_states_np : float32 array of shape (N, bars_count, 6)
      sample_meta    : metadata dict summarizing sampled symbols and counts
    """
    split_norm = str(split).strip().lower()
    assert split_norm in ("train", "test", "all"), (
        f"SPLIT must be 'train', 'test', or 'all', got '{split}'"
    )
    pos_mode = str(position_mode).strip().lower()
    assert pos_mode in ("binary", "uniform", "zero"), (
        f"POSITION_MODE must be 'binary', 'uniform', or 'zero', got '{position_mode}'"
    )
    assert int(num_symbols) >= 1, "NUM_SYMBOLS must be >= 1"
    assert int(samples_per_symbol) >= 1, "SAMPLES_PER_SYMBOL must be >= 1"

    bars_count = int(cfg.BARS_COUNT)
    episode_steps = int(getattr(cfg, "EPISODE_STEPS", 512))
    exclude_set = set(str(s) for s in (getattr(cfg, "EXCLUDE_SYMBOLS", None) or []))

    data_subdir = "test_data" if split_norm == "test" else "train_data"
    data_dir = os.path.join(REPO_ROOT, "Brain", "simulation", data_subdir)
    assert os.path.isdir(data_dir), f"Simulation data directory not found: {data_dir}"

    if symbols is not None and len(symbols) > 0:
        candidate_symbols = [
            s[:-4] if s.endswith(".csv") else s for s in symbols
        ]
    else:
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        candidate_symbols = [
            os.path.basename(f)[:-4]
            for f in csv_files
            if os.path.basename(f)[:-4] not in exclude_set
        ]

    assert len(candidate_symbols) > 0, f"No candidate symbols found in {data_dir}"

    rng = np.random.default_rng(int(seed))
    if symbols is None or len(symbols) == 0:
        order = rng.permutation(len(candidate_symbols))
        candidate_symbols = [candidate_symbols[int(i)] for i in order]

    train_end_ts = (
        pd.to_datetime(cfg.TRAIN_END_DATE)
        if getattr(cfg, "TRAIN_END_DATE", None)
        else None
    )
    test_start_ts = (
        pd.to_datetime(cfg.TEST_START_DATE)
        if getattr(cfg, "TEST_START_DATE", None)
        else None
    )
    test_end_ts = (
        pd.to_datetime(cfg.TEST_END_DATE)
        if getattr(cfg, "TEST_END_DATE", None)
        else None
    )

    feature_loader = OriginalDataFeature()
    state_encoder = State_time_step(
        bars_count=bars_count,
        commission_perc=float(cfg.MODEL_DEFAULT_COMMISSION_PERC_TRAING),
        model_train=False,
        default_slippage=float(cfg.DEFAULT_SLIPPAGE),
        N_steps=episode_steps,
        deadzone_threshold=float(cfg.DEADZONE_THRESHOLD),
        rebalance_threshold=float(cfg.REBALANCE_THRESHOLD),
        w1_step_return=float(cfg.W1_STEP_RETURN),
        downside_window_size=int(cfg.DOWNSIDE_WINDOW_SIZE),
        w2_downside_ratio=float(cfg.W2_DOWNSIDE_RATIO),
        weight_scale=float(cfg.WEIGHT_SCALE),
    )

    states_list: List[np.ndarray] = []
    time_list: List[np.ndarray] = []
    used_symbols: List[str] = []
    per_symbol_counts: Dict[str, int] = {}

    target_symbol_count = len(candidate_symbols) if (symbols is not None and len(symbols) > 0) else int(num_symbols)

    for sym in candidate_symbols:
        if len(used_symbols) >= target_symbol_count:
            break
        csv_path = os.path.join(data_dir, f"{sym}.csv")
        if not os.path.isfile(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if "Datetime" not in df.columns or len(df) <= 360 + bars_count:
            continue
        df.set_index("Datetime", inplace=True)

        if split_norm == "train" and train_end_ts is not None:
            df = df[pd.to_datetime(df.index) <= train_end_ts]
            if len(df) <= 360 + bars_count:
                continue

        cleaned = feature_loader.cleanData(df)
        if len(cleaned) <= 360:
            continue
        cleaned = feature_loader.add_market_features(cleaned)
        if len(cleaned) <= bars_count + 1:
            continue
        cleaned = feature_loader.add_time_feature(cleaned)
        prices = feature_loader.load_relative(cleaned)
        n_bars = len(prices.close)

        if split_norm == "test":
            dt_idx = pd.to_datetime(cleaned.index)
            min_off = (
                max(bars_count, int(dt_idx.searchsorted(test_start_ts, side="left")))
                if test_start_ts is not None
                else bars_count
            )
            max_off = (
                min(n_bars - 1, int(dt_idx.searchsorted(test_end_ts, side="right")))
                if test_end_ts is not None
                else n_bars - 1
            )
        else:
            min_off = bars_count
            max_off = n_bars - 1

        if max_off <= min_off:
            continue

        valid_offsets = np.arange(min_off, max_off, dtype=np.int64)
        k_draw = min(int(samples_per_symbol), len(valid_offsets))
        chosen_offsets = rng.choice(valid_offsets, size=k_draw, replace=False)

        for idx_k, off in enumerate(chosen_offsets):
            state_encoder.reset(prices, int(off))
            if pos_mode == "binary":
                state_encoder.position = float(idx_k % 2)
            elif pos_mode == "uniform":
                state_encoder.position = float(rng.uniform(0.0, 1.0))
            else:
                state_encoder.position = 0.0
            obs = state_encoder.encode()
            states_list.append(obs["states"])
            time_list.append(obs["time_states"])

        used_symbols.append(sym)
        per_symbol_counts[sym] = int(k_draw)

    assert len(states_list) > 0, (
        f"Failed to sample any valid observation windows (split='{split_norm}', data_dir='{data_dir}')."
    )

    states_np = np.stack(states_list, axis=0).astype(np.float32)
    time_states_np = np.stack(time_list, axis=0).astype(np.float32)
    assert np.all(np.isfinite(states_np)), "Sampled states contain non-finite values"
    assert np.all(np.isfinite(time_states_np)), "Sampled time_states contain non-finite values"

    sample_meta = {
        "split": split_norm,
        "data_dir": data_dir,
        "n_windows": int(states_np.shape[0]),
        "seq_len": int(states_np.shape[1]),
        "n_symbols_used": len(used_symbols),
        "symbols_used": used_symbols,
        "per_symbol_counts": per_symbol_counts,
        "position_mode": pos_mode,
        "seed": int(seed),
    }
    return states_np, time_states_np, sample_meta


# =============================================================================
# A1: Input Gradients & Four-Metric Feature Attribution
# =============================================================================
def compute_input_gradients(
    model: MambaContinuousActorCritic,
    states_np: np.ndarray,
    time_states_np: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """
    Computes exact input gradients w.r.t. Actor mean (mu) and Critic value (V)
    using OOM-safe mini-batches with separate forward/backward passes (no retain_graph=True).

    Returns a dict of NumPy float32 arrays:
      - g_mu_states, g_v_states : (N, L, d_model)
      - g_mu_time,   g_v_time   : (N, L, time_features_in)
      - mu, value               : (N,)
    """
    assert states_np.ndim == 3, f"Expected states_np 3D (N, L, F), got {states_np.shape}"
    assert time_states_np.ndim == 3, f"Expected time_states_np 3D (N, L, T), got {time_states_np.shape}"
    assert states_np.shape[:2] == time_states_np.shape[:2], (
        f"Batch/seq mismatch: states {states_np.shape} vs time_states {time_states_np.shape}"
    )
    assert int(batch_size) >= 1, "batch_size must be >= 1"

    model.eval()
    orig_req_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    n_samples = int(states_np.shape[0])
    mb = int(batch_size)

    g_mu_s_chunks: List[np.ndarray] = []
    g_v_s_chunks: List[np.ndarray] = []
    g_mu_t_chunks: List[np.ndarray] = []
    g_v_t_chunks: List[np.ndarray] = []
    mu_chunks: List[np.ndarray] = []
    v_chunks: List[np.ndarray] = []

    try:
        for start in range(0, n_samples, mb):
            end = min(start + mb, n_samples)

            # Pass 1: Actor mu gradient (independent computation graph)
            sb_mu = torch.tensor(
                states_np[start:end], dtype=torch.float32, device=device, requires_grad=True
            )
            tb_mu = torch.tensor(
                time_states_np[start:end], dtype=torch.float32, device=device, requires_grad=True
            )
            mu_out, _, _ = model(sb_mu, tb_mu)
            mu_scalar = mu_out.sum()
            mu_scalar.backward()

            g_mu_s_chunks.append(sb_mu.grad.detach().cpu().numpy().astype(np.float32))
            g_mu_t_chunks.append(tb_mu.grad.detach().cpu().numpy().astype(np.float32))
            mu_chunks.append(mu_out.detach().squeeze(-1).cpu().numpy().astype(np.float32))
            del sb_mu, tb_mu, mu_out, mu_scalar

            # Pass 2: Critic value gradient (independent computation graph)
            sb_v = torch.tensor(
                states_np[start:end], dtype=torch.float32, device=device, requires_grad=True
            )
            tb_v = torch.tensor(
                time_states_np[start:end], dtype=torch.float32, device=device, requires_grad=True
            )
            _, _, v_out = model(sb_v, tb_v)
            v_scalar = v_out.sum()
            v_scalar.backward()

            g_v_s_chunks.append(sb_v.grad.detach().cpu().numpy().astype(np.float32))
            g_v_t_chunks.append(tb_v.grad.detach().cpu().numpy().astype(np.float32))
            v_chunks.append(v_out.detach().reshape(-1).cpu().numpy().astype(np.float32))
            del sb_v, tb_v, v_out, v_scalar
    finally:
        for p, flag in zip(model.parameters(), orig_req_grad):
            p.requires_grad_(flag)

    out = {
        "g_mu_states": np.concatenate(g_mu_s_chunks, axis=0),
        "g_v_states": np.concatenate(g_v_s_chunks, axis=0),
        "g_mu_time": np.concatenate(g_mu_t_chunks, axis=0),
        "g_v_time": np.concatenate(g_v_t_chunks, axis=0),
        "mu": np.concatenate(mu_chunks, axis=0),
        "value": np.concatenate(v_chunks, axis=0),
    }
    for k, arr in out.items():
        assert np.all(np.isfinite(arr)), f"Non-finite values detected in '{k}'"
    return out


def _batched_eval_outputs(
    model: MambaContinuousActorCritic,
    states_np: np.ndarray,
    time_states_np: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs no-grad batched forward passes returning (mu[N], value[N])."""
    n_samples = int(states_np.shape[0])
    mb = int(batch_size)
    mus: List[np.ndarray] = []
    vals: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n_samples, mb):
            end = min(start + mb, n_samples)
            sb = torch.as_tensor(states_np[start:end], dtype=torch.float32, device=device)
            tb = torch.as_tensor(time_states_np[start:end], dtype=torch.float32, device=device)
            mu_b, _, v_b = model(sb, tb)
            mus.append(mu_b.squeeze(-1).cpu().numpy().astype(np.float32))
            vals.append(v_b.reshape(-1).cpu().numpy().astype(np.float32))
    return np.concatenate(mus, axis=0), np.concatenate(vals, axis=0)


def _derangement_perm(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generates a random permutation without trivial identity mapping when n > 1."""
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    perm = rng.permutation(n)
    if np.array_equal(perm, np.arange(n)):
        perm = np.roll(perm, 1)
    return perm


def compute_feature_attribution(
    model: MambaContinuousActorCritic,
    states_np: np.ndarray,
    time_states_np: np.ndarray,
    grad_pack: Dict[str, np.ndarray],
    feature_names: Sequence[str],
    time_feature_names: Sequence[str],
    device: torch.device,
    batch_size: int = 32,
    perm_repeats: int = 3,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    A1: Computes 4-metric feature attribution + first-layer weight L2 norms
    for every channel in states (market + portfolio) and time_states, plus
    group-level permutation perturbations (market_all, portfolio_all, time_all).
    """
    n_samples, _, d_model = states_np.shape
    time_in = time_states_np.shape[2]
    assert len(feature_names) == d_model, (
        f"feature_names length ({len(feature_names)}) != d_model ({d_model})"
    )
    assert len(time_feature_names) == time_in, (
        f"time_feature_names length ({len(time_feature_names)}) != time_in ({time_in})"
    )
    assert int(perm_repeats) >= 1, "perm_repeats must be >= 1"

    g_mu_s = grad_pack["g_mu_states"]
    g_v_s = grad_pack["g_v_states"]
    g_mu_t = grad_pack["g_mu_time"]
    g_v_t = grad_pack["g_v_time"]
    mu0 = grad_pack["mu"]
    v0 = grad_pack["value"]

    # First-layer weight L2 norms
    with torch.no_grad():
        w_states_l2 = (
            model.market_embedding.weight.detach().float().norm(dim=0).cpu().numpy()
        )
        # SineActivation combines w (in_features, out-1) and w0 (in_features, 1)
        w_time_concat = torch.cat(
            [model.time_embedding.w.detach().float(), model.time_embedding.w0.detach().float()],
            dim=1,
        )
        w_time_l2 = w_time_concat.norm(dim=1).cpu().numpy()

    # Gradient-based metrics for states
    s_mean = states_np.mean(axis=(0, 1))
    s_std = states_np.std(axis=(0, 1))
    s_raw_mu = np.abs(g_mu_s).mean(axis=(0, 1))
    s_raw_v = np.abs(g_v_s).mean(axis=(0, 1))
    s_std_mu = s_raw_mu * s_std
    s_std_v = s_raw_v * s_std
    s_ixg_mu = np.abs(g_mu_s * states_np).mean(axis=(0, 1))
    s_ixg_v = np.abs(g_v_s * states_np).mean(axis=(0, 1))

    # Gradient-based metrics for time_states
    t_mean = time_states_np.mean(axis=(0, 1))
    t_std = time_states_np.std(axis=(0, 1))
    t_raw_mu = np.abs(g_mu_t).mean(axis=(0, 1))
    t_raw_v = np.abs(g_v_t).mean(axis=(0, 1))
    t_std_mu = t_raw_mu * t_std
    t_std_v = t_raw_v * t_std
    t_ixg_mu = np.abs(g_mu_t * time_states_np).mean(axis=(0, 1))
    t_ixg_v = np.abs(g_v_t * time_states_np).mean(axis=(0, 1))

    # Permutation importance (averaged over perm_repeats deterministic shuffles)
    rng = np.random.default_rng(int(seed) + 1009)
    perms = [_derangement_perm(rng, n_samples) for _ in range(int(perm_repeats))]

    s_perm_mu = np.zeros(d_model, dtype=np.float64)
    s_perm_v = np.zeros(d_model, dtype=np.float64)
    for j in range(d_model):
        dmu_acc, dv_acc = 0.0, 0.0
        for perm in perms:
            sp = states_np.copy()
            sp[:, :, j] = states_np[perm, :, j]
            mup, vp = _batched_eval_outputs(model, sp, time_states_np, device, batch_size)
            dmu_acc += float(np.abs(mup - mu0).mean())
            dv_acc += float(np.abs(vp - v0).mean())
        s_perm_mu[j] = dmu_acc / len(perms)
        s_perm_v[j] = dv_acc / len(perms)

    t_perm_mu = np.zeros(time_in, dtype=np.float64)
    t_perm_v = np.zeros(time_in, dtype=np.float64)
    for j in range(time_in):
        dmu_acc, dv_acc = 0.0, 0.0
        for perm in perms:
            tp = time_states_np.copy()
            tp[:, :, j] = time_states_np[perm, :, j]
            mup, vp = _batched_eval_outputs(model, states_np, tp, device, batch_size)
            dmu_acc += float(np.abs(mup - mu0).mean())
            dv_acc += float(np.abs(vp - v0).mean())
        t_perm_mu[j] = dmu_acc / len(perms)
        t_perm_v[j] = dv_acc / len(perms)

    # Group-level permutation perturbations
    market_dim = int(model.market_dim)
    group_perm: Dict[str, Dict[str, float]] = {}

    for gname in ("market_all", "portfolio_all", "time_all"):
        if gname == "portfolio_all" and market_dim >= d_model:
            group_perm[gname] = {"perm_dmu": 0.0, "perm_dv": 0.0}
            continue
        dmu_acc, dv_acc = 0.0, 0.0
        for perm in perms:
            if gname == "market_all":
                sp = states_np.copy()
                sp[:, :, :market_dim] = states_np[perm, :, :market_dim]
                mup, vp = _batched_eval_outputs(model, sp, time_states_np, device, batch_size)
            elif gname == "portfolio_all":
                sp = states_np.copy()
                sp[:, :, market_dim:] = states_np[perm, :, market_dim:]
                mup, vp = _batched_eval_outputs(model, sp, time_states_np, device, batch_size)
            else:
                tp = time_states_np[perm]
                mup, vp = _batched_eval_outputs(model, states_np, tp, device, batch_size)
            dmu_acc += float(np.abs(mup - mu0).mean())
            dv_acc += float(np.abs(vp - v0).mean())
        group_perm[gname] = {
            "perm_dmu": dmu_acc / len(perms),
            "perm_dv": dv_acc / len(perms),
        }

    states_rows: List[Dict[str, Any]] = []
    for i, name in enumerate(feature_names):
        states_rows.append(
            {
                "index": i,
                "name": str(name),
                "stream": "market" if i < market_dim else "portfolio",
                "x_mean": float(s_mean[i]),
                "x_std": float(s_std[i]),
                "weight_l2": float(w_states_l2[i]),
                "raw_grad_mu": float(s_raw_mu[i]),
                "std_grad_mu": float(s_std_mu[i]),
                "input_x_grad_mu": float(s_ixg_mu[i]),
                "perm_dmu": float(s_perm_mu[i]),
                "raw_grad_v": float(s_raw_v[i]),
                "std_grad_v": float(s_std_v[i]),
                "input_x_grad_v": float(s_ixg_v[i]),
                "perm_dv": float(s_perm_v[i]),
            }
        )

    time_rows: List[Dict[str, Any]] = []
    for i, name in enumerate(time_feature_names):
        time_rows.append(
            {
                "index": i,
                "name": str(name),
                "stream": "time",
                "x_mean": float(t_mean[i]),
                "x_std": float(t_std[i]),
                "weight_l2": float(w_time_l2[i]),
                "raw_grad_mu": float(t_raw_mu[i]),
                "std_grad_mu": float(t_std_mu[i]),
                "input_x_grad_mu": float(t_ixg_mu[i]),
                "perm_dmu": float(t_perm_mu[i]),
                "raw_grad_v": float(t_raw_v[i]),
                "std_grad_v": float(t_std_v[i]),
                "input_x_grad_v": float(t_ixg_v[i]),
                "perm_dv": float(t_perm_v[i]),
            }
        )

    return {
        "perm_repeats": int(perm_repeats),
        "states": states_rows,
        "time_states": time_rows,
        "group_permutation": group_perm,
    }


# =============================================================================
# A2: Temporal Saliency Profile
# =============================================================================
def _effective_horizon_bars(per_step_mass: np.ndarray, target_frac: float) -> int:
    """
    Minimum number of most-recent bars (counting backward from the last bar L-1)
    required to accumulate >= target_frac of total gradient mass.
    """
    total = float(per_step_mass.sum())
    if total <= 0.0:
        return 0
    rev_cum = np.cumsum(per_step_mass[::-1]) / total
    idx = int(np.searchsorted(rev_cum, target_frac, side="left"))
    return min(len(per_step_mass), idx + 1)


def compute_temporal_saliency_profile(
    g_mu_states: np.ndarray,
    g_v_states: np.ndarray,
    buckets: Sequence[Tuple[str, int, Optional[int]]] = TEMPORAL_BUCKETS,
) -> Dict[str, Any]:
    """
    A2: Aggregates |d(mu)/dx| and |d(V)/dx| along the temporal axis L,
    reporting normalized bucket shares (summing to 1.0) and 50%/80% effective lookback horizons.
    """
    assert g_mu_states.ndim == 3 and g_v_states.ndim == 3
    seq_len = int(g_mu_states.shape[1])

    # Per-timestep total absolute gradient mass across samples and features: shape (L,)
    mu_mass = np.abs(g_mu_states).sum(axis=(0, 2)).astype(np.float64)
    v_mass = np.abs(g_v_states).sum(axis=(0, 2)).astype(np.float64)
    mu_tot = float(mu_mass.sum())
    v_tot = float(v_mass.sum())

    bucket_shares_mu: Dict[str, float] = {}
    bucket_shares_v: Dict[str, float] = {}
    bucket_ranges: Dict[str, List[int]] = {}

    # Clamp negative indices cleanly into [0, seq_len] so shorter synthetic sequences also partition exactly
    for name, start_idx, end_idx in buckets:
        s = max(0, seq_len + start_idx) if start_idx < 0 else min(seq_len, start_idx)
        if end_idx is None:
            e = seq_len
        else:
            e = max(0, seq_len + end_idx) if end_idx < 0 else min(seq_len, end_idx)
        e = max(s, e)
        bucket_ranges[name] = [int(s), int(e)]
        bucket_shares_mu[name] = float(mu_mass[s:e].sum() / mu_tot) if mu_tot > 0.0 else 0.0
        bucket_shares_v[name] = float(v_mass[s:e].sum() / v_tot) if v_tot > 0.0 else 0.0

    return {
        "seq_len": seq_len,
        "bucket_ranges": bucket_ranges,
        "actor_mu_share": bucket_shares_mu,
        "critic_v_share": bucket_shares_v,
        "actor_bars_for_50pct": _effective_horizon_bars(mu_mass, 0.50),
        "actor_bars_for_80pct": _effective_horizon_bars(mu_mass, 0.80),
        "critic_bars_for_50pct": _effective_horizon_bars(v_mass, 0.50),
        "critic_bars_for_80pct": _effective_horizon_bars(v_mass, 0.80),
    }


# =============================================================================
# A3: Policy Behavior, Buy vs. Cash Comparison & Diagnostic Verdicts
# =============================================================================
def compute_policy_behavior_summary(
    states_np: np.ndarray,
    mu: np.ndarray,
    value: np.ndarray,
    model: MambaContinuousActorCritic,
    feature_names: Sequence[str],
    deadzone_threshold: float,
    buy_threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    A3: Summarizes policy distribution (log_std, mu/action quantiles, deadzone rate)
    and compares last-bar feature means between Buy (sigmoid(mu) > buy_threshold)
    and Cash (sigmoid(mu) <= buy_threshold) windows.
    """
    mu_arr = np.asarray(mu, dtype=np.float64).reshape(-1)
    v_arr = np.asarray(value, dtype=np.float64).reshape(-1)
    actions = 1.0 / (1.0 + np.exp(-np.clip(mu_arr, -60.0, 60.0)))

    with torch.no_grad():
        raw_log_std = float(model.log_std.detach().mean().item())
        clamped_log_std = float(
            torch.clamp(model.log_std.detach(), model.log_std_min, model.log_std_max)
            .mean()
            .item()
        )
        policy_std = float(np.exp(clamped_log_std))

    q_levels = [0.01, 0.10, 0.50, 0.90, 0.99]
    q_keys = ["p01", "p10", "p50", "p90", "p99"]
    mu_q = dict(zip(q_keys, [float(x) for x in np.quantile(mu_arr, q_levels)]))
    act_q = dict(zip(q_keys, [float(x) for x in np.quantile(actions, q_levels)]))
    val_q = dict(zip(q_keys, [float(x) for x in np.quantile(v_arr, q_levels)]))

    deadzone_mask = actions <= float(deadzone_threshold)
    buy_mask = actions > float(buy_threshold)
    cash_mask = ~buy_mask

    n_total = len(mu_arr)
    n_buy = int(buy_mask.sum())
    n_cash = int(cash_mask.sum())

    last_bar = states_np[:, -1, :]  # (N, d_model)
    buy_vs_cash: List[Dict[str, Any]] = []
    for i, name in enumerate(feature_names):
        buy_mean = float(last_bar[buy_mask, i].mean()) if n_buy > 0 else None
        buy_std = float(last_bar[buy_mask, i].std()) if n_buy > 0 else None
        cash_mean = float(last_bar[cash_mask, i].mean()) if n_cash > 0 else None
        cash_std = float(last_bar[cash_mask, i].std()) if n_cash > 0 else None
        diff = (
            float(buy_mean - cash_mean)
            if (buy_mean is not None and cash_mean is not None)
            else None
        )
        buy_vs_cash.append(
            {
                "name": str(name),
                "buy_mean": buy_mean,
                "buy_std": buy_std,
                "cash_mean": cash_mean,
                "cash_std": cash_std,
                "diff_buy_minus_cash": diff,
            }
        )

    return {
        "n_windows": int(n_total),
        "raw_log_std": raw_log_std,
        "clamped_log_std": clamped_log_std,
        "policy_std": policy_std,
        "mu_mean": float(mu_arr.mean()),
        "mu_std": float(mu_arr.std()),
        "mu_quantiles": mu_q,
        "action_mean": float(actions.mean()),
        "action_quantiles": act_q,
        "value_mean": float(v_arr.mean()),
        "value_std": float(v_arr.std()),
        "value_quantiles": val_q,
        "deadzone_threshold": float(deadzone_threshold),
        "deadzone_frac": float(deadzone_mask.mean()) if n_total > 0 else 0.0,
        "buy_threshold": float(buy_threshold),
        "n_buy": n_buy,
        "n_cash": n_cash,
        "buy_frac": float(n_buy / n_total) if n_total > 0 else 0.0,
        "buy_vs_cash": buy_vs_cash,
    }


def compute_verdicts(a1: Dict[str, Any], a3: Dict[str, Any]) -> Dict[str, Any]:
    """Derives automated, transparent diagnostic verdicts from A1 and A3."""
    market_rows = [r for r in a1["states"] if r["stream"] == "market"]
    sorted_v = sorted(market_rows, key=lambda r: r["perm_dv"], reverse=True)
    top1_v = sorted_v[0] if len(sorted_v) >= 1 else None
    top2_v = sorted_v[1] if len(sorted_v) >= 2 else None
    v_ratio = (
        float(top1_v["perm_dv"] / max(top2_v["perm_dv"], 1e-12))
        if (top1_v is not None and top2_v is not None)
        else 0.0
    )

    dead_channels = [
        r["name"]
        for r in market_rows
        if (
            r["perm_dmu"] < DEAD_CHANNEL_PERM_DMU_THRESHOLD
            and r["perm_dv"] < DEAD_CHANNEL_PERM_DV_THRESHOLD
        )
    ]

    deadzone_triggered = bool(
        a3["deadzone_frac"] >= DEADZONE_COLLAPSE_FRAC_THRESHOLD
        or a3["mu_quantiles"]["p50"] <= DEADZONE_COLLAPSE_MEDIAN_MU_THRESHOLD
    )
    critic_dom_triggered = bool(
        top1_v is not None
        and top1_v["perm_dv"] > 1e-4
        and v_ratio >= CRITIC_DOMINANCE_RATIO_THRESHOLD
    )

    return {
        "deadzone_collapse": {
            "triggered": deadzone_triggered,
            "deadzone_frac": a3["deadzone_frac"],
            "mu_p50": a3["mu_quantiles"]["p50"],
            "policy_std": a3["policy_std"],
            "threshold": (
                f"deadzone_frac >= {DEADZONE_COLLAPSE_FRAC_THRESHOLD:.2f} "
                f"or mu_p50 <= {DEADZONE_COLLAPSE_MEDIAN_MU_THRESHOLD:.1f}"
            ),
        },
        "critic_single_feature_dominance": {
            "triggered": critic_dom_triggered,
            "top1_feature": top1_v["name"] if top1_v else None,
            "top1_perm_dv": top1_v["perm_dv"] if top1_v else 0.0,
            "top2_feature": top2_v["name"] if top2_v else None,
            "top2_perm_dv": top2_v["perm_dv"] if top2_v else 0.0,
            "top1_over_top2_ratio": v_ratio,
            "threshold": f"top1/top2 >= {CRITIC_DOMINANCE_RATIO_THRESHOLD:.2f}",
        },
        "dead_channels": {
            "triggered": len(dead_channels) > 0,
            "count": len(dead_channels),
            "features": dead_channels,
            "threshold": (
                f"perm_dmu < {DEAD_CHANNEL_PERM_DMU_THRESHOLD} "
                f"and perm_dv < {DEAD_CHANNEL_PERM_DV_THRESHOLD}"
            ),
        },
    }


# =============================================================================
# Console Reporting & Top-Level Orchestrator
# =============================================================================
def _fmt_opt(val: Optional[float], fmt: str = "+.3f") -> str:
    return f"{val:{fmt}}" if val is not None else "N/A"


def print_report(report: Dict[str, Any]) -> None:
    """Prints formatted, aligned attribution tables and verdicts to stdout."""
    bar = "=" * 118
    ck = report["checkpoint_meta"]
    sm = report["sample_meta"]
    print(f"\n{bar}")
    print(
        f"PPO2 FEATURE ATTRIBUTION DIAGNOSTICS | ckpt={os.path.basename(ck['checkpoint_path'])} "
        f"(iter={ck['iteration']}, steps={ck['total_steps']})"
    )
    print(
        f"Split={sm['split']} | Windows={sm['n_windows']} ({sm['n_symbols_used']} symbols, L={sm['seq_len']}) | "
        f"PosMode={sm['position_mode']} | PermRepeats={report['A1']['perm_repeats']} | Device={report['device']}"
    )
    print(bar)

    # A1: States attribution (sorted by perm_dv desc for immediate diagnosis)
    print("\n[A1-a] States Feature Attribution (sorted by Critic Perm_dV desc):")
    header = (
        f"  {'Feature':<17} {'Stream':<9} {'x_std':>6} {'W_L2':>6} | "
        f"{'|dmu/dx|':>9} {'s*|dmu|':>9} {'|x*dmu|':>9} {'Perm_dmu':>9} | "
        f"{'|dV/dx|':>9} {'s*|dV|':>9} {'|x*dV|':>9} {'Perm_dV':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in sorted(report["A1"]["states"], key=lambda x: x["perm_dv"], reverse=True):
        print(
            f"  {r['name']:<17} {r['stream']:<9} {r['x_std']:>6.3f} {r['weight_l2']:>6.2f} | "
            f"{r['raw_grad_mu']:>9.5f} {r['std_grad_mu']:>9.5f} {r['input_x_grad_mu']:>9.5f} {r['perm_dmu']:>9.4f} | "
            f"{r['raw_grad_v']:>9.5f} {r['std_grad_v']:>9.5f} {r['input_x_grad_v']:>9.5f} {r['perm_dv']:>9.4f}"
        )

    print("\n[A1-b] Time States Attribution:")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in report["A1"]["time_states"]:
        print(
            f"  {r['name']:<17} {r['stream']:<9} {r['x_std']:>6.3f} {r['weight_l2']:>6.2f} | "
            f"{r['raw_grad_mu']:>9.5f} {r['std_grad_mu']:>9.5f} {r['input_x_grad_mu']:>9.5f} {r['perm_dmu']:>9.4f} | "
            f"{r['raw_grad_v']:>9.5f} {r['std_grad_v']:>9.5f} {r['input_x_grad_v']:>9.5f} {r['perm_dv']:>9.4f}"
        )

    gp = report["A1"]["group_permutation"]
    print("\n[A1-c] Group-Level Permutation Perturbation:")
    for gname, vals in gp.items():
        print(f"  {gname:<16} Perm_dmu={vals['perm_dmu']:.4f} | Perm_dV={vals['perm_dv']:.4f}")

    # A2: Temporal profile
    a2 = report["A2"]
    print(f"\n[A2] Temporal Saliency Profile (seq_len={a2['seq_len']}):")
    for bname, rng_pair in a2["bucket_ranges"].items():
        print(
            f"  {bname:<14} idx=[{rng_pair[0]:3d}:{rng_pair[1]:3d}) | "
            f"Actor mu share={a2['actor_mu_share'][bname] * 100:5.1f}% | "
            f"Critic V share={a2['critic_v_share'][bname] * 100:5.1f}%"
        )
    print(
        f"  Effective lookback horizon | Actor 50%/80%: {a2['actor_bars_for_50pct']}/{a2['actor_bars_for_80pct']} bars | "
        f"Critic 50%/80%: {a2['critic_bars_for_50pct']}/{a2['critic_bars_for_80pct']} bars"
    )

    # A3: Policy behavior & Buy vs. Cash
    a3 = report["A3"]
    mq = a3["mu_quantiles"]
    aq = a3["action_quantiles"]
    print("\n[A3] Policy Behavior & Buy vs. Cash Conditional Feature Means:")
    print(
        f"  log_std={a3['clamped_log_std']:+.4f} (std={a3['policy_std']:.4f}) | "
        f"Deadzone(<={a3['deadzone_threshold']}): {a3['deadzone_frac'] * 100:.1f}% | "
        f"Buy(>{a3['buy_threshold']}): {a3['n_buy']}/{a3['n_windows']} ({a3['buy_frac'] * 100:.1f}%)"
    )
    print(
        f"  mu quantiles [p01/p10/p50/p90/p99]: "
        f"[{mq['p01']:+.3f}, {mq['p10']:+.3f}, {mq['p50']:+.3f}, {mq['p90']:+.3f}, {mq['p99']:+.3f}]"
    )
    print(
        f"  act quantiles[p01/p10/p50/p90/p99]: "
        f"[{aq['p01']:.4f}, {aq['p10']:.4f}, {aq['p50']:.4f}, {aq['p90']:.4f}, {aq['p99']:.4f}]"
    )
    print(f"  {'Feature':<18} {'BuyMean':>9} {'CashMean':>9} {'Buy-Cash':>10}")
    for row in a3["buy_vs_cash"]:
        print(
            f"  {row['name']:<18} {_fmt_opt(row['buy_mean']):>9} "
            f"{_fmt_opt(row['cash_mean']):>9} {_fmt_opt(row['diff_buy_minus_cash']):>10}"
        )

    # Automated Verdicts
    print("\n[Verdicts]")
    for k, v in report["verdicts"].items():
        flag = "TRIGGERED" if v["triggered"] else "OK"
        details = ", ".join(
            f"{dk}={dv:.4f}" if isinstance(dv, float) else f"{dk}={dv}"
            for dk, dv in v.items()
            if dk not in ("triggered", "threshold")
        )
        print(f"  {k:<32} [{flag:<9}] {details}  ({v['threshold']})")


def run_diagnostics(
    checkpoint_path: str = CHECKPOINT_PATH,
    config_path: str = CONFIG_PATH,
    split: str = SPLIT,
    symbols: Optional[List[str]] = SYMBOLS,
    num_symbols: int = NUM_SYMBOLS,
    samples_per_symbol: int = SAMPLES_PER_SYMBOL,
    batch_size: int = BATCH_SIZE,
    perm_repeats: int = PERM_REPEATS,
    position_mode: str = POSITION_MODE,
    buy_threshold: float = BUY_THRESHOLD,
    device_spec: str = DEVICE,
    seed: int = SEED,
    output_path: Optional[str] = OUTPUT_PATH,
    save_json: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Executes the full A1/A2/A3 read-only attribution diagnostic pipeline."""
    cfg_full_path = resolve_path(config_path)
    assert os.path.isfile(cfg_full_path), f"Config file not found: {cfg_full_path}"
    cfg = OmegaConf.load(cfg_full_path)

    device = resolve_device(device_spec)
    model, ckpt_meta = load_model_from_checkpoint(checkpoint_path, cfg, device)
    states_np, time_states_np, sample_meta = sample_observation_windows(
        cfg=cfg,
        split=split,
        symbols=symbols,
        num_symbols=num_symbols,
        samples_per_symbol=samples_per_symbol,
        position_mode=position_mode,
        seed=seed,
    )

    feature_names = get_state_feature_names(ckpt_meta["portfolio_dim"])
    time_feature_names = list(BasicState.STATIONARY_TIME_FEATURES)

    grad_pack = compute_input_gradients(
        model=model,
        states_np=states_np,
        time_states_np=time_states_np,
        device=device,
        batch_size=batch_size,
    )
    a1 = compute_feature_attribution(
        model=model,
        states_np=states_np,
        time_states_np=time_states_np,
        grad_pack=grad_pack,
        feature_names=feature_names,
        time_feature_names=time_feature_names,
        device=device,
        batch_size=batch_size,
        perm_repeats=perm_repeats,
        seed=seed,
    )
    a2 = compute_temporal_saliency_profile(
        g_mu_states=grad_pack["g_mu_states"],
        g_v_states=grad_pack["g_v_states"],
    )
    a3 = compute_policy_behavior_summary(
        states_np=states_np,
        mu=grad_pack["mu"],
        value=grad_pack["value"],
        model=model,
        feature_names=feature_names,
        deadzone_threshold=float(cfg.DEADZONE_THRESHOLD),
        buy_threshold=buy_threshold,
    )
    verdicts = compute_verdicts(a1, a3)

    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "config_path": cfg_full_path,
        "checkpoint_meta": ckpt_meta,
        "sample_meta": sample_meta,
        "A1": a1,
        "A2": a2,
        "A3": a3,
        "verdicts": verdicts,
    }

    if verbose:
        print_report(report)

    if save_json:
        out_file = (
            resolve_path(output_path)
            if output_path is not None
            else os.path.join(
                REPO_ROOT,
                "results",
                "feature_attribution",
                f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
            )
        )
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        report["output_path"] = out_file
        if verbose:
            print(f"\n[Feature Attribution] JSON report written to: {out_file}")

    return report


def main() -> None:
    run_diagnostics()


if __name__ == "__main__":
    main()
