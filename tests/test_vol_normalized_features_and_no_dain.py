"""
Comprehensive verification suite for Option B:
  - Volatility-normalized stationary 14-dim market features (DataFeature.py / BasicState)
  - Causal cleaning without bfill look-ahead bias
  - Train/Inference bit-for-bit observation parity (State_time_step vs PPO2DataAdapter)
  - Removal of DAIN_Layer from MambaContinuousActorCritic & global gradient clipping
  - Full-dataset empirical validation across all non-stablecoin CSV files
"""

import glob
import os
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from omegaconf import OmegaConf

from Brain.Common.DataFeature import (
    OriginalDataFeature,
    Prices,
    SOFT_CLIP_LIMIT,
    VOL_REF,
)
from Brain.Common.env_components import BasicState
from Brain.Common.grad_diagnostics import build_param_groups, group_grad_norms
from Brain.Common.optimization import build_optimizer
from Brain.PPO2.lib.Agent import PPO2Agent
from Brain.PPO2.lib.data_adapter import PPO2DataAdapter
from Brain.PPO2.lib.environment import State_time_step


def _make_synthetic_ohlcv(
    n: int = 800,
    base_price: float = 100.0,
    sigma: float = 0.008,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2025-01-01 00:00:00", periods=n, freq="30min")
    rets = rng.normal(0.0, sigma, size=n)
    close = base_price * np.exp(np.cumsum(rets))
    gap = rng.normal(0.0, sigma * 0.1, size=n)
    open_p = np.roll(close, 1) * np.exp(gap)
    open_p[0] = base_price
    up = np.abs(rng.normal(0.0, sigma * 0.5, size=n))
    dn = np.abs(rng.normal(0.0, sigma * 0.5, size=n))
    high = np.maximum(open_p, close) * np.exp(up)
    low = np.minimum(open_p, close) * np.exp(-dn)
    quote_av = rng.lognormal(mean=10.0, sigma=0.7, size=n)
    volume = quote_av / close
    trades = np.maximum(1.0, np.round(rng.lognormal(mean=5.0, sigma=0.6, size=n)))
    tb_ratio = rng.uniform(0.3, 0.7, size=n)
    tb_quote_av = quote_av * tb_ratio
    tb_base_av = volume * tb_ratio

    return pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "quote_av": quote_av,
            "trades": trades,
            "tb_base_av": tb_base_av,
            "tb_quote_av": tb_quote_av,
        },
        index=dt.astype(str),
    )


def test_single_source_of_truth_and_shapes():
    assert len(BasicState.MARKET_FEATURES) == 14
    assert len(BasicState.STATIONARY_TIME_FEATURES) == 6
    assert len(Prices._fields) == 24
    assert Prices._fields == (
        "open",
        "high",
        "low",
        "close",
        *BasicState.MARKET_FEATURES,
        *BasicState.STATIONARY_TIME_FEATURES,
    )

    state = State_time_step(
        bars_count=300,
        commission_perc=0.0005,
        model_train=True,
        default_slippage=0.0005,
        N_steps=512,
        deadzone_threshold=0.03,
        rebalance_threshold=0.05,
        w1_step_return=1.0,
        downside_window_size=60,
        w2_downside_ratio=0.25,
        weight_scale=100.0,
    )
    assert state.info_list == list(BasicState.MARKET_FEATURES)
    assert state.timelist == list(BasicState.STATIONARY_TIME_FEATURES)
    assert state.getStateShape() == (300, 15)
    assert state.getTimeShape() == (300, 6)


def test_price_scale_invariance():
    """Scaling OHLC by 1e-4 or 1e+4 must leave all 14 market features invariant (< 1e-6)."""
    df_real = pd.read_csv("Brain/simulation/train_data/UBUSDT-F-30-Min.csv")
    df_real.set_index("Datetime", inplace=True)

    loader = OriginalDataFeature()
    base_prices = loader.get_train_net_work_data_by_pd("UB", df_real)["UB"]

    for scale in (1e-4, 1e4):
        df_scaled = df_real.copy()
        for col in ("Open", "High", "Low", "Close"):
            df_scaled[col] = df_scaled[col] * scale
        scaled_prices = OriginalDataFeature().get_train_net_work_data_by_pd(
            "UB", df_scaled
        )["UB"]
        for feat in BasicState.MARKET_FEATURES:
            diff = np.max(
                np.abs(getattr(base_prices, feat) - getattr(scaled_prices, feat))
            )
            assert diff < 1e-6, f"Scale {scale} broke invariance for {feat}: diff={diff}"


def test_strict_causality_zero_lookahead():
    """Perturbing bars strictly after index t must produce 0.0 change on bars <= t."""
    df = _make_synthetic_ohlcv(n=700, base_price=50.0, sigma=0.01, seed=7)
    p_orig = OriginalDataFeature().get_train_net_work_data_by_pd("SYN", df)["SYN"]

    # Choose cut index well after 360 warmup
    raw_cut = 550
    df_mod = df.copy()
    df_mod.iloc[raw_cut:, df_mod.columns.get_loc("Open")] *= 3.7
    df_mod.iloc[raw_cut:, df_mod.columns.get_loc("High")] *= 4.2
    df_mod.iloc[raw_cut:, df_mod.columns.get_loc("Low")] *= 0.25
    df_mod.iloc[raw_cut:, df_mod.columns.get_loc("Close")] *= 2.9
    df_mod.iloc[raw_cut:, df_mod.columns.get_loc("quote_av")] *= 15.0
    df_mod.iloc[raw_cut:, df_mod.columns.get_loc("trades")] *= 8.0
    df_mod.iloc[raw_cut:, df_mod.columns.get_loc("tb_quote_av")] *= 0.1

    p_mod = OriginalDataFeature().get_train_net_work_data_by_pd("SYN", df_mod)["SYN"]

    # First valid row after SMA-360 dropna corresponds to raw index 359
    valid_prefix_len = raw_cut - 359
    assert valid_prefix_len > 100
    for feat in BasicState.MARKET_FEATURES:
        arr_orig = getattr(p_orig, feat)[:valid_prefix_len]
        arr_mod = getattr(p_mod, feat)[:valid_prefix_len]
        assert np.array_equal(
            arr_orig, arr_mod
        ), f"Look-ahead leak detected in feature '{feat}'!"


def test_candlestick_geometry_and_zero_point_semantics():
    df = _make_synthetic_ohlcv(n=800, base_price=1.2, sigma=0.012, seed=19)
    loader = OriginalDataFeature()
    cleaned = loader.cleanData(df)
    enriched = loader.add_market_features(cleaned)

    assert np.all(enriched["upper_wick"].to_numpy() >= 0.0)
    assert np.all(enriched["lower_wick"].to_numpy() >= 0.0)

    # Zero-point sign preservation for candle_body
    bull = enriched["Close"] > enriched["Open"]
    bear = enriched["Close"] < enriched["Open"]
    assert np.all(enriched.loc[bull, "candle_body"] > 0.0)
    assert np.all(enriched.loc[bear, "candle_body"] < 0.0)

    # Zero-point sign preservation for all SMA deviations
    close_s = cleaned["Close"]
    for p in (30, 60, 120, 240, 360):
        sma = close_s.rolling(window=p).mean().loc[enriched.index]
        above = enriched["Close"] > sma
        below = enriched["Close"] < sma
        assert np.all(enriched.loc[above, f"ma_dev_{p}"] > 0.0)
        assert np.all(enriched.loc[below, f"ma_dev_{p}"] < 0.0)


def test_volatility_regime_awareness():
    """High-volatility series must produce strictly higher vol_regime_abs than low-vol."""
    df_low = _make_synthetic_ohlcv(n=700, base_price=100.0, sigma=0.001, seed=101)
    df_high = _make_synthetic_ohlcv(n=700, base_price=100.0, sigma=0.020, seed=101)

    p_low = OriginalDataFeature().get_train_net_work_data_by_pd("LOW", df_low)["LOW"]
    p_high = OriginalDataFeature().get_train_net_work_data_by_pd("HIGH", df_high)["HIGH"]

    assert np.median(p_low.vol_regime_abs) < 0.0
    assert np.median(p_high.vol_regime_abs) > 0.0
    assert np.median(p_high.vol_regime_abs) - np.median(p_low.vol_regime_abs) > 2.0


def test_boundary_zero_vol_bad_prices_and_extreme_spike():
    df = _make_synthetic_ohlcv(n=600, base_price=10.0, sigma=0.008, seed=33)

    # Leading NaN / <=0 prices: must be dropped, NEVER bfilled from future
    df.iloc[0, df.columns.get_loc("Close")] = np.nan
    df.iloc[1, df.columns.get_loc("Open")] = -5.0
    df.iloc[2, df.columns.get_loc("Low")] = 0.0
    # Mid-series inf and zero-volume bars
    df.iloc[400, df.columns.get_loc("Close")] = np.inf
    df.iloc[450:455, df.columns.get_loc("Volume")] = 0.0
    df.iloc[450:455, df.columns.get_loc("quote_av")] = 0.0
    df.iloc[450:455, df.columns.get_loc("trades")] = 0.0
    df.iloc[450:455, df.columns.get_loc("tb_quote_av")] = 0.0
    df.iloc[450:455, df.columns.get_loc("tb_base_av")] = 0.0
    # Extreme 100-sigma spike
    df.iloc[520, df.columns.get_loc("Close")] *= 5.0
    df.iloc[520, df.columns.get_loc("High")] *= 6.0

    loader = OriginalDataFeature()
    cleaned = loader.cleanData(df)
    # Leading 3 invalid rows dropped (not bfilled)
    assert len(cleaned) == len(df) - 3
    assert cleaned.index[0] == df.index[3]

    prices = loader.get_train_net_work_data_by_pd("EDGE", df)["EDGE"]
    for feat in BasicState.MARKET_FEATURES:
        arr = getattr(prices, feat)
        assert np.all(np.isfinite(arr)), f"Non-finite values in {feat}"
        assert np.all(np.abs(arr) <= SOFT_CLIP_LIMIT + 1e-12), f"Out of soft_clip bounds in {feat}"

    # Zero-volume rows must have taker_imbalance == 0.0
    zero_vol_indices = [
        i for i, idx in enumerate(loader.df.index) if idx in df.index[450:455]
    ]
    assert len(zero_vol_indices) == 5
    for idx in zero_vol_indices:
        assert prices.taker_imbalance[idx] == 0.0


def test_train_vs_inference_and_entry_point_parity():
    """
    Verify:
      1. get_train_net_work_data_by_path == get_train_net_work_data_by_pd bit-for-bit.
      2. State_time_step.encode() == PPO2DataAdapter.extract_latest_obs() bit-for-bit.
    """
    sym = "ETHUSDT-F-30-Min"
    p_path = OriginalDataFeature().get_train_net_work_data_by_path(
        [sym], end_date="2025-12-31 23:59:59"
    )[sym]

    df_raw = pd.read_csv(f"Brain/simulation/train_data/{sym}.csv")
    df_raw_cut = df_raw[
        pd.to_datetime(df_raw["Datetime"]) <= pd.Timestamp("2025-12-31 23:59:59")
    ].copy()
    df_idx = df_raw_cut.set_index("Datetime")
    p_pd = OriginalDataFeature().get_train_net_work_data_by_pd(sym, df_idx)[sym]
    p_adapter = PPO2DataAdapter.process_df_to_prices(OriginalDataFeature(), df_raw_cut)

    for field in Prices._fields:
        assert np.array_equal(getattr(p_path, field), getattr(p_pd, field)), f"path vs pd mismatch: {field}"
        assert np.array_equal(getattr(p_path, field), getattr(p_adapter, field)), f"path vs adapter mismatch: {field}"

    bars_count = 300
    pos_val = 0.65
    state = State_time_step(
        bars_count=bars_count,
        commission_perc=0.0005,
        model_train=False,
        default_slippage=0.0005,
        N_steps=512,
        deadzone_threshold=0.03,
        rebalance_threshold=0.05,
        w1_step_return=1.0,
        downside_window_size=60,
        w2_downside_ratio=0.25,
        weight_scale=100.0,
    )
    state.reset(p_adapter, offset=len(p_adapter.close))
    state.position = pos_val
    obs_env = state.encode()
    obs_inf = PPO2DataAdapter.extract_latest_obs(
        p_adapter, bars_count=bars_count, position=pos_val
    )

    assert obs_env["states"].shape == (bars_count, 15)
    assert obs_env["time_states"].shape == (bars_count, 6)
    assert np.array_equal(obs_env["states"], obs_inf["states"])
    assert np.array_equal(obs_env["time_states"], obs_inf["time_states"])


def test_model_without_dain_forward_backward_and_grad_diagnostics():
    cfg = OmegaConf.load("configs/training/ppo_v2.yaml")
    assert not hasattr(cfg, "BASE_LR"), "BASE_LR should be removed from ppo_v2.yaml"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bars_count = int(cfg.BARS_COUNT)
    ob_space = spaces.Dict(
        {
            "states": spaces.Box(
                low=-np.inf, high=np.inf, shape=(bars_count, 15), dtype=np.float32
            ),
            "time_states": spaces.Box(
                low=-np.inf, high=np.inf, shape=(bars_count, 6), dtype=np.float32
            ),
        }
    )
    ac_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    ssm_cfg = OmegaConf.to_container(cfg.SSM_CFG, resolve=True)

    agent = PPO2Agent(
        ob_space=ob_space,
        ac_space=ac_space,
        device=device,
        seq_dim=bars_count,
        hidden_size=int(cfg.HIDDEN_SIZE),
        nlayers=int(cfg.NLAYERS),
        time_features_out=int(cfg.TIME_FEATURES_OUT),
        init_log_std=float(cfg.INIT_LOG_STD),
        portfolio_dim=int(cfg.PORTFOLIO_DIM),
        ssm_cfg=ssm_cfg,
        log_std_min=float(cfg.LOG_STD_MIN),
        log_std_max=float(cfg.LOG_STD_MAX),
    )

    assert not hasattr(agent.model, "dean"), "MambaContinuousActorCritic must not contain DAIN ('dean')"
    for name, _ in agent.model.named_parameters():
        assert "dean" not in name, f"Residual DAIN parameter found: {name}"

    optimizer = build_optimizer(
        agent.model, lr=float(cfg.LEARNING_RATE), weight_decay=float(cfg.WEIGHT_DECAY)
    )
    # Without DAIN, optimizer has only 2 parameter groups (decay & no_decay)
    assert len(optimizer.param_groups) == 2

    param_groups = build_param_groups(agent.model)
    assert len(param_groups["dain_mean"]) == 0
    assert len(param_groups["dain_scale"]) == 0
    assert len(param_groups["dain_gate"]) == 0

    # Build real observation batch from ETHUSDT & UBUSDT
    obs_states = []
    obs_times = []
    for sym in ("ETHUSDT-F-30-Min", "UBUSDT-F-30-Min"):
        df = pd.read_csv(f"Brain/simulation/train_data/{sym}.csv")
        p = PPO2DataAdapter.process_df_to_prices(OriginalDataFeature(), df)
        o = PPO2DataAdapter.extract_latest_obs(p, bars_count=bars_count, position=0.5)
        obs_states.append(o["states"])
        obs_times.append(o["time_states"])

    states_t = torch.as_tensor(np.stack(obs_states, axis=0), dtype=torch.float32, device=device)
    time_t = torch.as_tensor(np.stack(obs_times, axis=0), dtype=torch.float32, device=device)

    out = agent.get_action({"states": states_t, "time_states": time_t}, stochastic=True)
    log_prob, entropy, values = agent.evaluate_actions(states_t, time_t, out.action)

    loss = -(log_prob.mean()) + 0.5 * values.pow(2).mean() - 0.001 * entropy.mean()
    assert torch.isfinite(loss), f"Non-finite loss: {loss}"

    optimizer.zero_grad()
    loss.backward()
    gn = group_grad_norms(param_groups)
    assert gn["dain"] == 0.0
    assert gn["total"] > 0.0 and np.isfinite(gn["total"])
    assert gn["actor"] > 0.0 and gn["critic"] > 0.0 and gn["backbone"] > 0.0

    clipped = torch.nn.utils.clip_grad_norm_(
        agent.model.parameters(), float(cfg.MAX_GRAD_NORM)
    )
    assert np.isfinite(float(clipped.item()))
    optimizer.step()


def test_full_real_dataset_empirical_audit():
    """All 177 non-stablecoin CSVs must be 100% finite, bounded in [-5, 5], and O(1) aligned."""
    cfg = OmegaConf.load("configs/training/ppo_v2.yaml")
    exclude = set(cfg.EXCLUDE_SYMBOLS or [])
    files = sorted(glob.glob("Brain/simulation/train_data/*.csv"))

    stds_per_sym = []
    tested = 0
    for f in files:
        sym = os.path.basename(f).replace(".csv", "")
        if sym in exclude:
            continue
        df = pd.read_csv(f)
        if len(df) <= 360:
            continue
        prices = PPO2DataAdapter.process_df_to_prices(OriginalDataFeature(), df)
        arr = np.column_stack(
            [getattr(prices, col).astype(np.float32) for col in BasicState.MARKET_FEATURES]
        )
        assert arr.shape[0] > 0
        assert np.all(np.isfinite(arr)), f"Non-finite feature in {sym}"
        assert np.all(np.abs(arr) <= SOFT_CLIP_LIMIT + 1e-6), f"Out of bounds in {sym}"
        stds_per_sym.append(arr.std(axis=0))
        tested += 1

    assert tested >= 170, f"Expected >= 170 symbols tested, got {tested}"
    std_mat = np.vstack(stds_per_sym)
    med_stds = dict(zip(BasicState.MARKET_FEATURES, np.median(std_mat, axis=0)))
    for col in ("candle_body", "ma_dev_30", "ma_dev_60", "ma_dev_120", "ma_dev_240", "ma_dev_360"):
        assert 0.7 <= med_stds[col] <= 1.1, f"{col} median std out of O(1) target: {med_stds[col]:.3f}"


if __name__ == "__main__":
    test_single_source_of_truth_and_shapes()
    test_price_scale_invariance()
    test_strict_causality_zero_lookahead()
    test_candlestick_geometry_and_zero_point_semantics()
    test_volatility_regime_awareness()
    test_boundary_zero_vol_bad_prices_and_extreme_spike()
    test_train_vs_inference_and_entry_point_parity()
    test_model_without_dain_forward_backward_and_grad_diagnostics()
    test_full_real_dataset_empirical_audit()
    print("ALL VERIFICATION CHECKS PASSED!")
