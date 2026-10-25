# Specification: PPO2 Long-Only Portfolio Accounting, Calendar Leakage Removal & Out-of-Time Split

- **Category**: `bugfix`
- **Spec Path**: `spec/bugfix/ppo2_env_accounting_and_temporal_split.md`
- **Status**: `Implemented & Verified`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
Empirical evaluation of `checkpoint_iter_6570.pt` on `AAVEUSDT` (`104,086` 30-minute bars from `2020-10-16` to `2026-09-24`) exposed five compounding defects across action-space / portfolio return accounting, temporal feature leakage, training/testing dataset partitioning, fee realism, and live inference parity:

1. **Biased Linear-in-Position Log-Return Accounting & Asymmetric Short Tail Risk**:
   [`State_time_step.step()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L178-L184) previously operated over `[-1.0, 1.0]` and computed portfolio log return as $p \cdot \ln(P_1 / P_0) = p \cdot \ln(1 + r)$ instead of the true compounded equity log return $\ln(1 + p \cdot r)$, injecting $\frac{p(p-1)}{2}r^2$ bias ($+r^2 > 0$ phantom volatility subsidy on shorts, inflating gross `AAVEUSDT` equity by `32.2x`, while also mispricing fractional sizing $p \in (0, 1)$). Furthermore, in `[-1.0, 1.0]`, flat cash (`0.0`) sits as an interior knife-edge (`[-0.01, +0.01]`), causing stochastic Gaussian exploration to flip-flop across zero (`Position == 0.0` on only `0.01%` of test bars) while exposing shorts to unbounded intra-bar squeeze wicks (`+80%` to `+237.7%` in 30 minutes).
   - **Architectural Resolution (Long-Only `[0.0, 1.0]`)**: Transition PPO2 to a **Long-or-Cash continuous allocation space `[0.0, 1.0]`** with exact simple-to-log portfolio equity compounding. In `[0.0, 1.0]`:
     - `0.0` (100% USDT cash) becomes the **left clamp boundary**, allowing Gaussian exploration to park cleanly in cash with zero turnover fees during bearish/choppy regimes.
     - Unleveraged long equity multiplier $(1 - p) + p(P_1 / P_0)$ is strictly positive for all $P > 0$, completely eliminating short-squeeze bankruptcy, intra-bar liquidation complexity, and negative-argument `np.log` singularities.
2. **Cross-Sectional Calendar Leakage & New-Listing Sampling Skew**:
   `Brain/simulation/train_data` and `Brain/simulation/test_data` are partitioned by symbol across the identical calendar span (`2020` to `2026-09-24`). Because 68.4% of `train_data` symbols were listed in 2025–2026 and [`SymbolQueue`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L31-L50) samples every symbol equally, 84.74% of training episodes fell in 2025–2026. Combined with absolute calendar coordinates (`age_log_minutes`, `age_years`, `month_sin/cos`, `day_sin/cos`, `week_sin/cos`) in [`BasicState.timelist`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py#L41-L56), the network memorized 2025–2026 market-wide crypto rally/crash dates (shifting calendar dates by 97 days collapsed 2026 Sharpe from `+1.16` to `-0.43`).
3. **Unrealistic Fee Friction in Training Config**:
   [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L7-L8) sets one-way fee + slippage to `2 bps` (`0.0001 + 0.0001`), far below realistic Binance USDⓈ-M Perpetual Futures VIP0 market-order friction (`~7 bps`: `0.00045` taker fee + `0.00025` slippage).
4. **Formal Live Inference State-Loss Bug**:
   [`PPO2Engine._extract_latest_obs()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L240-L279) and [`get_if_order_map()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L441-L529) hardcode `[position, unrealized_return, trade_bar/48]` to `[0.0, 0.0, 0.0]` and bypass `rebalance_threshold`, breaking parity with stateful [`ProductionEnv`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L378-L401).
5. **Missing `Open` Price in Exported Order CSVs**:
   [`PPO2Evaluator.evaluate_symbol()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L66-L143) executes trades at `Open_t` inside `State_time_step` but exports only `Close_t`, creating a 30-minute execution-price mismatch for downstream external backtesters.

### 1.2 Current vs. Expected Behavior
| Dimension | Current Behavior | Expected Behavior (Long-Only `[0.0, 1.0]`) |
| :--- | :--- | :--- |
| **Action & Position Space** | Bidirectional `Box(-1.0, 1.0)`; interior `0.0` cash knife-edge; short flip-flop whipsaws | Long-or-Cash `Box(0.0, 1.0)`; left-boundary `0.0` cash state (`clipped_action <= deadzone_threshold -> 0.0`); zero short exposure |
| **Portfolio Step Return** | $p_{\text{prev}} \ln(\frac{O_t}{C_{t-1}}) + p_{\text{target}} \ln(\frac{C_t}{O_t}) - \text{cost}_t$ | Exact compounded equity log return $\ln\big(\max((1 + p_{\text{prev}} r_{\text{gap}})(1 - |\Delta p| c)(1 + p_{\text{target}} r_{\text{bar}}), 10^{-8})\big)$ |
| **Unrealized PnL Feature** | $p \cdot (\ln C_{t-1} - \ln P_{\text{open}})$ | Exact equity log ratio $\ln\big(\max(1 + p \cdot \frac{C_{t-1} - P_{\text{open}}}{P_{\text{open}}}, 10^{-8})\big)$ (strictly $> 0$ for $p \in (0, 1]$) |
| **Time Features (`timelist`)** | 14 dims including `age_years`, `age_log_minutes`, `month`, `day`, `week` | Overridden in `State_time_step` to 6 stationary cyclical dims (`hour_sin/cos`, `minute_sin/cos`, `dayofweek_sin/cos`) |
| **Temporal Split** | Full overlap up to `2026-09-24` across `train_data` and `test_data` | Dual Out-of-Sample via `pd.to_datetime`: `train_data` truncated at `TRAIN_END_DATE` (`2025-12-31 23:59:59`); `test_data` evaluated within `[TEST_START_DATE, TEST_END_DATE]` (`2026-01-01 00:00:00` onward) using prior bars for 300-bar lookback |
| **Symbol Sampling** | Equal weight per symbol regardless of bar count; stale `UNIQUE_SYMBOLS` | Length-weighted round-robin in `SymbolQueue` (`1..5` slots); `cfg_training.UNIQUE_SYMBOLS` synchronized with filtered `preloaded_data` keys |
| **CSV Schema & Engine Robustness** | Exports `[Datetime, Symbol, Close, Action, Position]`; Formal mode zeroes portfolio state; crashes on legacy 14-dim `.pt` | Exports `[Datetime, Symbol, Open, Close, Action, Position]`; Formal mode accepts `portfolio_state` + applies `rebalance_threshold`; `run_test()` gracefully skips incompatible legacy checkpoints and empty evaluation DataFrames |

### 1.3 Technical Root Cause & References
- Bidirectional action space & biased linear-in-position log return: [`Brain/PPO2/lib/environment.py:L104-L191`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L104-L191), [`L260-L291`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L260-L291)
- Hardcoded `[-1.0, 1.0]` action clamping in agent & training loop: [`Brain/PPO2/lib/Agent.py:L94`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L94), [`PPO2_rl_train.py:L331`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L331)
- Biased unrealized log return in observation encoding: [`Brain/PPO2/lib/environment.py:L229-L243`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L229-L243)
- Calendar-memorizable time feature list inherited from `BasicState`: [`Brain/Common/env_components.py:L41-L56`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py#L41-L56)
- Unfiltered full-history training preload & unweighted `SymbolQueue`: [`Brain/Common/DataFeature.py:L77-L95`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L77-L95), [`PPO2_rl_train.py:L31-L114`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L31-L114)
- Missing `Open` export & `TEST_START_DATE` / `TEST_END_DATE` window handling: [`Brain/PPO2/lib/Evaluator.py:L51-L143`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L51-L143)
- Hardcoded `Box(-1.0, 1.0)`, zeroed portfolio features, rigid column casing & unguarded legacy checkpoint loading in `PPO2Engine`: [`Brain/PPO2/lib/engine.py:L204`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L204), [`L240-L375`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L240-L375), [`L402-L529`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L402-L529)
- Underestimated friction in config: [`configs/training/ppo_v2.yaml:L7-L8`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L7-L8)

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix
| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) | **In-Scope** | 1. Override `self.timelist` in `State_time_step` to 6 cyclical features.<br/>2. Switch `BaseTradingEnv.action_space` and clamping to Long-Only `[0.0, 1.0]`.<br/>3. Implement exact simple-to-log Long-Only equity compounding in `step()` and aligned `unrealized_return` in `encode()`.<br/>4. Support `train_end_date` in `TrainingEnv` and `start_offset` in `ProductionEnv.reset()`. |
| **Item 2** | [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py) | **In-Scope** | Clamp `clipped_action` dynamically using `self.ac_space.low` and `self.ac_space.high` in `get_action()` instead of hardcoded `[-1.0, 1.0]`. |
| **Item 3** | [`Brain/Common/DataFeature.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py) | **In-Scope** | Add backward-compatible `end_date: Optional[str] = None` (parsed safely via `pd.to_datetime`) and `min_bars: int = 0` filtering with short-DataFrame guards in `get_train_net_work_data_by_path()`. |
| **Item 4** | [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) | **In-Scope** | Pass `TRAIN_END_DATE` and `min_bars` in `_prepare_symbols()`; synchronize `self.cfg_training.UNIQUE_SYMBOLS` with retained `preloaded_data` keys; upgrade `SymbolQueue` to bar-length-weighted round-robin (`max_weight=5`); update fallback action clip to `[0.0, 1.0]`. |
| **Item 5** | [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py) | **In-Scope** | Export `Open` column (`[Datetime, Symbol, Open, Close, Action, Position]`); support `test_start_date` and optional `test_end_date` (`pd.to_datetime` safe); clamp actions to `[0.0, 1.0]` and apply `prev_position` + `rebalance_threshold` in `evaluate_single_step()`. |
| **Item 6** | [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py), [`PPO2_rl_test.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py) | **In-Scope** | Set `ac_space = Box(0.0, 1.0)` in `build_agent()`; align `_extract_latest_obs()` with 6-dim `timelist` and real `portfolio_state`; normalize lowercase OHLCV/date columns; gracefully warn & skip incompatible legacy 14-dim checkpoints and empty `order_df` in `run_test()`; wire `test_start_date` / `test_end_date` (with optional CLI overrides in `PPO2_rl_test.py` for validation/test windowing). |
| **Item 7** | [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) | **In-Scope** | Set `MODEL_DEFAULT_COMMISSION_PERC_TRAING: 0.00045`, `DEFAULT_SLIPPAGE: 0.00025`, `TRAIN_END_DATE: "2025-12-31 23:59:59"`, `TEST_START_DATE: "2026-01-01 00:00:00"`, `TEST_END_DATE: null`. |
| **Item 8** | `tests/test_ppo2_accounting_and_split.py` | **In-Scope [NEW]** | Automated unit tests covering Long-Only `[0.0, 1.0]` clamping, fractional long compounding, 6-dim time features, `TRAIN_END_DATE` / `TEST_START_DATE` / `TEST_END_DATE` isolation, CSV `Open` schema, legacy checkpoint skip safety, and formal portfolio state parity. |
| **Item 9** | [`Brain/Common/env_components.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py), [`DQN_rl_test.py`](file:///home/b0457812963/Mamba3RL/SynapseX/DQN_rl_test.py), [`TradingSysteam.py`](file:///home/b0457812963/Mamba3RL/SynapseX/TradingSysteam.py) | **Out-of-Scope** | Strictly preserved to guarantee zero regression on legacy DQN, A2C, HopeDQN, and EIIE pipelines. |
| **Item 10** | [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) | **Out-of-Scope** | Actor variance bounds & `[0.0, 1.0]` interval soft margin regularization are covered in [`spec/feature/ppo2_actor_margin_bound_and_std_control.md`](file:///home/b0457812963/Mamba3RL/SynapseX/spec/feature/ppo2_actor_margin_bound_and_std_control.md). |

### 2.2 Architectural Boundaries & Non-Goals
- **Long-Only Allocation Invariant (`position in [0.0, 1.0]`)**:
  Shorting (`position < 0.0`) and leverage `> 1.0x` are excluded by design. Because $1 + p \cdot r = (1 - p) + p(P_1 / P_0) > 0$ for all $p \in [0.0, 1.0]$ and positive prices, intra-bar liquidation mechanics (`LIQUIDATION_EQUITY_FLOOR`) are unnecessary and intentionally omitted.
- **Intentional Checkpoint Dimension Break (`time_features_in: 14 -> 6`)**:
  Reducing `State_time_step.timelist` from 14 features to 6 stationary cyclical features changes `ob_space["time_states"].shape` from `(300, 14)` to `(300, 6)`. `PPO2Engine.run_test()` will catch `RuntimeError` during `load_checkpoint()`, print a clear warning, and skip legacy 14-dim checkpoints instead of crashing.
- **Dual Out-of-Sample & Validation Window Support**:
  `train_data` and `test_data` remain disjoint by symbol, while `TRAIN_END_DATE` (`2025-12-31 23:59:59`) and `TEST_START_DATE` (`2026-01-01 00:00:00`) enforce strict chronological separation. Optional `TEST_END_DATE` (and CLI `--test_start_date` / `--test_end_date`) allows evaluating checkpoints on `test_data` during `2025` as an out-of-symbol validation set before touching `2026` holdout data.

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Upgrade `State_time_step` & `BaseTradingEnv` to Long-Only `[0.0, 1.0]` Accounting & 6-Dim Time Features**
  - [x] Override `self.timelist` in `State_time_step.__init__` to `STATIONARY_TIME_FEATURES` (`6` cyclical features).
  - [x] Update `BaseTradingEnv.action_space` to `spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)` and clip actions to `[0.0, 1.0]` in both `BaseTradingEnv.step()` and `State_time_step.step()`.
  - [x] Simplify `State_time_step.step()` position transitions for `[0.0, 1.0]` (remove negative flip branch) and compute exact compounded equity multipliers across gap, rebalance friction, and bar close.
  - [x] Refactor `State_time_step.encode()` `unrealized_return` to $\ln(\max(1.0 + p \cdot (C_{t-1} - P_{\text{open}}) / P_{\text{open}}, 10^{-8}))$ when `self.position > 0.0 and self.open_price > 0.0`.
  - [x] Update `ProductionEnv.reset(symbol=None, start_offset=None)` to support custom starting offset $\ge \text{bars\_count}$, and wire `train_end_date` in `TrainingEnv._load_data_for_instrument()`.
  - [x] Update `PPO2Agent.get_action()` in [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py) to clamp `clipped_action` dynamically via `self.ac_space.low` / `self.ac_space.high`.
- [x] **Step 2: Add Timestamp-Safe Temporal Cutoff & Weighted `SymbolQueue` in `DataFeature.py` and `PPO2_rl_train.py`**
  - [x] Extend `OriginalDataFeature.get_train_net_work_data_by_path()` with `end_date: Optional[str] = None` (using `pd.to_datetime` comparison) and `min_bars: int = 0`, skipping symbols with insufficient pre-cutoff rows (`<= 360` raw rows or `< min_bars` post-MA rows).
  - [x] Update `PPO2._prepare_symbols()` to pass `TRAIN_END_DATE` and `min_bars = BARS_COUNT + EPISODE_STEPS + 2`, then synchronize `self.cfg_training.UNIQUE_SYMBOLS = sorted(list(self.preloaded_data.keys()))` (asserting non-empty).
  - [x] Upgrade `SymbolQueue` to accept optional `symbol_weights: Optional[Dict[str, int]] = None` (`1..5` slots per cycle proportional to usable training bars), and update fallback `np.clip` in `PPO2.train()` to `[0.0, 1.0]`.
- [x] **Step 3: Align `PPO2Evaluator`, `PPO2Engine`, `PPO2_rl_test.py`, and `ppo_v2.yaml`**
  - [x] In `Evaluator.py`, export `Open` column (`[Datetime, Symbol, Open, Close, Action, Position]`), resolve `start_offset` and `end_offset` via `pd.to_datetime` for `test_start_date` / `test_end_date`, skip writing CSV when zero evaluation bars remain, and apply `[0.0, 1.0]` clip + `rebalance_threshold` in `evaluate_single_step()`.
  - [x] In `engine.py`, set `ac_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)`, update `_extract_latest_obs()` to 6-dim `STATIONARY_TIME_FEATURES` and real `(position, open_price, trade_bar)` state, normalize lowercase OHLCV/date columns in `_process_df_to_prices()`, expose `portfolio_state` in `get_if_order_map()`, and guard `run_test()` against incompatible legacy checkpoints and empty `order_df`.
  - [x] Add optional `--test_start_date` and `--test_end_date` CLI flags in `PPO2_rl_test.py` forwarding to `PPO2Engine`.
  - [x] Update `configs/training/ppo_v2.yaml` with `MODEL_DEFAULT_COMMISSION_PERC_TRAING: 0.00045`, `DEFAULT_SLIPPAGE: 0.00025`, `TRAIN_END_DATE: "2025-12-31 23:59:59"`, `TEST_START_DATE: "2026-01-01 00:00:00"`, and `TEST_END_DATE: null`.
- [x] **Step 4: Verification Suite Creation & Execution**
  - [x] Create `tests/test_ppo2_accounting_and_split.py` and run with `/home/b0457812963/Mamba3RL/bin/python`.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)
```python
STATIONARY_TIME_FEATURES = [
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "dayofweek_sin",
    "dayofweek_cos",
]
```
Inside `State_time_step.__init__`:
```python
        self.timelist = list(STATIONARY_TIME_FEATURES)
```
Inside `State_time_step.step(self, action: float)`:
```python
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
```
Inside `State_time_step.encode(self)`:
```python
        if self.position > 0.0 and self.open_price > 0.0:
            prev_close = max(float(self._prices.close[self._offset - 1]), 1e-8)
            safe_open = max(float(self.open_price), 1e-8)
            raw_equity_ratio = 1.0 + self.position * ((prev_close - safe_open) / safe_open)
            data_res[:, len(self.info_list) + 1] = float(np.log(max(raw_equity_ratio, 1e-8)))
            data_res[:, len(self.info_list) + 2] = float(self.trade_bar) / 48.0
        else:
            data_res[:, len(self.info_list) + 1] = 0.0
            data_res[:, len(self.info_list) + 2] = 0.0
```

#### 3.2.2 [`Brain/Common/DataFeature.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py)
```python
    def get_train_net_work_data_by_path(
        self,
        symbolNames: list,
        typeName: str = "train_data",
        end_date: str = None,
        min_bars: int = 0,
    ) -> dict:
        out_dict = {}
        cutoff_ts = pd.to_datetime(end_date) if end_date is not None else None
        for symbolName in symbolNames:
            df = pd.read_csv(f"Brain/simulation/{typeName}/{symbolName}.csv")
            df.set_index("Datetime", inplace=True)
            if cutoff_ts is not None:
                df = df[pd.to_datetime(df.index) <= cutoff_ts]
            if len(df) <= 360:
                continue
            self.df = self.cleanData(df)
            self.df = self.add_average_metric(self.df)
            if len(self.df) < max(1, int(min_bars)):
                continue
            self.df = self.add_time_feature(self.df)
            out_dict[symbolName] = self.load_relative()
        return out_dict
```

#### 3.2.3 [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py)
- Export columns: `["Datetime", "Symbol", "Open", "Close", "Action", "Position"]`.
- Convert `datetime_series` via `pd.to_datetime(datetime_series)` when resolving `test_start_date` (`side="left"`) and optional `test_end_date` (`side="right"`), clamping `start_offset = max(self.bars_count, start_idx)` and `end_offset = min(len(prices.close) - 1, end_idx)`. If `start_offset >= end_offset`, return an empty 6-column DataFrame without writing an empty CSV artifact.

---

## 4. Validation & Verification Plan

### 4.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST adhere to repository environment standards using the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python`
> Relative path: `../bin/python`

```bash
/home/b0457812963/Mamba3RL/bin/python -m unittest tests/test_ppo2_accounting_and_split.py
```

### 4.2 Acceptance Verification Checklist
- [x] **1. Long-Only Bounds & Exact Fractional Compounding**:
  - Passing negative actions (e.g., `-1.0`, `-0.35`) to `BaseTradingEnv.step()` or `State_time_step.step()` strictly clamps `position` to `0.0` with zero fees when starting from cash.
  - Holding fractional `position = 0.5` with zero fees across a $+25\%$ bar ($100 \to 125$) followed by a $-20\%$ bar ($125 \to 100$) yields cumulative step log return exactly equal to $\ln(1.125 \times 0.90) = \ln(1.0125) \approx +0.012423$ (exact rebalanced equity).
  - Holding `position = 1.0` across an extreme intra-bar flash wick (`Open = 100, Low = 15, Close = 95`) does NOT trigger false liquidation and settles at exact $\ln(0.95)$ (minus entry cost if newly opened).
- [x] **2. Calendar Feature & Out-of-Time Split Isolation**:
  - `State_time_step.getTimeShape()` returns `(300, 6)`.
  - Loading `train_data` with `end_date="2025-12-31 23:59:59"` (or date-only `"2025-12-31"`) and `min_bars=814` cleanly filters out post-2025 symbols without `AssertionError` or `KeyError`, synchronizes `UNIQUE_SYMBOLS`, and ensures every retained bar timestamp $\le$ cutoff.
  - Evaluating `AAVEUSDT` with `test_start_date="2026-01-01 00:00:00"` produces its first CSV row at `2026-01-01 00:00:00` with schema `[Datetime, Symbol, Open, Close, Action, Position]`.
- [x] **3. Formal Live Mode State Parity & Legacy Checkpoint Safety**:
  - `PPO2Engine._extract_latest_obs()` with `(position, open_price, trade_bar)` produces bit-for-bit identical `states` and `time_states` tensors to `State_time_step.encode()`, applies `rebalance_threshold` in `evaluate_single_step()`, and handles lowercase DataFrame columns (`date`, `open`, `high`, `low`, `close`, `volume`) without `KeyError`.
  - `PPO2Engine.run_test()` logs a warning and skips incompatible 14-dim checkpoints without aborting evaluation of valid 6-dim models.
