# Specification: PPO2 Anomaly Symbol Filtering, Volatility Floor Guard, and Comprehensive Convergence Diagnostics

- **Category**: `feature`
- **Spec Path**: `spec/feature/ppo2_symbol_filtering_and_training_metrics.md`
- **Status**: `Implemented & Verified`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
During analysis of the 78-iteration (323,584 steps, 632 episodes) training log ([`train.log`](file:///home/b0457812963/Mamba3RL/SynapseX/train.log)) produced by [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py), two critical obstacles prevented clear assessment of model convergence and degraded Critic stability:

1. **Severe Reward & Value Loss Distortion from Non-Directional / Ultra-Low-Volatility Symbols**:
   In [`State_time_step.calculate_asset_sigma`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L66-L81) and [`DownsidePenaltyReward.step`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/reward.py#L20-L25), single-step net log returns $r_t = p_{t-1} r_{\text{gap}} + p_t r_{\text{bar}} - \text{cost}_t$ are normalized by ex-ante rolling 96-bar volatility $\sigma_t$ with a tiny floor of `1e-5` (`0.001%`), followed by an asymmetric quadratic penalty:
   $$z_t = \frac{r_t}{\max(\sigma_t, 10^{-5})}, \quad \text{reward}_t = \text{clip}\left(z_t - \frac{1}{2} \lambda_{\text{risk}} \min(0, z_t)^2, -5.0, +5.0\right)$$
   A full scan of all 179 CSV instruments in `Brain/simulation/train_data` revealed three distinct anomaly tiers:
   - **Tier 1 — Pegged Stablecoins & Fixed-Income Par Preferred Stock (Must Remove)**:
     - **`USDCUSDT-F-30-Min`** (USD Stablecoin pair): Price is pegged within `[0.9539, 1.0039]`, median 30-min volatility is $\sigma_{\text{med}} = 0.000054$ (`99.92%` of 96-bar windows have $\sigma_t < 0.0005$), and **99.5% of bars move less than round-trip friction (`0.04%`)**. Whenever [`SymbolQueue`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L31-L50) sampled `USDCUSDT` (Ep #11, #313, #411, #577), friction divided by $\sigma_t \approx 10^{-5}$ hit the `-5.0` clip on nearly every bar, producing episode returns of **`-1873.62`, `-1440.35`, `-1419.11`, and `-585.78` (mean `-1329.71`)**. These 4 episodes caused Critic `VfLoss` to explode from `~2.6` to **`228.90`, `139.58`, `135.67`, and `24.27`**, collapsing `ExplVar` to `0.00`.
     - **`STRCUSDT-F-30-Min`** (Strategy Perpetual Preferred Stock, \$100 par fixed-income instrument): In the second half of its history, price converges asymptotically toward par (`$97.0–$99.1`) with bond-like low volatility (`39.1%` of windows have $\sigma_t < 0.0010$, `10.4%` have $\sigma_t < 0.0005$). In [`train.log`](file:///home/b0457812963/Mamba3RL/SynapseX/train.log), `STRCUSDT` was the second worst outlier across all 179 symbols, with all 4 episodes yielding **`-181.08`, `-178.53`, `-140.56`, and `-91.16` (mean `-147.83`)**, more than $4\times$ worse than normal instruments (`~ -35`).
   - **Tier 2 — Ultra-Low-Volatility Broad Market ETFs & Tokenized Gold**:
     - **`SPYUSDT-F-30-Min`** ($\sigma_{\text{med}} = 0.00083$, `69.8%` windows $< 0.0010$, mean return `-109.99`), **`PAXGUSDT-F-30-Min`** ($\sigma_{\text{med}} = 0.00146$, mean return `-93.65`), **`QQQUSDT-F-30-Min`** ($\sigma_{\text{med}} = 0.00133$, mean return `-83.44`), and **`XAUTUSDT-F-30-Min`** ($\sigma_{\text{med}} = 0.00157$, mean return `-82.52`).
     - During off-market hours or quiet regimes, $\sigma_t$ drops to `0.0002–0.0005`, amplifying transaction costs by $5\times\text{--}10\times$ compared to standard crypto perpetuals ($\sigma_{\text{med}} \approx 0.0045$). Raising the volatility denominator floor from `1e-5` to a configurable `MIN_ASSET_SIGMA: 0.001` (`0.1%`) bounds friction amplification while keeping configurable symbol exclusion in [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml).

2. **Insufficient Training Observability (`train.log` Metrics Gap)**:
   - Currently, [`PPO2.train`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L346-L403) only prints synthetic `Return` (which conflates directional PnL, transaction costs, volatility scaling, and quadratic downside penalties) and `End Pos` (a single step's position at $t = 512$).
   - Because `Return` is negatively biased by $-0.5 \lambda_{\text{risk}} z_t^2$, even a break-even policy with Gaussian exploration (`Std ≈ 0.59`) logs `Return ≈ -35`, making it impossible to see:
     - Whether actual unpenalized portfolio return (`NetPnL%` and `GrossPnL%`) is improving or turning positive.
     - How much return is lost to transaction costs (`Cost%`) and how frequently the agent rebalances (`Trades` count and `Turnover` $\sum |\Delta p_t|$).
     - Whether the agent is holding extreme positions across the entire episode (`Avg|Pos|`) or only at the final bar (`End Pos`).
     - Whether rolling performance is improving on a robust outlier-resistant basis (`MedRet(100)`, `NetPnL%(100)`, `Win%(100)`) and whether gradient norms (`GradNorm`) remain healthy during [`PPO2.update_ppo`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L153-L254).

### 1.2 Current vs. Expected Behavior
| Dimension | Current Behavior | Expected Behavior |
| :--- | :--- | :--- |
| **Symbol Filtering (`PPO2._prepare_symbols`)** | Loads all 179 CSVs indiscriminately, including stablecoin `USDCUSDT-F-30-Min` and fixed-income preferred stock `STRCUSDT-F-30-Min`. | Filters out symbols listed in `cfg.training.EXCLUDE_SYMBOLS` (defaulting to `["USDCUSDT-F-30-Min", "STRCUSDT-F-30-Min"]`) prior to preloading and `SymbolQueue` initialization. |
| **Volatility Floor (`calculate_asset_sigma` & `DownsidePenaltyReward`)** | Hardcodes `1e-5` (`0.001%`) floor, allowing low-volatility windows to amplify `0.02%` friction cost by up to $20\times$ before quadratic penalty. | Uses configurable `MIN_ASSET_SIGMA` (default `0.001` = `0.1%`) from `ppo_v2.yaml` across `State_time_step` and `DownsidePenaltyReward`. |
| **Episode-Level Logging (`[Ep #xxxx]`)** | Only logs `Steps`, `Return`, `End Pos`, `TotalSteps`. | Logs `Steps`, `Return`, `NetPnL%`, `Cost%`, `Trades`, `Turnover`, `Avg\|Pos\|`, `EndPos`, and `TotalSteps`. |
| **Iteration-Level Logging (`[Iter #xxxx]`)** | Only logs `PiLoss`, `VfLoss`, `Ent`, `KL`, `ClipFrac`, `ExplVar`, `Std`, and `AvgRet(20)`. | Logs `PiLoss`, `VfLoss`, `Ent`, `KL`, `ClipFrac`, `ExplVar`, `Std`, `GradNorm`, plus rolling convergence summary: `AvgRet(20)`, `MedRet(100)`, `NetPnL%(100)`, `Cost%(100)`, `Win%(100)`, `Trades(100)`, `Avg\|Pos\|(100)`. |
| **KL Early Stopping** | Not implemented. | Explicitly kept **Out-of-Scope** per user directive (no early stopping added to `update_ppo`). |

### 1.3 Technical Root Cause & References
- Unfiltered symbol discovery and preload: [`PPO2_rl_train.py:L73-L100`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L73-L100)
- Hardcoded `1e-5` volatility floor in environment and reward: [`Brain/PPO2/lib/environment.py:L66-L81`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L66-L81), [`Brain/PPO2/lib/reward.py:L14-L25`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/reward.py#L14-L25)
- Missing step-level PnL/cost/turnover metadata in `State_time_step.step` and `BaseTradingEnv.step`: [`Brain/PPO2/lib/environment.py:L83-L186`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L83-L186), [`L261-L277`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L261-L277)
- Uncaptured gradient norm and minimal episode/iteration metrics in `PPO2.update_ppo` and `PPO2.train`: [`PPO2_rl_train.py:L153-L254`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L153-L254), [`L279-L403`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L279-L403)

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix
| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) | **In-Scope** | Add `EXCLUDE_SYMBOLS` list and `MIN_ASSET_SIGMA: 0.001` under environment parameters. |
| **Item 2** | [`Brain/PPO2/lib/reward.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/reward.py) | **In-Scope** | Add `min_sigma: float = 1e-3` parameter to `DownsidePenaltyReward.__init__` and `step()`. |
| **Item 3** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) | **In-Scope** | Wire `min_sigma` into `State_time_step` and `TrainingEnv`; track step-level accounting (`last_step_return`, `last_gross_return`, `last_cost`, `last_turnover`, `last_traded`) and include them in `BaseTradingEnv.step()` `info` dict. |
| **Item 4** | [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) | **In-Scope** | Filter `EXCLUDE_SYMBOLS` in `_prepare_symbols`; capture `grad_norm` in `update_ppo`; accumulate and log enriched per-episode and rolling 100-episode convergence metrics in `train()`. |
| **Item 5** | [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) & [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py) | **In-Scope** | Forward `min_sigma` from `ppo_v2.yaml` to `State_time_step` for strict training/evaluation config alignment. |
| **Item 6** | [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) | **Out-of-Scope** | Neural network architecture remains untouched. |
| **Item 7** | [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py) | **Out-of-Scope** | Policy sampling and action clipping remain untouched. |
| **Item 8** | [`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py) | **Out-of-Scope** | GAE and rollout buffer logic remain untouched. |

### 2.2 Architectural Boundaries & Non-Goals
- **In-Scope Boundaries**:
  - Configurable symbol exclusion (`EXCLUDE_SYMBOLS`) and volatility floor (`MIN_ASSET_SIGMA`).
  - Non-intrusive step accounting metadata exposed via the existing Gym 5-tuple `info` dictionary.
  - Comprehensive episode-level and iteration-level convergence diagnostics in stdout / `train.log`.
- **Out-of-Scope (Non-Goals)**:
  - **No KL Early Stopping (`target_kl`)**: Per explicit user instruction, `PPO2.update_ppo` will execute all configured `PPO_EPOCHS` without early termination.
  - **No Return Signature Changes**: `State_time_step.step(action)` continues to return the 3-tuple `(reward, terminated, truncated)` and `BaseTradingEnv.step(action)` continues to return `(obs, reward, terminated, truncated, info)`.

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Update Configuration in [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml)**
  - [x] Add `EXCLUDE_SYMBOLS` list containing `["USDCUSDT-F-30-Min", "STRCUSDT-F-30-Min"]`.
  - [x] Add `MIN_ASSET_SIGMA: 0.001` under `# Environment Penalty & Rebalance Parameters`.
- [x] **Step 2: Add Configurable `min_sigma` to [`Brain/PPO2/lib/reward.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/reward.py)**
  - [x] Update `DownsidePenaltyReward.__init__` to accept `min_sigma: float = 1e-3` with assertion `assert min_sigma > 0.0`.
  - [x] Use `self.min_sigma` in `DownsidePenaltyReward.step()`.
- [x] **Step 3: Enrich Step Accounting & Wire `min_sigma` in [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)**
  - [x] Add `min_sigma: float = 1e-3` to `State_time_step.__init__`, pass it to `DownsidePenaltyReward`, and apply `max(sigma, self.min_sigma)` in `calculate_asset_sigma()`.
  - [x] Track `last_step_return`, `last_gross_return`, `last_cost`, `last_turnover`, and `last_traded` on `State_time_step` during `reset()` and `step()`.
  - [x] Expose `step_return`, `gross_return`, `cost`, `turnover`, and `traded` in the `info` dict inside `BaseTradingEnv.step()`.
  - [x] Wire `min_sigma` in `TrainingEnv.__init__`, `PPO2Engine`, and `PPO2Evaluator`.
- [x] **Step 4: Implement Symbol Exclusion & Rich Convergence Metrics in [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)**
  - [x] Filter out `EXCLUDE_SYMBOLS` in `PPO2._prepare_symbols` and log both excluded and active symbol counts.
  - [x] Record pre-clipping total gradient norm (`grad_norm`) in `PPO2.update_ppo` and return `grad_norm` in the metrics dict.
  - [x] Track per-worker episode accumulators (`episode_net_pnl`, `episode_cost`, `episode_trades`, `episode_turnover`, `episode_abs_pos_sum`) and print enriched `[Ep #xxxx]` and `[Iter #xxxx]` diagnostic lines.
- [x] **Step 5: Clean Up & Defensiveness**
  - [x] Ensure all new comments and docstrings are written in English in compliance with `.agents/rules/rules.md`.
  - [x] Add defensive assertions for positive `min_sigma` and non-empty filtered `unique_symbols`.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml)
```yaml
SAVES_TAG: saves
BARS_COUNT: 300
UNIQUE_SYMBOLS:
EXCLUDE_SYMBOLS:
  - "USDCUSDT-F-30-Min"
  - "STRCUSDT-F-30-Min"
```
And under `# Environment Penalty & Rebalance Parameters`:
```yaml
# Environment Penalty & Rebalance Parameters
DEADZONE_THRESHOLD: 0.01
REBALANCE_THRESHOLD: 0.03
W1_STEP_RETURN: 1.0
VOL_WINDOW: 96
MIN_ASSET_SIGMA: 0.001
RISK_LAMBDA: 0.5
REWARD_CLIP: 5.0
```

#### 3.2.2 [`Brain/PPO2/lib/reward.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/reward.py)
```python
class DownsidePenaltyReward:
    def __init__(
        self,
        risk_lambda: float = 0.5,
        clip: float = 5.0,
        min_sigma: float = 1e-3,
    ):
        assert risk_lambda >= 0.0, "risk_lambda must be non-negative"
        assert clip > 0.0, "clip must be positive"
        assert min_sigma > 0.0, "min_sigma must be positive"
        self.risk_lambda = float(risk_lambda)
        self.clip = float(clip)
        self.min_sigma = float(min_sigma)

    def step(self, step_return: float, asset_sigma: float) -> float:
        safe_sigma = max(float(asset_sigma), self.min_sigma)
        z_return = float(step_return) / safe_sigma
        downside = min(0.0, z_return)
        reward = z_return - 0.5 * self.risk_lambda * (downside * downside)
        return float(np.clip(reward, -self.clip, self.clip))
```

#### 3.2.3 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)
In `State_time_step`:
```python
        # Per-step diagnostic accounting attributes
        self.last_step_return = 0.0
        self.last_gross_return = 0.0
        self.last_cost = 0.0
        self.last_turnover = 0.0
        self.last_traded = False
```
In `State_time_step.step()`:
```python
        gross_p_return = float(prev_pos * gap_log_return + target_pos * bar_log_return)
        current_p_return = float(gross_p_return - cost_t)

        self.last_step_return = current_p_return
        self.last_gross_return = gross_p_return
        self.last_cost = float(cost_t)
        self.last_turnover = float(abs(delta_p))
        self.last_traded = bool(abs(delta_p) > 1e-8)
```
In `BaseTradingEnv.step()`:
```python
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
```

#### 3.2.4 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)
In `_prepare_symbols()`:
```python
        all_symbols = sorted(list(set(f.split(".")[0] for f in symbol_files)))
        exclude_cfg = getattr(self.cfg_training, "EXCLUDE_SYMBOLS", None) or []
        exclude_set = set(str(s) for s in exclude_cfg)
        unique_symbols = [s for s in all_symbols if s not in exclude_set]
        assert len(unique_symbols) > 0, "No symbols remaining after applying EXCLUDE_SYMBOLS"
```
In `update_ppo()`:
```python
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.model.parameters(), max_grad_norm
                )
                self.optimizer.step()
                grad_norms.append(float(grad_norm.item()))
```
In `train()` episode & iteration logging:
```python
# Episode print format:
# [Ep #0001 | W0] Symbol: ETHUSDT-F-30-Min     | Steps: 0512 | Return: -24.73 | NetPnL: -1.42% | Cost: 0.85% | Trades: 112 | Avg|Pos|: 0.48 | EndPos: -1.00 | TotalSteps: 0004096

# Iteration print format:
# [Iter #0001] Steps: 0004096 | FPS: 425.0 | PiLoss: -0.0045 | VfLoss: 2.5720 | Ent: 0.9170 | KL: 0.00249 | ClipFrac: 0.024 | ExplVar: +0.175 | Std: 0.605 | GradNorm: 0.412 | AvgRet(20): -34.12 | MedRet(100): -33.50 | NetPnL(100): -1.85% | Cost(100): 1.12% | Win(100): 18.0% | Trades(100): 145.2 | Avg|Pos|(100): 0.51
```

---

## 4. Validation & Verification Plan

### 4.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST adhere to repository environment standards using the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python`
> Relative path: `../bin/python`

```bash
# 1. Verify existing PPO2 environment test suite passes without regressions:
../bin/python -m unittest test_ppo2_env.py

# 2. Verify symbol filtering, min_sigma floor guard, step info accounting, and grad_norm tracking:
../bin/python -c "
import numpy as np
from omegaconf import OmegaConf
from Brain.PPO2.lib.reward import DownsidePenaltyReward
from Brain.PPO2.lib.environment import State_time_step, ProductionEnv
from test_ppo2_env import create_dummy_prices

cfg = OmegaConf.load('configs/training/ppo_v2.yaml')
assert 'USDCUSDT-F-30-Min' in cfg.EXCLUDE_SYMBOLS
assert 'STRCUSDT-F-30-Min' in cfg.EXCLUDE_SYMBOLS
assert np.isclose(cfg.MIN_ASSET_SIGMA, 0.001)

r_fn = DownsidePenaltyReward(risk_lambda=cfg.RISK_LAMBDA, clip=cfg.REWARD_CLIP, min_sigma=cfg.MIN_ASSET_SIGMA)
# With min_sigma=0.001, a 0.0002 friction loss on a flat bar (sigma=1e-5) yields z = -0.2 instead of z = -20.0
r_val = r_fn.step(-0.0002, 1e-5)
assert np.isclose(r_val, -0.2 - 0.5 * 0.5 * 0.04), f'Unexpected reward: {r_val}'

state = State_time_step(
    bars_count=10,
    commission_perc=0.0001,
    model_train=True,
    default_slippage=0.0001,
    N_steps=50,
    min_sigma=cfg.MIN_ASSET_SIGMA,
)
env = ProductionEnv({'TEST': create_dummy_prices(100)}, state)
obs = env.reset('TEST')
obs, reward, terminated, truncated, info = env.step(0.5)
for key in ['step_return', 'gross_return', 'cost', 'turnover', 'traded']:
    assert key in info, f'Missing key in info: {key}'
assert info['traded'] is True and np.isclose(info['turnover'], 0.5) and np.isclose(info['cost'], 0.0001)
print('All verification checks passed!')
"
```

### 4.2 Acceptance Verification Checklist
- [x] **1. Functional Correctness**:
  - [x] `EXCLUDE_SYMBOLS` properly filters out `USDCUSDT-F-30-Min` and `STRCUSDT-F-30-Min` (reducing 179 symbols to 177 clean symbols).
  - [x] `MIN_ASSET_SIGMA: 0.001` prevents micro-volatility denominator explosion during quiet market hours.
  - [x] `BaseTradingEnv.step()` returns accurate `step_return`, `gross_return`, `cost`, `turnover`, and `traded` fields in `info`.
- [x] **2. Boundary & Edge Case Handling**:
  - [x] Existing `test_ppo2_env.py` helper compatibility and `PPO2Engine` config binding pass 100% with zero regressions.
  - [x] `PPO2.update_ppo` does not perform early stopping (`PPO_EPOCHS` runs to completion).
- [x] **3. Runtime & Empirical Verification**:
  - [x] Episode and Iteration log lines display all new convergence and PnL diagnostics cleanly.
