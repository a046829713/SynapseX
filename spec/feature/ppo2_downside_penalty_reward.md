# Specification: PPO2 Ex-Ante Volatility-Normalized Downside Penalty Reward & Dead Code Cleanup

- **Category**: `feature`
- **Spec Path**: `spec/feature/ppo2_downside_penalty_reward.md`
- **Status**: `Implemented & Verified`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
In multi-instrument reinforcement learning ([`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)), raw logarithmic returns (`current_p_return`) exhibit vastly different volatility scales across cryptocurrency pairs (e.g., BTCUSDT vs. high-beta altcoins). Furthermore, the legacy rolling-window downside risk penalty (`calculate_step_downside_penalty`) suffered from non-Markovian window-boundary artifacts when historical drawdowns exited the 60-bar sliding window.

To address these limitations, [`Brain/PPO2/lib/reward.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/reward.py) and [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) introduce an **Ex-Ante Volatility-Normalized Asymmetric Downside Penalty Reward (`DownsidePenaltyReward`)**:
1. **Ex-Ante Asset Volatility (`calculate_asset_sigma`)**: Computes the standard deviation $\sigma_t$ of historical log-returns strictly prior to the current bar (`close[offset - vol_window - 1 : offset]`), preventing look-ahead bias.
2. **Z-Score Normalization & Quadratic Downside Penalty**:
   $$z_t = \frac{r_t}{\sigma_t}, \quad \text{reward}_t = \text{clip}\left(z_t - \frac{1}{2} \lambda_{\text{risk}} \min(0, z_t)^2, -C, +C\right)$$
   Because $z_t \propto p_t$, the quadratic term $\min(0, z_t)^2 \propto p_t^2$ penalizes oversized losing positions without penalizing profitable trend holding.

Following the latest code inspection, four remaining issues must be resolved to complete this feature cleanly:
1. **High-Frequency Debug Prints in `reward.py`**: [`DownsidePenaltyReward.step()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/reward.py#L21-L26) contains three `print()` calls per step, generating $8 \times 512 \times 3 = 12,288$ stdout writes per rollout iteration and severely degrading `SubprocVecEnv` throughput.
2. **Legacy Rolling Downside Dead Code in `environment.py`**: [`State_time_step.step()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L204-L208) still appends to `self.return_history` and computes `downside_penalty = self.calculate_step_downside_penalty()`, discarding the result on every step.
3. **Unconnected Config Parameters**: While `VOL_WINDOW: 96`, `RISK_LAMBDA: 0.5`, and `REWARD_CLIP: 5.0` were added to [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L23-L25), they are not yet wired into [`TrainingEnv.__init__`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L339-L350), [`PPO2Engine`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L70-L84), or [`PPO2Evaluator`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L18-L43).
4. **Mini-Batch Shuffling Disabled in `PPO2_rl_train.py`**: [`PPO2.update_ppo()`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L173) was modified to `shuffle=False`, causing mini-batches to contain temporally contiguous, highly correlated slices rather than i.i.d. samples across the horizon.

### 1.2 Current vs. Expected Behavior
| Dimension | Current Behavior | Expected Behavior |
| :--- | :--- | :--- |
| **Reward Computation (`reward.py`)** | Executes 3 `print()` statements on every step; no floor guard inside `DownsidePenaltyReward.step()`. | Silent, vectorized/fast scalar math with defensive `max(asset_sigma, 1e-5)` denominator guard and English docstrings. |
| **Legacy Downside Risk (`environment.py`)** | `return_history`, `prev_downside_risk`, `calculate_downside_risk_numpy`, and `calculate_step_downside_penalty` execute every step as unused dead code. | Completely removed along with unused imports (`import time`, `deque`) and obsolete `w2_downside_ratio` / `downside_window_size`. |
| **Config Propagation** | `VOL_WINDOW`, `RISK_LAMBDA`, `REWARD_CLIP` in `ppo_v2.yaml` are ignored by `TrainingEnv`, `PPO2Engine`, and `PPO2Evaluator`. | Strictly bound and propagated from `ppo_v2.yaml` across `TrainingEnv`, `PPO2Engine`, and `PPO2Evaluator`. |
| **PPO Mini-Batch Sampling** | `self.buffer.get_batches(batch_size=batch_size, shuffle=False)` yields chronologically ordered mini-batches. | Restored to `shuffle=True` to preserve i.i.d. mini-batch gradient updates. |

### 1.3 Technical Root Cause & References
- Debug `print()` bottlenecks in reward calculation: [`Brain/PPO2/lib/reward.py:L20-L29`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/reward.py#L20-L29)
- Unused legacy rolling downside risk state & methods: [`Brain/PPO2/lib/environment.py:L2`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L2), [`L10`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L10), [`L24-L26`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L24-L26), [`L42-L48`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L42-L48), [`L56`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L56), [`L73-L100`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L73-L100), [`L204-L208`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L204-L208)
- Missing hyperparameter wiring in `TrainingEnv`: [`Brain/PPO2/lib/environment.py:L339-L350`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L339-L350)
- Missing hyperparameter wiring in `PPO2Engine` & `PPO2Evaluator`: [`Brain/PPO2/lib/engine.py:L70-L84`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L70-L84), [`L180-L191`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L180-L191), [`L296-L308`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L296-L308), [`L455-L467`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L455-L467), [`Brain/PPO2/lib/Evaluator.py:L18-L43`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L18-L43), [`L76-L87`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L76-L87)
- Unshuffled PPO mini-batches: [`PPO2_rl_train.py:L173`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L173)

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix
| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`Brain/PPO2/lib/reward.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/reward.py) | **In-Scope** | Remove debug `print()` calls; add defensive `safe_sigma = max(asset_sigma, 1e-5)` check; standardize English docstrings/comments. |
| **Item 2** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) | **In-Scope** | Remove dead rolling downside risk code (`return_history`, `prev_downside_risk`, `calculate_downside_risk_numpy`, `calculate_step_downside_penalty`, `downside_window_size`, `w2_downside_ratio`, unused `time`/`deque` imports); wire `vol_window`, `risk_lambda`, `reward_clip` in `TrainingEnv`; standardize English docstrings on new methods. |
| **Item 3** | [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) | **In-Scope** | Organize `VOL_WINDOW`, `RISK_LAMBDA`, `REWARD_CLIP`, `W1_STEP_RETURN` under `# Environment Penalty & Rebalance Parameters` and remove obsolete `DOWNSIDE_WINDOW_SIZE` and `W2_DOWNSIDE_RATIO`. |
| **Item 4** | [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) | **In-Scope** | Replace obsolete `downside_window_size` / `w2_downside_ratio` bindings with `vol_window`, `risk_lambda`, `reward_clip`. |
| **Item 5** | [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py) | **In-Scope** | Update `PPO2Evaluator.__init__` and `evaluate_symbol` to forward `vol_window`, `risk_lambda`, `reward_clip` to `State_time_step`. |
| **Item 6** | [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) | **In-Scope** | Restore `shuffle=True` in `self.buffer.get_batches(batch_size=batch_size, shuffle=True)`. |
| **Item 7** | [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) | **Out-of-Scope** | Mamba2 neural network architecture remains untouched. |
| **Item 8** | [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py) | **Out-of-Scope** | Continuous Actor-Critic distribution and action clipping remain untouched. |
| **Item 9** | [`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py) | **Out-of-Scope** | `RolloutBuffer` GAE computation and storage remain untouched. |

### 2.2 Architectural Boundaries & Non-Goals
- **In-Scope Boundaries**:
  - Clean integration of `DownsidePenaltyReward` and `calculate_asset_sigma` into `State_time_step`.
  - Complete removal of superseded rolling downside risk state variables and methods.
  - Full hyperparameter alignment across `ppo_v2.yaml`, `TrainingEnv`, `PPO2Engine`, and `PPO2Evaluator`.
- **Out-of-Scope (Non-Goals)**:
  - No changes to observation space shapes (`getStateShape()`, `getTimeShape()`), action space bounds (`[-1.0, 1.0]`), or the 3-way rebalance clamping logic.

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Refine `Brain/PPO2/lib/reward.py`**
  - [x] Remove the 3 debug `print()` statements inside `DownsidePenaltyReward.step()`.
  - [x] Add defensive floor `safe_sigma = max(float(asset_sigma), 1e-5)` and `assert clip > 0.0`.
  - [x] Convert docstring and inline comments to English in compliance with `.agents/rules/rules.md`.
- [x] **Step 2: Clean Up & Wire `Brain/PPO2/lib/environment.py`**
  - [x] Remove unused `import time` and `from collections import deque`.
  - [x] Remove obsolete `downside_window_size`, `w2_downside_ratio`, `self.prev_downside_risk`, `self.return_history`, `calculate_downside_risk_numpy()`, and `calculate_step_downside_penalty()`.
  - [x] Remove unused `self.return_history.append(current_p_return)` and `downside_penalty = self.calculate_step_downside_penalty()` from `State_time_step.step()`.
  - [x] Convert `calculate_asset_sigma()` docstring to English.
  - [x] Update `TrainingEnv.__init__` `state_params` to pass `vol_window`, `risk_lambda`, and `reward_clip` from `cfg_training`.
- [x] **Step 3: Synchronize `configs/training/ppo_v2.yaml`, `PPO2Engine`, and `PPO2Evaluator`**
  - [x] Consolidate `VOL_WINDOW: 96`, `RISK_LAMBDA: 0.5`, `REWARD_CLIP: 5.0`, and `W1_STEP_RETURN: 1.0` under `# Environment Penalty & Rebalance Parameters` in `configs/training/ppo_v2.yaml`, removing obsolete `DOWNSIDE_WINDOW_SIZE` and `W2_DOWNSIDE_RATIO`.
  - [x] Update `PPO2Engine` and `PPO2Evaluator` to bind and forward `vol_window`, `risk_lambda`, and `reward_clip`.
- [x] **Step 4: Restore Mini-Batch Shuffle in `PPO2_rl_train.py`**
  - [x] Set `shuffle=True` in `self.buffer.get_batches(batch_size=batch_size, shuffle=True)`.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 [`Brain/PPO2/lib/reward.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/reward.py)
```python
import numpy as np


class DownsidePenaltyReward:
    """
    Ex-ante volatility-normalized asymmetric downside penalty reward (Sortino utility):
      - Profitable or flat step (z >= 0): reward = z
      - Losing step (z < 0): reward = z - 0.5 * risk_lambda * z^2

    Since z = (p * r_market - cost) / sigma_asset, z^2 is proportional to p^2,
    which quadratically penalizes large losing positions without penalizing winning holds.
    """

    def __init__(self, risk_lambda: float = 0.5, clip: float = 5.0):
        assert risk_lambda >= 0.0, "risk_lambda must be non-negative"
        assert clip > 0.0, "clip must be positive"
        self.risk_lambda = float(risk_lambda)
        self.clip = float(clip)

    def step(self, step_return: float, asset_sigma: float) -> float:
        safe_sigma = max(float(asset_sigma), 1e-5)
        z_return = float(step_return) / safe_sigma
        downside = min(0.0, z_return)
        reward = z_return - 0.5 * self.risk_lambda * (downside * downside)
        return float(np.clip(reward, -self.clip, self.clip))
```

#### 3.2.2 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)
```python
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
            "w1_step_return": w1_step_return,
        }

        self.position = 0.0
        self.open_price = 0.0
        self.trade_bar = 0
        self.game_steps = 0
        self._prev_close = 0.0

        self.vol_window = int(vol_window)
        self.reward_fn = DownsidePenaltyReward(
            risk_lambda=risk_lambda,
            clip=reward_clip,
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
            return 1e-2

        sigma = float(np.std(np.diff(np.log(closes))))
        return max(sigma, 1e-5)
```

In `TrainingEnv.__init__`:
```python
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
        }
```

---

## 4. Validation & Verification Plan

### 4.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST adhere to repository environment standards using the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python`
> Relative path: `../bin/python`

```bash
# Run unit verification of DownsidePenaltyReward, calculate_asset_sigma (no look-ahead), dead code removal, and config wiring:
../bin/python -c "
import numpy as np
from omegaconf import OmegaConf
from Brain.PPO2.lib.reward import DownsidePenaltyReward
from Brain.PPO2.lib.environment import State_time_step, TrainingEnv
from Brain.PPO2.lib.engine import PPO2Engine

# 1. Verify DownsidePenaltyReward math (z >= 0 vs z < 0 and clipping)
r_fn = DownsidePenaltyReward(risk_lambda=0.5, clip=5.0)
assert np.isclose(r_fn.step(0.02, 0.01), 2.0), 'Positive z should not be penalized'
assert np.isclose(r_fn.step(-0.02, 0.01), -2.0 - 0.5 * 0.5 * 4.0), 'Negative z should incur quadratic penalty (-3.0)'
assert np.isclose(r_fn.step(-0.10, 0.01), -5.0), 'Reward should clip at -5.0'

# 2. Verify hygiene (dead attributes/methods removed)
state = State_time_step(bars_count=10, commission_perc=0.0001, model_train=True, default_slippage=0.0001, N_steps=20)
assert not hasattr(state, 'return_history')
assert not hasattr(state, 'prev_downside_risk')
assert not hasattr(state, 'calculate_step_downside_penalty')

# 3. Verify YAML and PPO2Engine config binding
cfg = OmegaConf.load('configs/training/ppo_v2.yaml')
engine = PPO2Engine(config=cfg)
assert engine.vol_window == 96 and np.isclose(engine.risk_lambda, 0.5) and np.isclose(engine.reward_clip, 5.0)
print('All verification checks passed successfully!')
"
```

### 4.2 Acceptance Verification Checklist
- [x] **1. Functional & Mathematical Correctness**:
  - [x] For $z \ge 0$, `DownsidePenaltyReward.step()` returns $z$ (clipped to `reward_clip`).
  - [x] For $z < 0$, `DownsidePenaltyReward.step()` returns $z - 0.5 \cdot \text{risk\_lambda} \cdot z^2$ (clipped to `-reward_clip`).
  - [x] `calculate_asset_sigma()` strictly slices `close[start : self._offset]`, invariant to changes in `close[self._offset:]` (zero look-ahead bias).
- [x] **2. Hygiene & Performance Verification**:
  - [x] Zero `print()` calls inside `Brain/PPO2/lib/reward.py`.
  - [x] Legacy attributes (`return_history`, `prev_downside_risk`, `calculate_step_downside_penalty`, `calculate_downside_risk_numpy`) are completely removed from `State_time_step`.
- [x] **3. Configuration & End-to-End Integration**:
  - [x] `TrainingEnv`, `PPO2Engine`, and `PPO2Evaluator` properly ingest `VOL_WINDOW`, `RISK_LAMBDA`, and `REWARD_CLIP` from `configs/training/ppo_v2.yaml`.
  - [x] `PPO2_rl_train.py` uses `shuffle=True` in `self.buffer.get_batches()`.
