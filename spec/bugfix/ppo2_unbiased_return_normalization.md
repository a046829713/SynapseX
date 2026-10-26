# Specification: PPO2 Removal of DownsidePenaltyReward & Direct Unbiased Log-Return Objective

- **Category**: `bugfix`
- **Spec Path**: `spec/bugfix/ppo2_unbiased_return_normalization.md`
- **Status**: `Implemented & Verified`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
Empirical analysis of multi-instrument training logs ([`train.txt`](file:///home/b0457812963/Mamba3RL/SynapseX/train.txt), Iterations `#0034`–`#0061`, 232 episodes) revealed a persistent structural force driving portfolio allocation (`MeanMu`, `MeanAct`, `Avg|Pos|`) toward `0.0` even after lowering transaction friction (`MODEL_DEFAULT_COMMISSION_PERC_TRAING: 0.0001`, `DEFAULT_SLIPPAGE: 0.0001`):

1. **Severe Sign Inversion Between Net Profit and RL Return**:
   - Out of 232 training episodes, 75 episodes achieved positive net profit (`NetPnL > 0`, mean `+5.18%`), yet **74.7% of those profitable episodes received a negative RL `Return` (`mean = -3.625`)**—strictly worse than holding `0.0` Cash (`Return = 0.0000`).
   - Even episodes with strong net gains such as `+21.58%` (`Ep #0311 FORMUSDT`) and `+11.93%` (`Ep #0409 AXSUSDT`) were penalized with negative RL returns (`-2.8709` and `-11.2988`).
2. **Time-Scale ($\Delta t$) Mismatch in [`DownsidePenaltyReward`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/reward.py#L4-L32)**:
   - Currently, single-step net log-return $r_t$ is normalized by ex-ante 30-minute volatility $\sigma_t \propto \sqrt{\Delta t}$ into $z_t = r_t / \sigma_t$, followed by a quadratic downside penalty:
     $$R_t = z_t - \frac{1}{2}\lambda_{\text{risk}} \min(0, z_t)^2 \quad (\lambda_{\text{risk}} = 0.5)$$
   - While expected drift scales as $O(\sqrt{\Delta t})$ in Z-score space, the quadratic term $\mathbb{E}[\min(0, z_t)^2] \approx \frac{1}{2} p^2$ is $O(1)$ **per 30-minute bar**, subtracting $\approx -0.125\,p^2$ every step (`-64.0` per 512-step episode at $p = 1.0$) even at **zero commission and zero slippage**.
   - Dividing $r_t^2$ by $\sigma_t^2$ inflates the effective Merton risk-aversion coefficient to $\gamma_{\text{eff}} \approx \lambda_{\text{risk}} / \sigma_t \approx 60 \sim 500$, forcing optimal allocation $p^* \to 0$ across bull-market regimes (`SOLUSDT` +377% log-return receives `-56.30` per 512 steps at $p=1.0$ vs. `0.00` for Cash).
3. **Why Raw `current_p_return` Needs Zero Additional Scaling (`No ReturnNormalizer`)**:
   - **Natural Shared-Trunk Gradient Balance**: Direct GPU gradient diagnostics on [`MambaContinuousActorCritic`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L293-L470) (`5,106,333` parameters) show:
     - Policy loss gradient norm (`PiLoss` alone) is **`1.0135`**, which is already 100% scale-invariant due to mini-batch advantage normalization `(advantages_b - mean) / (std + 1e-8)` in [`PPO2_rl_train.py:L293-L295`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L293-L295) (`std ≈ 0.015 >> 1e-8`).
     - Under raw `current_p_return` (`GAMMA = 0.99`, typical discounted return residual `~0.05`), Critic value loss gradient norm (`VfLoss` after initial warmup) settles at **`0.3475`**, yielding a balanced total pre-clip `GradNorm ≈ 1.07` instead of the `31.7 ~ 171.4` Critic gradient explosion observed in `train.txt` (or the `~6.95` Critic dominance that `VecNormalize` 20x upscaling would re-introduce).
   - **Stationary Economic Value Targets**: Because `current_p_return = log(m_net)` is already a dimensionless, scale-invariant logarithmic growth rate (`0.01 = 1%` across all assets and eras), Critic targets $V(s)$ remain strictly stationary regardless of changes in the policy's average exposure $\bar{p}$, avoiding `VecNormalize`'s position-dependent denominator drift.
   - **Minimal Blast Radius**: Removing `DownsidePenaltyReward` directly leaves [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) **100% untouched** and establishes an exact 1-to-1 identity between logged RL `Return` and `NetPnL` (`NetPnL% = Return * 100` when `W1_STEP_RETURN = 1.0`).

### 1.2 Current vs. Expected Behavior
| Dimension | Current Behavior | Expected Behavior (Direct Unbiased Log-Return) |
| :--- | :--- | :--- |
| **Environment Step Reward** | Distorted asymmetric Z-score utility $z_t - \frac{1}{2}\lambda_{\text{risk}}\min(0, z_t)^2$ via [`DownsidePenaltyReward`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/reward.py#L4-L32) | Exact unbiased single-step net portfolio log-return `w1_step_return * current_p_return` ($\ln(E_t / E_{t-1})$) |
| **Profit / Reward Identity** | 74.7% of `NetPnL > 0` episodes receive `Return < 0` (worse than `0.0` Cash) | Exact parity: `Return > 0` iff `NetPnL > 0`, `Return == 0` iff `NetPnL == 0` (`NetPnL% == Return * 100` at `W1_STEP_RETURN = 1.0`) |
| **Critic Target & `GradNorm`** | Cumulative Z-penalties (`-15 ~ -49`) inflate Critic `GradNorm` to `31.7 ~ 171.4`, crushing Actor gradients via `clip_grad_norm_(0.5)` | Stationary log-return Critic targets (`±0.05` typical) keep adapted Critic `GradNorm ≈ 0.35` alongside Policy `GradNorm ≈ 1.01` |
| **Per-Step Env CPU Overhead** | Computes 96-bar `np.log`, `np.diff`, and `np.std` on every environment step | Zero rolling-volatility overhead in `State_time_step.step()` |
| **Training Loop (`PPO2_rl_train.py`)** | Receives distorted Z-score rewards | **100% Unmodified** (`0` lines changed) |

### 1.3 Technical Root Cause & References
- $\Delta t$-mismatched quadratic downside penalty module: [`Brain/PPO2/lib/reward.py:L1-L32`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/reward.py#L1-L32)
- Ex-ante rolling volatility calculation & reward invocation in environment: [`Brain/PPO2/lib/environment.py:L9`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L9), [`L32-L35`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L32-L35), [`L68-L75`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L68-L75), [`L92-L108`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L92-L108), [`L176-L183`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L176-L183), [`L316-L320`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L316-L320)
- Obsolete downside-penalty config bindings in evaluation engine, evaluator, & YAML: [`Brain/PPO2/lib/engine.py:L83-L86`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L83-L86), [`L205-L208`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L205-L208), [`L414-L417`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L414-L417), [`L609-L612`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L609-L612), [`Brain/PPO2/lib/Evaluator.py:L28-L31`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L28-L31), [`L44-L47`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L44-L47), [`L170-L173`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L170-L173), [`configs/training/ppo_v2.yaml:L57-L60`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L57-L60)

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix
| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`Brain/PPO2/lib/reward.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/reward.py) | **In-Scope [DELETE]** | Delete obsolete `DownsidePenaltyReward` module completely to eliminate dead code. |
| **Item 2** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) | **In-Scope** | Remove `DownsidePenaltyReward` import, `calculate_asset_sigma()`, and `vol_window` / `risk_lambda` / `reward_clip` / `min_sigma` attributes; set `reward = float(self.weights["w1_step_return"] * current_p_return)` in `State_time_step.step()`; remove obsolete entries from `TrainingEnv.__init__` `state_params`. |
| **Item 3** | [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) & [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py) | **In-Scope** | Remove obsolete `VOL_WINDOW`, `RISK_LAMBDA`, `REWARD_CLIP`, `MIN_ASSET_SIGMA` (`vol_window`, `risk_lambda`, `reward_clip`, `min_sigma`) bindings and constructor arguments while keeping `W1_STEP_RETURN` and all execution/deadzone/rebalance logic intact. |
| **Item 4** | [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) | **In-Scope** | Remove obsolete `VOL_WINDOW`, `MIN_ASSET_SIGMA`, `RISK_LAMBDA`, and `REWARD_CLIP` keys. |
| **Item 5** | `tests/test_ppo2_unbiased_log_return.py` | **In-Scope [NEW]** | Automated unit tests verifying exact `reward == w1_step_return * current_p_return` parity, positive RL return on profitable trajectories, dead-code removal, and `TrainingEnv` / `PPO2Engine` / `PPO2Evaluator` config alignment. |
| **Item 6** | [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py), [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py), [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py), [`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py), [`Brain/Common/DataFeature.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py), [`Brain/Common/dain.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/dain.py) | **Out-of-Scope** | Explicitly preserved with **zero modifications** (`0` lines changed) to strictly isolate the reward change. |

### 2.2 Architectural Boundaries & Non-Goals
- **In-Scope Boundaries**:
  - Direct unbiased single-bar net portfolio log-return reward (`w1_step_return * current_p_return`) in `State_time_step`.
  - Complete removal of `DownsidePenaltyReward`, `calculate_asset_sigma`, and associated YAML/engine/evaluator parameters.
- **Out-of-Scope (Non-Goals)**:
  - **No reward scaling wrapper (`ReturnNormalizer`)** and **zero modifications** to [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py).
  - **No changes** to `MambaContinuousActorCritic` architecture, DAIN normalization, or `DataFeature` inputs.
  - **No changes** to `PPO2Agent` action distribution (`Squashed-Gaussian`), `ENT_COEF`, `INIT_LOG_STD`, or PPO surrogate/value/entropy loss equations.
  - **No changes** to Long-Only `[0.0, 1.0]` action bounds, `deadzone_threshold` (`0.01`), `rebalance_threshold` (`0.03`), or `m_net` equity compounding math.

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Remove Obsolete [`Brain/PPO2/lib/reward.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/reward.py)**
  - [x] Delete `Brain/PPO2/lib/reward.py` (`DownsidePenaltyReward`).
- [x] **Step 2: Simplify [`State_time_step`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L21-L192) & [`TrainingEnv`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L290-L360) in [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)**
  - [x] Remove `from Brain.PPO2.lib.reward import DownsidePenaltyReward`.
  - [x] Remove `vol_window`, `risk_lambda`, `reward_clip`, `min_sigma`, and `self.reward_fn` from `State_time_step.__init__` (preserving `w1_step_return: float = 1.0` and `**kwargs`).
  - [x] Remove `calculate_asset_sigma()` from `State_time_step`.
  - [x] Set `reward = float(self.weights["w1_step_return"] * current_p_return)` in `State_time_step.step()`.
  - [x] Remove `vol_window`, `risk_lambda`, `reward_clip`, `min_sigma` from `TrainingEnv.__init__` `state_params`.
- [x] **Step 3: Clean Up Config & Evaluation Bindings ([`ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml), [`engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py), [`Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py))**
  - [x] Remove `VOL_WINDOW`, `MIN_ASSET_SIGMA`, `RISK_LAMBDA`, and `REWARD_CLIP` from `configs/training/ppo_v2.yaml`.
  - [x] Remove `vol_window`, `risk_lambda`, `reward_clip`, `min_sigma` bindings and arguments from `PPO2Engine` and `PPO2Evaluator`.
- [x] **Step 4: Verification Suite Creation & Empirical Validation**
  - [x] Create `tests/test_ppo2_unbiased_log_return.py` and run full verification with `/home/b0457812963/Mamba3RL/bin/python`.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)
In `State_time_step.__init__`:
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
```

In `State_time_step.step()`:
```python
        # Unbiased single-step net portfolio log-return
        reward = float(self.weights["w1_step_return"] * current_p_return)
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
        }
```

#### 3.2.2 [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml)
```yaml
# Environment Rebalance & Reward Parameters
DEADZONE_THRESHOLD: 0.01
REBALANCE_THRESHOLD: 0.03
W1_STEP_RETURN: 1.0
```

#### 3.2.3 [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) & [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py)
- Remove `self.vol_window`, `self.risk_lambda`, `self.reward_clip`, and `self.min_sigma` from `PPO2Engine.__init__`, `PPO2Engine.build_agent`, `PPO2Engine.run_test`, `PPO2Engine.get_if_order_map`, `PPO2Evaluator.__init__`, and `PPO2Evaluator.evaluate_symbol`.

---

## 4. Validation & Verification Plan

### 4.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST adhere to repository environment standards using the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python`
> Relative path: `../bin/python`

```bash
/home/b0457812963/Mamba3RL/bin/python -m unittest tests/test_ppo2_unbiased_log_return.py
```

### 4.2 Acceptance Verification Checklist
- [x] **1. Exact Unbiased Log-Return Parity**:
  - [x] `State_time_step.step()` outputs `reward == w1_step_return * current_p_return` within `< 1e-12` across flat (`0.0`), profitable, and losing steps.
  - [x] Over multi-step episodes with `W1_STEP_RETURN = 1.0`, cumulative RL `Return` equals cumulative `info["step_return"]` (`NetPnL% == Return * 100.0`) within `< 1e-7`, guaranteeing profitable episodes always receive `Return > 0`.
- [x] **2. Clean Hygiene & Dead-Code Removal**:
  - [x] `Brain/PPO2/lib/reward.py` is deleted, and `State_time_step` has no `reward_fn`, `calculate_asset_sigma`, `vol_window`, `risk_lambda`, `reward_clip`, or `min_sigma` attributes.
- [x] **3. End-to-End Config & Evaluation Compatibility**:
  - [x] `TrainingEnv`, `PPO2Engine`, and `PPO2Evaluator` instantiate and run deterministically using `configs/training/ppo_v2.yaml` without any `DownsidePenaltyReward` parameters, while [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) remains 100% untouched.
