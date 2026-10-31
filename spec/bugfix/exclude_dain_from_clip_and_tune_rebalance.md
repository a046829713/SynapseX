# Specification: Exclude DAIN from Gradient Clipping and Tune Rebalance Threshold

- **Category**: `bugfix`
- **Spec Path**: `spec/bugfix/exclude_dain_from_clip_and_tune_rebalance.md`
- **Status**: `Implemented & Verified`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
Telemetry from initial training iterations (Iter #0001–#0008) empirically verified two critical bottlenecks severely hindering PPO2 convergence:

1. **DAIN Gradient Throttling Policy & Value Learning**:
   The telemetry `GN[dain/rest/actor/critic/backbone]` confirmed that DAIN parameters generate gradient norms between `6.1` and `46.9`, accounting for **over 98% of the total network gradient norm** ($\text{DAIN}^2 / \text{Total}^2 \ge 98\%$).
   Because [`PPO2_rl_train.py:L334`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L334) applied global `torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), max_grad_norm=0.5)`, every parameter was scaled down by `ClipCoef = 0.0215 ~ 0.0911` (a 10× to 46× reduction).
   However, the non-DAIN network (`rest`: Actor, Critic, Mamba backbone) only had gradient norms of `1.1 ~ 1.5`, and the Actor head specifically had gradient norms of `0.33 ~ 0.44` (well below `0.5`, requiring zero clipping). Global clipping inadvertently starved the Actor and Critic of gradient signal, resulting in negligible KL divergences (`0.0001 ~ 0.002`) and stagnant value function learning (`ExplainedVariance` around `0.1 ~ 0.2`).

2. **Severe Churn and Friction Decay (Rebalance Threshold Too Low)**:
   During early exploration (`MeanMu ≈ 0.0`, `Std ≈ 1.0`), latent action sampling $u \sim \mathcal{N}(0, 1)$ mapped via Sigmoid to actions fluctuating around `0.5`. With `REBALANCE_THRESHOLD = 0.03`, action noise triggered continuous rebalancing in **471 out of 512 steps (92% of all bars)**, producing an episode turnover exceeding `120` and eating **2.4% ~ 2.5% in fees and slippage per episode** (`Cost(100) ≈ 2.45%`). Over 65% of the initial episode losses were pure transaction friction.

### 1.2 Current vs. Expected Behavior
| Dimension | Current Behavior | Expected Behavior |
| :--- | :--- | :--- |
| **Gradient Clipping Target** | `clip_grad_norm_` covers all model parameters including DAIN (`mean_layer`, `scaling_layer`, `gating_layer`). | `clip_grad_norm_` strictly operates on non-DAIN parameters (`self._rest_params`: Mamba backbone, temporal pool, portfolio encoder, actor, critic, log_std). DAIN is excluded. |
| **Clip Coefficient Telemetry** | `clip_coef` computed against total norm (`group_norms["total"]`), showing false suppression dominated by DAIN. | `clip_coef` and primary `GradNorm` reflect the active non-DAIN optimization parameters (`group_norms["rest"]`). |
| **Rebalance Deadband** | `REBALANCE_THRESHOLD = 0.03` allows tiny exploratory oscillations ($|\Delta a| > 0.03$) to trigger trading execution. | `REBALANCE_THRESHOLD = 0.05` filters exploratory jitter below 5% position change, reducing friction and unnecessary trade churn. |
| **Telemetry Alignment** | Telemetry logs `GradNorm` dominated by DAIN. | Telemetry logs `GradNorm` as the effective `rest` norm being clipped to `MAX_GRAD_NORM: 0.5`, with full `GN[dain/rest/actor/critic/backbone]` breakdown preserved. |

### 1.3 Technical Root Cause & References
- Global gradient clipping in [`PPO2_rl_train.py:L333-L338`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L333-L338):
  ```python
  grad_norm = torch.nn.utils.clip_grad_norm_(
      self.agent.model.parameters(), max_grad_norm
  )
  ```
- DAIN learning rate in [`Brain/Common/optimization.py:L48-L60`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py#L48-L60):
  DAIN parameters have dedicated learning rates scaled by $10^{-5}$ (`lr = 1e-4 * 1e-5 = 1e-9`), meaning DAIN weights barely move. Its large gradients are mathematical artifacts of window standard deviation division, not meaningful policy signals.
- Environment rebalance threshold configured in [`configs/training/ppo_v2.yaml:L55`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L55):
  ```yaml
  REBALANCE_THRESHOLD: 0.03
  ```
  Consumed by [`Brain/PPO2/lib/environment.py:L97`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L97).

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix
| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) | **In-Scope** | Cache `self._rest_params` excluding DAIN groups; clip only `self._rest_params` with `max_grad_norm`; update `clip_coef` to monitor `rest` norm. |
| **Item 2** | [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) | **In-Scope** | Change `REBALANCE_THRESHOLD` from `0.03` to `0.05`. |
| **Item 3** | `tests/test_exclude_dain_clip.py` (new) | **In-Scope** | Unit test verifying `clip_grad_norm_` strictly limits non-DAIN grad norm to $\le 0.5$ while preserving unclipped DAIN gradients without side effects. |
| **Item 4** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) | **Out-of-Scope** | Environment logic already dynamically reads `rebalance_threshold` from config; no code modifications needed. |
| **Item 5** | [`Brain/Common/dain.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/dain.py), [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) | **Out-of-Scope** | Model architecture, forward passes, and DAIN operations remain 100% unmodified. |

### 2.2 Architectural Boundaries & Non-Goals
- **In-Scope Boundaries**:
  - Only modify gradient clipping target in `update_ppo` of `PPO2_rl_train.py`.
  - Maintain exact parameter group mappings from `Brain/Common/grad_diagnostics.py`.
  - Update `REBALANCE_THRESHOLD: 0.05` in YAML configuration.
- **Out-of-Scope (Non-Goals)**:
  - Do not change optimizer parameter groups, learning rates, or weight decay in `build_optimizer`.
  - Do not alter `DEADZONE_THRESHOLD: 0.01` or reward formulation `w1_step_return`.
  - Do not alter state representation or Mamba SSM architecture.

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Cache `_rest_params` in `PPO2._prepare_agent`**
  - [x] Use `DAIN_GROUPS` from `Brain.Common.grad_diagnostics` to filter `self._param_groups`.
  - [x] Construct `self._rest_params = [p for g, ps in self._param_groups.items() if g not in DAIN_GROUPS for p in ps]`.
  - [x] Add defensive assertion ensuring `len(self._rest_params) > 0` and total parameter counts add up.
- [x] **Step 2: Update `update_ppo` in `PPO2_rl_train.py`**
  - [x] Change `torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), max_grad_norm)` to `torch.nn.utils.clip_grad_norm_(self._rest_params, max_grad_norm)`.
  - [x] Change `clip_coefs.append(clip_coef(group_norms["total"], max_grad_norm))` to `clip_coefs.append(clip_coef(group_norms["rest"], max_grad_norm))`.
  - [x] Ensure reported `GradNorm` reflects the clipped `rest` norm (with `GN[...]` continuing to show all individual groups).
- [x] **Step 3: Update `configs/training/ppo_v2.yaml`**
  - [x] Set `REBALANCE_THRESHOLD: 0.05`.
- [x] **Step 4: Unit Testing & Verification**
  - [x] Implement `tests/test_exclude_dain_clip.py` verifying that:
    1. Gradients on non-DAIN parameters have norm $\le \text{max\_grad\_norm} + \epsilon$.
    2. DAIN gradients remain unclipped and are not modified by the clip call.
    3. Rebalance deadband of 0.05 properly ignores changes $< 0.05$.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 `PPO2_rl_train.py`
```python
# In PPO2._prepare_agent():
from Brain.Common.grad_diagnostics import (
    DAIN_GROUPS,
    build_param_groups,
    clip_coef,
    group_grad_norms,
)

# After self._param_groups = build_param_groups(self.agent.model):
self._rest_params = [
    p for g, ps in self._param_groups.items()
    if g not in DAIN_GROUPS for p in ps
]
assert len(self._rest_params) > 0, "No non-DAIN parameters found for clipping"

# In PPO2.update_ppo():
total_loss.backward()
group_norms = group_grad_norms(self._param_groups)

# Exclude DAIN from clipping; clip only non-DAIN (rest) parameters
grad_norm = torch.nn.utils.clip_grad_norm_(
    self._rest_params, max_grad_norm
)
self.optimizer.step()
group_norm_records.append(group_norms)
clip_coefs.append(clip_coef(group_norms["rest"], max_grad_norm))
```

#### 3.2.2 `configs/training/ppo_v2.yaml`
```yaml
# Environment Rebalance & Reward Parameters
DEADZONE_THRESHOLD: 0.01
REBALANCE_THRESHOLD: 0.05
W1_STEP_RETURN: 100.0
```

---

## 4. Validation & Verification Plan

### 4.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST adhere to repository environment standards using the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python`
> Relative path: `../bin/python`

```bash
# 1. Run unit test for excluded DAIN clipping and rebalance threshold
../bin/python -m unittest tests/test_exclude_dain_clip.py -v

# 2. Run existing unit tests to prevent regressions
../bin/python -m unittest tests/test_grad_diagnostics.py -v
```

### 4.2 Acceptance Verification Checklist
- [x] **1. Gradient Independence**:
  - [x] Pre-clip `rest` norm $\approx 1.2 \sim 1.5$ is scaled to $\le 0.5$ without being affected by DAIN's norm of $10 \sim 40$.
  - [x] Actor gradient receives full intended update proportion (no longer crushed to $0.015$).
- [x] **2. Rebalance Threshold**:
  - [x] Environment configured with `0.05` filters position updates where $|\Delta a| < 0.05$.
  - [x] Episodes with exploratory noise exhibit reduced turnover and trades count.
- [x] **3. Runtime Safety**:
  - [x] PPO training step completes without dimension mismatch or gradient leakage.
  - [x] Checkpoints continue saving and loading seamlessly.
