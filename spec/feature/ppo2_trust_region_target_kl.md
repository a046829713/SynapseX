# Specification: PPO2 Trust-Region TARGET_KL Early-Stopping Brake & Telemetry

- **Category**: `feature`
- **Spec Path**: `spec/feature/ppo2_trust_region_target_kl.md`
- **Status**: `Completed`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
Review of multi-instrument continuous PPO2 training telemetry ([`train.txt`](file:///home/b0457812963/Mamba3RL/SynapseX/train.txt), Iterations `#0423`–`#0432`, `~1.77M` steps) reveals severe, unconstrained policy drift during within-iteration gradient updates:
- **Rollout & Update Structure**: Each iteration collects `4096` transitions (`8` parallel workers $\times$ `512` horizon steps). The policy executes `2` PPO epochs with `batch_size: 256`, resulting in `32` sequential AdamW mini-batch updates (`16` batches/epoch $\times$ `2` epochs) on the same rollout experience.
- **Uncontrolled Trust-Region Violation**:
  - In [`PPO2_rl_train.py:L307`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L307), `approx_kl = ((ratio - 1.0) - log_ratio).mean().item()` is computed strictly for metric logging. There is **no mechanism to halt updates when the policy strays from the data-collection policy $\pi_{\text{old}}$**.
  - Mean `approx_kl` across batches routinely fluctuates between `0.030` and `0.052`, and surges to **`0.08124` in Iteration `#0430`** (and `0.05008` in Iteration `#0431`).
  - Because `approx_kl` starts near `0.0` on the first mini-batch, a mean of `0.08124` implies that by epoch 2, the late-batch KL divergence reaches $D_{KL}(\pi_{\text{old}} \parallel \pi) > 0.12 \sim 0.15$—drastically exceeding standard PPO trust-region bounds (`0.010 ~ 0.020`).
- **Destructive Cascade on Value Estimation & Policy Stability**:
  - Whenever `approx_kl` spikes (e.g. Iter `#0425: KL=0.0525`, Iter `#0430: KL=0.0812`), `ValueLoss` suffers violent shocks (`ValueLoss: 4.5535` in `#0425` and `0.8757` in `#0430` vs typical `0.08 ~ 0.20`).
  - `ExplainedVariance` collapses from `+0.233` down to `+0.075` (`#0425`) and `+0.099` (`#0430`).
  - Net performance destabilizes, with `NetPnL(100)` dropping from `-1.06%` down to `-2.20%`.

### 1.2 Current vs. Expected Behavior
| Dimension | Current Behavior | Expected Behavior |
| :--- | :--- | :--- |
| **PPO Trust-Region Control** | Blindly runs all `32` mini-batch updates regardless of policy divergence; mean `approx_kl` surges up to `0.0812`. | Pre-step `TARGET_KL` early-stopping brake monitors mini-batch `approx_kl`. Halts update loop when `approx_kl > 1.5 * TARGET_KL` (`0.0225` for `TARGET_KL=0.015`). |
| **Update Execution Guarantee** | Fixed `32` updates every iteration, risking severe over-optimization on stale rollout ratios. | Dynamic update count ($1 \le N \le 32$). Mini-batch 0 is guaranteed to execute ($KL \approx 0$ initially), preventing 0-update deadlocks. |
| **Telemetry & Observability** | Iteration log reports mean `KL` only; no indication of how many updates executed or whether drift occurred. | Log displays completed update count `Upd: {n_updates:02d}/32` and early-stopping status `[KL Stop]` alongside mean `KL`. |
| **Scope Isolation** | Previous draft bundled environment hysteresis and variance clamp changes. | **Pure KL focus**: strictly isolated to training update logic (`PPO2_rl_train.py`) and training hyperparameters (`ppo_v2.yaml`). |

### 1.3 Technical Root Cause & References
- Affected Location: [`PPO2_rl_train.py:L279-L355`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L279-L355).
- In the inner mini-batch loop:
  ```python
  log_ratio = log_probs - old_log_probs_b
  ratio = torch.exp(log_ratio)
  with torch.no_grad():
      approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
      approx_kls.append(approx_kl)
  ...
  self.optimizer.step()
  ```
  Even though `approx_kl` is computed, execution unconditionally falls through to `total_loss.backward()` and `self.optimizer.step()`.
- Affected Location: [`configs/training/ppo_v2.yaml:L16-L32`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L16-L32).
  Missing `TARGET_KL` hyperparameter configuration.

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix
| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`PPO2_rl_train.py:L258-L395, L551-L601`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L258-L601) | **In-Scope** | 1. Add `target_kl: Optional[float] = None` to `update_ppo()`.<br/>2. Add pre-step early-stopping check: `if target_kl is not None and target_kl > 0.0 and approx_kl > 1.5 * target_kl:` break both loops.<br/>3. Return `n_updates`, `total_updates`, and `kl_early_stop` in metrics dict.<br/>4. Pass `TARGET_KL` from `self.cfg_training` in `PPO2.train()`.<br/>5. Update iteration log string to display `Upd: XX/YY` and early-stopping indicator. |
| **Item 2** | [`configs/training/ppo_v2.yaml:L24-L30`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L24-L30) | **In-Scope** | Add `TARGET_KL: 0.015` under `# PPO Hyperparameters`. |
| **Item 3** | `tests/test_ppo2_target_kl.py` | **In-Scope [NEW]** | Unit test suite validating:<br/>- Early-stopping triggers when `approx_kl > 1.5 * target_kl`.<br/>- First mini-batch always executes ($N \ge 1$).<br/>- `target_kl=None` runs full epochs without early exit.<br/>- Returned metrics accurately reflect `n_updates` and `kl_early_stop`. |
| **Item 4** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py), [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py) | **Out-of-Scope** | Preserved completely untouched. No modifications to trade thresholds, reward calculations, or state transitions. |
| **Item 5** | [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py), [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) | **Out-of-Scope** | Policy architecture, distribution sampling, squashed Gaussian math, and `log_std` parameters remain unmodified. |
| **Item 6** | [`Brain/Common/DataFeature.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py), [`Brain/Common/dain.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/dain.py) | **Out-of-Scope** | Normalization layers and data pipelines remain unmodified. |

### 2.2 Architectural Boundaries & Non-Goals
- **In-Scope Boundaries**:
  - Clean trust-region brake in accordance with standard PPO literature (Schulman et al., 2017; CleanRL; OpenAI Spinning Up).
  - Unbiased estimator: `approx_kl = ((ratio - 1.0) - log_ratio).mean().item()` remains standard.
  - Telemetry transparency: engineers can immediately inspect how many mini-batches executed in every training iteration.
- **Out-of-Scope (Non-Goals)**:
  - Do NOT modify environment deadzones or rebalance thresholds (`DEADZONE_THRESHOLD`, `REBALANCE_THRESHOLD`).
  - Do NOT modify entropy coefficient (`ENT_COEF`) or policy standard deviation (`INIT_LOG_STD`, `LOG_STD_MAX`, `LOG_STD_MIN`).
  - Do NOT alter model architecture, network layers, or checkpoint schema.

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Trust-Region Early-Stopping & Telemetry in `PPO2_rl_train.py`**
  - [x] Add `target_kl: Optional[float] = None` parameter to [`PPO2.update_ppo`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L258-L266).
  - [x] Initialize `n_updates = 0`, `kl_early_stop = False`, and compute `total_expected_updates = ppo_epochs * math.ceil(buffer_size / batch_size)` (or track batch count per epoch).
  - [x] Inside the inner mini-batch loop, immediately after computing `approx_kl`:
    ```python
    if target_kl is not None and target_kl > 0.0 and approx_kl > 1.5 * target_kl:
        kl_early_stop = True
        break
    ```
  - [x] In the outer epoch loop, check `if kl_early_stop: break`.
  - [x] Increment `n_updates += 1` after `self.optimizer.step()`.
  - [x] Add `n_updates`, `total_updates`, and `kl_early_stop` to the returned metrics dictionary in `update_ppo()`.
  - [x] In [`PPO2.train`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L552-L560), read `target_kl = getattr(self.cfg_training, "TARGET_KL", None)` and pass to `update_ppo()`.
  - [x] In [`PPO2.train`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L586-L601), update the iteration log format:
    Include `Upd: {metrics['n_updates']:02d}/{metrics['total_updates']:02d}{' [KL Stop]' if metrics['kl_early_stop'] else ''}`.
- [x] **Step 2: Hyperparameter Configuration in `configs/training/ppo_v2.yaml`**
  - [x] Add `TARGET_KL: 0.015` under `# PPO Hyperparameters`.
- [x] **Step 3: Unit Testing & Defensive Checks**
  - [x] Create `tests/test_ppo2_target_kl.py`.
  - [x] Test brake triggering when KL exceeds threshold.
  - [x] Test backward compatibility when `TARGET_KL` is None / 0.0.
  - [x] Test that initial batch ($N=1$) always executes even under synthetic high-KL data.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)

```python
    def update_ppo(
        self,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
    ):
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        clipfracs = []
        expl_vars = []
        grad_norms = []
        mean_actions = []
        mean_mus = []
        group_norm_records = []
        clip_coefs = []

        n_updates = 0
        kl_early_stop = False
        num_batches_per_epoch = max(1, len(self.buffer) // batch_size)
        total_expected_updates = ppo_epochs * num_batches_per_epoch

        for epoch in range(ppo_epochs):
            if kl_early_stop:
                break
            for (
                states_b,
                time_states_b,
                actions_b,
                old_log_probs_b,
                returns_b,
                advantages_b,
                old_values_b,
            ) in self.buffer.get_batches(batch_size=batch_size, shuffle=True):
                states_b = states_b.to(self.device)
                time_states_b = time_states_b.to(self.device)
                actions_b = actions_b.to(self.device)
                old_log_probs_b = old_log_probs_b.to(self.device)
                returns_b = returns_b.to(self.device)
                advantages_b = advantages_b.to(self.device)
                old_values_b = old_values_b.to(self.device)

                log_probs, entropy, values, stats = self.agent.evaluate_actions(
                    states_b, time_states_b, actions_b, return_stats=True
                )

                # Ratio
                log_ratio = log_probs - old_log_probs_b
                ratio = torch.exp(log_ratio)

                # Approximate KL divergence & Clip Fraction for monitoring
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clipfrac = torch.mean(
                        (torch.abs(ratio - 1.0) > clip_eps).float()
                    ).item()
                    expl_var = explained_variance(values, returns_b)

                # Trust-Region Early Stopping Check (before backward/step)
                # Guarantees at least 1 update has occurred before early stopping
                if (
                    target_kl is not None
                    and target_kl > 0.0
                    and approx_kl > 1.5 * target_kl
                    and n_updates > 0
                ):
                    kl_early_stop = True
                    break

                approx_kls.append(approx_kl)
                clipfracs.append(clipfrac)
                expl_vars.append(expl_var)

                # Mini-batch Advantage Normalization
                mb_advantages = (advantages_b - advantages_b.mean()) / (
                    advantages_b.std() + 1e-8
                )

                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clipped Value Function loss
                v_clipped = old_values_b + torch.clamp(
                    values - old_values_b, -clip_eps, clip_eps
                )
                vf_loss_unclipped = (values - returns_b).pow(2)
                vf_loss_clipped = (v_clipped - returns_b).pow(2)
                value_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                total_loss = (
                    policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                group_norms = group_grad_norms(self._param_groups)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._rest_params, max_grad_norm
                )
                self.optimizer.step()
                n_updates += 1

                group_norm_records.append(group_norms)
                clip_coefs.append(clip_coef(group_norms["rest"], max_grad_norm))
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                grad_norms.append(float(grad_norm.item()))
                mean_actions.append(stats["mean_action"])
                mean_mus.append(stats["mean_mu"])

        ...
        return {
            ...
            "n_updates": n_updates,
            "total_updates": total_expected_updates,
            "kl_early_stop": kl_early_stop,
        }
```

In `PPO2.train()`:
```python
                target_kl = getattr(self.cfg_training, "TARGET_KL", None)
                metrics = self.update_ppo(
                    ppo_epochs=ppo_epochs,
                    batch_size=batch_size,
                    clip_eps=clip_eps,
                    vf_coef=vf_coef,
                    ent_coef=ent_coef,
                    max_grad_norm=max_grad_norm,
                    target_kl=target_kl,
                )
...
                upd_str = f"Upd: {metrics['n_updates']:02d}/{metrics['total_updates']:02d}"
                if metrics["kl_early_stop"]:
                    upd_str += " [KL Stop]"

                print(
                    f"[Iter #{iteration:04d}] Steps: {total_steps:07d} | FPS: {fps:6.1f} | "
                    f"{upd_str} | "
                    f"PiLoss: {metrics['policy_loss']:+.4f} | ValueLoss: {metrics['value_loss']:.4f} | "
                    f"Entropy: {metrics['entropy']:+.4f} | KL: {metrics['approx_kl']:.5f} | "
                    ...
```

#### 3.2.2 [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml)

```yaml
# PPO Hyperparameters
LEARNING_RATE: 0.0002
WEIGHT_DECAY: 0
BASE_LR: 0.0001
BATCH_SIZE: 256
PPO_EPOCHS: 2
CLIP_EPS: 0.2
VF_COEF: 0.5
ENT_COEF: 0.001
INIT_LOG_STD: 0.0
LOG_STD_MIN: -5.0
LOG_STD_MAX: 1.0
TARGET_KL: 0.015
MAX_GRAD_NORM: 0.5
CHECKPOINT_EVERY_ITER: 10
GAMMA: 0.99
GAE_LAMBDA: 0.95
```

---

## 4. Validation & Verification Plan

### 4.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST adhere to repository environment standards using the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python`
> Relative path: `../bin/python`

```bash
# 1. Run dedicated Target KL unit tests
/home/b0457812963/Mamba3RL/bin/python -m unittest tests/test_ppo2_target_kl.py -v

# 2. Verify Hydra config loading
/home/b0457812963/Mamba3RL/bin/python -c "
from hydra import compose, initialize_config_dir
import os
with initialize_config_dir(config_dir=os.path.abspath('configs'), version_base=None):
    cfg = compose(config_name='config')
assert hasattr(cfg.training, 'TARGET_KL'), 'Missing TARGET_KL in config'
assert cfg.training.TARGET_KL == 0.015, f'Expected 0.015, got {cfg.training.TARGET_KL}'
print('Hydra TARGET_KL configuration verified!')
"
```

### 4.2 Acceptance Verification Checklist
- [x] **1. Trust-Region Brake Functionality**:
  - [x] When `approx_kl > 1.5 * TARGET_KL` (`0.0225`), update terminates immediately before executing gradient step on the out-of-bound batch.
  - [x] `kl_early_stop` is set to `True`.
  - [x] Remaining batches and subsequent epochs are skipped.
- [x] **2. Boundary & Invariant Guarantees**:
  - [x] Minimum update count guarantee: `n_updates >= 1` always holds (no zero-update iterations).
  - [x] When `TARGET_KL` is `None` or `0.0`, all `32` updates run to completion (`kl_early_stop == False`).
  - [x] When `approx_kl <= 1.5 * TARGET_KL`, all `32` updates run to completion.
- [x] **3. Telemetry Transparency**:
  - [x] `metrics` returned by `update_ppo()` contains integer `n_updates`, `total_updates`, and boolean `kl_early_stop`.
  - [x] `print()` output includes `Upd: XX/32` and `[KL Stop]` if early stopping occurred.
