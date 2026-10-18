# Specification: PPO2 OpenAI Baselines Alignment & Actor Tanh Removal

- **Category**: `bugfix`
- **Spec Path**: `spec/bugfix/ppo2_openai_baseline_alignment.md`
- **Status**: `Completed`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Mathematical & Algorithmic Foundations](#2-mathematical--algorithmic-foundations)
3. [3. Target Scope & Architecture Boundaries](#3-target-scope--architecture-boundaries)
4. [4. Implementation Checklist & Detailed Blueprint](#4-implementation-checklist--detailed-blueprint)
5. [5. Validation & Verification Plan](#5-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
During continuous development and architecture upgrades (incorporating Mamba2, `TemporalAttentionPool`, and decoupled portfolio stream late fusion), an architectural regression occurred in [`Brain/PPO2/lib/model.py:L418`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L418):
An `nn.Tanh()` squashing activation was inadvertently reintroduced at the output of `self.fc_actor`.

While [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py) and [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) correctly implemented unclipped action sampling and rollout storage (as specified in [`spec/bugfix/ppo2_unclipped_action_storage.md`](file:///home/b0457812963/Mamba3RL/SynapseX/spec/bugfix/ppo2_unclipped_action_storage.md)), having `nn.Tanh()` on the policy mean $\mu$ re-introduced:
1. **Mean Squashing & Gradient Vanishing**: $\mu$ is confined to $(-1.0, 1.0)$. In directional market regimes where full allocation (+1.0 or -1.0) is optimal, $(1 - \mu^2) \to 0$ causes severe quadratic vanishing gradients near boundaries.
2. **Inability to Commit / Action Churn**: With non-zero Gaussian variance ($\sigma \approx 0.6$), capping $\mu < 1.0$ forces $\approx 37\%$ of actions to jitter across positions, incurring excessive transaction slippage and commission fees. Removing `Tanh` enables $\mu$ to reach high confidence (e.g. $\mu \ge 2.0$), ensuring stable maximal positioning.

Additionally, to ensure production stability and convergence consistency, the user requested full alignment with official **OpenAI Baselines (`ppo2`) / CleanRL PPO standards**:
- **Item A (Mini-Batch Advantage Normalization)**: Advantage estimates must be normalized at the mini-batch level during surrogate objective optimization.
- **Item B (Clipped Value Loss with 0.5 Factor)**: Match OpenAI Baselines `ppo2` value function loss formulation:
  $$L^{VF}(\theta) = \frac{1}{2} \mathbb{E} \left[ \max\left( (V_\theta(s) - R)^2, (V_{\text{clipped}}(s) - R)^2 \right) \right]$$
  where $V_{\text{clipped}}(s) = V_{\text{old}}(s) + \text{clip}(V_\theta(s) - V_{\text{old}}(s), -\epsilon, \epsilon)$.
- **Item C (Mini-Batch Divisibility at Batch Size 256)**: Prevent sample weight imbalance caused by fractional trailing batches. Adjust `HORIZON_STEPS` and `EPISODE_STEPS` to 512, such that $8 \times 512 = 4096 = 16 \times 256$ exact mini-batches.
- **Item E (OpenAI Monitoring Metrics & AdamW Epsilon)**: Monitor `explained_variance`, `clipfrac`, and policy `std`, while setting AdamW $\epsilon = 10^{-5}$ for numerical stability.

Items D (`target_kl` early stopping) and F (`MlpPolicy` legacy cleanup) were explicitly designated out-of-scope by the user.

---

## 2. Mathematical & Algorithmic Foundations

### 2.1 Mini-batch Advantage Normalization
GAE advantages are computed across rollout trajectories:
$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$
In official OpenAI Baselines `ppo2` and CleanRL Detail #7, normalizing $\hat{A}$ at the mini-batch level:
$$\hat{A}_{\text{mb}} \leftarrow \frac{\hat{A}_{\text{mb}} - \mu(\hat{A}_{\text{mb}})}{\sigma(\hat{A}_{\text{mb}}) + 10^{-8}}$$
ensures uniform gradient magnitude across diverse mini-batches, insulating the policy update from localized volatility clustering in financial time series.

### 2.2 Clipped Value Loss Function with 0.5 Coefficient
In OpenAI Baselines `ppo2`, the Critic loss is clipped around the rollout predictions $V_{\text{old}}$:
$$V_{\text{clipped}} = V_{\text{old}} + \text{clip}(V_\theta(s) - V_{\text{old}}, -\epsilon, \epsilon)$$
$$L^{VF}(\theta) = \frac{1}{2} \frac{1}{|B|} \sum_{i \in B} \max\left( (V_\theta(s_i) - R_i)^2, (V_{\text{clipped}}(s_i) - R_i)^2 \right)$$
The $\frac{1}{2}$ multiplier ensures the derivative w.r.t. $V_\theta$ is $(V_\theta - R)$ rather than $2(V_\theta - R)$, correctly scaling with `VF_COEF = 0.5`.

### 2.3 Explained Variance Metric
$$\text{ExplVar}(V, R) = 1 - \frac{\text{Var}(R - V)}{\text{Var}(R)}$$
- If $\text{ExplVar} = 1.0$: Critic perfectly predicts returns.
- If $\text{ExplVar} \le 0.0$: Critic predictions are worse than or equal to predicting a constant mean.

---

## 3. Target Scope & Architecture Boundaries

### 3.1 File & Module Scope Matrix
| Index | Target File | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) | **In-Scope** | Remove `nn.Tanh()` from `self.fc_actor` (Line 418), leaving `init_(nn.Linear(head_h2, action_dim), std=0.01)` as pure linear output. Update docstrings and comments. |
| **Item 2** | [`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py) | **In-Scope** | In `get_batches()`, yield `old_values[b_idx]` to enable clipped value loss. Preserve raw GAE advantages in `self.advantages` (or provide unnormalized advantage arrays). Enforce dataset size divisibility assertion. |
| **Item 3** | [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) | **In-Scope** | 1. Implement mini-batch advantage normalization in `update_ppo`.<br/>2. Implement clipped value loss with 0.5 factor using `old_values_b`.<br/>3. Compute and log `explained_variance`, `clipfrac`, and `policy_std`. |
| **Item 4** | [`Brain/Common/optimization.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py) | **In-Scope** | Set AdamW default `eps=1e-5` for policy gradient stability (aligned with OpenAI Baselines). |
| **Item 5** | [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) | **In-Scope** | Set `HORIZON_STEPS: 512` and `EPISODE_STEPS: 512` so that $8 \times 512 = 4096 = 16 \times 256$, ensuring zero residual batch and uniform weighting. |
| **Item 6** | [`tests/test_ppo2_unclipped_action.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_ppo2_unclipped_action.py) | **In-Scope [NEW]** | Construct unit test suite verifying: pure linear $\mu$, unclipped action storage, clipped value loss calculation, mini-batch advantage normalization, and metric reporting. |
| **Item 7** | [`Brain/PPO2/lib/model.py:L91-L245`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L91-L245) | **Out-of-Scope (Preserved)** | Legacy `MlpPolicy` code is left untouched as instructed by user ("F: 跳過"). |
| **Item 8** | `TARGET_KL` early stopping in `PPO2_rl_train.py` | **Out-of-Scope (Preserved)** | Skipped as instructed by user ("D: 跳過"). |

---

## 4. Implementation Checklist & Detailed Blueprint

### 4.1 Step-by-Step Task Checklist
- [x] **Step 1: Remove `nn.Tanh()` from `fc_actor` in `Brain/PPO2/lib/model.py`**
  - [x] Remove `nn.Tanh()` from `fc_actor` sequence.
  - [x] Update docstring in `MambaContinuousActorCritic` to state pure linear projection.
- [x] **Step 2: Update `RolloutBuffer` in `Brain/PPO2/lib/experience.py`**
  - [x] In `get_batches()`, yield `values[b_idx]` as `old_values_b`.
  - [x] In `get_batches()`, add assertion ensuring `dataset_size % batch_size == 0`.
- [x] **Step 3: Update `PPO2.update_ppo` in `PPO2_rl_train.py`**
  - [x] Receive `old_values_b` from `buffer.get_batches()`.
  - [x] Normalize `advantages_b` at the mini-batch level: `(advantages_b - advantages_b.mean()) / (advantages_b.std() + 1e-8)`.
  - [x] Compute clipped value loss with 0.5 multiplier:
    ```python
    v_clipped = old_values_b + torch.clamp(values - old_values_b, -clip_eps, clip_eps)
    vf_loss_unclipped = (values - returns_b).pow(2)
    vf_loss_clipped = (v_clipped - returns_b).pow(2)
    value_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped).mean()
    ```
  - [x] Calculate `clipfrac = torch.mean((torch.abs(ratio - 1.0) > clip_eps).float()).item()`.
  - [x] Calculate `explained_variance(values, returns_b)`.
  - [x] Record current policy exploration `std = self.agent.model.log_std.exp().mean().item()`.
  - [x] Include `clipfrac`, `explained_variance`, and `std` in training metrics and CLI logs.
- [x] **Step 4: Update Optimizer Epsilon & Hyperparameter Config**
  - [x] In [`Brain/Common/optimization.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py), set AdamW default `eps=1e-5`.
  - [x] In [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml), update `HORIZON_STEPS: 512` and `EPISODE_STEPS: 512`.
- [x] **Step 5: Create Test Suite & Run Verification**
  - [x] Create [`tests/test_ppo2_unclipped_action.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_ppo2_unclipped_action.py).
  - [x] Execute test suite using virtual environment interpreter `/home/b0457812963/Mamba3RL/bin/python`.

---

## 5. Validation & Verification Plan

### 5.1 Verification Commands
```bash
/home/b0457812963/Mamba3RL/bin/python -m unittest tests/test_ppo2_unclipped_action.py
```

### 5.2 Acceptance Verification Criteria
1. **Unbounded Mean Output**: `model.fc_actor` produces values outside $[-1.0, 1.0]$ without asymptotic saturation.
2. **Batch Divisibility**: `HORIZON_STEPS=512`, `NUM_WORKERS=8` yields exactly 16 batches of size 256.
3. **Clipped Value Loss & Mini-Batch Advantage Normalization**: Value loss matches official OpenAI formulation; mini-batch advantages possess zero mean and unit variance.
4. **Metric Correctness**: `explained_variance` correctly outputs 1.0 for perfect predictions and $\le 0.0$ for uncorrelated predictions; `clipfrac` is within $[0.0, 1.0]$.
