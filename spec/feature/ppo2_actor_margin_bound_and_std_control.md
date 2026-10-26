# Specification: PPO2 Sigmoid Squashed-Gaussian Policy & Bounded Action-Space Entropy for Long-Only [0.0, 1.0]

- **Category**: `feature`
- **Spec Path**: `spec/feature/ppo2_actor_margin_bound_and_std_control.md`
- **Status**: `Implemented & Verified`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
Diagnostics on `checkpoint_iter_6570.pt` (26.91M training steps) revealed severe policy degeneration into a discrete bang-bang controller (`|Action| == 1.0` on `98.77%` of test bars, `Position == 0.0` on `0.01%` of bars, intermediate sizing on only `1.16%` of bars). Inspection of model parameters and raw actor forward passes identified an **Unconstrained Mean–Variance Co-Explosion** (`log_std -> +0.894`, $\sigma = 2.445$, $|\mu| \to 20.5 \sim 31.5$) caused by a fundamental **Probability Distribution Support Mismatch**:

1. **Support Mismatch Between Policy Distribution and Bounded Action Space**:
   While [`BaseTradingEnv`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) operates on the bounded Long-Only allocation interval `[0.0, 1.0]`, [`PPO2Agent.get_action()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L79-L115) models the policy as an unbounded Gaussian $\mathcal{N}(\mu_\theta(s), \sigma^2)$ supported on $(-\infty, +\infty)$ and applies `torch.clamp(action, 0.0, 1.0)` *outside* the probabilistic model. As established in bounded-control policy gradient theory (Chou et al., ICML 2017), evaluating $\log \pi_\theta$ on unbounded samples while executing clamped actions induces an intrinsic boundary score-function bias ($\nabla_\mu \log \mathcal{N}(u; \mu, \sigma^2) = \frac{u - \mu}{\sigma^2}$) that continuously pushes $\mu \to \pm\infty$ whenever a boundary allocation (`0.0` Cash or `1.0` Full Long) outperforms interior noise.
2. **Wrong-Space Entropy Bonus (`(-\infty, +\infty)` Instead of `(0.0, 1.0)`)**:
   [`PPO2Agent.evaluate_actions()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L117-L128) computes unclipped Gaussian differential entropy $\mathcal{H}(U) = \frac{1}{2}\ln(2\pi e \sigma^2)$, which is invariant to $\mu$ and unbounded above ($\frac{\partial \mathcal{H}(U)}{\partial \log \sigma} = +1 > 0$). Under [`ENT_COEF: 0.01`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L24), the policy harvests free latent-space entropy by pushing $\sigma \to \sigma_{\max}$ while simultaneously pushing $|\mu| \to 20+$ so executed actions remain pinned at the boundaries without suffering random turnover fees.

### 1.2 Architectural Selection: Pure PPO Loss + Sigmoid Squashed-Gaussian Policy
To preserve the exact theoretical integrity of PPO's Clipped Surrogate Objective (Schulman et al., 2017)—**strictly prohibiting any heuristic penalty terms on `total_loss`**—we align the policy distribution directly with the bounded action space $(0.0, 1.0)$ via a **Bijective Sigmoid Change-of-Variables (Squashed-Gaussian) Policy** (Haarnoja et al., ICML 2018):

1. **Exact Change-of-Variables Likelihood on `(low, high)` (`(0.0, 1.0)`)**:
   - Sample latent variable $u \sim \mathcal{N}(\mu_\theta(s), \sigma^2)$ (or deterministic $u = \mu_\theta(s)$ at inference) and map bijectively onto the bounded action space with width $w = \text{high} - \text{low}$ (`w = 1.0` for `[0.0, 1.0]`):
     $$a = g(u) = \text{low} + w \cdot \text{sigmoid}(u) \in (\text{low}, \text{high})$$
   - Store latent $u \in \mathbb{R}$ in [`RolloutBuffer`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py) (`out.action = u`) to avoid `float32` `logit` inversion rounding at boundaries, while executing `out.clipped_action = a` in the environment.
   - By the Change-of-Variables Theorem ($g'(u) = w \cdot \text{sigmoid}(u)(1 - \text{sigmoid}(u))$), the exact log-density of executed action $a$ is:
     $$\log \pi_\theta(a \mid s) = \log \mathcal{N}(u; \mu_\theta(s), \sigma^2) - \ln w + \text{softplus}(u) + \text{softplus}(-u)$$
   - In [`PPO2.update_ppo()`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L270-L271), the parameter-free Jacobian term $-\ln w + \text{softplus}(u_t) + \text{softplus}(-u_t)$ cancels identically in $\log r_t(\theta) = \log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)$, guaranteeing exact importance-sampling ratios and smooth Gaussian KL curvature.
2. **Exact Bounded Action-Space Differential Entropy ($\mathcal{H}(A \mid s) \le \ln w = 0$)**:
   - The true differential entropy of the executed action $A = g(U)$ on $(0.0, 1.0)$ is:
     $$\mathcal{H}(A \mid s) = \frac{1}{2}\ln(2\pi e \sigma^2) + \ln w - \mathbb{E}_{z \sim \mathcal{N}(0, 1)}\Big[\text{softplus}(\mu + \sigma z) + \text{softplus}\big(-(\mu + \sigma z)\big)\Big] \le 0$$
   - Evaluated deterministically via **16-point Probabilists' Gauss–Hermite Quadrature** (zero Monte-Carlo variance, machine-precision accuracy $\sim 10^{-9}$, differentiable w.r.t. both $\mu$ and $\log \sigma$):
     - **Automatic Variance Self-Regulation**: When $\sigma$ grows beyond $\approx 1.7$, probability mass piles up at $0$ and $1$, causing $\mathcal{H}(A \mid s)$ to *decrease* ($\frac{\partial \mathcal{H}(A)}{\partial \log \sigma} < 0$). PPO's native entropy bonus automatically prevents variance runaway.
     - **Automatic Mean Self-Regulation & Clean Cash Parking**: Because $g'(u) = a(1 - a)$, at $\mu \approx -6.0, \sigma \approx 0.6$, over `99%` of rollout samples satisfy $a \le 0.01$ (`deadzone_threshold`), parking cleanly in `0.0` Cash with zero turnover fee bleed, while further outward drift ($|\mu| \to 20$) linearly decreases $\mathcal{H}(A \mid s) \approx \text{const} - |\mu|$ and is naturally arrested by PPO's native entropy bonus.

### 1.3 Current vs. Expected Behavior
| Dimension | Current Behavior | Expected Behavior (Sigmoid Squashed-Gaussian) |
| :--- | :--- | :--- |
| **PPO Surrogate Loss** | `policy_loss + vf_coef * value_loss + ent_coef * entropy_loss` | **100% Preserved Unchanged** (zero artificial penalty terms) |
| **Policy Distribution $\pi_\theta(a \mid s)$** | Unbounded $\mathcal{N}(\mu, \sigma^2)$ on $(-\infty, +\infty)$ + external `clamp(0.0, 1.0)` | Exact Sigmoid Squashed-Gaussian bijection onto `(0.0, 1.0)` with analytical Jacobian correction |
| **Entropy Objective $\mathcal{H}$** | Unbounded latent entropy $\frac{1}{2}\ln(2\pi e \sigma^2) \in (-\infty, +\infty)$ (invariant to $\mu$) | Exact action-space differential entropy $\mathcal{H}(A \mid s) \le 0$ via 16-point Gauss–Hermite quadrature (depends on both $\mu$ and $\sigma$) |
| **Latent `log_std` Config** | Hardcoded `init_log_std=-0.5, log_std_min=-5.0, log_std_max=1.0` in model only | Configurable via YAML (`INIT_LOG_STD: 0.0`, `LOG_STD_MIN: -5.0`, `LOG_STD_MAX: 1.0`) wired through train & engine |
| **Risk Utility (`RISK_LAMBDA`)** | `RISK_LAMBDA: 0.5` | **Preserved at `0.5`** to isolate policy distribution impact |
| **Training Diagnostics** | Logs unclamped `Std` only; allocation center is invisible | Logs clamped latent `Std`, bounded `Ent`, `MeanAct` (`sigmoid(mu)`), and `MeanMu` |

### 1.4 Technical Root Cause & References
- Unbounded Gaussian sampling & missing Jacobian / squashed entropy in agent: [`Brain/PPO2/lib/Agent.py:L79-L128`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L79-L128)
- Hardcoded `init_log_std` & missing defensive bound checks in model: [`Brain/PPO2/lib/model.py:L322-L342`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L322-L342), [`L458-L474`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L458-L474)
- Missing `log_std` config forwarding, unclamped `policy_std` readout, and missing `MeanAct` telemetry: [`PPO2_rl_train.py:L204-L220`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L204-L220), [`L323-L334`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L323-L334), [`L528-L538`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L528-L538)
- Missing `log_std` config bindings in inference/test engine: [`Brain/PPO2/lib/engine.py:L73-L88`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L73-L88), [`L224-L235`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L224-L235)
- Uncalibrated `ENT_COEF: 0.01` for normalized advantages: [`configs/training/ppo_v2.yaml:L24`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L24)

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix
| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) | **In-Scope** | Update `MambaContinuousActorCritic.__init__` default `init_log_std=0.0` with defensive assertions (`log_std_min < log_std_max`, `log_std_min <= init_log_std <= log_std_max`); preserve linear `fc_actor` and identical `state_dict` keys; update `AgentOutput` docstring to English. |
| **Item 2** | [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py) | **In-Scope** | Forward `init_log_std`, `log_std_min`, `log_std_max`; validate finite `Box` bounds; implement Sigmoid Squashed-Gaussian transformation + exact Jacobian `log_prob` in `get_action()` and `evaluate_actions()`; compute deterministic 16-point Gauss–Hermite squashed entropy; support optional `return_stats: bool = False` for read-only telemetry. |
| **Item 3** | [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) | **In-Scope** | Wire `INIT_LOG_STD`, `LOG_STD_MIN`, `LOG_STD_MAX` in `_prepare_agent()`; **keep `total_loss` formula strictly untouched**; fix `policy_std` readout to use clamped `log_std`; log `MeanAct` and `MeanMu` in iteration diagnostics. |
| **Item 4** | [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) | **In-Scope** | Bind `INIT_LOG_STD`, `LOG_STD_MIN`, `LOG_STD_MAX` via `getattr` defaults and pass to `PPO2Agent` in `build_agent()`. |
| **Item 5** | [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) | **In-Scope** | Set `ENT_COEF: 0.002`, `INIT_LOG_STD: 0.0`, `LOG_STD_MIN: -5.0`, `LOG_STD_MAX: 1.0`; keep `RISK_LAMBDA: 0.5` unchanged. |
| **Item 6** | `tests/test_ppo2_squashed_gaussian_policy.py` | **In-Scope [NEW]** | Automated unit tests verifying PDF normalization ($\int_0^1 \pi(a|s)\,da = 1$), Gauss–Hermite entropy accuracy vs. numerical integration, entropy concavity ($\mathcal{H} \le 0$ and inward restoring gradients at large $|\mu|$ and $\sigma$), `[0.0, 1.0]` action bounds, and pure PPO update integrity. |
| **Item 7** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py), [`Brain/PPO2/lib/reward.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/reward.py), [`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py), [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py) | **Out-of-Scope** | Strictly preserved with zero modifications. |

### 2.2 Architectural Boundaries & Non-Goals
- **Strict Preservation of Original PPO Objective**:
  `total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss` in [`PPO2.update_ppo()`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L307-L309) is **NOT** modified; no auxiliary actor margin/bound losses are introduced.
- **Reward Function Stability (`RISK_LAMBDA: 0.5`)**:
  [`DownsidePenaltyReward`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/reward.py) parameters remain fixed (`RISK_LAMBDA: 0.5`) to maintain strict experimental control while observing training dynamics under the squashed-Gaussian policy.
- **Full `state_dict` Compatibility**:
  Quadrature nodes/weights and action-bound tensors live on `PPO2Agent` rather than `nn.Module` persistent buffers, keeping `MambaContinuousActorCritic.state_dict()` keys 100% unchanged.

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Update `MambaContinuousActorCritic` Defaults & Defensive Assertions**
  - [x] In [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py), update `init_log_std: float = 0.0` (preserving `log_std_min: float = -5.0`, `log_std_max: float = 1.0`) and add `assert log_std_min < log_std_max` and `assert log_std_min <= init_log_std <= log_std_max`.
  - [x] Update `AgentOutput` docstring in English to document `action` (pre-squash latent $u \in \mathbb{R}$ for exact `RolloutBuffer` replay) and `clipped_action` (squashed bounded action $a \in (\text{low}, \text{high})$ for environment execution).
- [x] **Step 2: Implement Sigmoid Squashed-Gaussian Policy & Gauss–Hermite Entropy in `PPO2Agent`**
  - [x] In [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py), add `init_log_std: float = 0.0`, `log_std_min: float = -5.0`, `log_std_max: float = 1.0` to `PPO2Agent.__init__` and forward to `MambaContinuousActorCritic`.
  - [x] Assert finite `Box` bounds (`ac_space.high > ac_space.low`) in `_validate_spaces()`, and precompute `_act_low`, `_act_scale`, `_log_act_scale` plus 16-point probabilists' Gauss–Hermite nodes/weights (`np.polynomial.hermite_e.hermegauss(16)`) on `self.device`.
  - [x] Implement `_squash_action(u)` and `_squashed_log_prob(dist, u)` using numerically stable `F.softplus(u) + F.softplus(-u) - self._log_act_scale`.
  - [x] Implement `_squashed_entropy(dist, mu, std)` via 16-point Gauss–Hermite quadrature.
  - [x] Update `get_action()` and `evaluate_actions(states, time_states, actions, return_stats: bool = False)` (defaulting to the backward-compatible 3-tuple `(log_prob, entropy, value)`).
- [x] **Step 3: Wire Configs & Diagnostic Telemetry in `PPO2_rl_train.py`, `engine.py`, and `ppo_v2.yaml`**
  - [x] Pass `INIT_LOG_STD`, `LOG_STD_MIN`, `LOG_STD_MAX` via `getattr` in `PPO2._prepare_agent()` and `PPO2Engine.build_agent()`.
  - [x] In `PPO2.update_ppo()`, leave `total_loss` completely untouched; record read-only `mean_action` (`squashed_mu`) and `mean_mu` metrics, fix `policy_std` to read clamped `log_std`, and display `MeanAct` and `MeanMu` in iteration printouts.
  - [x] Update [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) with `ENT_COEF: 0.002`, `INIT_LOG_STD: 0.0`, `LOG_STD_MIN: -5.0`, `LOG_STD_MAX: 1.0` (keeping `RISK_LAMBDA: 0.5`).
- [x] **Step 4: Verification Suite Creation & Execution**
  - [x] Create `tests/test_ppo2_squashed_gaussian_policy.py` and verify all mathematical and runtime invariants using `/home/b0457812963/Mamba3RL/bin/python`.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py)
```python
        low_np = np.asarray(self.ac_space.low, dtype=np.float32)
        high_np = np.asarray(self.ac_space.high, dtype=np.float32)
        scale_np = high_np - low_np

        self._act_low = torch.as_tensor(low_np, dtype=torch.float32, device=self.device)
        self._act_scale = torch.as_tensor(scale_np, dtype=torch.float32, device=self.device)
        self._log_act_scale = torch.log(self._act_scale)

        gh_nodes_np, gh_weights_np = np.polynomial.hermite_e.hermegauss(16)
        gh_weights_np = gh_weights_np / np.sqrt(2.0 * np.pi)
        self._gh_nodes = torch.as_tensor(
            gh_nodes_np, dtype=torch.float32, device=self.device
        ).view(1, 1, -1)
        self._gh_weights = torch.as_tensor(
            gh_weights_np, dtype=torch.float32, device=self.device
        ).view(1, 1, -1)
```
And in action evaluation / sampling methods:
```python
    def _squash_action(self, latent_u: torch.Tensor) -> torch.Tensor:
        return self._act_low + self._act_scale * torch.sigmoid(latent_u)

    def _squashed_log_prob(
        self, dist: torch.distributions.Normal, latent_u: torch.Tensor
    ) -> torch.Tensor:
        neg_log_det_jac = (
            F.softplus(latent_u) + F.softplus(-latent_u) - self._log_act_scale
        )
        return (dist.log_prob(latent_u) + neg_log_det_jac).sum(dim=-1)

    def _squashed_entropy(
        self,
        dist: torch.distributions.Normal,
        mu: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        u_quad = mu.unsqueeze(-1) + std.unsqueeze(-1) * self._gh_nodes
        expected_neg_log_jac = (
            (F.softplus(u_quad) + F.softplus(-u_quad)) * self._gh_weights
        ).sum(dim=-1) - self._log_act_scale
        return (dist.entropy() - expected_neg_log_jac).sum(dim=-1)
```

#### 3.2.2 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)
```python
                log_probs, entropy, values, stats = self.agent.evaluate_actions(
                    states_b, time_states_b, actions_b, return_stats=True
                )
                ...
                # Pure PPO Objective (100% unmodified)
                entropy_loss = -entropy.mean()
                total_loss = (
                    policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
                )
```

#### 3.2.3 [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml)
```yaml
ENT_COEF: 0.002
INIT_LOG_STD: 0.0
LOG_STD_MIN: -5.0
LOG_STD_MAX: 1.0
...
RISK_LAMBDA: 0.5
```

---

## 4. Validation & Verification Plan

### 4.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST adhere to repository environment standards using the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python`
> Relative path: `../bin/python`

```bash
/home/b0457812963/Mamba3RL/bin/python -m unittest tests/test_ppo2_squashed_gaussian_policy.py
```

### 4.2 Acceptance Verification Checklist
- [x] **1. Exact Change-of-Variables Density Normalization**:
  - Numerical integration of $\exp(\log \pi_\theta(a \mid s))$ across $a \in (0.0, 1.0)$ equals `1.0` within `< 1e-5` across interior ($\mu = 0$), bearish cash ($\mu = -6$), and bullish full-long ($\mu = +6$) regimes.
  - Importance sampling log-ratio $\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)$ remains finite with zero `NaN`/`Inf` even for extreme latent samples $u = \pm 50.0$.
- [x] **2. Bounded Action-Space Entropy & Self-Regulating Gradients**:
  - For all $(\mu, \sigma)$ on `Box(0.0, 1.0)`, `entropy <= 0.0` (matching exact numerical integration within `< 1e-4` for operating $\sigma \le 3.0$).
  - Gradient check on `-entropy.mean()`:
    - At large positive $\mu = +8.0$, entropy gradient pulls $\mu$ **downward** ($\frac{\partial (-\mathcal{H})}{\partial \mu} > 0$); at large negative $\mu = -8.0$, entropy gradient pulls $\mu$ **upward** ($\frac{\partial (-\mathcal{H})}{\partial \mu} < 0$).
    - At excessive $\sigma = 2.5$ (`log_std = 0.916`), entropy gradient pulls `log_std` **downward** ($\frac{\partial (-\mathcal{H})}{\partial \log \sigma} > 0$), proving variance runaway is impossible without modifying PPO Loss.
- [x] **3. Pure PPO Update & Inference Parity**:
  - `PPO2Agent.get_action()` returns `clipped_action` strictly inside `(0.0, 1.0)` in both stochastic and deterministic modes, and `PPO2.update_ppo` completes cleanly with finite `mean_action` and `mean_mu` diagnostics.
