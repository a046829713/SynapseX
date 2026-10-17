# Specification: PPO2 Unclipped Action Storage & Actor Linear Output Alignment

- **Category**: `bugfix`
- **Spec Path**: `spec/bugfix/ppo2_unclipped_action_storage.md`
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
In continuous action reinforcement learning algorithms such as Proximal Policy Optimization (PPO), continuous action distributions are typically parameterized by a diagonal Gaussian (Normal) policy:
$$\pi_\theta(a \mid s) = \mathcal{N}\left(a; \mu_\theta(s), \sigma_\theta(s)^2\right)$$
Because Gaussian distributions possess infinite support on $\mathbb{R}$, sampled actions $a_{\text{raw}} \sim \mathcal{N}(\mu, \sigma^2)$ can exceed the bounded physical domain of the environment (for instance, portfolio allocation weights constrained to $[-1.0, 1.0]$ in trading environments).

In canonical implementations established by **OpenAI Baselines**, **OpenAI Spinning Up**, and **CleanRL (Detail 18 of the 37 Implementation Details of PPO)**:
1. **Unbounded Mean Projection**: The policy network's final layer for $\mu$ is a pure linear layer without non-linear squashing activations (no `nn.Tanh()`). This allows $\mu$ to move freely in $\mathbb{R}$.
2. **Unclipped Sampling**: Actions are sampled from the unbounded Gaussian distribution: $a_{\text{raw}} \sim \mathcal{N}(\mu, \sigma^2)$.
3. **Unbiased Log-Prob**: The log-probability is evaluated directly on the unclipped sample: $\log \pi(a_{\text{raw}})$.
4. **Environment-Only Clipping**: The action is clipped strictly when passed to the environment: $a_{\text{env}} = \text{clip}(a_{\text{raw}}, -1.0, 1.0)$.
5. **Unclipped Buffer Storage**: The **unclipped action** $a_{\text{raw}}$ and its corresponding $\log \pi(a_{\text{raw}})$ are stored in `RolloutBuffer`.
6. **Consistent Ratio Updates**: During PPO updates, surrogate importance sampling ratios are evaluated on the stored unclipped actions:
   $$r_t(\theta) = \frac{\pi_\theta(a_{\text{raw}} \mid s_t)}{\pi_{\theta_{\text{old}}}(a_{\text{raw}} \mid s_t)}$$

---

### 1.2 Current Failures in SynapseX
Two compounding architectural flaws exist in the current SynapseX continuous PPO implementation:

#### Flaw A: `nn.Tanh()` Squeezing the Gaussian Mean $\mu$ in `fc_actor`
In [`Brain/PPO2/lib/model.py:L327-L336`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L327-L336):
```python
# Actor 頭 (輸出連續動作均值 mu，經 Tanh 限制在 [-1.0, 1.0])
self.fc_actor = nn.Sequential(
    nn.Linear(flat_dim, 512),
    nn.LayerNorm(512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.LayerNorm(256),
    nn.ReLU(),
    nn.Linear(256, action_dim),
    nn.Tanh(),  # <--- Flaw A: Locks Gaussian mean inside (-1.0, 1.0)
)
```
- **Does `nn.Tanh()` prevent out-of-bound actions? No.**
  Even though $\mu \in (-1, 1)$, actions are sampled from $a \sim \mathcal{N}(\mu, \sigma^2)$. With default `init_log_std = -0.5` ($\sigma \approx 0.6065$), whenever the policy commits towards a directional trade (e.g., $\mu = 0.8$):
  $$P(a > 1.0) = P\left(Z > \frac{1.0 - 0.8}{0.6065}\right) \approx 37\%$$
  Over 35% of sampled actions STILL spill over the $[-1.0, 1.0]$ boundary and require clipping.
- **Double Vanishing Gradient**:
  The gradient w.r.t. pre-activation $z$ is:
  $$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \mu} \cdot \frac{\partial \mu}{\partial z} = \frac{a_{\text{clip}} - \mu}{\sigma^2} \cdot (1 - \mu^2)$$
  As $\mu \to 1.0$, $(1.0 - \mu) \to 0$ and $(1 - \mu^2) \to 0$, producing an $O((1 - \mu)^2)$ quadratic vanishing gradient. The policy becomes paralyzed near the boundary.
- **Inability to Commit / Position Jitter**:
  In single-direction trending markets, the optimal strategy is continuous 100% long ($+1.0$). If $\mu$ is capped at $0.99$, high variance ($\sigma \approx 0.6$) causes subsequent actions to randomly jitter between $1.0, 0.6, 0.2$, causing catastrophic turnover and transaction fees. With unconstrained $\mu$, the network can learn $\mu = 2.0$, ensuring $>98\%$ of samples clamp to $+1.0$ (firm position holding).

#### Flaw B: Clamping Actions Before Log-Prob and Storing Clamped Actions in Buffer
In [`Brain/PPO2/lib/Agent.py:L78-L81`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L78-L81):
```python
# 連續動作裁剪至合法邊界 [-1.0, 1.0]
action = torch.clamp(action, -1.0, 1.0)
log_prob = dist.log_prob(action).sum(dim=-1)
```
And in [`PPO2_rl_train.py:L246-L276`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L246-L276):
```python
self.buffer.store(
    obs,
    out.action.cpu(),     # <--- Storing clamped action [-1.0, 1.0]
    out.log_prob.cpu(),   # <--- Log-prob evaluated at clamped boundary
    ...
)
```
- **Boundary Gradient Sign Inversion**:
  If $\mu = 1.1$ and a sample $a_{\text{raw}} = 1.5$ incurs a loss ($\hat{A} = -1.0$):
  - Unclipped: $(1.5 - 1.1) \times (-1.0) = -0.4 < 0 \implies \mu$ decreases (correct).
  - Clipped: $(1.0 - 1.1) \times (-1.0) = +0.1 > 0 \implies \mu$ increases (catastrophic inversion).
- **Probability Density vs. Point Mass Contradiction**:
  A clamped Gaussian is a mixed distribution with discrete point mass at $\pm 1.0$:
  $$P(A = 1.0) = 1 - \Phi\left(\frac{1.0 - \mu}{\sigma}\right)$$
  Substituting continuous density $f(1.0)$ destroys the validity of the importance sampling ratio $r_t(\theta)$.

---

### 1.3 Current vs. Expected Behavior Matrix
| Dimension | Current Behavior (SynapseX) | Expected Behavior (OpenAI / CleanRL Standard) |
| :--- | :--- | :--- |
| **Actor Head Projection** | Ends with `nn.Tanh()`, artificially capping $\mu \in (-1, 1)$. | Pure linear projection `nn.Linear(256, action_dim)` without `Tanh`. |
| **Mean Parametrization** | Saturated at $\pm 1.0$, suffering double vanishing gradient. | Unconstrained $\mu \in \mathbb{R}$, allowing high-confidence boundary saturation. |
| **Action Sampling** | Sample $a_{\text{raw}}$, then immediately `torch.clamp` to $[-1, 1]$. | Sample $a_{\text{raw}} \sim \mathcal{N}(\mu, \sigma^2)$; keep $a_{\text{raw}}$ unclipped. |
| **Log-Prob Evaluation** | Evaluated on clamped action: `dist.log_prob(clamp(a))`. | Evaluated on raw action: `dist.log_prob(a_raw)`. |
| **AgentOutput Dataclass** | `action` holds clamped action. No separate field for clipped/unclipped. | `action` holds unclipped raw action; `clipped_action` holds clamp to $[-1, 1]$. |
| **RolloutBuffer Storage** | Buffer stores clamped action. | Buffer stores unclipped raw action $a_{\text{raw}}$ and its true $\log \pi(a_{\text{raw}})$. |
| **Environment Step** | Receives clamped action (or relies on internal `np.clip`). | Receives explicit `out.clipped_action` (guaranteed $[-1.0, 1.0]$). |
| **PPO Policy Update** | `evaluate_actions` evaluates `dist.log_prob(actions_b)` on clamped actions. | `evaluate_actions` evaluates `dist.log_prob(actions_b)` on unclipped actions. |

---

## 2. Mathematical & Algorithmic Foundations

### 2.1 The Policy Gradient Theorem on Unconstrained Action Parametrization
Let the policy $\pi_\theta(a \mid s)$ be an unconstrained Gaussian distribution over $a \in \mathbb{R}^d$.
The policy gradient theorem establishes that:
$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) Q^\pi(s, a) \right]$$
The score function identity $\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a \mid s)] = 0$ requires that expectation is taken w.r.t. the exact distribution $\pi_\theta$ from which $a$ is drawn.

When the environment applies an action clipping function $g(a) = \text{clip}(a, -1, 1)$, the transition dynamics and reward function become composite functions of $g(a)$:
$$r(s, a) = r(s, g(a)), \quad P(s' \mid s, a) = P(s' \mid s, g(a))$$
By treating $g(\cdot)$ as part of the environment dynamics:
1. The policy parameter space remains completely unconstrained.
2. The actor network can output arbitrary $\mu \in \mathbb{R}$ with smooth, linear gradients.
3. The sample $a_t \sim \pi_\theta(\cdot \mid s_t)$ is the true action taken by the agent.
4. The likelihood ratio $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ is mathematically exact and numerically stable.

---

## 3. Target Scope & Architecture Boundaries

### 3.1 File & Module Scope Matrix
| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`Brain/PPO2/lib/model.py:L327-L336`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L327-L336) | **In-Scope** | Remove `nn.Tanh()` from `self.fc_actor` in `MambaContinuousActorCritic`, leaving a pure linear projection for $\mu$. Initialize final linear layer with orthogonal gain 0.01. |
| **Item 2** | [`Brain/PPO2/lib/model.py:L371-L378`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L371-L378) | **In-Scope** | Extend `AgentOutput` dataclass to include `clipped_action: Optional[torch.Tensor] = None`. |
| **Item 3** | [`Brain/PPO2/lib/Agent.py:L67-L94`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L67-L94) | **In-Scope** | In `get_action`: compute `log_prob` using raw unclipped action; generate `clipped_action = torch.clamp(action, -1.0, 1.0)`; return both in `AgentOutput`. |
| **Item 4** | [`PPO2_rl_train.py:L244-L278`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L244-L278) | **In-Scope** | Pass `out.clipped_action.cpu().numpy()` to `vec_env.step(...)`; store `out.action.cpu()` (unclipped) in `self.buffer.store(...)`. |
| **Item 5** | [`tests/test_ppo2_unclipped_action.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_ppo2_unclipped_action.py) | **In-Scope [NEW]** | Create dedicated unit tests verifying linear $\mu$ output, unclipped storage, clipped environment execution, gradient correctness, and KL consistency. |
| **Item 6** | [`Brain/PPO2/lib/environment.py:L293-L295`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L293-L295) | **Out-of-Scope (Preserved)** | Preserve `np.clip(action_val, -1.0, 1.0)` in `BaseTradingEnv.step()` as an additional defensive boundary guard. |
| **Item 7** | [`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py) | **Out-of-Scope (Preserved)** | `RolloutBuffer` already accepts generic action tensors; buffer structure and GAE logic remain unchanged. |
| **Item 8** | Loss & Optimizer in [`PPO2_rl_train.py:L150-L208`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L150-L208) | **Out-of-Scope (Preserved)** | Clipped surrogate ratio calculation and value loss remain unchanged. |

---

## 4. Implementation Checklist & Detailed Blueprint

### 4.1 Step-by-Step Task Checklist
- [x] **Step 1: Remove `nn.Tanh()` from `fc_actor` and Extend `AgentOutput`**
  - [x] In [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py), remove `nn.Tanh()` from `self.fc_actor`.
  - [x] Initialize the final linear layer `nn.Linear(256, action_dim)` using `init_(..., std=0.01)` so initial policy mean is centered at 0.
  - [x] Add `clipped_action: Optional[torch.Tensor] = None` to `AgentOutput` dataclass.
- [x] **Step 2: Update `PPO2Agent.get_action`**
  - [x] Sample `raw_action = dist.sample() if stochastic else mu`.
  - [x] Compute `log_prob = dist.log_prob(raw_action).sum(dim=-1)`.
  - [x] Compute `clipped_action = torch.clamp(raw_action, -1.0, 1.0)`.
  - [x] Return `AgentOutput(action=raw_action, clipped_action=clipped_action, value=value, log_prob=log_prob)`.
- [x] **Step 3: Update `PPO2_rl_train.py` Rollout Loop**
  - [x] Use `out.clipped_action.cpu().numpy()` for `vec_env.step(...)`.
  - [x] Store `out.action.cpu()` (the unclipped tensor) into `self.buffer.store(...)`.
- [x] **Step 4: Create Comprehensive Unit Test**
  - [x] Create [`tests/test_ppo2_unclipped_action.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_ppo2_unclipped_action.py).
  - [x] Test 1: Verify `fc_actor` output is unconstrained linear (can output values outside $[-1, 1]$).
  - [x] Test 2: Verify `AgentOutput` contains unclipped `action` and strictly bounded `clipped_action`.
  - [x] Test 3: Verify `log_prob` matches `Normal(mu, std).log_prob(raw_action)` exactly.
  - [x] Test 4: Verify gradient direction under negative advantage when $\mu > 1.0$ pushes $\mu$ downwards (preventing sign reversal).
  - [x] Test 5: Verify initial rollout and `evaluate_actions` produce ratio $r_t = 1.0 \pm 1e-6$ and approximate $KL \approx 0.0$.

---

### 4.2 Detailed Code Modifications Blueprint

#### 4.2.1 [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py)
```python
        # Critic 頭 (估計狀態價值 V(s))
        self.fc_val = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Actor 頭 (輸出連續動作均值 mu，純線性投影無 Tanh，對齊 OpenAI Baselines / CleanRL)
        actor_out_layer = nn.Linear(256, action_dim)
        init_(actor_out_layer, std=0.01)  # 初始權重微小化，使初期均值趨近於 0
        self.fc_actor = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            actor_out_layer,
        )

...

@dataclass
class AgentOutput:
    """Agent在與環境互動時的輸出。"""

    action: torch.Tensor          # 未裁剪的原始動作 (Unclipped / Raw Action)，用於 Buffer 儲存與策略訓練
    value: torch.Tensor           # 狀態價值估計 V(s)
    log_prob: torch.Tensor        # 基於未裁剪動作計算的高斯對數機率 log π(action)
    clipped_action: Optional[torch.Tensor] = None  # 裁剪至合法邊界 [-1.0, 1.0] 的動作，專供環境執行
```

#### 4.2.2 [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py)
```python
    def get_action(self, obs, stochastic: bool = True) -> AgentOutput:
        self.model.eval()
        states, time_states, is_single = self._preprocess_obs(obs)
        with torch.no_grad():
            mu, std, value = self.model(states, time_states)
            dist = torch.distributions.Normal(mu, std)
            if stochastic:
                action = dist.sample()
            else:
                action = mu

            # 根據 OpenAI Baselines / CleanRL 標準實作：
            # 1. log_prob 必須基於未裁剪的原始高斯採樣動作計算
            # 2. clipped_action 供環境 step 執行，確保環境介面在合法範圍內
            log_prob = dist.log_prob(action).sum(dim=-1)
            clipped_action = torch.clamp(action, -1.0, 1.0)

        if is_single:
            return AgentOutput(
                action=action.squeeze(0),                  # 原始未裁剪動作
                clipped_action=clipped_action.squeeze(0),  # 裁剪後動作
                value=value.squeeze(0),
                log_prob=log_prob.squeeze(0),
            )

        return AgentOutput(
            action=action,                                 # 原始未裁剪動作 (N, action_dim)
            clipped_action=clipped_action,                 # 裁剪後動作 (N, action_dim)
            value=value,                                   # (N,)
            log_prob=log_prob,                             # (N,)
        )
```

#### 4.2.3 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)
```python
                # 1. Rollout sampling across all N workers
                for _ in range(horizon_steps):
                    out = self.agent.get_action(obs, stochastic=True)
                    # 傳入環境執行已裁剪動作
                    env_actions_np = (
                        out.clipped_action.cpu().numpy()
                        if out.clipped_action is not None
                        else np.clip(out.action.cpu().numpy(), -1.0, 1.0)
                    )
                    next_obs, rewards, terminateds, truncateds, infos = self.vec_env.step(env_actions_np)

                    ...

                    # 存入 Buffer 的是未裁剪原始動作 out.action 與對應的 out.log_prob
                    self.buffer.store(
                        obs,
                        out.action.cpu(),
                        out.log_prob.cpu(),
                        stored_rewards,
                        next_obs,
                        dones,
                        out.value.cpu(),
                    )
```

---

## 5. Validation & Verification Plan

### 5.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST adhere to repository environment standards using the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python`
> Relative path: `../bin/python`

```bash
# Execute newly created unit tests for unclipped action storage and linear mu output:
/home/b0457812963/Mamba3RL/bin/python -m unittest tests/test_ppo2_unclipped_action.py
```

### 5.2 Acceptance Verification Checklist
- [x] **1. Linear Policy Output**:
  - [x] `fc_actor` has no `Tanh` activation; raw $\mu$ can smoothly take values $> 1.0$ or $< -1.0$ without boundary saturation.
  - [x] Final layer initialization with `std=0.01` ensures initial $\mu$ is centered close to 0.0.
- [x] **2. Dual-Action Integrity**:
  - [x] `AgentOutput.action` retains raw samples that can take values outside $[-1.0, 1.0]$.
  - [x] `AgentOutput.clipped_action` is strictly bounded within $[-1.0, 1.0]$ at all times.
- [x] **3. Log-Probability Mathematical Consistency**:
  - [x] `AgentOutput.log_prob` strictly equals $\sum \log \mathcal{N}(\text{action}; \mu, \sigma)$.
  - [x] Re-evaluating stored unclipped actions via `evaluate_actions(states, time_states, actions)` produces initial ratio $r_t = \exp(\log \pi_{\text{new}} - \log \pi_{\text{old}}) = 1.0 \pm 1e-6$ and approximate $KL \le 1e-6$.
- [x] **4. Prevention of Boundary Gradient Sign Inversion**:
  - [x] Set $\mu = 1.2, \sigma = 0.5$. Sample $a = 1.6$, assign negative advantage $\hat{A} = -1.0$.
  - [x] Verify that policy gradient w.r.t. $\mu$ evaluates to $\frac{1.6 - 1.2}{0.5^2} \times (-1.0) = -1.6 < 0$ (pushing $\mu$ downward).
  - [x] Confirm that with linear $\mu$, the gradient does not vanish at $\mu \approx 1.0$.
- [x] **5. End-to-End Vector Rollout Execution**:
  - [x] Vector environment (`vec_env.step`) receives valid bounded $[-1.0, 1.0]$ floats without runtime errors or warning logs.
  - [x] One full PPO rollout collection and update cycle runs stably without NaN values.
