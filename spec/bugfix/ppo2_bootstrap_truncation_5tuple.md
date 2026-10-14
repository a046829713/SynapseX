# Specification: PPO2 Gymnasium 5-Tuple Interface and Value Bootstrapping Fix (Option A)

- **Category**: `bugfix`
- **Spec Path**: `spec/bugfix/ppo2_bootstrap_truncation_5tuple.md`
- **Status**: `Completed`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Option A Mathematical Foundation & Mechanism](#2-option-a-mathematical-foundation--mechanism)
3. [3. Target Scope & Architecture Boundaries](#3-target-scope--architecture-boundaries)
4. [4. Implementation Checklist & Detailed Blueprint](#4-implementation-checklist--detailed-blueprint)
5. [5. Validation & Verification Plan](#5-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
Reinforcement learning relies heavily on the correct calculation of Temporal Difference (TD) errors and Generalized Advantage Estimation (GAE).
In the legacy Gym interface, `step()` returns 4 items:
```python
obs, reward, done, info = env.step(action)
```
The boolean flag `done` conflates two fundamentally different termination conditions:
1. **True Natural Termination (`terminated = True`)**: The environment reaches an absorbing state where no future states or transitions exist (e.g., reaching the final candle of historical price data in [`Brain/PPO2/lib/environment.py:L213`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L213)). In this case, the true continuation value is zero: $V(s_{\text{terminal}}) = 0$. Future returns must **not** be bootstrapped.
2. **Artificial Time-Limit Truncation (`truncated = True`)**: The episode is prematurely cut off solely due to a training horizon limit (e.g., `game_steps >= N_steps`, 1000 steps in [`Brain/PPO2/lib/environment.py:L214`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L214)). The market and financial process continue unabated. The next state $s_{t+1}$ possesses a non-zero continuation value $V(s_{t+1}) \neq 0$. Future returns **must** be bootstrapped from $V(s_{t+1})$.

### 1.2 Current Failures in PPO2
In the current implementation of [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py), [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py), and [`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py):

#### Failure Mode 1: Edge-of-Cliff Value Collapse (Truncation Zero-Masking)
When `game_steps >= N_steps`, `done` is set to `True`.
In [`Brain/PPO2/lib/experience.py:L115-L122`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py#L115-L122):
```python
next_non_terminal = 1.0 - dones[step]
delta = rewards[step] + self.gamma * next_val * next_non_terminal - values[step]
last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae
```
When `dones[step] == True`, `next_non_terminal` is $0.0$.
This zeros out `next_val`, forcing:
$$\delta_t = r_t - V(s_t)$$
The critic network is falsely taught that market value drops to zero at step 1000. As a result, the agent exhibits severe myopic distortions and abnormal dumping/gambling behavior near the 1000-step boundary.

#### Failure Mode 2: Multi-Worker Vectorized Cross-Instrument Value Contamination
When an episode ends in [`PPO2_rl_train.py:L264-L283`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L264-L283), worker $i$ is immediately reset to a **different financial symbol** from the round-robin queue:
```python
next_sym = self.symbol_queue.get_next()
new_obs_i = self.vec_env.reset_at(i, symbol=next_sym)
obs["states"][i] = new_obs_i["states"]
```
If one were to naively set `next_non_terminal = 1.0 - terminated` without special handling for truncation:
- At step $t+1$, worker $i$ observes the initial state of the **new symbol** (e.g., SOL at step 0).
- Step $t$ (e.g., BTC at step 1000) would use $V(s_{t+1, i}^{\text{SOL}})$ to bootstrap BTC's terminal return!
- Furthermore, the future advantage $\hat{A}_{t+1, i}$ of SOL would propagate backwards across the reset boundary into BTC's trajectory via $\gamma \lambda \hat{A}_{t+1}$.
This constitutes severe data leakage and cross-asset distortion.

---

## 2. Option A Mathematical Foundation & Mechanism

### 2.1 Mathematical Formulation of Option A
Option A implements the canonical CleanRL / Stable-Baselines3 terminal value adjustment technique:
At step $t$, worker $i$:
1. The vectorized environment executes action $a_{t, i}$ and yields:
   $$s_{t+1, i}, r_{t, i}, \text{terminated}_{t, i}, \text{truncated}_{t, i}, \text{info}_{t, i}$$
   Crucially, $s_{t+1, i}$ (`next_obs[i]`) is the **true terminal observation of the old instrument before `reset_at` is invoked**.
2. If `truncated[i] == True` and `terminated[i] == False`:
   We compute the true continuation value of the truncated state using the Critic:
   $$V_{\text{trunc}, i} = V(s_{t+1, i})$$
3. We adjust the stored reward for step $t$, worker $i$:
   $$r_{t, i}^{\text{adjusted}} = r_{t, i} + \gamma V_{\text{trunc}, i}$$
4. We record the transition into [`RolloutBuffer`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py) with:
   $$\text{done}_{t, i} = \text{terminated}_{t, i} \lor \text{truncated}_{t, i} = \text{True}$$
   (which corresponds to `next_non_terminal = 0.0`).

### 2.2 Proof of Exact Equivalence & Zero Leakage
Let us evaluate the GAE recurrence equation in [`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py) under Option A:

1. **TD Error $\delta_{t, i}$**:
   $$\begin{aligned}
   \delta_{t, i} &= r_{t, i}^{\text{adjusted}} + \gamma \cdot V_{t+1, i} \cdot (1 - \text{done}_{t, i}) - V_{t, i} \\
   &= (r_{t, i} + \gamma V_{\text{trunc}, i}) + \gamma \cdot V_{t+1, i} \cdot 0 - V_{t, i} \\
   &= r_{t, i} + \gamma V_{\text{trunc}, i} - V_{t, i}
   \end{aligned}$$
   $\implies$ The TD error is mathematically identical to the exact 1-step bootstrap equation! The next state's value $V_{t+1, i}$ (which belongs to the new instrument) is multiplied by 0 and has **zero impact**.

2. **Advantage Propagation $\hat{A}_{t, i}$**:
   $$\begin{aligned}
   \hat{A}_{t, i} &= \delta_{t, i} + (\gamma \lambda) \cdot (1 - \text{done}_{t, i}) \cdot \hat{A}_{t+1, i} \\
   &= \delta_{t, i} + (\gamma \lambda) \cdot 0 \cdot \hat{A}_{t+1, i} \\
   &= \delta_{t, i}
   \end{aligned}$$
   $\implies$ Because the episode ended at step $t$, the future advantage $\hat{A}_{t+1, i}$ of the new instrument is strictly blocked from propagating back to step $t$.

3. **Target Return $R_{t, i}$**:
   $$R_{t, i} = \hat{A}_{t, i} + V_{t, i} = (r_{t, i} + \gamma V_{\text{trunc}, i} - V_{t, i}) + V_{t, i} = r_{t, i} + \gamma V_{\text{trunc}, i}$$
   $\implies$ The critic's regression target $R_{t, i}$ exactly equals the discounted sum of returns including the bootstrapped continuation value!

4. **Preservation of Raw Financial Metrics**:
   The cumulative episodic return tracking in [`PPO2_rl_train.py:L259`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L259):
   ```python
   episode_rewards += raw_rewards  # Raw reward without bootstrap adjustment!
   ```
   Ensures that training logs and console statistics reflect true trading PnL without artificial inflation.

---

## 3. Target Scope & Architecture Boundaries

### 3.1 File & Module Scope Matrix
| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`Brain/PPO2/lib/environment.py:L210-L218`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L210-L218) | **In-Scope** | Update `State_time_step.step()` to return `(reward, terminated, truncated)`. |
| **Item 2** | [`Brain/PPO2/lib/environment.py:L289-L307`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L289-L307) | **In-Scope** | Update `BaseTradingEnv.step()` to unpack 3 values from `_state.step` and return 5-tuple: `(obs, reward, terminated, truncated, info)`. |
| **Item 3** | [`Brain/PPO2/lib/environment.py:L402-L432`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L402-L432) | **In-Scope** | Update `_subproc_worker` to receive and send 5-tuple on `cmd == "step"`. |
| **Item 4** | [`Brain/PPO2/lib/environment.py:L499-L515`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L499-L515) | **In-Scope** | Update `SubprocVecEnv.step()` to return `(obs, rewards, terminateds, truncateds, infos)`. |
| **Item 5** | [`PPO2_rl_train.py:L245-L283`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L245-L283) | **In-Scope** | Unpack 5 values from `vec_env.step()`. For truncated workers, compute $V(s_{t+1})$ and add $\gamma V$ to stored rewards. Set `dones = terminateds | truncateds`. Log raw unadjusted returns. |
| **Item 6** | [`test_ppo2_env.py`](file:///home/b0457812963/Mamba3RL/SynapseX/test_ppo2_env.py) | **In-Scope** | Update unit tests to expect 5-tuple outputs from `env.step()` and `state.step()`. |
| **Item 7** | [`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py) | **Out-of-Scope (Preserved)** | Thanks to Option A mathematical equivalence, `RolloutBuffer` structure remains untouched, maintaining optimal memory locality. |
| **Item 8** | [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py) | **Out-of-Scope (Preserved)** | `PPO2Agent.get_action` already supports batched observation inputs and value extraction. |

### 3.2 Non-Goals & Architectural Invariants
- **Non-Goal 1**: No changes to neural network architectures (Mamba Continuous Actor-Critic) or action space bounds $[-1.0, 1.0]$.
- **Non-Goal 2**: No modifications to the discrete DQN or A2C environments.
- **Non-Goal 3**: No changes to financial mark-to-market formulas or slippage calculations.

---

## 4. Implementation Checklist & Detailed Blueprint

### 4.1 Step-by-Step Task Checklist
- [x] **Step 1: Environment 5-Tuple Modernization in [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)**
  - [x] In `State_time_step.step()`:
    - `terminated = bool(self._offset >= self._prices.close.shape[0] - 1)`
    - `truncated = bool(self.game_steps >= self.N_steps and self.model_train)`
    - Return `(reward, terminated, truncated)`
  - [x] In `BaseTradingEnv.step()`:
    - Unpack `reward, terminated, truncated = self._state.step(action_val)`
    - Return `(obs, reward, terminated, truncated, info)`
  - [x] In `_subproc_worker`:
    - Unpack `obs, reward, terminated, truncated, info = env.step(data)`
    - Send 5-tuple back through pipe
  - [x] In `SubprocVecEnv.step()`:
    - Unpack `obs_list, rewards, terminateds, truncateds, infos = zip(*results)`
    - Return `self._stack_obs(obs_list), np.array(rewards, dtype=np.float32), np.array(terminateds, dtype=bool), np.array(truncateds, dtype=bool), list(infos)`
- [x] **Step 2: Implement Option A Value Bootstrapping in [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)**
  - [x] Unpack 5-tuple: `next_obs, rewards, terminateds, truncateds, infos = self.vec_env.step(actions_np)`
  - [x] Identify truncated workers: `is_trunc = truncateds & (~terminateds)`
  - [x] When `np.any(is_trunc)`:
    - Slice `next_obs` for truncated indices
    - Evaluate continuation values using `self.agent.get_action(trunc_obs, stochastic=False).value`
    - Add $\gamma \times V_{\text{trunc}}$ to `stored_rewards`
  - [x] Compute `dones = terminateds | truncateds`
  - [x] Store transitions into `self.buffer.store(obs, out.action.cpu(), out.log_prob.cpu(), stored_rewards, next_obs, dones, out.value.cpu())`
  - [x] Maintain unadjusted raw `rewards` in `episode_rewards += rewards`
  - [x] When `done` occurs, log episode returns, reset worker via `reset_at`, and update `obs`
- [x] **Step 3: Update Test Suite in [`test_ppo2_env.py`](file:///home/b0457812963/Mamba3RL/SynapseX/test_ppo2_env.py)**
  - [x] Update all `state.step()` calls to unpack `reward, terminated, truncated`
  - [x] Update all `env.step()` calls to unpack `obs, reward, terminated, truncated, info`
  - [x] Add explicit unit test asserting `terminated` vs `truncated` separation
- [x] **Step 4: Verification and Smoke Testing**
  - [x] Run `test_ppo2_env.py` using `../bin/python`
  - [x] Run dedicated 5-tuple bootstrap verification test using `../bin/python`
  - [x] Run multi-worker PPO2 training smoke test using `../bin/python PPO2_rl_train.py`

### 4.2 Detailed Code Blueprint

#### 4.2.1 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)
```python
# In State_time_step.step(self, action: float):
        # 6. 推進時間步與判斷終止
        self._offset += 1
        self.game_steps += 1
        terminated = bool(self._offset >= self._prices.close.shape[0] - 1)
        truncated = bool(self.game_steps >= self.N_steps and self.model_train)

        return reward, terminated, truncated

# In BaseTradingEnv.step(self, action):
        action_val = float(np.asarray(action).squeeze())
        action_val = float(np.clip(action_val, -1.0, 1.0))
        reward, terminated, truncated = self._state.step(action_val)
        obs = self._state.encode()

        info = {
            "instrument": self._instrument,
            "offset": self._state._offset,
            "position": float(self._state.position),
            "postion": float(self._state.position),
        }

        return obs, reward, terminated, truncated, info

# In _subproc_worker:
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                remote.send((obs, reward, terminated, truncated, info))

# In SubprocVecEnv.step:
        obs_list, rewards, terminateds, truncateds, infos = zip(*results)
        return (
            self._stack_obs(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=bool),
            np.array(truncateds, dtype=bool),
            list(infos),
        )
```

#### 4.2.2 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)
```python
                # 1. Rollout sampling across all N workers
                for _ in range(horizon_steps):
                    out = self.agent.get_action(obs, stochastic=True)
                    actions_np = out.action.cpu().numpy()
                    next_obs, rewards, terminateds, truncateds, infos = self.vec_env.step(actions_np)

                    # Option A: Terminal Value Bootstrap Adjustment
                    stored_rewards = rewards.copy()
                    is_trunc = truncateds & (~terminateds)
                    if np.any(is_trunc):
                        trunc_indices = np.where(is_trunc)[0]
                        trunc_obs = {
                            "states": next_obs["states"][trunc_indices],
                            "time_states": next_obs["time_states"][trunc_indices],
                        }
                        with torch.no_grad():
                            trunc_out = self.agent.get_action(trunc_obs, stochastic=False)
                            trunc_values = trunc_out.value.cpu().numpy().reshape(-1)

                        gamma = float(self.cfg_training.GAMMA)
                        stored_rewards[trunc_indices] += gamma * trunc_values

                    dones = terminateds | truncateds

                    self.buffer.store(
                        obs,
                        out.action.cpu(),
                        out.log_prob.cpu(),
                        stored_rewards,
                        next_obs,
                        dones,
                        out.value.cpu(),
                    )
                    episode_rewards += rewards  # Record raw trading returns
                    episode_steps += 1
                    total_steps += num_workers
                    obs = next_obs

                    for i, done in enumerate(dones):
                        if done:
                            episode_count += 1
                            recent_episode_returns.append(float(episode_rewards[i]))
                            sym = infos[i].get("instrument", current_symbols[i])
                            pos = infos[i].get("position", 0.0)
                            print(
                                f"[Ep #{episode_count:04d} | W{i}] Symbol: {sym:<20s} | "
                                f"Steps: {episode_steps[i]:04d} | Return: {episode_rewards[i]:+.4f} | "
                                f"End Pos: {pos:+.2f} | TotalSteps: {total_steps:07d}"
                            )
                            # Reset worker i with next symbol from round-robin queue
                            next_sym = self.symbol_queue.get_next()
                            current_symbols[i] = next_sym
                            new_obs_i = self.vec_env.reset_at(i, symbol=next_sym)
                            obs["states"][i] = new_obs_i["states"]
                            obs["time_states"][i] = new_obs_i["time_states"]
                            episode_rewards[i] = 0.0
                            episode_steps[i] = 0
```

---

## 5. Validation & Verification Plan

### 5.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST adhere to repository environment standards using the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python`
> Relative path: `../bin/python`

```bash
# 1. Run unit test suite
../bin/python -m unittest test_ppo2_env.py

# 2. Run dedicated 5-tuple bootstrap verification test
../bin/python test_ppo2_bootstrap.py

# 3. Multi-worker smoke training test (2 iterations)
../bin/python PPO2_rl_train.py training.HORIZON_STEPS=16 training.PPO_EPOCHS=1
```

### 5.2 Acceptance Verification Criteria
1. **5-Tuple Signature**: `BaseTradingEnv.step()` and `SubprocVecEnv.step()` return exactly 5 elements `(obs, rewards, terminateds, truncateds, infos)`.
2. **Proper Separation of Termination**:
   - When `_offset == len - 1`: `terminated == True`, `truncated == False`.
   - When `game_steps == N_steps`: `truncated == True`, `terminated == False`.
3. **Accurate Bootstrapping without Leakage**:
   - Truncated transitions receive $\gamma V(s_{t+1})$ added to the TD target.
   - Transitions from newly reset instruments do not contaminate prior episodes in GAE calculation.
4. **Clean Execution**: Zero runtime crashes, dimension mismatches, or multi-process pipe deadlocks.
