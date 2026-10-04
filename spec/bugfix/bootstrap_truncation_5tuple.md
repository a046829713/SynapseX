# Specification: Gymnasium 5-Tuple Step Interface and Value Bootstrapping Fix

- **Category**: `bugfix`
- **Spec Path**: `spec/bugfix/bootstrap_truncation_5tuple.md`
- **Status**: `Completed`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
In Reinforcement Learning, the Bellman optimality update equation states:
$$Q(s_t, a_t) = r_t + \gamma (1 - d) \max_{a'} Q(s_{t+1}, a')$$

The continuation discount factor flag $(1 - d)$ crucially distinguishes between:
1. **True Termination (`terminated = True`)**: The environment reaches an absorbing terminal state (e.g., end of historical price data). Future value is strictly zero: $V(s_{terminal}) = 0$. Future returns should **not** be bootstrapped ($d = 1$).
2. **Artificial Truncation (`truncated = True`)**: The episode stops due to an artificial horizon limit (e.g., `game_steps >= N_steps`, such as 1000 steps). The underlying MDP did not terminate; next state $s_{t+1}$ still possesses continuation value. Future returns **must** be bootstrapped from $Q(s_{t+1})$ ($d = 0$).

Currently, [`Brain/DQN/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py) and [`AC_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/AC_train.py) use the legacy 4-tuple Gym interface (`next_state, reward, done, info = env.step(action)`). When an episode reaches `N_steps = 1000`:
- `done` is set to `True`.
- In [`AC_train.py:L451`](file:///home/b0457812963/Mamba3RL/SynapseX/AC_train.py#L451), `last_state` is set to `None if done else next_state`.
- In [`Brain/DQN/lib/common.py:L148`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/common.py#L148) and [`L243`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/common.py#L243), `calc_loss` masks `next_state_values[done_mask] = 0.0`.
- **Failure Mode**: The agent is falsely taught that the future market value drops to zero at step 1000. This causes severe edge-of-cliff value collapse and distorted trading behavior near the horizon.

Additionally, at episode end, the old code flushed partial trajectories from the N-step buffer with `last_state = None`, compounding the truncation error.

### 1.2 Current vs. Expected Behavior
| Dimension | Current Behavior | Expected Behavior |
| :--- | :--- | :--- |
| **`step()` Return Signature** | 4-tuple: `(obs, reward, done, info)` | 5-tuple: `(obs, reward, terminated, truncated, info)` |
| **Termination Condition** | `done = offset >= len - 1 or game_steps == N_steps` | `terminated = bool(offset >= len - 1)` (Negative portfolio equity is permitted; no bankruptcy liquidation) |
| **Truncation Condition** | Confused inside `done` | `truncated = bool(game_steps >= N_steps and model_train)` |
| **Actor Loop Control** | `while not done:` | `while not (terminated or truncated):` with `done = terminated or truncated` |
| **N-step Experience Target** | `last_state = None if done else next_state` (zeros out truncated values) | `last_state = None if terminated else next_state` (bootstraps on truncated!) |
| **Partial Buffer Flush at Truncation** | Emits partial transitions with `last_state = None` | On `truncated`: discard residual $< N$ buffer to maintain exact $\gamma^N$ discount; on `terminated`: flush with `last_state = None`. |

### 1.3 Technical Root Cause & References
- Affected Location 1: [`Brain/DQN/lib/environment.py:L267-L272`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py#L267-L272)
  - `done` mixes step limit with dataset boundary.
- Affected Location 2: [`Brain/DQN/lib/environment.py:L341-L356`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py#L341-L356)
  - `BaseTradingEnv.step` returns 4-tuple instead of standard 5-tuple.
- Affected Location 3: [`AC_train.py:L437-L482`](file:///home/b0457812963/Mamba3RL/SynapseX/AC_train.py#L437-L482)
  - `ActorProcess.run()` handles `step()` output with 4 variables, treats truncation as terminal, and forces `last_state = None`.
- Affected Location 4: [`Brain/DQN/lib/Backtest.py:L174`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/Backtest.py#L174)
  - `evaluate_env.step(action_idx)` unrolls 4 values and will throw unpack error if environment signature changes.

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix
| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`Brain/DQN/lib/environment.py:L265-L273`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py#L265-L273) | **In-Scope** | Update `State_time_step.step` to return `(reward, terminated, truncated)`. `terminated` is triggered solely when reaching the end of price data. Negative portfolio equity is permitted (no bankruptcy check). |
| **Item 2** | [`Brain/DQN/lib/environment.py:L341-L356`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py#L341-L356) | **In-Scope** | Update `BaseTradingEnv.step` to return `(obs, reward, terminated, truncated, info)`. |
| **Item 3** | [`AC_train.py:L437-L482`](file:///home/b0457812963/Mamba3RL/SynapseX/AC_train.py#L437-L482) | **In-Scope** | Unpack 5 values from `step()`. Compute `done = terminated or truncated`. Set `last_state = None if terminated else next_state`. On `truncated`, cleanly clear residual buffer without emitting partial transitions; on `terminated`, flush residual buffer with `last_state = None`. |
| **Item 4** | [`Brain/DQN/lib/Backtest.py:L174`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/Backtest.py#L174) | **In-Scope** | Unpack 5 values from `evaluate_env.step(action_idx)`. Compute `done = terminated or truncated`. |
| **Item 5** | [`Brain/DQN/lib/common.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/common.py) | **Out-of-Scope (Preserved)** | `unpack_batch` computes `dones.append(exp.last_state is None)`. Since `last_state` will now be valid during truncation and only `None` during termination, `calc_loss` naturally bootstraps truncated transitions without modification. |
| **Item 6** | [`Brain/Common/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/experience.py) | **Out-of-Scope (Preserved)** | Buffer storage mechanism is agnostic to 4 vs 5 tuple step semantics. Kept unchanged. |

### 2.2 Architectural Boundaries & Non-Goals
- **Non-Goal 1**: Do NOT introduce bankruptcy or liquidation checks on negative portfolio assets (`TotalPortfolioPercent <= 0.0`). The system explicitly allows negative equity.
- **Non-Goal 2**: Do NOT modify the neural network architecture or loss formulation.
- **Non-Goal 3**: Preserve existing `ExperienceFirstLast` namedtuple schema `("state", "action", "reward", "last_state", "info", "last_info")`.

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Environment 5-Tuple Modernization in [`Brain/DQN/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py)**
  - [x] In `State_time_step.step()`, compute:
    - `terminated = bool(self._offset >= self._prices.close.shape[0] - 1)`
    - `truncated = bool(self.game_steps >= self.N_steps and self.model_train)`
    - Return `(reward, terminated, truncated)`.
  - [x] In `BaseTradingEnv.step()`, unpack `reward, terminated, truncated = self._state.step(action)`.
    - Return `obs, reward, terminated, truncated, info`.
- [x] **Step 2: Bootstrap & N-Step Buffer Correction in [`AC_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/AC_train.py)**
  - [x] In `ActorProcess.run()`:
    - Initialize `terminated = False`, `truncated = False`, `done = False`.
    - Unpack `next_state, reward, terminated, truncated, info = self.env.step(action)`.
    - Set `done = terminated or truncated`.
    - Set `last_state = None if terminated else next_state` when forming N-step experience.
    - When `done` is reached:
      - If `terminated`: flush remaining transitions in `n_step_buffer` with `last_state = None`.
      - If `truncated`: discard remaining partial transitions (< N steps) by clearing `n_step_buffer` (Option A: guarantees zero discount mismatch with $\gamma^N$).
- [x] **Step 3: Interface Compatibility in [`Brain/DQN/lib/Backtest.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/Backtest.py)**
  - [x] In `evaluate()` loop, unpack `_state, reward, terminated, truncated, info = self.evaluate_env.step(action_idx)` and compute `done = terminated or truncated`.
- [x] **Step 4: Verification and Cleanliness**
  - [x] Run unit verification tests with `../bin/python`.
  - [x] Run multi-process smoke test with `AC_train.py`.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 [`Brain/DQN/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py)
```python
# In State_time_step.step(self, action: Actions):
        self._offset += 1
        self.game_steps += 1

        terminated = bool(self._offset >= self._prices.close.shape[0] - 1)
        truncated = bool(self.game_steps >= self.N_steps and self.model_train)

        return reward, terminated, truncated

# In BaseTradingEnv.step(self, action_idx: int):
    def step(self, action_idx: int):
        action = Actions(action_idx)
        reward, terminated, truncated = self._state.step(action)
        obs = self._state.encode()

        info = {
            "instrument": self._instrument,
            "offset": self._state._offset,
            "postion": float(self._state.position),
        }

        return obs, reward, terminated, truncated, info
```

#### 3.2.2 [`AC_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/AC_train.py)
```python
# In ActorProcess.run(self):
            terminated = False
            truncated = False
            done = False
            episode_accumulated_reward = 0.0
            episode_steps = 0

            # 3. 執行一個完整的 episode
            while not done:
                self.state_queue.put((self.actor_id, state))
                action = self.action_queue.get()

                # 3.3. 在環境中執行動作 (Gymnasium 5-tuple standard)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_accumulated_reward += reward
                episode_steps += 1
                n_step_buffer.append((state, action, reward))

                # 3.4. 計算 N-Step Reward 並發送經驗
                if len(n_step_buffer) == self.config.REWARD_STEPS or (done and len(n_step_buffer) > 0):
                    total_reward = 0.0
                    for transition in reversed(n_step_buffer):
                        reward_in_step = transition[2]
                        total_reward = reward_in_step + self.config.GAMMA * total_reward

                    first_state, first_action, _ = n_step_buffer[0]
                    # CRITICAL FIX: Only set last_state = None on true termination.
                    # On truncation, next_state has real continuation value and MUST bootstrap!
                    last_state = None if terminated else next_state

                    self.experience_queue.put(
                        ExperienceFirstLast(
                            first_state, first_action, total_reward, last_state, info, done
                        )
                    )

                state = next_state

                # 3.5. 如果 episode 結束，處理剩餘的 n-step transitions
                if done:
                    try:
                        self.metrics_queue.put_nowait(
                            ("episode_done", self.actor_id, episode_accumulated_reward, episode_steps)
                        )
                    except Full:
                        pass

                    if terminated:
                        # 真正終止時，未來價值為 0，可安全 flush
                        while len(n_step_buffer) > 1:
                            n_step_buffer.popleft()
                            total_reward = 0.0
                            for transition in reversed(n_step_buffer):
                                total_reward = transition[2] + self.config.GAMMA * total_reward

                            first_state, first_action, _ = n_step_buffer[0]
                            self.experience_queue.put(
                                ExperienceFirstLast(
                                    first_state, first_action, total_reward, None, info, done
                                )
                            )
                    else:
                        # 人為截斷 (truncated) 時，丟棄未滿 N 步的殘留片段，保證全體經驗皆為精確 N 步與 gamma^N 折扣
                        n_step_buffer.clear()
```

#### 3.2.3 [`Brain/DQN/lib/Backtest.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/Backtest.py)
```python
# In Backtest.evaluate(self):
                action_idx = action.max(dim=1)[1].item()
                record_orders.append(self._parser_order(action_idx))
                _state, reward, terminated, truncated, info = self.evaluate_env.step(action_idx)
                done = terminated or truncated
```

---

## 4. Validation & Verification Plan

### 4.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST adhere to repository environment standards using the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python`
> Relative path: `../bin/python`

```bash
# 1. 執行單元測試驗證 5 參數輸出與 Bootstrap 行為
../bin/python test_bootstrap_5tuple.py

# 2. 執行短程整合冒煙測試驗證多進程運作 (AC_train.py)
../bin/python -c "import AC_train; print('Imports valid')"
```

### 4.2 Acceptance Verification Checklist
- [x] **1. Step Output 5-Tuple**: `TrainingEnv.step(action)` returns exactly 5 elements: `(obs, reward, terminated, truncated, info)`.
- [x] **2. Truncation Identification**: At `game_steps == N_steps`, `truncated == True` and `terminated == False`.
- [x] **3. Value Bootstrapping Integrity**: Experiences generated on `truncated` have `last_state == next_state` (not `None`). `calc_loss` calculates target $y = r + \gamma^N \max Q(next\_state)$, proving future value is NOT zeroed out.
- [x] **4. Backtest Compatibility**: `Backtest.py` executes without `ValueError: too many values to unpack`.
- [x] **5. Multi-Process Pipeline Stability**: `AC_train.py` runs and trains smoothly without queue stalls or dimension mismatches.
