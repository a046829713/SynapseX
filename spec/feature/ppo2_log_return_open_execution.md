# Specification: PPO2 Log Return & Market-Open Execution Model

- **Category**: `feature`
- **Spec Path**: `spec/feature/ppo2_log_return_open_execution.md`
- **Status**: `Approved`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
In reinforcement learning environments for financial trading, accurate return formulation and realistic execution causality are paramount. The current implementation in [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) computes returns using simple arithmetic returns ($R_t = \frac{Close_t - Close_{t-1}}{Close_{t-1}}$) and attributes returns to the previous step's position (`prev_pos`). This presents two major structural issues:
1. **Compounding & Additivity Asymmetry**: Simple arithmetic returns are not time-additive ($\sum R_t \neq R_{0 \to T}$) and introduce asymmetric reward scale bias under volatile market swings. Log returns ($r_t = \ln(P_t / P_{t-1})$) are strictly time-additive, symmetric for positive and negative fluctuations, and well-behaved for continuous neural policy optimization.
2. **Execution Timing & Causality (Missing Open Price)**: In realistic quantitative trading, after an agent observes historical features up to bar $t-1$, its target position for bar $t$ should execute at market open ($Open_t$). The price movement during bar $t$ consists of two distinct components:
   - **Overnight / Inter-bar Gap**: $Close_{t-1} \to Open_t$ (experienced while holding `prev_pos`).
   - **Intrabar Price Movement**: $Open_t \to Close_t$ (experienced by the new target position `target_pos`).
Currently, the environment omits $Open_t$ entirely, and executes trades at $Close_t$, delaying the effect of action decisions by an entire bar.

### 1.2 Current vs. Expected Behavior
| Dimension | Current Behavior | Expected Behavior (Option A) |
| :--- | :--- | :--- |
| **Return Formula** | Simple arithmetic: $(Close_t - Close_{t-1}) / Close_{t-1}$ | Logarithmic return: $\ln(P_2 / P_1)$ |
| **Execution Point** | Executed at $Close_t$ with 1-bar delay | Executed at $Open_t$ immediately upon bar $t$ open |
| **Return Attribution** | $r_t = prev\_pos \cdot \frac{Close_t - Close_{t-1}}{Close_{t-1}} - cost_t$ | $r_t = prev\_pos \cdot \ln\left(\frac{Open_t}{Close_{t-1}}\right) + target\_pos \cdot \ln\left(\frac{Close_t}{Open_t}\right) - cost_t$ |
| **Position Basis (`open_price`)** | Updated to $Close_t$ | Updated to $Open_t$ upon entry/rebalance |
| **Realized PnL (`closecash`)** | Settled at $Close_t$ with percentage formula | Settled at $Open_t$ with log return formula |
| **Unrealized PnL in Observation** | Percentage return from `open_price` to `close[t-1]` | Log return: $pos \cdot (\ln(Close_{t-1}) - \ln(open\_price))$ |
| **Reset Initialization** | Sets `_prev_close = close[offset]` (coinciding with step 0 close) | Sets `_prev_close = close[offset - 1]` (prior completed bar) |

### 1.3 Technical Root Cause & References
- Affected Location: [`State_time_step.step()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L98-L217)
  - Line 109 & Line 127/149: Trade execution and cost basis rely strictly on `close_price`.
  - Line 196-198: Return is calculated as `prev_pos * (close_price - safe_prev_close) / safe_prev_close - cost_t`. The newly chosen `target_pos` earns zero price return in the current step.
- Affected Location: [`State_time_step.reset()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L59-L73)
  - Line 72: `self._prev_close = float(self._prices.close[self._offset])` duplicates the current step's close, creating an offset mismatch with the observation window (`self._offset - self.bars_count : self._offset`).
- Affected Location: [`State_time_step.encode()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L236-L242)
  - Floating PnL feature still uses simple arithmetic returns, creating an inconsistency with the reward signal.

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix
| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`Brain/PPO2/lib/environment.py:State_time_step`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L12-L253) | **In-Scope** | Implement Option A log-return execution, update `reset()`, `step()`, and `encode()` |
| **Item 2** | [`Brain/PPO2/lib/environment.py:BaseTradingEnv`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L255-L315) | **Out-of-Scope** | Gym interface, action spaces, and stepping wrapper remain intact |
| **Item 3** | [`Brain/PPO2/lib/environment.py:SubprocVecEnv`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L432-L530) | **Out-of-Scope** | Multiprocessing IPC mechanism preserved without modification |
| **Item 4** | [`Brain/Common/DataFeature.py:Prices`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L10-L48) | **Out-of-Scope** | Data loader already supplies `open` and `close` price arrays |
| **Item 5** | [`Brain/PPO2/PPO2_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/PPO2_train.py) | **Out-of-Scope** | Training loop and Mamba Actor-Critic architectures require no schema change |

### 2.2 Architectural Boundaries & Non-Goals
- **In-Scope Boundaries**:
  - Exact formulation of logarithmic returns for both gap and intrabar price intervals.
  - Refinement of `open_price` cost basis updates to reflect $Open_t$ trade execution.
  - Consistent update of `encode()` unrealized return feature using logarithmic scale.
  - Fixing `reset()` to align `_prev_close` with bar index `offset - 1`.
- **Out-of-Scope (Non-Goals)**:
  - Do NOT modify observation space dimension or shape (`getStateShape()`, `getTimeShape()`).
  - Do NOT alter neural network model structure or hyperparameters in YAML configs.
  - Do NOT modify other environment files (e.g., DQN, A2C, EIIE).

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Fix `State_time_step.reset()` initial previous close**
  - [x] Update `self._prev_close = float(self._prices.close[self._offset - 1])` to align with the pre-decision state.
- [x] **Step 2: Refactor `State_time_step.step()` trade execution and return**
  - [x] Extract `open_price = float(self._prices.open[self._offset])` and `close_price = float(self._prices.close[self._offset])`.
  - [x] Update position basis `self.open_price = open_price` upon new position or flip.
  - [x] Update realized PnL settling logic to use `open_price` with log return formulation.
  - [x] Implement Option A log return decomposition:
    $$r_{gap} = \ln\left(\frac{Open_t}{Close_{t-1}}\right)$$
    $$r_{bar} = \ln\left(\frac{Close_t}{Open_t}\right)$$
    $$current\_p\_return = prev\_pos \cdot r_{gap} + target\_pos \cdot r_{bar} - cost_t$$
  - [x] Accumulate log returns into `self.TotalPortfolioPercent += current_p_return` and `self.return_history`.
  - [x] Update `self._prev_close = close_price`.
- [x] **Step 3: Update `State_time_step.encode()` feature consistency**
  - [x] Convert `unrealized_return` calculation to log return:
    $$unrealized\_return = self.position \cdot \left(\ln(Close_{t-1}) - \ln(self.open\_price)\right)$$
- [x] **Step 4: Clean up & Defensiveness**
  - [x] Add defensive bounds `max(price, 1e-8)` before `np.log` to prevent negative or zero logarithm runtime warnings.
  - [x] Ensure all return variables remain `float64` / `float32` compatible.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)

```python
    def reset(self, prices: Prices, offset: int):
        assert offset >= self.bars_count - 1
        self._prices = prices
        self._offset = offset
        self.position = 0.0
        self.open_price = 0.0
        self.trade_bar = 0
        self.TotalPortfolioPercent = 1.0
        self.cost_sum = 0.0
        self.closecash = 0.0
        self.game_steps = 0
        self.return_history.clear()
        self.prev_downside_risk = 0.0
        # Align previous close with bar t-1 (last bar observed in encode)
        self._prev_close = float(self._prices.close[self._offset - 1])

    def step(self, action: float):
        # 1. Action deadzone filter
        if abs(action) <= self.deadzone_threshold:
            target_pos = 0.0
        else:
            target_pos = float(action)

        open_price = float(self._prices.open[self._offset])
        close_price = float(self._prices.close[self._offset])
        prev_pos = self.position
        safe_open_price = max(self.open_price, 1e-8)

        # 2. Friction cost (Commission + Slippage) paid at market open
        delta_p = target_pos - prev_pos
        friction_rate = self.commission_perc + self.default_slippage
        cost_t = abs(delta_p) * friction_rate
        self.cost_sum += cost_t

        # 3. Position state transition, cost basis tracking, and realized PnL at Open_t
        if prev_pos == 0.0 and target_pos == 0.0:
            self.trade_bar = 0
            self.open_price = 0.0
        elif prev_pos == 0.0 and target_pos != 0.0:
            self.open_price = open_price
            self.trade_bar = 1
        elif prev_pos * target_pos < 0.0:
            # Flip: settle previous position at open_price using log return
            closed_size = abs(prev_pos)
            log_pnl = np.log(max(open_price, 1e-8)) - np.log(safe_open_price)
            realized_pnl = closed_size * log_pnl if prev_pos > 0 else closed_size * (-log_pnl)
            self.closecash += realized_pnl

            # Establish new position at open_price
            self.open_price = open_price
            self.trade_bar = 1
        else:
            prev_size = abs(prev_pos)
            curr_size = abs(target_pos)
            if curr_size > prev_size:
                # Add to position: weighted average cost basis at open_price
                added_size = curr_size - prev_size
                self.open_price = (
                    prev_size * self.open_price + added_size * open_price
                ) / curr_size
                self.trade_bar += 1
            elif curr_size < prev_size:
                # Reduce position: cost basis unchanged, settle portion at open_price
                reduced_size = prev_size - curr_size
                log_pnl = np.log(max(open_price, 1e-8)) - np.log(safe_open_price)
                realized_pnl = reduced_size * log_pnl if prev_pos > 0 else reduced_size * (-log_pnl)
                self.closecash += realized_pnl

                if target_pos == 0.0:
                    self.open_price = 0.0
                    self.trade_bar = 0
                else:
                    self.trade_bar += 1
            else:
                self.trade_bar += 1

        self.position = target_pos

        # 4. Mark-to-Market Log Return Decomposition (Option A)
        safe_prev_close = max(self._prev_close, 1e-8)
        safe_curr_open = max(open_price, 1e-8)
        safe_curr_close = max(close_price, 1e-8)

        gap_log_return = float(np.log(safe_curr_open / safe_prev_close))
        bar_log_return = float(np.log(safe_curr_close / safe_curr_open))

        current_p_return = float(prev_pos * gap_log_return + target_pos * bar_log_return - cost_t)

        self.TotalPortfolioPercent += current_p_return
        self.return_history.append(current_p_return)
        self._prev_close = close_price

        # 5. Downside risk penalty and reward
        downside_penalty = self.calculate_step_downside_penalty()
        reward = float(
            self.weights["w1_step_return"] * current_p_return - downside_penalty
        )

        # 6. Step offset and termination checks
        self._offset += 1
        self.game_steps += 1
        terminated = bool(self._offset >= self._prices.close.shape[0] - 1)
        truncated = bool(self.game_steps >= self.N_steps and self.model_train)

        return reward, terminated, truncated
```

---

## 4. Validation & Verification Plan

### 4.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST adhere to repository environment standards using the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python`
> Relative path: `../bin/python`

```bash
# Execute standalone verification test suite:
../bin/python spec/feature/verify_ppo2_log_return.py
```

### 4.2 Acceptance Verification Checklist
- [x] **1. Return Additivity Verification**:
  - Under constant position $pos = 1.0$, verified that $\sum r_{step}$ matches the full cumulative log return $\ln(Close_T / Close_0)$ minus cumulative transaction friction.
- [x] **2. Gap & Intrabar Separation Verification**:
  - Test Case 1 (Buy at Open): `prev_pos = 0.0`, `target_pos = 1.0`. Verify return matches $\ln(Close_t / Open_t) - cost$, completely immune to gap $(Open_t / Close_{t-1})$.
  - Test Case 2 (Sell at Open): `prev_pos = 1.0`, `target_pos = 0.0`. Verify return captures $\ln(Open_t / Close_{t-1}) - cost$, completely immune to $(Close_t / Open_t)$.
  - Test Case 3 (Flip at Open): `prev_pos = 1.0`, `target_pos = -1.0`. Verify return captures long gap and short intrabar move.
- [x] **3. Feature Matrix Consistency**:
  - Verify that `unrealized_return` in `encode()` returns exact log returns and causes no NaN/Inf or shape mismatches.
- [x] **4. End-to-End Environment Reset & Step**:
  - Verify `TrainingEnv` and `ProductionEnv` initialize and step across multiple episodes without warnings or exceptions.
