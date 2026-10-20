# Specification: PPO2 Environment Dead Code Cleanup & Rebalance Clamping Scheme

- **Category**: `feature`
- **Spec Path**: `spec/feature/ppo2_rebalance_clamp_and_cleanup.md`
- **Status**: `Implemented & Verified`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
In [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py), two distinct architectural issues have been identified:

1. **Dead Code & Computational Overhead**:
   State variables `self.closecash`, `self.cost_sum`, and `self.TotalPortfolioPercent` are legacy remnants from discrete DQN equity-accounting algorithms ($\text{equity} = 1.0 - \text{cost\_sum} + \text{closecash} + \text{opencash\_diff}$). Under the recently adopted **Option A (Mark-to-Market Logarithmic Return Decomposition)**, step returns are directly computed per bar as:
   $$\text{current\_p\_return} = \text{prev\_pos} \cdot r_{\text{gap}} + \text{target\_pos} \cdot r_{\text{bar}} - \text{cost}_t$$
   These cumulative variables are never read, exported, or used for reward, observation, or termination. Crucially, in `Case C` (flip) and `Case D` (reduce), redundant logarithmic calculations (`np.log(open_price) - np.log(safe_open_price)`) and floating-point multiplications are executed solely to update `self.closecash`, creating unnecessary CPU overhead on every step.

2. **Micro-Rebalancing Fee Friction (Turnover Drag)**:
   In continuous reinforcement learning with actions $a_t \in [-1.0, 1.0]$, neural network outputs naturally experience continuous micro-fluctuations (e.g., $0.501 \to 0.498 \to 0.505$). Currently, only an absolute zero-deadzone filter is applied (`abs(action) <= deadzone_threshold -> 0.0`). Because there is no threshold on position differences, every minute adjustment ($|\Delta p| \approx 0.005 \sim 0.02$) incurs transaction costs:
   $$\text{cost}_t = |\Delta p| \times (\text{commission} + \text{slippage})$$
   Over an episode of hundreds or thousands of steps, this persistent churn erodes returns through unnecessary transaction friction.

### 1.2 Current vs. Expected Behavior
| Dimension | Current Behavior | Expected Behavior |
| :--- | :--- | :--- |
| **Dead Variables** | `self.closecash`, `self.cost_sum`, `self.TotalPortfolioPercent` maintained at runtime; redundant `np.log` in flips/reductions. | Completely eliminated. No unused state tracking or wasted math operations. |
| **Micro-Adjustment Handling** | Any continuous action change triggers a rebalance and charges friction fee $\text{cost}_t$. | Position changes $|\Delta p| < \text{rebalance\_threshold}$ (default `0.03`) are clamped to `prev_pos` ($\text{cost}_t = 0.0$). |
| **Full Exit Handling** | Zero-deadzone sets $a \to 0.0$, but without distinction from rebalancing threshold. | Clean exit: if candidate position is $0.0$, always liquidate to $0.0$ without being blocked by rebalance threshold. |
| **Directional Flip Handling** | Flips trigger rebalance and redundant `realized_pnl` math. | Flips ($\text{candidate\_pos} \times \text{prev\_pos} < 0.0$) execute immediately without redundant math. |
| **Step Info Dictionary** | Contains duplicate typo `"postion"` alongside `"position"`. | Cleared of typo; clean and standard dictionary output. |

### 1.3 Technical Root Cause & References
- Legacy PnL accounting & dead variables: [`Brain/PPO2/lib/environment.py:L47-L49`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L47-L49), [`L66-L68`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L66-L68), [`L132-L140`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L132-L140), [`L159-L167`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L159-L167)
- Missing rebalance clamping filter: [`Brain/PPO2/lib/environment.py:L102-L118`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L102-L118)
- Dead property `have_position`: [`Brain/PPO2/lib/environment.py:L56-L57`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L56-L57)
- Typo `"postion"` in info dictionary: [`Brain/PPO2/lib/environment.py:L299`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L299)

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix
| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) | **In-Scope** | 1. Remove `closecash`, `cost_sum`, `TotalPortfolioPercent`, `have_position`, `safe_open_price`.<br/>2. Add `rebalance_threshold: float = 0.03`.<br/>3. Implement rebalance deadband clamp in `step()`.<br/>4. Fix `"postion"` typo in `info`. |
| **Item 2** | [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) | **In-Scope** | Add `REBALANCE_THRESHOLD: 0.03` under environment parameters. |
| **Item 3** | [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) | **In-Scope** | Bind `self.rebalance_threshold` from config and propagate to `State_time_step` and `PPO2Evaluator`. |
| **Item 4** | [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py) | **In-Scope** | Accept `rebalance_threshold` in `PPO2Evaluator.__init__` and pass when constructing `State_time_step`. |
| **Item 5** | [`tests/test_ppo2_rebalance_clamp.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_ppo2_rebalance_clamp.py) | **In-Scope** | Unit tests verifying clamp suppression, zero exit, sign flip, and dead variable removal. |
| **Item 6** | [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) | **Out-of-Scope** | Neural network backbone is strictly preserved. |
| **Item 7** | [`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py) | **Out-of-Scope** | `RolloutBuffer` memory layout is strictly preserved. |
| **Item 8** | [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py) | **Out-of-Scope** | Agent interface and action distribution remain intact. |

### 2.2 Architectural Boundaries & Non-Goals
- **Preservation of Observation Features**:
  `self.open_price` and `self.trade_bar` are required by `encode()` (features 2 and 3) and must continue to be updated accurately.
- **Backward Compatibility**:
  Setting `rebalance_threshold = 0.0` disables clamping completely, recovering previous behavior.

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Configuration Binding in `configs/training/ppo_v2.yaml`**
  - [x] Add `REBALANCE_THRESHOLD: 0.03`.
- [x] **Step 2: Clean Up Dead Code in `Brain/PPO2/lib/environment.py`**
  - [x] Remove `self.closecash`, `self.cost_sum`, `self.TotalPortfolioPercent` from `__init__` and `reset`.
  - [x] Remove `have_position` property.
  - [x] Remove `safe_open_price` from `step()`.
  - [x] Remove dead math (`np.log` and `realized_pnl` calculation) from `Case C` (flip) and `Case D` (reduce).
  - [x] Remove `self.cost_sum += cost_t` and `self.TotalPortfolioPercent += current_p_return`.
  - [x] Fix typo `"postion"` in `BaseTradingEnv.step()`.
- [x] **Step 3: Implement Rebalance Clamping Scheme in `State_time_step`**
  - [x] Add `rebalance_threshold: float = 0.03` parameter to `State_time_step.__init__`.
  - [x] Implement 3-way arbitration logic in `State_time_step.step()`:
    1. Zero deadzone filter produces `candidate_pos`.
    2. If `candidate_pos == 0.0` (exit), `prev_pos == 0.0` (entry), or `candidate_pos * prev_pos < 0.0` (flip), set `target_pos = candidate_pos`.
    3. Else if `abs(candidate_pos - prev_pos) < self.rebalance_threshold`, set `target_pos = prev_pos` ($\Delta p = 0$, $\text{cost}_t = 0$).
    4. Else, set `target_pos = candidate_pos`.
  - [x] Pass `rebalance_threshold` from training config in `TrainingEnv.__init__`.
- [x] **Step 4: Update Engine & Evaluator Integration**
  - [x] In `Brain/PPO2/lib/engine.py`, load `self.rebalance_threshold` from `cfg` and pass to `State_time_step` and `PPO2Evaluator`.
  - [x] In `Brain/PPO2/lib/Evaluator.py`, accept `rebalance_threshold` and pass to `State_time_step`.
- [x] **Step 5: Verification Suite Creation & Empirical Execution**
  - [x] Create `tests/test_ppo2_rebalance_clamp.py` covering all state transitions, fee suppression, edge cases, and hygiene.
  - [x] Execute tests via `/home/b0457812963/Mamba3RL/bin/python`.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 `Brain/PPO2/lib/environment.py` (Rebalance Clamping & Cleanup)
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
        downside_window_size: int = 60,
        w1_step_return: float = 1.0,
        w2_downside_ratio: float = 1.0,
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
        self.downside_window_size = downside_window_size
        self.prev_downside_risk = 0.0

        self.weights = {
            "w1_step_return": w1_step_return,
            "w2_downside_ratio": w2_downside_ratio,
        }

        # Active state variables
        self.position = 0.0
        self.open_price = 0.0
        self.trade_bar = 0
        self.game_steps = 0
        self.return_history = deque(maxlen=max(N_steps, self.downside_window_size))
        self._prev_close = 0.0

    def reset(self, prices: Prices, offset: int):
        assert offset >= self.bars_count - 1
        self._prices = prices
        self._offset = offset
        self.position = 0.0
        self.open_price = 0.0
        self.trade_bar = 0
        self.game_steps = 0
        self.return_history.clear()
        self.prev_downside_risk = 0.0
        self._prev_close = float(self._prices.close[self._offset - 1])

    def step(self, action: float):
        # 1. Action Deadzone Filter
        if abs(action) <= self.deadzone_threshold:
            candidate_pos = 0.0
        else:
            candidate_pos = float(action)

        open_price = float(self._prices.open[self._offset])
        close_price = float(self._prices.close[self._offset])
        prev_pos = self.position

        # 2. Rebalance Clamping Scheme (Micro-Adjustment Friction Suppression)
        if candidate_pos == 0.0 or (candidate_pos * prev_pos < 0.0):
            target_pos = candidate_pos
        elif abs(candidate_pos - prev_pos) < self.rebalance_threshold:
            target_pos = prev_pos
        else:
            target_pos = candidate_pos

        # 3. Position Delta and Friction Cost
        delta_p = target_pos - prev_pos
        friction_rate = self.commission_perc + self.default_slippage
        cost_t = abs(delta_p) * friction_rate

        # 4. State Transitions & Open Price Maintenance
        if prev_pos == 0.0 and target_pos == 0.0:
            self.trade_bar = 0
            self.open_price = 0.0
        elif prev_pos == 0.0 and target_pos != 0.0:
            self.open_price = open_price
            self.trade_bar = 1
        elif prev_pos * target_pos < 0.0:
            # Direction flip: establish new position at open_price without dead closecash math
            self.open_price = open_price
            self.trade_bar = 1
        else:
            prev_size = abs(prev_pos)
            curr_size = abs(target_pos)
            if curr_size > prev_size:
                added_size = curr_size - prev_size
                self.open_price = (
                    prev_size * self.open_price + added_size * open_price
                ) / curr_size
                self.trade_bar += 1
            elif curr_size < prev_size:
                if target_pos == 0.0:
                    self.open_price = 0.0
                    self.trade_bar = 0
                else:
                    self.trade_bar += 1
            else:
                self.trade_bar += 1

        self.position = target_pos

        # 5. Log Return Decomposition (Option A)
        safe_prev_close = max(self._prev_close, 1e-8)
        safe_curr_open = max(open_price, 1e-8)
        safe_curr_close = max(close_price, 1e-8)

        gap_log_return = float(np.log(safe_curr_open / safe_prev_close))
        bar_log_return = float(np.log(safe_curr_close / safe_curr_open))

        current_p_return = float(
            prev_pos * gap_log_return + target_pos * bar_log_return - cost_t
        )

        self.return_history.append(current_p_return)
        self._prev_close = close_price

        # 6. Downside Risk Penalty & Step Progression
        downside_penalty = self.calculate_step_downside_penalty()
        reward = float(
            self.weights["w1_step_return"] * current_p_return - downside_penalty
        )

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
# 1. Run dedicated rebalance clamp and cleanup unit test suite:
../bin/python -m unittest tests/test_ppo2_rebalance_clamp.py

# 2. Run engine evaluation check to verify end-to-end integration:
../bin/python PPO2_rl_test.py --config configs/training/ppo_v2.yaml
```

### 4.2 Acceptance Verification Checklist
- [x] **1. Deadband Suppression Test**:
  - Position changes with $|\Delta p| < 0.03$ (e.g. $0.50 \to 0.52$) maintain $p = 0.50$, $\Delta p = 0.0$, $\text{cost}_t = 0.0$, `open_price` unchanged, `trade_bar += 1`.
- [x] **2. Significant Adjustment Test**:
  - Position changes with $|\Delta p| \ge 0.03$ (e.g. $0.50 \to 0.55$) trigger rebalance with $\Delta p = 0.05$, correct weighted cost basis, and fee charged.
- [x] **3. Exit Immunity Test**:
  - When candidate action is $0.0$ (or within zero deadzone $\le 0.01$), position fully closes to $0.0$, resetting `open_price = 0.0` and `trade_bar = 0`, even if $|0.0 - \text{prev\_pos}| < 0.03$.
- [x] **4. Direction Flip Immunity Test**:
  - Reversing direction (e.g. $+0.02 \to -0.02$) flips cleanly to $-0.02$, resetting `trade_bar = 1` and `open_price = open_price`.
- [x] **5. Hygiene Audit**:
  - Assert that `hasattr(state, "closecash") is False`, `hasattr(state, "cost_sum") is False`, `hasattr(state, "TotalPortfolioPercent") is False`, `hasattr(state, "have_position") is False`.
- [x] **6. Config & Engine Parity**:
  - Verify that `TrainingEnv`, `PPO2Engine`, and `PPO2Evaluator` correctly ingest `rebalance_threshold` from YAML and behave identically.
