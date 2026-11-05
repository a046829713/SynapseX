# Specification: Remove Obsolete `trade_bar` Feature and Align Portfolio Observation Dimension to 2 in PPO2

- **Category**: `feature`
- **Spec Path**: `spec/feature/ppo2_remove_trade_bar_observation.md`
- **Status**: `Approved / Completed`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
Historically, trading environments in this repository supported discrete action spaces (e.g., in DQN environments where actions correspond to discrete operations such as Flat, Long, Short). Under discrete operations, each trade possesses a discrete ticket-like lifecycle: entering a position, holding it over discrete time intervals, and completely exiting. The `trade_bar` feature was originally designed to count how many consecutive bars the agent had been holding the position (normalized by 48 bars, representing 24 hours of 30-minute intervals), allowing the agent to learn time-based exits or holding-duration penalties.

Following the upgrade of PPO2 to a pure continuous action space ($a_t \in [0.0, 1.0]$ Long-Only rebalancing), portfolio management became a continuous dynamic asset allocation problem:
- The agent continuously rebalances portfolio weights at each bar (e.g., adjusting exposure from 0.4 to 0.7 or 0.3).
- Because the position rarely drops to exactly $0.0$, `self.trade_bar` continues to increment monotonically without resetting.
- Over extended runs or long episodes, `float(self.trade_bar) / 48.0` drifts unbounded (e.g., reaching 5.0, 10.0, 20.0+), injecting a non-stationary and arbitrary feature into the neural network observation space.
- The concept of a discrete "ticket holding duration" is semantically invalid under continuous fractional rebalancing.

Therefore, `trade_bar` must be removed from the environment observation space and state tracking, reducing `PORTFOLIO_DIM` from 3 to 2:
1. `position` $\in [0.0, 1.0]$
2. `unrealized_return` (log equity ratio relative to weighted average entry price)

### 1.2 Current vs. Expected Behavior
| Dimension | Current Behavior | Expected Behavior |
| :--- | :--- | :--- |
| **Portfolio Feature Vector** | Shape is `(bars_count, 3)`: `[position, unrealized_return, trade_bar / 48.0]` | Shape is `(bars_count, 2)`: `[position, unrealized_return]` |
| **Holding Duration Tracking** | `self.trade_bar` maintained and incremented in `State_time_step` | `self.trade_bar` completely eliminated; no duration counter maintained |
| **State Shape Definition** | `State_time_step` inherits `BasicState.getStateShape()` returning `(bars_count, len(info_list) + 3)` | `State_time_step.getStateShape()` explicitly overridden to return `(self.bars_count, len(self.info_list) + 2)` |
| **Observation Encoding** | Writes `data_res[:, len(info_list) + 2] = float(trade_bar) / 48.0` in both `environment.py` and `engine.py` | Eliminates index `+ 2`; writes only `len(info_list)` (position) and `len(info_list) + 1` (unrealized return) |
| **Configuration Alignment** | `configs/training/ppo_v2.yaml` sets `PORTFOLIO_DIM: 3` | `configs/training/ppo_v2.yaml` sets `PORTFOLIO_DIM: 2` |
| **Network & Engine Defaults** | `PPONetwork`, `PPOAgent`, and `PPO2Engine` default `portfolio_dim = 3` | `portfolio_dim` defaults to `2` across all model layers and evaluation engine |

### 1.3 Technical Root Cause & References
- Environment state maintenance: [`Brain/PPO2/lib/environment.py:L52`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L52), [`L69`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L69), [`L108-L118`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L108-L118)
- Environment observation encoding: [`Brain/PPO2/lib/environment.py:L177-L180`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L177-L180)
- Template default state shape: [`Brain/Common/env_components.py:L58-L60`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py#L58-L60) (`len(self.info_list) + 3`)
- Engine observation encoding: [`Brain/PPO2/lib/engine.py:L265`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L265), [`L300`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L300), [`L316-L332`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L316-L332)
- Training configuration: [`configs/training/ppo_v2.yaml:L38`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L38)

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix
| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) | **In-Scope** | 1. Remove `self.trade_bar` from `__init__`, `reset`, and `step`.<br/>2. Add explicit `getStateShape(self)` returning `(self.bars_count, len(self.info_list) + 2)`.<br/>3. Remove `trade_bar` assignment at index `len(self.info_list) + 2` in `encode()`. |
| **Item 2** | [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) | **In-Scope** | Update `PORTFOLIO_DIM: 2`. |
| **Item 3** | [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) | **In-Scope** | Update default parameter `portfolio_dim: int = 2` and class docstrings. |
| **Item 4** | [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py) | **In-Scope** | Update default parameter `portfolio_dim: int = 2`. |
| **Item 5** | [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) | **In-Scope** | 1. Update `encode_observation()` to drop `trade_bar` parameter and feature encoding index.<br/>2. Update `_resolve_symbol_portfolio_state()` return tuple to `(position, open_price)`.<br/>3. Align all call-sites in `run_evaluation` / step loops. |
| **Item 6** | [`Brain/Common/env_components.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py) | **Out-of-Scope** | Shared base classes remain untouched to ensure DQN / HopeDQN environments are not broken. |
| **Item 7** | Checkpoint weights (`checkpoint_iter_*.pt`) | **Out-of-Scope (Breaking)** | Legacy checkpoints trained with `portfolio_dim = 3` are incompatible and will not be migrated; fresh training runs will start with 2 dimensions. |

### 2.2 Architectural Boundaries & Non-Goals
- **In-Scope Boundaries**:
  - The observation space shape for `states` must strictly become `(bars_count, len(info_list) + 2)`.
  - Bit-for-bit parity between `State_time_step.encode()` and `PPO2Engine.encode_observation()` must be strictly maintained.
  - Parameterized `PORTFOLIO_DIM` in `configs/training/ppo_v2.yaml` flows dynamically to `PPO2Agent`, `PPONetwork`, and `PPO2Engine`.
- **Out-of-Scope (Non-Goals)**:
  - Do not modify legacy discrete environments (`Brain/DQN`, `Brain/HopeDQN`).
  - Do not alter `time_states` stationary time feature encoding (6 dimensions remain untouched).
  - Do not alter continuous action bounds (`[0.0, 1.0]`) or reward formulation (`w1_step_return * current_p_return`).

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Environment Refactoring in [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)**
  - [x] Remove `self.trade_bar` initialization in `__init__` and `reset`.
  - [x] Remove `self.trade_bar` mutation logic in `step()`.
  - [x] Implement `getStateShape()` in `State_time_step`:
    ```python
    def getStateShape(self):
        return self.bars_count, len(self.info_list) + 2
    ```
  - [x] Remove line `data_res[:, len(self.info_list) + 2] = ...` in `encode()`.
- [x] **Step 2: Configuration Update in [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml)**
  - [x] Change `PORTFOLIO_DIM: 3` to `PORTFOLIO_DIM: 2`.
- [x] **Step 3: Model and Agent Defaults in [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) & [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py)**
  - [x] Update `MambaContinuousActorCritic.__init__` default `portfolio_dim: int = 2`.
  - [x] Update `PPO2Agent.__init__` default `portfolio_dim: int = 2`.
  - [x] Update docstrings describing the portfolio stream as `[position, unrealized_return]`.
- [x] **Step 4: Engine Alignment in [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py)**
  - [x] In `encode_observation()`: remove `trade_bar: int = 0` parameter and assignment `data_res[:, len(info_list) + 2] = ...`.
  - [x] In `_resolve_symbol_portfolio_state()`: return `(position, open_price)` instead of 3-tuple.
  - [x] Update call sites in `_evaluate_portfolio_vectorized` and single-symbol evaluation methods.
- [x] **Step 5: Code Cleanliness and Defensive Assertions**
  - [x] Ensure no lingering references to `trade_bar` exist anywhere in `Brain/PPO2/`.
  - [x] Validate shape assertions on observation dictionaries.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)
```python
class State_time_step(State_time_step_template):
    def __init__(self, ...):
        ...
        self.position = 0.0
        self.open_price = 0.0
        self.game_steps = 0
        self._prev_close = 0.0
        # self.trade_bar removed

    def getStateShape(self):
        return self.bars_count, len(self.info_list) + 2

    def reset(self, prices: Prices, offset: int):
        ...
        self.position = 0.0
        self.open_price = 0.0
        self.game_steps = 0
        # self.trade_bar removed

    def step(self, action: float):
        ...
        if target_pos == 0.0:
            self.open_price = 0.0
        elif prev_pos == 0.0:
            self.open_price = open_price
        else:
            if target_pos > prev_pos:
                added_size = target_pos - prev_pos
                self.open_price = (
                    prev_pos * self.open_price + added_size * open_price
                ) / target_pos
        self.position = target_pos
        ...

    def encode(self):
        data_res = np.zeros(shape=self.getStateShape(), dtype=np.float32)
        time_res = np.zeros(shape=self.getTimeShape(), dtype=np.float32)

        start = self._offset - self.bars_count
        end = self._offset

        for idx, field in enumerate(self.info_list):
            data_res[:, idx] = getattr(self._prices, field)[start:end]

        data_res[:, len(self.info_list)] = self.position

        if self.position > 0.0 and self.open_price > 0.0:
            prev_close = max(float(self._prices.close[self._offset - 1]), 1e-8)
            safe_open = max(float(self.open_price), 1e-8)
            raw_equity_ratio = 1.0 + self.position * ((prev_close - safe_open) / safe_open)
            data_res[:, len(self.info_list) + 1] = float(np.log(max(raw_equity_ratio, 1e-8)))
        else:
            data_res[:, len(self.info_list) + 1] = 0.0

        for idx, field in enumerate(self.timelist):
            time_res[:, idx] = getattr(self._prices, field)[start:end]

        return {"states": data_res, "time_states": time_res}
```

#### 3.2.2 [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py)
```python
    def encode_observation(
        self,
        prices: Prices,
        offset: int,
        info_list: List[str],
        time_list: List[str],
        position: float = 0.0,
        open_price: float = 0.0,
        portfolio_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        ...
        data_res[:, len(info_list)] = position
        if position > 0.0 and open_price > 0.0:
            prev_close = max(float(prices.close[offset - 1]), 1e-8)
            safe_open = max(float(open_price), 1e-8)
            raw_equity_ratio = 1.0 + position * ((prev_close - safe_open) / safe_open)
            data_res[:, len(info_list) + 1] = float(np.log(max(raw_equity_ratio, 1e-8)))
        else:
            data_res[:, len(info_list) + 1] = 0.0
        ...
```

---

## 4. Validation & Verification Plan

### 4.1 Verification Environment & Commands
> [!IMPORTANT]
> In accordance with repository rules, all verification must execute using the project virtual environment Python binary:
> Relative: `../bin/python`
> Absolute: `/home/b0457812963/Mamba3RL/bin/python`

```bash
# 1. Verification of observation space shape and State_time_step bit-level encoding
../bin/python -c "
import numpy as np
from Brain.PPO2.lib.environment import State_time_step
from Brain.Common.DataFeature import Prices

N = 100
dummy_prices = Prices(
    open=np.ones(N, dtype=np.float32)*100.0,
    high=np.ones(N, dtype=np.float32)*105.0,
    low=np.ones(N, dtype=np.float32)*95.0,
    close=np.ones(N, dtype=np.float32)*100.0,
    volume=np.ones(N, dtype=np.float32)*1000.0,
    hour_sin=np.zeros(N, dtype=np.float32),
    hour_cos=np.ones(N, dtype=np.float32),
    minute_sin=np.zeros(N, dtype=np.float32),
    minute_cos=np.ones(N, dtype=np.float32),
    dayofweek_sin=np.zeros(N, dtype=np.float32),
    dayofweek_cos=np.ones(N, dtype=np.float32),
)
state = State_time_step(bars_count=10, commission_perc=0.0001, model_train=True, default_slippage=0.0001, N_steps=50)
state.reset(dummy_prices, offset=10)
obs = state.encode()
print('State shape:', obs['states'].shape)
assert obs['states'].shape == (10, len(state.info_list) + 2), f'Mismatch shape: {obs[\"states\"].shape}'
print('Validation 1 Passed!')
"

# 2. Parity check between State_time_step.encode() and PPO2Engine.encode_observation()
../bin/python -c "
import numpy as np
from omegaconf import OmegaConf
from Brain.PPO2.lib.environment import State_time_step
from Brain.PPO2.lib.engine import PPO2Engine
from Brain.Common.DataFeature import Prices

cfg = OmegaConf.load('configs/training/ppo_v2.yaml')
assert int(cfg.PORTFOLIO_DIM) == 2, f'Expected PORTFOLIO_DIM 2, got {cfg.PORTFOLIO_DIM}'
engine = PPO2Engine(config=cfg)
print('Validation 2 Passed!')
"

# 3. Model forward/backward pass with portfolio_dim=2
../bin/python -c "
import torch
from Brain.PPO2.lib.model import PPONetwork
from omegaconf import OmegaConf

cfg = OmegaConf.load('configs/training/ppo_v2.yaml')
d_model = 32 + 2 # market + portfolio
net = PPONetwork(
    d_model=d_model,
    time_features_in=6,
    time_features_out=int(cfg.TIME_FEATURES_OUT),
    hidden_size=int(cfg.HIDDEN_SIZE),
    nlayers=int(cfg.NLAYERS),
    ssm_cfg=OmegaConf.to_container(cfg.SSM_CFG),
    portfolio_dim=int(cfg.PORTFOLIO_DIM),
)
states = torch.randn(2, 300, d_model)
time_states = torch.randn(2, 300, 6)
action, value = net(states, time_states)
print('Forward pass action shape:', action.shape, 'value shape:', value.shape)
assert action.shape == (2, 1) and value.shape == (2, 1)
print('Validation 3 Passed!')
"
```

### 4.2 Acceptance Verification Checklist
- [x] **1. Functional Correctness**:
  - [x] `State_time_step.getStateShape()` returns `(bars_count, len(info_list) + 2)`.
  - [x] `obs["states"]` contains exactly `len(info_list) + 2` columns.
  - [x] `engine.encode_observation()` matches `state.encode()` bit-for-bit.
- [x] **2. Cleanliness & Zero-Dead-Code**:
  - [x] Grepping `trade_bar` across `Brain/PPO2/` yields zero references.
  - [x] No unbounded values or drifting time features inside `states`.
- [x] **3. Runtime & Empirical Verification**:
  - [x] All three verification scripts run successfully under `../bin/python` with exit code 0.
