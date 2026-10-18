# Specification: Strict Fail-Fast YAML Configuration Binding and Environment Parity for PPO2 Test Engine

- **Category**: `feature`
- **Spec Path**: `spec/feature/strict_ppo2_test_engine_yaml_alignment.md`
- **Status**: `Implemented & Verified`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
In [`spec/feature/strict_ppo2_yaml_configuration.md`](file:///home/b0457812963/Mamba3RL/SynapseX/spec/feature/strict_ppo2_yaml_configuration.md), the repository established the strict principle of **Fail-Fast YAML Configuration Binding** and **Single Source of Truth** for the PPO2 continuous RL framework. All silent default fallbacks (e.g. `getattr(cfg, key, default)`) were eliminated from [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) and [`TrainingEnv`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L317-L343), guaranteeing that missing or misspelled parameters immediately halt execution.

However, during the recent implementation of the dedicated testing engine in [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) and [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py):
1. **Silent Fallback Anti-Pattern Resurfaced**:
   - In [`Brain/PPO2/lib/engine.py:L54-L77`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L54-L77), an entire hardcoded fallback dictionary `OmegaConf.create({...})` was reintroduced if the configuration file is not found.
   - Throughout `PPO2Engine.__init__` and `build_agent()`, parameter lookups rely on `self.cfg.get("PARAM", default)` (e.g., `BARS_COUNT`, `HIDDEN_SIZE`, `DEFAULT_SLIPPAGE`, `MODEL_DEFAULT_COMMISSION_PERC_TRAING`).
2. **Train/Test Environment Asymmetry (Isomorphism Violation)**:
   - In [`TrainingEnv`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L330-L340), `State_time_step` is instantiated with full YAML-declared parameters:
     ```python
     state_params = {
         "bars_count": cfg_training.BARS_COUNT,
         "commission_perc": cfg_training.MODEL_DEFAULT_COMMISSION_PERC_TRAING,
         "model_train": True,
         "default_slippage": cfg_training.DEFAULT_SLIPPAGE,
         "N_steps": cfg_training.EPISODE_STEPS,
         "deadzone_threshold": cfg_training.DEADZONE_THRESHOLD,
         "downside_window_size": cfg_training.DOWNSIDE_WINDOW_SIZE,
         "w1_step_return": cfg_training.W1_STEP_RETURN,
         "w2_downside_ratio": cfg_training.W2_DOWNSIDE_RATIO,
     }
     ```
   - In contrast, in [`Brain/PPO2/lib/Evaluator.py:L66-L73`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L66-L73) and `engine.py:L181-L187`:
     - `N_steps=500` is hardcoded rather than bound to `EPISODE_STEPS`.
     - `downside_window_size`, `w1_step_return`, and `w2_downside_ratio` are completely omitted and fall back to internal defaults inside [`State_time_step.__init__`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L13-L25).
     - `PPO2Evaluator.__init__` defaulted `commission_perc = 0.0` and `default_slippage = 0.0`, creating potential zero-friction discrepancy when initialized directly.
3. **Configuration Interface Disconnect**:
   - `PPO2_rl_train.py` passes a Hydra `DictConfig` containing `cfg.training`.
   - `PPO2Engine` expects a string file path and does not natively unwrap top-level Hydra `DictConfig` objects if passed directly by external orchestration or training integration hooks.

### 1.2 Current vs. Expected Behavior
| Dimension | Current Behavior | Expected Behavior |
| :--- | :--- | :--- |
| **Missing Config Key** | Silently falls back to hardcoded defaults via `.get(key, default)`. | **Fail-Fast**: Directly accesses attributes (`self.cfg.KEY`), immediately raising `ConfigAttributeError` if undefined. |
| **Missing Config File** | Silently creates an in-line default dictionary via `OmegaConf.create({...})`. | Raises `FileNotFoundError` immediately. No hardcoded fallback dictionaries. |
| **Configuration Ingestion** | Only accepts a file path string `config_path`. | Accepts `Optional[Union[DictConfig, str]] = None`. Automatically extracts `config.training` if a top-level Hydra config is provided. Defaults to `configs/training/ppo_v2.yaml`. |
| **State Machine Parameters** | `Evaluator` and `dummy_state` omit downside risk parameters (`downside_window_size`, `w1_step_return`, `w2_downside_ratio`) and hardcode `N_steps=500`. | **100% Isomorphic**: All state parameters declared in YAML (`EPISODE_STEPS`, `DOWNSIDE_WINDOW_SIZE`, `W1_STEP_RETURN`, `W2_DOWNSIDE_RATIO`, etc.) are passed directly into `PPO2Evaluator` and `State_time_step`. |
| **CLI Config Customization** | [`PPO2_rl_test.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py) lacks `--config` argument. | Supports `--config` argument (default: `configs/training/ppo_v2.yaml`) passed through to `PPO2Engine`. |

### 1.3 Technical Root Cause & References
- [`Brain/PPO2/lib/engine.py:L52-L85`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L52-L85): Hardcoded fallback dictionary and `.get(..., default)` calls.
- [`Brain/PPO2/lib/engine.py:L181-L187`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L181-L187): `dummy_state` initialized with hardcoded `N_steps=500` and missing downside risk parameters.
- [`Brain/PPO2/lib/engine.py:L210-L214`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L210-L214): `build_agent()` reads architecture hyperparameters with `.get(..., default)`.
- [`Brain/PPO2/lib/Evaluator.py:L21-L26`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L21-L26): Default `commission_perc=0.0` and `default_slippage=0.0`; omission of downside risk parameters.
- [`Brain/PPO2/lib/Evaluator.py:L66-L73`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L66-L73): `State_time_step` constructed without `downside_window_size`, `w1_step_return`, `w2_downside_ratio`, and with hardcoded `N_steps=500`.
- [`PPO2_rl_test.py:L7-L42`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py#L7-L42): Missing `--config` CLI option.

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix
| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) | **In-Scope** | 1. Support `config: Optional[Union[DictConfig, str]] = None`.<br>2. Eliminate fallback dictionary and all `.get(..., default)` calls.<br>3. Directly bind all parameters via `self.cfg.<KEY>`.<br>4. Pass complete environment parameters to `PPO2Evaluator` and `dummy_state`. |
| **Item 2** | [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py) | **In-Scope** | 1. Extend `PPO2Evaluator.__init__` to accept `downside_window_size`, `w1_step_return`, `w2_downside_ratio`, and `episode_steps`.<br>2. Pass all parameters into `State_time_step` inside `evaluate_symbol()` to ensure complete isomorphism with `TrainingEnv`. |
| **Item 3** | [`PPO2_rl_test.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py) | **In-Scope** | Add `--config` argument (default: `"configs/training/ppo_v2.yaml"`) and pass it to `PPO2Engine(config=args.config)`. |
| **Item 4** | `tests/test_ppo2_evaluator.py` | **In-Scope** | Add comprehensive unit tests verifying fail-fast behavior upon missing keys, config attribute resolution, and parameter parity between training and testing states. |
| **Item 5** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) | **Out-of-Scope (Preserved)** | `TrainingEnv` and `ProductionEnv` classes preserved unchanged. |
| **Item 6** | [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) | **Out-of-Scope (Preserved)** | PPO2 training script already follows strict YAML binding. |
| **Item 7** | [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) | **Out-of-Scope (Preserved)** | Single source of truth preserved. |

### 2.2 Architectural Boundaries & Non-Goals
- **In-Scope Boundaries**:
  - Guaranteeing identical environment configuration between training and testing.
  - Ensuring strict fail-fast configuration binding across `PPO2Engine` and `PPO2Evaluator`.
  - Adding customizable config path loading in `PPO2_rl_test.py`.
- **Out-of-Scope (Non-Goals)**:
  - Do not modify mathematical calculations in `State_time_step` or `ProductionEnv`.
  - Do not alter neural network architecture in `Brain/PPO2/lib/model.py` or agent logic in `Brain/PPO2/lib/Agent.py`.
  - Do not reintroduce financial PnL calculations or plotting to `PPO2Evaluator`.

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Refactor `PPO2Engine` in [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py)**
  - [x] Update `__init__` signature to `(formal: bool = False, config: Optional[Union[DictConfig, str]] = None, device: Optional[torch.device] = None, base_data_dir: str = "Brain/simulation/test_data")`.
  - [x] Implement robust configuration resolution:
    - If `config` is a `DictConfig` and has attribute `training`, use `self.cfg = config.training`.
    - If `config` is a `DictConfig` without `training`, use `self.cfg = config`.
    - If `config` is `None`, default to `"configs/training/ppo_v2.yaml"`.
    - If `config` is a path string, load via `OmegaConf.load(config_path)`. If file does not exist, raise `FileNotFoundError`.
  - [x] Replace all `.get(..., default)` with direct fail-fast attribute access:
    - `self.bars_count = int(self.cfg.BARS_COUNT)`
    - `self.deadzone_threshold = float(self.cfg.DEADZONE_THRESHOLD)`
    - `self.commission_perc = float(self.cfg.MODEL_DEFAULT_COMMISSION_PERC_TRAING)`
    - `self.default_slippage = float(self.cfg.DEFAULT_SLIPPAGE)`
    - `self.hidden_size = int(self.cfg.HIDDEN_SIZE)`
    - `self.nlayers = int(self.cfg.NLAYERS)`
    - `self.time_features_out = int(self.cfg.TIME_FEATURES_OUT)`
    - `self.portfolio_dim = int(self.cfg.PORTFOLIO_DIM)`
    - `self.downside_window_size = int(self.cfg.DOWNSIDE_WINDOW_SIZE)`
    - `self.w1_step_return = float(self.cfg.W1_STEP_RETURN)`
    - `self.w2_downside_ratio = float(self.cfg.W2_DOWNSIDE_RATIO)`
    - `self.episode_steps = int(self.cfg.EPISODE_STEPS)`
  - [x] In `build_agent()`, pass all parameters to `dummy_state` matching `State_time_step` parameters.
  - [x] In `run_test()` and `get_if_order_map()`, instantiate `PPO2Evaluator` passing all state and downside risk parameters.
- [x] **Step 2: Refactor `PPO2Evaluator` in [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py)**
  - [x] Extend `__init__` parameters:
    - `downside_window_size: int = 60`
    - `w1_step_return: float = 1.0`
    - `w2_downside_ratio: float = 1.0`
    - `episode_steps: int = 500`
  - [x] In `evaluate_symbol()`, construct `State_time_step` with:
    - `bars_count=self.bars_count`
    - `commission_perc=self.commission_perc`
    - `model_train=False`
    - `default_slippage=self.default_slippage`
    - `N_steps=self.episode_steps`
    - `deadzone_threshold=self.deadzone_threshold`
    - `downside_window_size=self.downside_window_size`
    - `w1_step_return=self.w1_step_return`
    - `w2_downside_ratio=self.w2_downside_ratio`
- [x] **Step 3: Update `PPO2_rl_test.py`**
  - [x] Add CLI flag `--config` with default `"configs/training/ppo_v2.yaml"`.
  - [x] Pass `config=args.config` to `PPO2Engine`.
- [x] **Step 4: Create automated unit tests in `tests/test_ppo2_evaluator.py`**
  - [x] Test 1: Fail-fast verification (omitted keys raise `ConfigAttributeError` or `AttributeError`).
  - [x] Test 2: Full parameter parity between `TrainingEnv` and `PPO2Evaluator` `State_time_step`.
  - [x] Test 3: Deterministic evaluation and output tabular order CSV schema validation.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 `Brain/PPO2/lib/engine.py` Configuration Binding
```python
class PPO2Engine:
    def __init__(
        self,
        formal: bool = False,
        config: Optional[Union[DictConfig, str]] = None,
        device: Optional[torch.device] = None,
        base_data_dir: str = "Brain/simulation/test_data",
    ):
        self.formal = formal
        self.base_data_dir = base_data_dir
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Resolve configuration strictly without silent fallbacks
        if config is None:
            config_path = "configs/training/ppo_v2.yaml"
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            raw_cfg = OmegaConf.load(config_path)
            self.cfg = raw_cfg.training if hasattr(raw_cfg, "training") else raw_cfg
        elif isinstance(config, str):
            if not os.path.exists(config):
                raise FileNotFoundError(f"Configuration file not found: {config}")
            raw_cfg = OmegaConf.load(config)
            self.cfg = raw_cfg.training if hasattr(raw_cfg, "training") else raw_cfg
        elif isinstance(config, DictConfig):
            self.cfg = config.training if hasattr(config, "training") else config
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        # Strict fail-fast binding of all model and environment hyperparameters
        self.bars_count = int(self.cfg.BARS_COUNT)
        self.deadzone_threshold = float(self.cfg.DEADZONE_THRESHOLD)
        self.commission_perc = float(self.cfg.MODEL_DEFAULT_COMMISSION_PERC_TRAING)
        self.default_slippage = float(self.cfg.DEFAULT_SLIPPAGE)
        self.hidden_size = int(self.cfg.HIDDEN_SIZE)
        self.nlayers = int(self.cfg.NLAYERS)
        self.time_features_out = int(self.cfg.TIME_FEATURES_OUT)
        self.portfolio_dim = int(self.cfg.PORTFOLIO_DIM)
        self.downside_window_size = int(self.cfg.DOWNSIDE_WINDOW_SIZE)
        self.w1_step_return = float(self.cfg.W1_STEP_RETURN)
        self.w2_downside_ratio = float(self.cfg.W2_DOWNSIDE_RATIO)
        self.episode_steps = int(self.cfg.EPISODE_STEPS)
```

#### 3.2.2 `Brain/PPO2/lib/Evaluator.py` Environment Alignment
```python
class PPO2Evaluator:
    def __init__(
        self,
        agent: PPO2Agent,
        bars_count: int,
        device: Optional[torch.device] = None,
        deadzone_threshold: float = 0.01,
        commission_perc: float = 0.0045,
        default_slippage: float = 0.0025,
        downside_window_size: int = 60,
        w1_step_return: float = 1.0,
        w2_downside_ratio: float = 1.0,
        episode_steps: int = 500,
    ):
        self.agent = agent
        self.bars_count = bars_count
        self.device = device or agent.device
        self.deadzone_threshold = deadzone_threshold
        self.commission_perc = commission_perc
        self.default_slippage = default_slippage
        self.downside_window_size = downside_window_size
        self.w1_step_return = w1_step_return
        self.w2_downside_ratio = w2_downside_ratio
        self.episode_steps = episode_steps

        self.agent.model.eval()

    def evaluate_symbol(
        self,
        symbol: str,
        raw_df: pd.DataFrame,
        prices: Prices,
        output_csv_path: Optional[str] = None,
    ) -> pd.DataFrame:
        self.agent.model.eval()

        state = State_time_step(
            bars_count=self.bars_count,
            commission_perc=self.commission_perc,
            model_train=False,
            default_slippage=self.default_slippage,
            N_steps=self.episode_steps,
            deadzone_threshold=self.deadzone_threshold,
            downside_window_size=self.downside_window_size,
            w1_step_return=self.w1_step_return,
            w2_downside_ratio=self.w2_downside_ratio,
        )
        env = ProductionEnv(prices_data={symbol: prices}, state=state)
        obs = env.reset(symbol=symbol)
        ...
```

---

## 4. Validation & Verification Plan

### 4.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST adhere to repository environment standards using the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python`
> Relative path: `../bin/python`

```bash
# 1. Run automated unit tests:
../bin/python -m unittest tests/test_ppo2_evaluator.py

# 2. Verify fail-fast behavior with a deliberately truncated configuration:
../bin/python -c "
from omegaconf import OmegaConf
from Brain.PPO2.lib.engine import PPO2Engine
cfg = OmegaConf.load('configs/training/ppo_v2.yaml')
del cfg.BARS_COUNT
try:
    engine = PPO2Engine(config=cfg)
    print('FAIL: Expected fail-fast error but succeeded.')
except Exception as e:
    print('PASS: Caught expected error:', type(e).__name__, e)
"

# 3. Verify parameter parity between TrainingEnv and PPO2Evaluator:
../bin/python -c "
import hydra
from omegaconf import OmegaConf
from Brain.PPO2.lib.engine import PPO2Engine
from Brain.PPO2.lib.environment import TrainingEnv

with hydra.initialize(version_base=None, config_path='configs'):
    hydra_cfg = hydra.compose(config_name='config')

engine = PPO2Engine(config=hydra_cfg)
agent = engine.build_agent()
evaluator = PPO2Evaluator(
    agent=agent,
    bars_count=engine.bars_count,
    deadzone_threshold=engine.deadzone_threshold,
    commission_perc=engine.commission_perc,
    default_slippage=engine.default_slippage,
    downside_window_size=engine.downside_window_size,
    w1_step_return=engine.w1_step_return,
    w2_downside_ratio=engine.w2_downside_ratio,
    episode_steps=engine.episode_steps,
)

assert evaluator.bars_count == hydra_cfg.training.BARS_COUNT
assert evaluator.commission_perc == hydra_cfg.training.MODEL_DEFAULT_COMMISSION_PERC_TRAING
assert evaluator.default_slippage == hydra_cfg.training.DEFAULT_SLIPPAGE
assert evaluator.downside_window_size == hydra_cfg.training.DOWNSIDE_WINDOW_SIZE
assert evaluator.w1_step_return == hydra_cfg.training.W1_STEP_RETURN
assert evaluator.w2_downside_ratio == hydra_cfg.training.W2_DOWNSIDE_RATIO
assert evaluator.episode_steps == hydra_cfg.training.EPISODE_STEPS
print('PASS: 100% parameter parity between Hydra training config and PPO2Evaluator.')
"

# 4. Verify end-to-end execution of formal and batch testing:
../bin/python PPO2_rl_test.py --formal
"
```

### 4.2 Acceptance Verification Checklist
- [x] **1. Fail-Fast Integrity**: If any required parameter in `ppo_v2.yaml` is missing, `PPO2Engine` immediately throws an error without silent default fallback.
- [x] **2. Single Source of Truth**: All fallback dictionaries inside `engine.py` are completely eliminated.
- [x] **3. Environment Isomorphism**: Every state parameter (`bars_count`, `commission_perc`, `default_slippage`, `N_steps`, `deadzone_threshold`, `downside_window_size`, `w1_step_return`, `w2_downside_ratio`) is bit-for-bit identical between `TrainingEnv` and `PPO2Evaluator`.
- [x] **4. CLI & API Compatibility**: `PPO2_rl_test.py --config` properly loads custom configs, and `PPO2Engine` accepts Hydra `DictConfig`, path strings, or default paths seamlessly.
