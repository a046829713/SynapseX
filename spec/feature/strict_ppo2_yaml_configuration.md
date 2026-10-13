# Specification: Strict Fail-Fast YAML Configuration Binding and Declarative SSM Architecture for PPO2

- **Category**: `feature`
- **Spec Path**: `spec/feature/strict_ppo2_yaml_configuration.md`
- **Status**: `Implemented`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
In [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) and [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py), hyperparameter lookups across model definition, environment configuration, buffer settings, and training updates use fallback defaults via `getattr(cfg_training, "PARAM_NAME", default_val)`. For instance:
- `bars_count = getattr(cfg_training, "BARS_COUNT", 300)`
- `gamma = getattr(cfg_training, "GAMMA", 0.99)`
- `lam = getattr(cfg_training, "GAE_LAMBDA", 0.95)`
- `hidden_size = getattr(cfg_training, "HIDDEN_SIZE", 96)`
- `nlayers = getattr(cfg_training, "NLAYERS", 2)`
- `lr = getattr(cfg_training, "LEARNING_RATE", 3e-4)`

Furthermore, in [`PPO2_rl_train.py:L121-L133`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L121-L133), the entire Mamba2 SSM configuration (`ssm_cfg_fast`) is hardcoded as an in-line Python dictionary rather than declared inside the Hydra YAML configuration:
```python
ssm_cfg_fast = {
    "layer": "Mamba2",
    "expand": 2,
    "d_ssm": 96,
    "headdim": 32,
    "d_state": 64,
    "ngroups": 1,
    "chunk_size": 64,
    "d_conv": 4,
    "rmsnorm": True,
    "bias": False,
    "conv_bias": True,
}
```

This presents two major drawbacks:
1. **Silent Fallback Hazards**: If a hyperparameter key in [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) is misspelled, missing, or omitted, Python silently falls back to the hardcoded default. The experimenter cannot guarantee whether the neural network, environment, or training loop actually executed with their intended parameters.
2. **Scattered Configuration**: Having hyperparameters divided between YAML (`ppo_v2.yaml`) and hardcoded script constants (`ssm_cfg_fast`) violates the single-source-of-truth principle, complicating hyperparameter sweeps, architecture ablation, and configuration version control.

### 1.2 Current vs. Expected Behavior
| Dimension | Current Behavior | Expected Behavior |
| :--- | :--- | :--- |
| **Missing Parameter Behavior** | Silently falls back to hardcoded default values via `getattr(cfg_training, key, default)`. | **Fail-Fast**: Immediately raises an error (e.g. `omegaconf.errors.ConfigAttributeError`) if any mandatory configuration key is missing or undefined. |
| **SSM / Mamba Architecture** | Hardcoded `ssm_cfg_fast` dictionary inside [`PPO2_rl_train.py:L121-L133`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L121-L133). | Fully declared inside [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) under `SSM_CFG` and converted via `OmegaConf.to_container(..., resolve=True)`. |
| **Missing Hyperparameters in YAML** | Keys like `GAMMA`, `GAE_LAMBDA`, `HIDDEN_SIZE`, `NLAYERS`, `TIME_FEATURES_OUT`, `DROPOUT`, `PORTFOLIO_DIM`, `DEADZONE_THRESHOLD`, `DOWNSIDE_WINDOW_SIZE`, `W1_STEP_RETURN`, `W2_DOWNSIDE_RATIO` are omitted from `ppo_v2.yaml`. | All training, agent, model, and environment hyperparameters are explicitly defined in `ppo_v2.yaml`. |
| **Environment State Params** | [`Brain/PPO2/lib/environment.py:L331-L345`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L331-L345) uses `getattr` fallbacks for `DEFAULT_SLIPPAGE`, `EPISODE_STEPS`, `DEADZONE_THRESHOLD`, `DOWNSIDE_WINDOW_SIZE`, `W1_STEP_RETURN`, `W2_DOWNSIDE_RATIO`. | Directly reads parameters from `cfg_training` without fallback defaults. |

### 1.3 Technical Root Cause & References
- [`PPO2_rl_train.py:L55-L59`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L55-L59): `RolloutBuffer` initialization uses `getattr` for `GAMMA` and `GAE_LAMBDA`.
- [`PPO2_rl_train.py:L95-L96`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L95-L96): `SAVES_TAG` and `BARS_COUNT` use `getattr`.
- [`PPO2_rl_train.py:L105-L106`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L105-L106): `NUM_WORKERS` uses `getattr`.
- [`PPO2_rl_train.py:L121-L146`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L121-L146): Hardcoded `ssm_cfg_fast` dictionary and `getattr` for model dimensions (`HIDDEN_SIZE`, `NLAYERS`, `TIME_FEATURES_OUT`, `DROPOUT`, `PORTFOLIO_DIM`).
- [`PPO2_rl_train.py:L154-L161`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L154-L161): `LEARNING_RATE`, `WEIGHT_DECAY`, and `BASE_LR` use `getattr`.
- [`PPO2_rl_train.py:L236-L246`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L236-L246): `HORIZON_STEPS`, `BATCH_SIZE`, `PPO_EPOCHS`, `CLIP_EPS`, `VF_COEF`, `ENT_COEF`, `MAX_GRAD_NORM`, `CHECKPOINT_EVERY_ITER` use `getattr`.
- [`Brain/PPO2/lib/environment.py:L326-L345`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L326-L345): Environment parameters fall back silently to internal defaults.

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix
| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) | **In-Scope** | Add missing training hyperparameters (`GAMMA`, `GAE_LAMBDA`), model hyperparameters (`HIDDEN_SIZE`, `NLAYERS`, `TIME_FEATURES_OUT`, `DROPOUT`, `PORTFOLIO_DIM`), `SSM_CFG` dictionary block, and environment downside parameters (`DEADZONE_THRESHOLD`, `DOWNSIDE_WINDOW_SIZE`, `W1_STEP_RETURN`, `W2_DOWNSIDE_RATIO`). |
| **Item 2** | [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) | **In-Scope** | 1. Replace all `getattr(cfg_training, ...)` occurrences with direct attribute access (`self.cfg_training.<KEY>`).<br>2. Remove hardcoded `ssm_cfg_fast` dictionary in `_prepare_agent()`.<br>3. Read `ssm_cfg` from `OmegaConf.to_container(self.cfg_training.SSM_CFG, resolve=True)`.<br>4. Cache `self.cfg_training = self.config.training` in `__init__`. |
| **Item 3** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L326-L345) | **In-Scope** | Update `TrainingEnv.__init__` to directly access `cfg_training.<KEY>` without fallback defaults. |
| **Item 4** | [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py) | **Out-of-Scope** | Preserved unchanged. `PPO2Agent` already accepts `ssm_cfg: Optional[dict] = None`. |
| **Item 5** | [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) | **Out-of-Scope** | Neural network backbone preserved unchanged. |

### 2.2 Architectural Boundaries & Non-Goals
- **In-Scope Boundaries**:
  - Guaranteeing fail-fast behavior across all PPO2 training and environment configurations.
  - Centralizing all PPO2 training, model, environment, and SSM configurations into `configs/training/ppo_v2.yaml`.
- **Out-of-Scope (Non-Goals)**:
  - Do not alter mathematical PPO update logic, loss calculations, or GAE computation.
  - Do not modify Mamba2 / SSM layer implementation in [`Brain/Common/ssm_tool.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/ssm_tool.py).
  - Do not modify non-PPO configurations (e.g. `dqn_pilot.yaml`).

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Enrich [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml)**
  - [x] Add `GAMMA: 0.99` and `GAE_LAMBDA: 0.95`.
  - [x] Add model parameters: `HIDDEN_SIZE: 96`, `NLAYERS: 2`, `TIME_FEATURES_OUT: 32`, `DROPOUT: 0.1`, `PORTFOLIO_DIM: 3`.
  - [x] Add `SSM_CFG` block containing Mamba2 hyperparameters.
  - [x] Add environment parameters: `DEADZONE_THRESHOLD: 0.01`, `DOWNSIDE_WINDOW_SIZE: 60`, `W1_STEP_RETURN: 1.0`, `W2_DOWNSIDE_RATIO: 1.0`.
- [x] **Step 2: Refactor [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)**
  - [x] In `PPO2.__init__`, define `self.cfg_training = self.config.training`.
  - [x] Replace `getattr(cfg_training, "GAMMA", 0.99)` with `self.cfg_training.GAMMA`.
  - [x] Replace `getattr(cfg_training, "GAE_LAMBDA", 0.95)` with `self.cfg_training.GAE_LAMBDA`.
  - [x] In `_prepare_symbols()`, replace `getattr` calls with `self.cfg_training.SAVES_TAG` and `self.cfg_training.BARS_COUNT`.
  - [x] In `_prepare_env()`, replace `getattr` call with `self.cfg_training.NUM_WORKERS`.
  - [x] In `_prepare_agent()`, remove hardcoded `ssm_cfg_fast` dictionary; read `ssm_cfg = OmegaConf.to_container(self.cfg_training.SSM_CFG, resolve=True)` and direct attribute access for `BARS_COUNT`, `HIDDEN_SIZE`, `NLAYERS`, `TIME_FEATURES_OUT`, `DROPOUT`, `PORTFOLIO_DIM`.
  - [x] In `_prepare_optimizer()`, direct attribute access for `LEARNING_RATE`, `WEIGHT_DECAY`, `BASE_LR`.
  - [x] In `train()`, direct attribute access for `HORIZON_STEPS`, `BATCH_SIZE`, `PPO_EPOCHS`, `CLIP_EPS`, `VF_COEF`, `ENT_COEF`, `MAX_GRAD_NORM`, `CHECKPOINT_EVERY_ITER`.
- [x] **Step 3: Refactor [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L326-L345)**
  - [x] In `TrainingEnv.__init__`, replace `getattr` fallbacks with direct attribute access `cfg_training.<KEY>`.
- [x] **Step 4: Clean up & Defensiveness**
  - [x] Validate fail-fast behavior: verify an omitted key raises an immediate error without starting training.

---

## 4. Validation & Verification Plan

### 4.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST adhere to repository environment standards using the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python`
> Relative path: `../bin/python`

```bash
# 1. Verification of clean configuration loading:
../bin/python -c "
import hydra
from omegaconf import OmegaConf
with hydra.initialize(version_base=None, config_path='configs'):
    cfg = hydra.compose(config_name='config')
    print('Configuration loaded successfully:')
    print(OmegaConf.to_yaml(cfg.training))
"

# 2. Verification of fail-fast behavior (deliberately deleting a key must raise ConfigAttributeError):
../bin/python -c "
import hydra
from omegaconf import OmegaConf
with hydra.initialize(version_base=None, config_path='configs'):
    cfg = hydra.compose(config_name='config')
    del cfg.training.BARS_COUNT
    try:
        val = cfg.training.BARS_COUNT
        print('FAIL: Expected error but got value', val)
    except Exception as e:
        print('PASS: Successfully caught expected error on missing key:', type(e).__name__)
"

# 3. Verification of agent and model initialization with YAML ssm_cfg:
../bin/python -c "
import hydra
from PPO2_rl_train import PPO2
with hydra.initialize(version_base=None, config_path='configs'):
    cfg = hydra.compose(config_name='config')
    ppo = PPO2(cfg)
    print('PPO2 initialized successfully with model:', ppo.agent.model.__class__.__name__)
    print('SSM config verify:', ppo.agent.model.backbone.layers[0].mixer.__class__.__name__)
    ppo.vec_env.close()
"
```

### 4.2 Acceptance Verification Checklist
- [ ] **1. Fail-Fast Integrity**: Omitting any key in `configs/training/ppo_v2.yaml` raises an immediate error without silent fallback.
- [ ] **2. Single Source of Truth**: `ssm_cfg_fast` is completely eliminated from `PPO2_rl_train.py` and populated entirely from `SSM_CFG` in YAML.
- [ ] **3. Runtime Compatibility**: `PPO2` initializes and starts workers normally when provided with the complete YAML configuration.
