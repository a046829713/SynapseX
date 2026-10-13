# Specification: Unify DAIN Optimizer into Brain/Common and Simplify PPO2 Training Script

- **Category**: `feature`
- **Spec Path**: `spec/feature/unify_dain_optimizer_common.md`
- **Status**: `Implemented`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
In the current repository, [`PPO2_rl_train.py:L43-L95`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L43-L95) contains a standalone 53-line implementation of `build_optimizer`. Concurrently, [`Brain/Common/optimization.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py) contains a class `opitmizar` that implements substantially the same DAIN (`dean`) parameter group splitting and weight decay filtering.

However, having duplicate optimizer logic across the codebase introduces maintenance overhead and code divergence risks:
1. **Missing `log_std` Weight Decay Exclusion in Common**: In [`Brain/Common/optimization.py:L43`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py#L43), parameters are split into decay vs. no-decay only using `if "norm" in name or name.endswith(".bias"):`. In continuous action Actor-Critic architectures such as PPO2, policy exploration variance is governed by `log_std`. Applying weight decay to `log_std` pulls exploration standard deviation towards 0, resulting in catastrophic loss of policy entropy and exploration collapse. In contrast, [`PPO2_rl_train.py:L66`](file:///file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L66) properly includes `or "log_std" in name`.
2. **Inflexible `base_lr` in Common**: In [`Brain/Common/optimization.py:L17`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py#L17), `_prepare_optimizer(self, base_lr=1e-4)` defines a `base_lr` default parameter, but `__init__` does not accept `base_lr` nor pass it to `_prepare_optimizer()`. Consequently, `base_lr` cannot be configured from training configurations.
3. **API Ergonomics & Typo**: [`Brain/Common/optimization.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py) wraps the optimizer in a class with a misspelled name (`class opitmizar`), which requires instantiation and `.get_optimizer()`. Providing a clean, standard factory function `build_optimizer(...)` makes usage intuitive across all training scripts while maintaining backward compatibility with `opitmizar`.

### 1.2 Current vs. Expected Behavior
| Dimension | Current Behavior | Expected Behavior |
| :--- | :--- | :--- |
| **Code Duplication** | Duplicate 53-line `build_optimizer` inside [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py). | Unified single source of truth in [`Brain/Common/optimization.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py); imported cleanly into `PPO2_rl_train.py`. |
| **`log_std` Weight Decay** | Common excludes only `norm` and `bias`. `log_std` would receive weight decay if Common is used. | Common excludes `norm`, `bias`, and `log_std` from weight decay, supporting both discrete and continuous RL models safely. |
| **`base_lr` Configuration** | Common hardcodes `base_lr=1e-4` in `__init__`. | Both `build_optimizer` and `opitmizar` support configurable `base_lr`. |
| **API Interface** | Common only exports `opitmizar` class (with typo). | Common exports `build_optimizer`, `Optimizer`, and preserves `opitmizar` alias for backward compatibility. |

### 1.3 Technical Root Cause & References
- [`PPO2_rl_train.py:L43-L95`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L43-L95): Duplicate definition of `build_optimizer`.
- [`Brain/Common/optimization.py:L43`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py#L43):
  ```python
  if "norm" in name or name.endswith(".bias"):
      no_decay_params.append(param)
  ```
  Missing `or "log_std" in name`.
- [`Brain/Common/optimization.py:L5-L12`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py#L5-L12):
  ```python
  def __init__(self, net, learning_rate: float, lambda_l2: float):
      ...
      self._prepare_optimizer()
  ```
  Missing `base_lr` argument in `__init__`.
- [`HopeModel_traingloop.py:L29, L362-L364`](file:///home/b0457812963/Mamba3RL/SynapseX/HopeModel_traingloop.py#L29): Relies on `from Brain.Common.optimization import opitmizar`. Backward compatibility must be strictly preserved.

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix
| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`Brain/Common/optimization.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py) | **In-Scope** | 1. Implement top-level factory function `build_optimizer(model, lr=3e-4, weight_decay=1e-4, base_lr=1e-4) -> optim.AdamW`.<br>2. Exclude `norm`, `bias`, and `log_std` from weight decay.<br>3. Refactor `opitmizar` to delegate to `build_optimizer` or support `base_lr`.<br>4. Expose `Optimizer = opitmizar` alias while keeping `opitmizar` for full backward compatibility. |
| **Item 2** | [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) | **In-Scope** | 1. Remove redundant local `build_optimizer` definition (lines 43-95).<br>2. Add import: `from Brain.Common.optimization import build_optimizer`.<br>3. Keep `PPO2._prepare_optimizer()` invoking `build_optimizer(...)`. |
| **Item 3** | [`HopeModel_traingloop.py`](file:///home/b0457812963/Mamba3RL/SynapseX/HopeModel_traingloop.py) | **Out-of-Scope** | Preserved unchanged. Relies on `opitmizar` which remains 100% compatible. |
| **Item 4** | [`Brain/Common/dain.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/dain.py) | **Out-of-Scope** | DAIN module definitions preserved without modification. |

### 2.2 Architectural Boundaries & Non-Goals
- **In-Scope Boundaries**:
  - Consolidation of DAIN-aware AdamW optimizer creation into `Brain/Common/optimization.py`.
  - Preservation of existing hyperparameter defaults (`lr=3e-4`, `weight_decay=1e-4`, `base_lr=1e-4`).
- **Out-of-Scope (Non-Goals)**:
  - Do not change training schedules, learning rate schedulers, or loss calculations in PPO2 or HopeModel.
  - Do not modify network architectures or DAIN layer implementations.

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Enhance [`Brain/Common/optimization.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py)**
  - [x] Add `build_optimizer(model: torch.nn.Module, lr: float = 3e-4, weight_decay: float = 1e-4, base_lr: float = 1e-4) -> optim.AdamW`.
  - [x] Ensure `no_decay_params` includes `"norm" in name`, `name.endswith(".bias")`, and `"log_std" in name`.
  - [x] Refactor `opitmizar` (and `Optimizer` alias) to use `build_optimizer` internally or support `base_lr`.
  - [x] Add type annotations and English docstrings.
- [x] **Step 2: Streamline [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)**
  - [x] Import `build_optimizer` from `Brain.Common.optimization`.
  - [x] Remove the 53-line duplicate `build_optimizer` function.
  - [x] Keep `self._prepare_optimizer()` in `PPO2` calling `build_optimizer(...)`.
- [x] **Step 3: Verification & Compatibility Testing**
  - [x] Run automated script to test parameter group separation on a model with `dean` and `log_std`.
  - [x] Verify `HopeModel_traingloop.py` imports without syntax or attribute errors.
  - [x] Verify `PPO2_rl_train.py` initializes optimizer cleanly.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 [`Brain/Common/optimization.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    base_lr: float = 1e-4,
) -> optim.AdamW:
    """
    Builds a specialized AdamW optimizer for models containing DAIN (dean) layers:
    1. Special layers in `dean` (mean_layer, scaling_layer, gating_layer) use dedicated
       learning rates (scaled by mean_lr, scale_lr, gate_lr) and 0.0 weight decay.
    2. LayerNorm/normalization layers, bias parameters, and `log_std` parameters are excluded from weight decay.
    3. Remaining model parameters receive standard weight decay.
    """
    dean_params_ids = set()
    if hasattr(model, "dean"):
        dean_params_ids.update(id(p) for p in model.dean.parameters())

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in dean_params_ids:
            continue
        if "norm" in name or name.endswith(".bias") or "log_std" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
    ]

    if hasattr(model, "dean"):
        param_groups.extend([
            {
                "params": list(model.dean.mean_layer.parameters()),
                "lr": base_lr * model.dean.mean_lr,
                "weight_decay": 0.0,
            },
            {
                "params": list(model.dean.scaling_layer.parameters()),
                "lr": base_lr * model.dean.scale_lr,
                "weight_decay": 0.0,
            },
            {
                "params": list(model.dean.gating_layer.parameters()),
                "lr": base_lr * model.dean.gate_lr,
                "weight_decay": 0.0,
            },
        ])

    return optim.AdamW(param_groups)


class opitmizar:
    """
    Backward-compatible wrapper for DAIN optimizer instantiation.
    """
    def __init__(
        self,
        net: nn.Module,
        learning_rate: float,
        lambda_l2: float,
        base_lr: float = 1e-4,
    ):
        self.net = net
        self.learning_rate = learning_rate
        self.lambda_l2 = lambda_l2
        self.base_lr = base_lr
        self.optimizer = build_optimizer(
            model=self.net,
            lr=self.learning_rate,
            weight_decay=self.lambda_l2,
            base_lr=self.base_lr,
        )

    def get_optimizer(self) -> optim.AdamW:
        return self.optimizer


# Proper alias for clean naming
Optimizer = opitmizar
```

#### 3.2.2 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)
```python
# Import from Common
from Brain.Common.optimization import build_optimizer

# Remove local lines 43-95:
# def build_optimizer(...)
```

---

## 4. Validation & Verification Plan

### 4.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST adhere to repository environment standards using the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python`
> Relative path: `../bin/python`

```bash
# Verification command:
../bin/python -c "
import torch
import torch.nn as nn
from Brain.Common.optimization import build_optimizer, opitmizar, Optimizer
from PPO2_rl_train import build_optimizer as ppo_build_optimizer

print('Imports successful!')
"
```

### 4.2 Acceptance Verification Checklist
- [x] **1. Functional Correctness**:
  - [x] `build_optimizer` successfully separates `dean` layers with 0 weight decay and custom learning rates.
  - [x] `norm`, `bias`, and `log_std` parameters are placed in `weight_decay = 0.0` group.
  - [x] Regular weight parameters are placed in `weight_decay = 1e-4` group.
- [x] **2. Backward Compatibility**:
  - [x] `opitmizar(net, learning_rate, lambda_l2).get_optimizer()` behaves identically to previous implementation.
  - [x] `HopeModel_traingloop.py` imports and instantiates `opitmizar` without regression.
- [x] **3. Code Cleanliness**:
  - [x] Duplicate definition in `PPO2_rl_train.py` is fully removed.
  - [x] Code compiles and executes without warnings or errors.
