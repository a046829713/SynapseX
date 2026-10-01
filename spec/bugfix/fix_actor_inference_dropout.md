# Specification: Fix Non-Deterministic Dropout During Actor Batch Inference

- **Category**: `bugfix`
- **Spec Path**: `spec/bugfix/fix_actor_inference_dropout.md`
- **Status**: `Completed / Verified`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
In [`AC_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/AC_train.py), the Learner process collects batched observation states from parallel Actor processes and executes inference via `_handle_inference_batch(self)` to choose actions.

The inference block currently relies on `with torch.no_grad():`:
```python
with torch.no_grad():
    q_values, _, imagined_features = self.net(states_v, time_states_v)
```

In PyTorch:
- `torch.no_grad()` **only** disables the autograd engine (disables computation graph recording and gradient caching).
- `torch.no_grad()` **does NOT** change the module mode (`model.training` remains `True`).
- Layers like `nn.Dropout` strictly evaluate `if self.training:` to determine whether to randomly zero-out activations.

Because `self.net` is instantiated with `dropout > 0` (e.g., `0.3` or `0.05` in [`mambaDuelingModel`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/model.py#L407-L510)) and defaults to `self.training == True`, and `self.net.eval()` is never invoked, **Dropout remains fully active during batch inference**.

### 1.2 Current vs. Expected Behavior

| Dimension | Current Behavior | Expected Behavior |
| :--- | :--- | :--- |
| **Model Mode During Inference** | `self.net.training == True` throughout inference. | `self.net.eval()` activated for inference, deterministically evaluating all neurons. |
| **Dropout Execution** | Up to 30% of features randomly dropped and remaining scaled by $1/(1-p)$ on every inference step. | Dropout behaves as an identity pass-through ($y = x$); 0% neuron drop. |
| **Greedy Policy Quality** | `greedy_actions = q_values.max(...)` produces noisy, sub-optimal actions corrupted by MC-Dropout noise. | `greedy_actions` strictly yields the deterministic argmax of expected Q-values $\arg\max_a Q(s, a)$. |
| **Exploration Protocol** | Agent exploration is distorted by compounding Dropout noise on top of $\epsilon$-greedy. | Exploration is governed cleanly and solely by the controlled $\epsilon$-greedy schedule. |
| **State Restoration Safety** | None | `self.net.train()` restored deterministically via `try...finally` block. |

### 1.3 Technical Root Cause & References
1. **Unchanged Training Mode**: In [`AC_train.py:L263-L265`](file:///home/b0457812963/Mamba3RL/SynapseX/AC_train.py#L263-L265), `_handle_inference_batch` directly calls `self.net(...)` under `torch.no_grad()` without setting `self.net.eval()`.
2. **Dropout Modules in Value & Advantage Heads**: In [`Brain/DQN/lib/model.py:L454, L458, L467, L471`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/model.py#L454-L471), `self.fc_val` and `self.fc_adv` contain multiple `nn.Dropout(dropout)` layers.
3. **Dropout Modules in Backbone MLP**: In [`Brain/Common/ssm_tool.py:L138, L144, L148`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/ssm_tool.py#L138-L148), `GatedMLP` inside `MixerModel` also incorporates `nn.Dropout(dropout)`.

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix

| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`AC_train.py:L263-L266`](file:///home/b0457812963/Mamba3RL/SynapseX/AC_train.py#L263-L266) | **In-Scope** | Wrap forward inference with `self.net.eval()` and ensure restoration to `self.net.train()` using `try...finally`. |
| **Item 2** | [`tests/test_inference_eval_mode.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_inference_eval_mode.py) | **In-Scope (Test)** | Create unit test verifying deterministic outputs under `eval()` and state recovery under normal & exception scenarios. |
| **Item 3** | [`Brain/DQN/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/model.py) | **Out-of-Scope** | Preserved without modifications. |
| **Item 4** | [`Brain/DQN/lib/common.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/common.py) | **Out-of-Scope** | Preserved without modifications. |

### 2.2 Architectural Boundaries & Non-Goals
- **In-Scope Boundaries**:
  - Guaranteeing `self.net` is in evaluation mode (`self.training == False`) when serving batch inference requests in `_handle_inference_batch`.
  - Guaranteeing `self.net` is in training mode (`self.training == True`) when computing gradients in the main training loop (`loss_v.backward()`).
  - Protecting against state leakage via `try...finally`.
- **Out-of-Scope (Non-Goals)**:
  - Modifying model architectures, layer definitions, or hyperparameters.
  - Altering `TargetNet` synchronization or loss computation in `calc_loss`.

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Update `_handle_inference_batch` in `AC_train.py`**
  - [x] Switch `self.net.eval()` immediately before the inference block.
  - [x] Execute `with torch.no_grad():` forward pass.
  - [x] Wrap in a `try...finally` structure ensuring `self.net.train()` is always called before exiting or continuing.
- [x] **Step 2: Implement Unit Verification in `tests/test_inference_eval_mode.py`**
  - [x] Create mock/minimal model with `nn.Dropout(0.5)`.
  - [x] Test that inference without `eval()` yields stochastic non-identical outputs.
  - [x] Test that the `eval()` + `try...finally` pattern yields 100% deterministic identical outputs ($\Delta = 0.0$).
  - [x] Test that model's `training` attribute is restored to `True` even if an exception occurs during inference.
- [x] **Step 3: Verification Execution**
  - [x] Execute test suite using `../bin/python -m unittest tests/test_inference_eval_mode.py`.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 [`AC_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/AC_train.py#L263-L266)

```python
        # 1. 切換為評估模式（關閉 Dropout，確保推理 Q 值確定性）
        self.net.eval()
        try:
            with torch.no_grad():
                q_values, _, imagined_features = self.net(states_v, time_states_v)
        finally:
            # 2. 確保一定切回訓練模式，保證後續 backward 與訓練正常
            self.net.train()
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
../bin/python -m unittest tests/test_inference_eval_mode.py
```

### 4.2 Acceptance Verification Checklist
- [x] **1. Inference Determinism Verification**:
  - Two consecutive forward passes on identical input batch in `eval()` mode produce identical Q-values ($\max |q_1 - q_2| == 0.0$).
  - Contrast with `train()` mode which exhibits stochastic variance ($\max |q_1 - q_2| > 0$).
- [x] **2. Training Mode Restoration Verification**:
  - `net.training` is `True` before `_handle_inference_batch`.
  - `net.training` is `True` after `_handle_inference_batch` finishes.
  - `net.training` is `True` even if an exception occurs during the forward pass.
- [x] **3. Zero Regressions**:
  - `AC_train.py` compiles and runs cleanly without syntax errors or signature changes.
