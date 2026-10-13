# Specification: Remove Dropout from PPO2 Configuration and Architecture

- **Category**: `bugfix`
- **Spec Path**: `spec/bugfix/remove_ppo2_dropout.md`
- **Status**: `Completed / Verified`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
During reinforcement learning training with PPO in [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py), an asymmetry exists between the experience rollout phase (sampling) and the policy update phase (learning).

1. In the **rollout phase** ([`PPO2_rl_train.py:L247`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L247)):
   [`PPO2Agent.get_action`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L70) switches the neural network to `self.model.eval()`. Under `eval()` mode, all Dropout layers are disabled. The agent samples actions from $\mathcal{N}(\mu_{\text{eval}}, \sigma)$ and stores $\text{old\_log\_prob} = \log \pi_{\text{eval}}(a_t \mid s_t)$ and $V_{\text{eval}}(s_t)$ into [`RolloutBuffer`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py).
2. In the **update phase** ([`PPO2_rl_train.py:L168`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L168)):
   [`PPO2Agent.evaluate_actions`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L98) switches the neural network to `self.model.train()`. Under `train()` mode, Dropout layers randomly zero out 10% of hidden activations and scale remaining features by $1 / (1 - 0.1)$. The network computes $\mu_{\text{train}}$ and $\log \pi_{\text{train}}(a_t \mid s_t)$.

### 1.2 Current vs. Expected Behavior

| Dimension | Current Behavior | Expected Behavior |
| :--- | :--- | :--- |
| **Rollout vs Update Invariance** | `get_action` uses `eval()` (Dropout 0%); `evaluate_actions` uses `train()` (Dropout 10%). | Identical deterministic forward pass in both phases; Dropout removed (0%). |
| **Initial Importance Sampling Ratio** | At epoch 0 (weights unchanged), $r_t = \frac{\pi_{\text{train}}}{\pi_{\text{eval}}} \neq 1.0$ due to random neuron drop. Triggers premature clipping. | At epoch 0, $r_t \equiv 1.0$ ($KL \equiv 0.0$), adhering strictly to PPO theoretical foundations. |
| **Gaussian Log-Prob Stability** | $\log \pi(a \mid s) \propto -\frac{1}{2}\frac{(a - \mu)^2}{\sigma^2}$. Dropout induces $\Delta \mu$, which is magnified quadratically by small $\sigma$, creating severe numerical instability. | $\mu(s)$ is deterministic given state $s$ and parameters $\theta$, eliminating artificial policy noise. |
| **Critic Target Alignment** | Critic updates fit $V_{\text{train}}(s)$ against GAE returns computed from $V_{\text{eval}}(s)$, introducing high gradient variance. | Critic predictions and targets operate under the exact same feature representations. |

### 1.3 Technical Root Cause & References
1. **Configuration**: [`configs/training/ppo_v2.yaml:L31`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L31) specifies `DROPOUT: 0.1`.
2. **Actor & Critic Heads**: [`Brain/PPO2/lib/model.py:L321, L325, L334, L338`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L321) instantiates `nn.Dropout(dropout)` in `self.fc_val` and `self.fc_actor`.
3. **SSM Backbone**: [`Brain/PPO2/lib/model.py:L309`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L309) passes `dropout=dropout` into [`MixerModel`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/ssm_tool.py#L309).
4. **Agent & Train Scripts**: [`Brain/PPO2/lib/Agent.py:L19, L44`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L19) and [`PPO2_rl_train.py:L124`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L124) wire the `dropout` parameter through constructors.

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix

| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`configs/training/ppo_v2.yaml:L31`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L31) | **In-Scope** | Remove `DROPOUT: 0.1` entry. |
| **Item 2** | [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) | **In-Scope** | Remove `dropout` parameter from `MambaContinuousActorCritic`, remove `nn.Dropout` from `fc_val` and `fc_actor`, pass `dropout=None` to `MixerModel`. |
| **Item 3** | [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py) | **In-Scope** | Remove `dropout` parameter from `PPO2Agent.__init__` and constructor call. |
| **Item 4** | [`PPO2_rl_train.py:L124`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L124) | **In-Scope** | Remove `dropout=self.cfg_training.DROPOUT` in `_prepare_agent()`. |
| **Item 5** | [`Brain/Common/ssm_tool.py:L143, L147`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/ssm_tool.py#L143) | **In-Scope** | Fix `GatedMLP` check from `if hasattr(self, 'dropout'):` to `if self.dropout is not None:` to prevent `TypeError` when `dropout=None`. |
| **Item 6** | [`test_ppo2_dropout_symmetry.py`](file:///home/b0457812963/Mamba3RL/SynapseX/test_ppo2_dropout_symmetry.py) | **In-Scope (Test)** | Create unit test verifying numerical identity between `train()` and `eval()` outputs ($r_t \equiv 1.0, KL \equiv 0.0$). |
| **Item 7** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) | **Out-of-Scope** | Preserved without modifications. |
| **Item 8** | [`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py) | **Out-of-Scope** | Preserved without modifications. |

### 2.2 Architectural Boundaries & Non-Goals
- **In-Scope Boundaries**:
  - Eliminating dropout completely from the PPO2 continuous actor-critic pipeline.
  - Ensuring exact numerical equivalence between rollout inference and batch training for identical inputs and actions.
- **Out-of-Scope (Non-Goals)**:
  - Do not modify other reinforcement learning algorithms (DQN, A2C, A3C, AC).
  - Do not alter the observation normalization pipeline (`DAIN_Layer`) or Mamba state space hyperparameters.

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Update Configuration [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml)**
  - [x] Remove `DROPOUT: 0.1` line.
- [x] **Step 2: Refactor [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py)**
  - [x] Remove `dropout: float = 0.1` from `MambaContinuousActorCritic.__init__`.
  - [x] Remove `dropout=dropout` in `MixerModel` call (pass `dropout=None` or omit).
  - [x] Remove 2 `nn.Dropout(dropout)` layers from `self.fc_val`.
  - [x] Remove 2 `nn.Dropout(dropout)` layers from `self.fc_actor`.
- [x] **Step 3: Refactor [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py)**
  - [x] Remove `dropout: float = 0.1` parameter from `PPO2Agent.__init__`.
  - [x] Remove `dropout=dropout` argument from `MambaContinuousActorCritic` instantiation.
- [x] **Step 4: Update [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)**
  - [x] Remove `dropout=self.cfg_training.DROPOUT` in `_prepare_agent()`.
- [x] **Step 5: Create Unit Test & Validate Symmetry**
  - [x] Create `test_ppo2_dropout_symmetry.py`.
  - [x] Verify `MambaContinuousActorCritic` forward output is strictly identical between `eval()` and `train()`.
  - [x] Verify `PPO2Agent.evaluate_actions` matches `get_action` outputs with $r_t = 1.0$ and $KL = 0.0$.
  - [x] Execute test using `/home/b0457812963/Mamba3RL/bin/python`.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L27-L33)
```yaml
# Model Architecture (Mamba2 Continuous Actor-Critic)
HIDDEN_SIZE: 96
NLAYERS: 2
TIME_FEATURES_OUT: 32
PORTFOLIO_DIM: 3
```

#### 3.2.2 [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L317-L342)
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

        # Actor 頭 (輸出連續動作均值 mu，經 Tanh 限制在 [-1.0, 1.0])
        self.fc_actor = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
```

#### 3.2.3 [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L9-L51)
```python
class PPO2Agent:
    def __init__(
        self,
        ob_space: gym.Space,
        ac_space: gym.Space,
        device: torch.device,
        seq_dim: int = 300,
        hidden_size: int = 96,
        nlayers: int = 2,
        time_features_out: int = 32,
        mode: str = "full",
        init_log_std: float = -0.5,
        portfolio_dim: int = 3,
        ssm_cfg: Optional[dict] = None,
        moe_cfg: Optional[dict] = None,
    ):
        ...
        self.model = MambaContinuousActorCritic(
            d_model=d_model,
            time_features_in=time_features_in,
            action_dim=action_dim,
            seq_dim=seq_dim,
            hidden_size=hidden_size,
            nlayers=nlayers,
            time_features_out=time_features_out,
            mode=mode,
            init_log_std=init_log_std,
            portfolio_dim=portfolio_dim,
            ssm_cfg=ssm_cfg,
            moe_cfg=moe_cfg,
        ).to(self.device)
```

#### 3.2.4 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L116-L127)
```python
        self.agent = PPO2Agent(
            ob_space=self.vec_env.observation_space,
            ac_space=self.vec_env.action_space,
            device=self.device,
            seq_dim=self.cfg_training.BARS_COUNT,
            hidden_size=self.cfg_training.HIDDEN_SIZE,
            nlayers=self.cfg_training.NLAYERS,
            time_features_out=self.cfg_training.TIME_FEATURES_OUT,
            portfolio_dim=self.cfg_training.PORTFOLIO_DIM,
            ssm_cfg=ssm_cfg,
        )
```

---

## 4. Validation & Verification Plan

### 4.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST adhere to repository environment standards using the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python`
> Relative path: `../bin/python`

```bash
# Execute unit verification test
/home/b0457812963/Mamba3RL/bin/python test_ppo2_dropout_symmetry.py
```

### 4.2 Acceptance Verification Checklist
- [x] **1. Model Symmetry Check**:
  - [x] `mu`, `std`, and `value` output from `MambaContinuousActorCritic` under `eval()` mode and `train()` mode for the exact same input batch must match with $\text{max absolute difference} < 1e-6$.
- [x] **2. Importance Sampling Symmetry Check**:
  - [x] For actions generated by `get_action`, evaluating them via `evaluate_actions` must yield $r_t = \exp(\log \pi_{\text{train}} - \log \pi_{\text{eval}}) = 1.0 \pm 1e-6$ and approximate $KL = 0.0 \pm 1e-6$.
- [x] **3. PPO2 Initialization Check**:
  - [x] Hydra configuration parsing in `PPO2_rl_train.py` executes without missing key exceptions after removing `DROPOUT`.
