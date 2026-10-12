# Specification: Fix PPO2 DAIN Portfolio Feature Leakage & Normalize Trade Bar Scale

- **Category**: `bugfix`
- **Spec Path**: `spec/bugfix/fix_ppo2_dain_portfolio_feature_leakage.md`
- **Status**: `Completed`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope & Architecture Boundaries](#2-target-scope--architecture-boundaries)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
During the reinforcement learning setup in [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py), the Actor-Critic model ([`MambaContinuousActorCritic`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L257-L365)) feeds all observation state dimensions (17 dims) directly into a Deep Adaptive Input Normalization layer ([`DAIN_Layer`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/dain.py#L7-L101)). The observation tensor consists of 14 sequential market features and 3 account/portfolio state features (`position`, `unrealized_return`, and `trade_bar`).

Just like the historical defect analyzed in DQN ([`spec/bugfix/fix_dain_portfolio_feature_leakage.md`](file:///home/b0457812963/Mamba3RL/SynapseX/spec/bugfix/fix_dain_portfolio_feature_leakage.md)), the 3 portfolio features are identical scalars broadcasted across all 300 sequence steps. DAIN's sequence-wise adaptive centering computes the temporal mean along the sequence axis and subtracts it:
$$x \leftarrow x - \text{Linear}(\mu), \quad \text{where } \mu = \text{mean}(x, \text{dim}=2)$$
Because the portfolio features are constant across the sequence axis, $x - \mu \approx c - c = 0$, completely wiping out the position state, floating return, and holding time signals. Consequently, the agent is blinded to its current position and profit/loss, preventing it from learning rational stop-loss, profit-taking, or rebalancing strategies.

Furthermore, in [`Brain/PPO2/lib/environment.py:L244`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L244), `trade_bar` is stored as an unscaled integer counter ($0 \dots 300+$). Once the portfolio features bypass DAIN, having an unscaled $O(10^2)$ feature enter the embedding layer directly creates massive numerical scale imbalance against returns ($O(10^{-2})$).

### 1.2 Current vs. Expected Behavior

| Dimension | Current Behavior | Expected Behavior |
| :--- | :--- | :--- |
| **`trade_bar` Scale** | Raw integer counter ($0 \dots 300+$) broadcasted into observation tensor. | Normalized by $48.0$ (`float(self.trade_bar) / 48.0`, representing holding duration in days where 1 day = 48 30m bars), keeping values balanced in $O(1)$ scale. |
| **DAIN Feature Scope** | All 17 dimensions (including account status) are processed through `DAIN_Layer`. | Only the 14 sequential market features enter `DAIN_Layer`. The 3 portfolio features bypass DAIN intact. |
| **Signal Retention** | `position`, `unrealized_return`, and `trade_bar` are centered to $\approx 0$ by DAIN temporal mean subtraction. | Intact signals preserved; the continuous Actor-Critic model can accurately observe current position weight, PnL, and holding duration. |
| **Feature Fusion** | `DAIN_Layer(input_dim=17)` outputs normalized tensor directly into `market_embedding`. | `DAIN_Layer(input_dim=14)` normalizes market features; concatenated with raw portfolio features `[B, L, 14 + 3]` before `market_embedding(17 -> hidden_size)`. |

### 1.3 Technical Root Cause & References
1. **Temporal Mean Wipeout in DAIN**: In [`Brain/Common/dain.py:L74-L80`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/dain.py#L74-L80), `avg = torch.mean(x, 2)` computes the mean along the sequence dimension. When a feature is constant across time, subtracting `adaptive_avg` collapses the signal to zero.
2. **DAIN Input Dimension in PPO2 Model**: In [`Brain/PPO2/lib/model.py:L287`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L287), `self.dean = DAIN_Layer(mode=mode, input_dim=d_model)` sets `input_dim=17`, and in [`Brain/PPO2/lib/model.py:L346-349`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L346-L349), the entire `states` tensor is passed through `self.dean`.
3. **Unscaled `trade_bar` in Observation**: In [`Brain/PPO2/lib/environment.py:L244`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L244), `data_res[:, len(self.info_list) + 2] = float(self.trade_bar)` uses the raw bar count instead of normalizing by 48.0.

---

## 2. Target Scope & Architecture Boundaries

### 2.1 File & Module Scope Matrix

| Index | Target File & Line | Scope | Planned Action |
| :---: | :--- | :---: | :--- |
| **Item 1** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L244) | **In-Scope** | In `State_time_step.encode()`, scale the third portfolio feature `trade_bar` by `48.0` (`float(self.trade_bar) / 48.0`). Internal counter `self.trade_bar` remains intact. |
| **Item 2** | [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L268-L350) | **In-Scope** | In `MambaContinuousActorCritic`: decouple market features (14 dims) and portfolio features (3 dims). Initialize `self.dean` with `input_dim=market_dim`. Forward passes market through DAIN and concatenates with portfolio features before `market_embedding`. |
| **Item 3** | [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L9-L49) | **In-Scope** | Add `portfolio_dim: int = 3` parameter in `PPO2Agent.__init__` and pass it to `MambaContinuousActorCritic`. |
| **Item 4** | [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L187-L198) | **In-Scope** | In `PPO2._prepare_agent()`, pass `portfolio_dim=getattr(cfg_training, "PORTFOLIO_DIM", 3)` to `PPO2Agent`. Ensure optimizer continues to track DAIN parameters without disruption. |
| **Item 5** | [`tests/test_ppo2_dain_decoupling.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_ppo2_dain_decoupling.py) | **In-Scope** | [NEW] Create unit test suite to verify signal preservation, numerical scale, and gradient flow. |
| **Item 6** | [`Brain/Common/dain.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/dain.py) | **Out-of-Scope** | Preserved intact. Generic DAIN normalization logic remains untouched. |
| **Item 7** | [`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py) | **Out-of-Scope** | Preserved intact. RolloutBuffer and GAE computation are agnostic to feature dimension slicing. |

### 2.2 Architectural Boundaries & Non-Goals
- **In-Scope Boundaries**:
  - Observation tensor structure and total dimensions remain `[B, L, 17]` where market features = 14 and portfolio features = 3.
  - Gym Dict observation space signatures (`spaces.Dict({"states": Box(...), "time_states": Box(...)})`) remain strictly identical.
  - Action space `Box(-1.0, 1.0, (1,))` and continuous Actor-Critic heads remain unchanged.
- **Out-of-Scope (Non-Goals)**:
  - No changes to DQN models or environments.
  - No changes to SSM/Mamba core layers (`MixerModel`).

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Normalize `trade_bar` in `Brain/PPO2/lib/environment.py`**
  - [x] Modify `State_time_step.encode()` to encode `float(self.trade_bar) / 48.0`.
  - [x] Keep `self.trade_bar` integer counter intact for transition rules and state accounting.
- [x] **Step 2: Decouple DAIN Input in `Brain/PPO2/lib/model.py`**
  - [x] In `MambaContinuousActorCritic.__init__`, add `portfolio_dim: int = 3`.
  - [x] Set `self.portfolio_dim = portfolio_dim` and `self.market_dim = d_model - self.portfolio_dim` (17 - 3 = 14).
  - [x] Initialize `self.dean = DAIN_Layer(mode=mode, input_dim=self.market_dim)`.
  - [x] Keep `self.market_embedding = nn.Linear(d_model, hidden_size)` (17 to `hidden_size`).
  - [x] In `MambaContinuousActorCritic.forward()`:
    - [x] Slice `market_raw = states[:, :, :self.market_dim].transpose(1, 2)`.
    - [x] Compute `market_norm = self.dean(market_raw).transpose(1, 2)`.
    - [x] Slice `portfolio_raw = states[:, :, self.market_dim:]`.
    - [x] Concatenate `combined_states = torch.cat([market_norm, portfolio_raw], dim=-1)`.
    - [x] Pass `market_emb = self.market_embedding(combined_states)`.
- [x] **Step 3: Update `Brain/PPO2/lib/Agent.py` & `PPO2_rl_train.py`**
  - [x] In `PPO2Agent.__init__`, accept `portfolio_dim: int = 3` and forward it to `MambaContinuousActorCritic`.
  - [x] In `PPO2_rl_train.py:_prepare_agent()`, pass `portfolio_dim=getattr(cfg_training, "PORTFOLIO_DIM", 3)`.
- [x] **Step 4: Clean up & Defensiveness**
  - [x] Add assertion in `MambaContinuousActorCritic.__init__` verifying `d_model > self.portfolio_dim`.
  - [x] Verify `build_optimizer()` in `PPO2_rl_train.py` attaches `mean_layer`, `scaling_layer`, and `gating_layer` correctly.
- [x] **Step 5: Create Unit Test Suite & Execute Verification**
  - [x] Create `tests/test_ppo2_dain_decoupling.py`.
  - [x] Run full test suite and confirm 100% pass rate.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L235-L248)
```python
        # 特徵矩陣部位狀態與浮動盈虧編碼
        data_res[:, len(self.info_list)] = self.position

        if abs(self.position) > 0 and self.open_price > 0:
            prev_close = float(self._prices.close[self._offset - 1])
            unrealized_return = (
                self.position
                * ((prev_close - self.open_price) / max(self.open_price, 1e-8))
            )
            data_res[:, len(self.info_list) + 1] = unrealized_return
            # 特徵 3: 持倉時間累計 (以日為基準標準化: 48 根 30m K 棒為 1 天)
            data_res[:, len(self.info_list) + 2] = float(self.trade_bar) / 48.0
        else:
            data_res[:, len(self.info_list) + 1] = 0.0
            data_res[:, len(self.info_list) + 2] = 0.0
```

#### 3.2.2 [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L268-L355)
```python
    def __init__(
        self,
        d_model: int,
        time_features_in: int,
        action_dim: int = 1,
        seq_dim: int = 300,
        hidden_size: int = 96,
        nlayers: int = 2,
        time_features_out: int = 32,
        dropout: float = 0.1,
        mode: str = "full",
        init_log_std: float = -0.5,
        portfolio_dim: int = 3,
        ssm_cfg: Optional[dict] = None,
        moe_cfg: Optional[dict] = None,
    ):
        super().__init__()
        assert d_model > portfolio_dim, f"d_model ({d_model}) must be greater than portfolio_dim ({portfolio_dim})"
        self.portfolio_dim = portfolio_dim
        self.market_dim = d_model - self.portfolio_dim

        self.time_embedding = SineActivation(
            in_features=time_features_in, out_features=time_features_out
        )
        self.dean = DAIN_Layer(mode=mode, input_dim=self.market_dim)  # DAIN 只處理市場數據 (14 維)
        self.market_embedding = nn.Linear(d_model, hidden_size)       # 接收拼接後的 17 維特徵
        self.time_emb_projection = nn.Linear(time_features_out, hidden_size)
...

    def forward(self, states: torch.Tensor, time_states: torch.Tensor):
        # 1. 提取時間特徵與市場特徵
        time_emb = self.time_embedding(time_states)
        time_proj = self.time_emb_projection(time_emb)

        # 市場數據流 (僅前 market_dim 維進入 DAIN 歸一化)
        market_raw = states[:, :, :self.market_dim].transpose(1, 2)
        market_norm = self.dean(market_raw).transpose(1, 2)

        # 帳戶/投資組合特徵 (後 portfolio_dim 維繞過 DAIN，保持真實數值信號)
        portfolio_raw = states[:, :, self.market_dim:]
        combined_states = torch.cat([market_norm, portfolio_raw], dim=-1)
        market_emb = self.market_embedding(combined_states)
```

#### 3.2.3 [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L9-L49)
```python
    def __init__(
        self,
        ob_space: gym.Space,
        ac_space: gym.Space,
        device: torch.device,
        seq_dim: int = 300,
        hidden_size: int = 96,
        nlayers: int = 2,
        time_features_out: int = 32,
        dropout: float = 0.1,
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
            dropout=dropout,
            mode=mode,
            init_log_std=init_log_std,
            portfolio_dim=portfolio_dim,
            ssm_cfg=ssm_cfg,
            moe_cfg=moe_cfg,
        ).to(self.device)
```

---

## 4. Validation & Verification Plan

### 4.1 Verification Environment & Commands
> [!IMPORTANT]
> All executions MUST adhere to repository environment standards using the virtual environment interpreter:
> Absolute path: `/home/b0457812963/Mamba3RL/bin/python`
> Relative path: `../bin/python`

```bash
# 1. Verification of DAIN decoupling and signal preservation
../bin/python -m unittest tests/test_ppo2_dain_decoupling.py

# 2. Existing PPO2 environment test regression check
../bin/python -m unittest test_ppo2_env.py

# 3. Existing PPO2 vector training test regression check
../bin/python -m unittest test_ppo2_vector_training.py
```

### 4.2 Acceptance Verification Checklist
- [x] **1. Signal Retention Verification**:
  - [x] Feed synthetic input tensor with `position = 0.5`, `unrealized_return = 0.05`, and `trade_bar / 48.0 = 0.5`.
  - [x] Verify that after passing through the decoupled feature pipeline, portfolio feature values remain exactly $0.5$, $0.05$, and $0.5$ (neither zeroed out nor modified by DAIN).
- [x] **2. Observation Encoding Scale Verification**:
  - [x] In `State_time_step`, step through holding periods of `trade_bar = 24` and `trade_bar = 48`.
  - [x] Verify `obs["states"][:, 16]` equals `0.5` and `1.0` respectively.
  - [x] Verify internal `state.trade_bar` remains integer `24` and `48`.
- [x] **3. End-to-End Gradient Flow Verification**:
  - [x] Instantiate `MambaContinuousActorCritic`, run forward pass, compute surrogate loss, and execute `loss.backward()`.
  - [x] Ensure non-zero gradients flow through DAIN parameters (`dean.mean_layer`, etc.) and `market_embedding` without runtime errors or dimension mismatch.
- [x] **4. Optimizer Compatibility Verification**:
  - [x] Verify `build_optimizer()` in `PPO2_rl_train.py` creates 5 parameter groups (standard decay, no decay, mean_layer, scaling_layer, gating_layer) without raising exceptions.
