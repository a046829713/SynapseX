# Specification: Fix DAIN Portfolio Feature Leakage & Normalize Trade Bar Scale

- **Category**: `bugfix`
- **Spec Path**: `spec/bugfix/fix_dain_portfolio_feature_leakage.md`
- **Status**: `Verified / Completed`

## 📋 Index / Contents
1. [1. Background & Root Cause Analysis](#1-background--root-cause-analysis)
2. [2. Target Scope](#2-target-scope)
3. [3. Implementation Checklist & Detailed Blueprint](#3-implementation-checklist--detailed-blueprint)
4. [4. Validation & Verification Plan](#4-validation--verification-plan)

---

## 1. Background & Root Cause Analysis

### 1.1 Problem Statement & Context
In the current reinforcement learning setup ([`AC_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/AC_train.py)), the agent's Q-network ([`mambaDuelingModel`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/model.py#L407-L508)) processes all state features using a Deep Adaptive Input Normalization layer ([`DAIN_Layer`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/dain.py#L7-L101)). The total input state consists of 17 dimensions: 14 sequential market features and 3 portfolio state features (`position`, `unrealized_return`, and `trade_bar`). 

Because the portfolio features are identical scalars broadcasted across all sequence steps (300 bars), DAIN's sequence-wise adaptive centering subtracts the temporal mean ($x - \mu \approx c - c = 0$), wiping out floating return (漲幅), position state, and holding time (持倉時間). Furthermore, `trade_bar` is unscaled (ranging from 0 to 300+), causing numerical imbalance relative to returns ($O(10^{-2})$).

### 1.2 Current vs. Expected Behavior

| Dimension | Current Behavior | Expected Behavior |
| :--- | :--- | :--- |
| **`trade_bar` Scaling** | Raw integer counter ($0 \dots 300+$) broadcasted into observation tensor. | Normalized by $48.0$ (representing holding duration in units of 24h days, where 1 day = 48 30m bars), keeping values in a balanced $O(1)$ range. |
| **DAIN Feature Scope** | All 17 dimensions (including account status) are processed through `DAIN_Layer`. | Only the 14 sequential market features are passed through `DAIN_Layer`. The 3 portfolio features bypass DAIN intact. |
| **Signal Retention** | `unrealized_return`, `position`, and `trade_bar` are centered to $\approx 0$ by temporal mean subtraction. | Intact signals preserved; the agent can accurately observe current PnL, position status, and holding duration. |

### 1.3 Technical Root Cause & References
1. **Temporal Mean Wipeout in DAIN**: In [`Brain/Common/dain.py:L74-L80`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/dain.py#L74-L80), `avg = torch.mean(x, 2)` computes the mean along the sequence dimension. When a feature is constant across time, subtracting `adaptive_avg` sets the feature to zero.
2. **Comment vs. Implementation Discrepancy**: In [`Brain/DQN/lib/model.py:L428`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/model.py#L428), the inline comment explicitly notes `# DAIN 只處理市場數據`, yet `self.dean` is initialized with `input_dim=d_model` (17) and invoked on the entire `src` tensor in [`model.py:L488-L490`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/model.py#L488-L490).

---

## 2. Target Scope

### 2.1 File Scope Matrix

| Target File | Planned Action |
| :--- | :--- |
| [`Brain/DQN/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py#L299) | In `State_time_step.encode()`, scale the third portfolio feature `trade_bar` by `48.0` (`float(self.trade_bar) / 48.0`). |
| [`Brain/DQN/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/model.py#L407-L508) | In `mambaDuelingModel`: decouple market features (14 dims) and portfolio features (3 dims). Only market features enter `DAIN_Layer`. Concatenate normalized market features with raw portfolio features before embedding. |

---

## 3. Implementation Checklist & Detailed Blueprint

### 3.1 Step-by-Step Task Checklist
- [x] **Step 1: Normalize `trade_bar` in `Brain/DQN/lib/environment.py`**
  - [x] Modify `State_time_step.encode()` to encode `float(self.trade_bar) / 48.0`.
  - [x] Keep `self.trade_bar` integer counter intact for audit logging and transition rules.
- [x] **Step 2: Decouple DAIN Input in `Brain/DQN/lib/model.py`**
  - [x] In `mambaDuelingModel.__init__`, configure `market_dim = 14` (or `d_model - portfolio_dim` where `portfolio_dim = 3`).
  - [x] Initialize `self.dean = DAIN_Layer(mode=mode, input_dim=market_dim)`.
  - [x] Keep `self.market_embedding = nn.Linear(d_model, hidden_size)` (17 to `hidden_size`).
  - [x] In `mambaDuelingModel.forward`:
    - Slice `market_data = src[:, :, :self.market_dim]`.
    - Transpose and pass through `self.dean`.
    - Slice `portfolio_data = src[:, :, self.market_dim:]`.
    - Concatenate `torch.cat([market_data, portfolio_data], dim=-1)` back into `[B, L, d_model]`.
    - Pass through `self.market_embedding`.

### 3.2 Detailed Code Modifications Blueprint

#### 3.2.1 [`Brain/DQN/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py#L298-L300)
```python
            # 特徵 3: 持倉時間累計 (以日為基準標準化: 48 根 30m K 棒為 1 天)
            data_res[:, len(self.info_list) + 2] = float(self.trade_bar) / 48.0
```

#### 3.2.2 [`Brain/DQN/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/model.py#L427-L492)
```python
        # In __init__:
        self.portfolio_dim = portfolio_dim  # default: 3
        self.market_dim = d_model - self.portfolio_dim  # 17 - 3 = 14
        self.time_embedding = SineActivation(in_features=time_features_in, out_features=time_features_out)
        self.dean = DAIN_Layer(mode=mode, input_dim=self.market_dim) # DAIN 只處理市場數據 (14 維)
        self.market_embedding = nn.Linear(d_model, hidden_size)

        # In forward:
        time_emb = self.time_embedding(time_tau)
        time_emb_proj = self.time_emb_projection(time_emb) # [B, L, hidden_size]
        
        # 市場數據流 (僅前 market_dim 維進入 DAIN)
        market_raw = src[:, :, :self.market_dim].transpose(1, 2)
        market_norm = self.dean(market_raw).transpose(1, 2)
        
        # 帳戶特徵 (後 portfolio_dim 維繞過 DAIN，保持真實數值信號)
        portfolio_raw = src[:, :, self.market_dim:]
        combined_src = torch.cat([market_norm, portfolio_raw], dim=-1)
        market_emb = self.market_embedding(combined_src) # [B, L, hidden_size]
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
../bin/python -m unittest tests/test_dain_decoupling.py
```

### 4.2 Acceptance Verification Checklist
- [x] **1. Portfolio Feature Retention Verification**:
  - [x] Feed synthetic input with `unrealized_return = 0.05` and `trade_bar / 48.0 = 0.5`.
  - [x] Verify that after the DAIN decoupling step, `unrealized_return` remains exactly $0.05$ and `trade_bar` remains exactly $0.5$ (not wiped to 0).
- [x] **2. Observation Encoding Scale Verification**:
  - [x] In `State_time_step`, simulate a step with `trade_bar = 24` and `trade_bar = 48`.
  - [x] Verify `obs[0][:, 16]` equals `0.5` and `1.0` respectively.
  - [x] Verify `state.trade_bar` remains integer `24` and `48` internally.
- [x] **3. End-to-End Forward & Backward Pass**:
  - [x] Instantiate `mambaDuelingModel` and run forward and backward passes.
  - [x] Ensure gradients flow back through both `market_raw` (via DAIN) and `portfolio_raw` (via embedding) without shape mismatch or runtime errors.
