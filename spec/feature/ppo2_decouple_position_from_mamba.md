# 規格文件 (Specification): PPO2 架構消融實驗第三步 — 解耦 `position` 倉位特徵不進入 Mamba2 骨幹

- **類別 (Category)**: `feature`
- **規格路徑 (Spec Path)**: `spec/feature/ppo2_decouple_position_from_mamba.md`
- **狀態 (Status)**: `Implemented & Verified`

## 📋 目錄 (Index / Contents)
1. [1. 背景與根本原因分析 (Background & Root Cause Analysis)](#1-背景與根本原因分析-background--root-cause-analysis)
2. [2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)](#2-改動範圍與架構邊界-target-scope--architecture-boundaries)
3. [3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)](#3-實作查核清單與詳細藍圖-implementation-checklist--detailed-blueprint)
4. [4. 驗證與測試計畫 (Validation & Verification Plan)](#4-驗證與測試計畫-validation--verification-plan)

---

## 1. 背景與根本原因分析 (Background & Root Cause Analysis)

### 1.1 問題陳述與情境脈絡 (Problem Statement & Context)
在完成前兩步單變因消融（[`spec/feature/ppo2_remove_vol_regime_abs.md`](file:///home/b0457812963/Mamba3RL/SynapseX/spec/feature/ppo2_remove_vol_regime_abs.md) 移除 `vol_regime_abs`、[`spec/feature/ppo2_remove_time_states.md`](file:///home/b0457812963/Mamba3RL/SynapseX/spec/feature/ppo2_remove_time_states.md) 移除 `time_states` 與 Time2Vec Gate 融合層）後，純 13 維市場量價 + 1 維倉位之 PPO2 模型在 2026 OOS 197 檔完全未見新標的上首次出現過關模型（`checkpoint_iter_9000`、`checkpoint_iter_7000`，`PBO = 0.0%`），且於外部回測 `DEADZONE_THRESHOLD = 0.15` 驗證下表現進一步改善（`iter_9000` 116 檔已輸出結果：`IQM Sharpe = +0.4597`、`IQM IR 2x = +0.3700`、`Beat B&H = 66.38%`）。

然而，對 [`Brain/PPO2/Meta/checkpoint_iter_9000.pt`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/Meta/checkpoint_iter_9000.pt) 進行特徵歸因審計（[`results/feature_attribution/20261128-102942.json`](file:///home/b0457812963/Mamba3RL/SynapseX/results/feature_attribution/20261128-102942.json)）、倉位遲滯實驗（Position Hysteresis Audit）與逐筆交易持倉分箱拆解後，發現一個明確的**架構層面殘留缺陷**正在放大深套死扛與單幣 MDD：

1. **`position` 單一通道支配 Actor 輸出，能量壓過全部 13 維市場特徵總和**：
   - 在 [`results/feature_attribution/20261128-102942.json`](file:///home/b0457812963/Mamba3RL/SynapseX/results/feature_attribution/20261128-102942.json) 的群組置換擾動中，`portfolio_all`（僅 1 維 `position`）造成 **`Perm_Δμ = 4.2987`**，高達全部 13 維市場特徵合計 `market_all`（**`Perm_Δμ = 2.6489`**）的 **`1.62×`**。
   - 在 Buy vs. Cash 條件對照中，買入樣本 `position` 均值高達 **`0.983`**，空手樣本均值僅 **`0.225`**。
2. **極端持倉遲滯（Position Hysteresis）與 `>96 bar` 深套死扛**：
   - 以相同 2026 OOS 市場視窗固定量價特徵、僅將 `position` 由 `0.0` 切換為 `1.0` 實測：`checkpoint_iter_9000.pt` 在 `pos = 0` 時確定性買進率僅 **`1.5%`**（`mu_med = -7.64`），但切換為 `pos = 1` 後續抱率暴衝至 **`77.5%`**（`mu_med = +1.52`）；`iter_9400` 與 `iter_10800` 甚至高達 **`100.0%`**。
   - 對應到實盤回測逐筆明細（`checkpoint_iter_9000`、`DEADZONE = 0.15`、116 檔共 656 筆交易）：
     - `6–24 bar`（73 筆）：勝率 **`94.52%`**、平均報酬 **`+3.51%`**、加總 **`+255.90%`**、P95 單筆 MDD 僅 **`2.92%`**；
     - `25–96 bar`（255 筆）：勝率 **`86.67%`**、平均報酬 **`+4.38%`**、加總 **`+1,117.68%`**、P95 單筆 MDD **`12.72%`**；
     - **`>96 bar`（272 筆）**：勝率斷崖崩落至 **`39.34%`**、平均報酬 **`-2.96%`**、總虧損 **`-805.96%`**（吞噬 `6–96 bar` 總獲利 `+1,373.6%` 的 **59%**），P95 單筆 MDD 飆升至 **`31.25%`**。
3. **架構根本原因：標量 `position` 被複製 300 步灌入 Mamba2 SSM 骨幹，造成雙重注入與直流偏置劫持**：
   - [`State_time_step.encode()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L206-L221) 將當前標量 `self.position` 廣播填滿整個序列欄位 `data_res[:, 13]`（連續 300 根 K 棒皆為同一常數，並非歷史倉位軌跡）。
   - 然而 [`MambaContinuousActorCritic.__init__()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L344) 仍以完整維度 `d_model = 14` 建立 `self.market_embedding = nn.Linear(d_model, hidden_size)`，並在 [`forward()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L416-L427) 將含 `position` 的完整 `states` 送入 4 層 Mamba2 SSM（`self.mixer`），之後又於 [`L425-L427`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L425-L427) 再次將 `states[:, -1, self.market_dim:]` 送入 `self.portfolio_encoder` 做晚融合。
   - Mamba2 狀態空間模型（SSM）本質上會對長序列（$L = 300$）輸入做指數遞迴積分，連續 300 步恆定常數輸入（`position = 1.0`）會直接對整個時序隱藏狀態注入強烈 DC 偏置，污染純市場表徵學習並過度放大持倉鎖定慣性。

為維持嚴謹的**單變因消融實驗（Controlled Single-Variable Ablation，方案 B）**，本規格第三步**僅將 `position` 從 `market_embedding -> Mamba2` 時序骨幹中徹底解耦**，讓 Mamba2 只處理純粹的 13 維市場量價序列，而 `position` 僅透過既有的晚融合 `portfolio_encoder` 進入 Actor-Critic 決策幹道；其餘市場特徵、觀測空間形狀、獎勵函數、訓練超參數與摩擦退火排程**全數維持不變**。

### 1.2 當前行為 vs. 預期行為 (Current vs. Expected Behavior)
| 維度 | 當前行為 (Current Behavior) | 預期行為 (Expected Behavior) |
| :--- | :--- | :--- |
| **`market_embedding` 輸入維度** | `nn.Linear(d_model, hidden_size)` = `Linear(14, 256)`，同時接收 13 維市場特徵 + 300 步重複的 `position` 常數 | **`nn.Linear(self.market_dim, hidden_size)`** = **`Linear(13, 256)`**，嚴格僅接收純市場量價特徵 `states[:, :, :self.market_dim]` |
| **資料流與數學不變性** | `position` 同時走「Mamba2 時序主幹」與「`portfolio_encoder` 晚融合」雙路徑；改變 `position` 會改寫全部 300 步 Mamba2 隱藏狀態 | **嚴格雙流解耦**：`states[:, :, :market_dim] -> market_embedding -> feature_embedding -> mixer -> temporal_pool`（對 `position` 梯度恆為嚴格 `0.0`，輸出 bit-for-bit 不受 `position` 影響）；`states[:, -1, market_dim:] -> portfolio_encoder` 僅於 trunk 晚融合 |
| **總參數量 (`MambaContinuousActorCritic`)** | `4,964,899` | **`4,964,643`**（精簡 `market_embedding.weight` 的 `256 × 1 = 256` 個參數） |
| **觀測空間與環境介面** | `observation_space["states"].shape == (300, 14)`；`RolloutBuffer` 6-tuple | **100% 維持不變**（`states` 仍為 `(300, 14)`，完全無需修改環境、緩衝區、DataAdapter、Agent、Evaluator、Engine） |
| **特徵歸因診斷 (`diagnose_ppo2_feature_attribution.py`)** | 驗證 `market_embedding.weight.shape[1] == d_model (14)` 並直接由 `market_embedding` 取全部 14 欄 `weight_l2`；A2 時序顯著度混入 `position` 梯度 | 驗證 `market_embedding.weight.shape[1] == market_dim (13)` 與 `portfolio_encoder.0.weight.shape[1] == portfolio_dim (1)`；`weight_l2` 依串流拼接 `market_embedding`（前 13 維）與 `portfolio_encoder[0]`（後 1 維）；A2 時序顯著度僅計算純市場通道 `[:market_dim]` |
| **其他所有特徵 / 超參數 / 獎勵 / 排程** | 現行狀態 | **100% 維持不變**（嚴格單變因隔離） |

### 1.3 技術根本原因與程式碼參照 (Technical Root Cause & References)
- 市場嵌入層維度耦合位置：[`Brain/PPO2/lib/model.py:L333-L348`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L333-L348)（[`L344`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L344) 定義 `self.market_embedding = nn.Linear(d_model, hidden_size)`）。
- 前向傳播雙重注入位置：[`Brain/PPO2/lib/model.py:L414-L429`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L414-L429)（[`L416`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L416) 傳入完整 `states` 至 `market_embedding`，[`L425`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L425) 再抽 `states[:, -1, self.market_dim:]` 進入 `portfolio_encoder`）。
- 環境標量廣播來源（保持介面相容不動）：[`Brain/PPO2/lib/environment.py:L219`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L219)（`data_res[:, len(self.info_list)] = self.position`）。
- 訓練初始化防禦斷言位置：[`PPO2_rl_train.py:L265-L273`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L265-L273)。
- 歸因工具受影響位置：[`tests/diagnose_ppo2_feature_attribution.py:L141-L145`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/diagnose_ppo2_feature_attribution.py#L141-L145)（checkpoint 維度斷言）、[`L488-L491`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/diagnose_ppo2_feature_attribution.py#L488-L491)（`w_states_l2` 長度會從 14 變 13 導致 `IndexError`）、[`L947-L950`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/diagnose_ppo2_feature_attribution.py#L947-L950)（A2 時序顯著度切片）。

---

## 2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)

### 2.1 檔案與模組改動矩陣 (File & Module Scope Matrix)
| 編號 | 目標檔案與行號 | 範圍 (Scope) | 規劃動作 (Planned Action) |
| :---: | :--- | :---: | :--- |
| **項目 1** | [`Brain/PPO2/lib/model.py:L291-L348,L414-L429`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L333-L348) | **In-Scope** | ① 加入防禦斷言 `assert portfolio_dim >= 0`；<br>② 將 `self.market_embedding` 改為 `nn.Linear(self.market_dim, hidden_size)`；<br>③ 在 `forward()` 中以 `market_states = states[:, :, : self.market_dim]` 餵入 `self.market_embedding`，確保 `position` 徹底不進入 Mamba2 骨幹，僅經 `self.portfolio_encoder` 晚融合；<br>④ 更新類別 docstring 說明雙流解耦架構。 |
| **項目 2** | [`PPO2_rl_train.py:L265-L273`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L265-L273) | **In-Scope** | 在 `_prepare_agent()` 加入一條防禦斷言，確認 `self.agent.model.market_embedding.in_features == self.agent.model.market_dim`（防止未來誤傳入含倉位維度之投影層）。不更動任何訓練迴圈、優化器或超參數邏輯。 |
| **項目 3** | [`tests/diagnose_ppo2_feature_attribution.py:L141-L145,L487-L492,L947-L950`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/diagnose_ppo2_feature_attribution.py#L141-L145) | **In-Scope** | ① `load_model_from_checkpoint()` 改驗證 `market_embedding.weight.shape[1] == market_dim` 與 `portfolio_encoder.0.weight.shape[1] == portfolio_dim`；<br>② `compute_feature_attribution()` 改為串流對齊拼接 `w_states_l2 = concat([market_embedding.norm(dim=0), portfolio_encoder[0].norm(dim=0)])` 以避免 `IndexError` 並精確量測各串流首層權重範數；<br>③ `run_diagnostics()` 呼叫 A2 時序顯著度時僅傳入市場通道梯度 `[:, :, :market_dim]`，避免單步晚融合的 `position` 梯度扭曲 `last_16` 時序佔比。 |
| **項目 4** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)、[`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py)、[`Brain/PPO2/lib/data_adapter.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py)、[`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py)、[`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py)、[`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py)、[`PPO2_rl_test.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py)、[`Brain/Common/`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/)、[`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml)、[`tests/diagnose_ppo2_gradients.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/diagnose_ppo2_gradients.py)、`Brain/DQN/*` | **Out-of-Scope** | **明確保留 100% 不更動**：觀測空間 `(300, 14)` 與所有張量介面保持完全不變；`grad_diagnostics.py` 與 `optimization.py` 自動相容無須調整；所有超參數、獎勵函數、特徵清單與 legacy DQN 零更動。 |

### 2.2 架構邊界與非目標 (Architectural Boundaries & Non-Goals)
- **範圍內邊界 (In-Scope Boundaries)**:
  - `MambaContinuousActorCritic` 內部資料流嚴格分離：
    - 市場時序支路：`states[:, :, :market_dim]` `(B, 300, 13)` → `market_embedding Linear(13, 256)` → `feature_embedding Linear(256, 256) + GELU` → `mixer (4-layer Mamba2)` → `temporal_pool` `(B, 768)`。
    - 帳戶狀態支路：`states[:, -1, market_dim:]` `(B, 1)` → `portfolio_encoder Linear(1, 32) + LayerNorm + SiLU` `(B, 32)`。
    - 晚融合幹道：`concat([temporal_feat, portfolio_feat])` `(B, 800)` → `trunk_norm` → `fc_actor` / `fc_val`。
  - 外部 `observation_space`、動作空間、`State_time_step.encode()`、`PPO2DataAdapter.extract_latest_obs()`、`RolloutBuffer` 介面 **100% 維持不變**。
- **非目標 (Out-of-Scope / Non-Goals)**:
  - **不更動** [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml)（不改動 `LEARNING_RATE`、`ENT_COEF`、`FRICTION_ANNEAL_*`、`NUM_WORKERS`、`HORIZON_STEPS`、`W2_DOWNSIDE_RATIO` 等任何設定）。
    > [!NOTE]
    > 檢查工作區 `git diff` 發現您目前在 [`configs/training/ppo_v2.yaml:L63`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L63) 有一行未提交的本地修改（`DEADZONE_THRESHOLD: 0.03 -> 0.15`）。本規格將該設定檔列為 **Out-of-Scope**，實作時不會覆寫它；開始下一輪訓練前，請依實驗需求自行決定要保留 `0.15` 或先還原回 `0.03` 以做最嚴格的單變因對照。
  - **不移除** `ret_gap` 或其他任何 13 維市場特徵。
  - **不更動** `PPO2_rl_train.py` 的 LR 排程、Entropy 排程或 Value Loss (`v_clipped`) 計算。
  - **不新增** 單邊持倉回撤特徵或 MDD / Drawdown 懲罰項（保留至後續步驟獨立驗證）。
  - **不回溯相容** 舊耦合 14 維 `market_embedding` checkpoint（如 `checkpoint_iter_9000.pt`）：本消融實驗將從零重新訓練解耦架構模型。

---

## 3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)

### 3.1 步驟清單 (Step-by-Step Task Checklist)
- [x] **步驟 1: 修改 [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py)**
  - [x] 在 `MambaContinuousActorCritic.__init__()` 加入 `assert portfolio_dim >= 0`，並將 `self.market_embedding` 由 `nn.Linear(d_model, hidden_size)` 改為 `nn.Linear(self.market_dim, hidden_size)`。
  - [x] 在 `MambaContinuousActorCritic.forward()` 將輸入明確切片為 `market_states = states[:, :, : self.market_dim]` 再餵入 `self.market_embedding(market_states)`；維持 `current_portfolio = states[:, -1, self.market_dim :]` 經 `self.portfolio_encoder` 晚融合。
  - [x] 更新 `MambaContinuousActorCritic` docstring 明確描述「市場時序流與倉位晚融合流嚴格解耦」。
- [x] **步驟 2: 加入訓練啟動防禦斷言 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)**
  - [x] 在 `PPO2._prepare_agent()` 加入防禦性斷言，確認 `self.agent.model.market_embedding.in_features == self.agent.model.market_dim`。
- [x] **步驟 3: 對齊離線歸因診斷工具 [`tests/diagnose_ppo2_feature_attribution.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/diagnose_ppo2_feature_attribution.py)**
  - [x] 更新 `load_model_from_checkpoint()` 驗證 `market_embedding.weight.shape[1] == len(BasicState.MARKET_FEATURES)`，且於 `portfolio_dim > 0` 時驗證 `portfolio_encoder.0.weight.shape[1] == portfolio_dim`。
  - [x] 更新 `compute_feature_attribution()` 依市場流／倉位流分別由 `model.market_embedding.weight` 與 `model.portfolio_encoder[0].weight` 計算首層權重 L2 範數並串接為 `w_states_l2`（長度等於 `d_model`）。
  - [x] 更新 `run_diagnostics()` 呼叫 `compute_temporal_saliency_profile()` 時僅傳入市場特徵梯度切片 `[:, :, :ckpt_meta["market_dim"]]`，並於 `A1` 註解說明 `position` 僅透過最後一步晚融合產生梯度。
- [x] **步驟 4: 代碼防禦與清理 (Clean up & Defensiveness)**
  - [x] 驗證 `portfolio_dim == 0` 與 `portfolio_dim == 1` 邊界條件皆不觸發形狀錯誤。
  - [x] 透過 `git diff` 確認 `Brain/Common/`、`Brain/PPO2/lib/{environment,experience,data_adapter,Agent,Evaluator,engine}.py`、`configs/training/ppo_v2.yaml` 零非預期更動。

### 3.2 詳細程式碼修改藍圖 (Detailed Code Modifications Blueprint)

#### 3.2.1 [`Brain/PPO2/lib/model.py:L327-L349,L414-L432`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L327-L349)
```python
        assert portfolio_dim >= 0, (
            f"portfolio_dim ({portfolio_dim}) must be >= 0"
        )
        assert d_model > portfolio_dim, (
            f"d_model ({d_model}) must be greater than portfolio_dim ({portfolio_dim})"
        )
        self.seq_dim = seq_dim
        self.hidden_size = hidden_size
        self.portfolio_dim = portfolio_dim
        self.market_dim = d_model - self.portfolio_dim
        ...
        # Pure market stream projection (strictly excludes portfolio_dim to prevent
        # constant-position DC bias from entering the Mamba2 temporal backbone)
        self.market_embedding = nn.Linear(self.market_dim, hidden_size)
        self.feature_embedding = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )

    def forward(self, states: torch.Tensor):
        # 1. Project pure market features only (portfolio stream bypasses Mamba2 completely)
        market_states = states[:, :, : self.market_dim]
        market_emb = self.market_embedding(market_states)
        fused = self.feature_embedding(market_emb)

        # 2. Mamba2 temporal backbone (strictly position-invariant market representation)
        out, _ = self.mixer(fused)

        # 3. Multi-horizon temporal pooling + late-fusion portfolio skip connection
        temporal_feat = self.temporal_pool(out)  # [B, 3 * hidden_size]
        if self.portfolio_encoder is not None:
            current_portfolio = states[:, -1, self.market_dim :]  # [B, portfolio_dim]
            portfolio_feat = self.portfolio_encoder(current_portfolio)  # [B, portfolio_emb_dim]
            trunk = torch.cat([temporal_feat, portfolio_feat], dim=-1)
        else:
            trunk = temporal_feat
```

#### 3.2.2 [`PPO2_rl_train.py:L265-L276`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L265-L276)
```python
        assert not hasattr(
            self.agent.model, "dean"
        ), "DAIN layer should be removed from PPO2 model"
        assert not hasattr(self.agent.model, "time_embedding") and not hasattr(
            self.agent.model, "gate_layer"
        ), "Time2Vec and gate fusion layers should be removed from PPO2 model"
        assert (
            self.agent.model.market_embedding.in_features == self.agent.model.market_dim
        ), (
            f"market_embedding must project pure market_dim ({self.agent.model.market_dim}) "
            f"without portfolio leakage, got {self.agent.model.market_embedding.in_features}"
        )
```

#### 3.2.3 [`tests/diagnose_ppo2_feature_attribution.py:L141-L150,L487-L499,L947-L952`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/diagnose_ppo2_feature_attribution.py#L141-L150)
```python
    # In load_model_from_checkpoint():
    expected_market_dim = len(BasicState.MARKET_FEATURES)
    ckpt_market_dim = int(state_dict["market_embedding.weight"].shape[1])
    assert ckpt_market_dim == expected_market_dim, (
        f"Checkpoint market_embedding input dim ({ckpt_market_dim}) != expected market_dim ({expected_market_dim}). "
        f"Legacy coupled checkpoints (in_features == d_model={d_model}) are incompatible with decoupled Mamba2 backbone."
    )
    if portfolio_dim > 0:
        assert "portfolio_encoder.0.weight" in state_dict, (
            "Checkpoint state_dict missing 'portfolio_encoder.0.weight'"
        )
        ckpt_port_dim = int(state_dict["portfolio_encoder.0.weight"].shape[1])
        assert ckpt_port_dim == portfolio_dim, (
            f"Checkpoint portfolio_encoder input dim ({ckpt_port_dim}) != expected portfolio_dim ({portfolio_dim})."
        )

    # In compute_feature_attribution():
    with torch.no_grad():
        w_market_l2 = (
            model.market_embedding.weight.detach().float().norm(dim=0).cpu().numpy()
        )
        if model.portfolio_encoder is not None:
            w_port_l2 = (
                model.portfolio_encoder[0].weight.detach().float().norm(dim=0).cpu().numpy()
            )
            w_states_l2 = np.concatenate([w_market_l2, w_port_l2], axis=0)
        else:
            w_states_l2 = w_market_l2

    # In run_diagnostics():
    market_dim = int(ckpt_meta["market_dim"])
    a2 = compute_temporal_saliency_profile(
        g_mu_states=grad_pack["g_mu_states"][:, :, :market_dim],
        g_v_states=grad_pack["g_v_states"][:, :, :market_dim],
    )
```

---

## 4. 驗證與測試計畫 (Validation & Verification Plan)

### 4.1 驗證環境與執行指令 (Verification Environment & Commands)
> [!IMPORTANT]
> 依據專案規範，所有執行必須使用虛擬環境之 Python 直譯器：
> 絕對路徑: `/home/b0457812963/Mamba3RL/bin/python`
> 相對路徑: `../bin/python`

```bash
# 1. 核心架構不變性、梯度隔離、參數量、GPU 前反向與邊界條件 (portfolio_dim=0/1) 驗證：
/home/b0457812963/Mamba3RL/bin/python -c "
import torch
from omegaconf import OmegaConf
from Brain.PPO2.lib.model import MambaContinuousActorCritic
from Brain.Common.grad_diagnostics import build_param_groups, group_grad_norms
from Brain.Common.optimization import build_optimizer

cfg = OmegaConf.load('configs/training/ppo_v2.yaml')
ssm_cfg = OmegaConf.to_container(cfg.SSM_CFG, resolve=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MambaContinuousActorCritic(
    d_model=14, action_dim=1, seq_dim=300,
    hidden_size=int(cfg.HIDDEN_SIZE), nlayers=int(cfg.NLAYERS),
    portfolio_dim=1, ssm_cfg=ssm_cfg,
    init_log_std=float(cfg.INIT_LOG_STD),
    log_std_min=float(cfg.LOG_STD_MIN), log_std_max=float(cfg.LOG_STD_MAX),
).to(device)

assert model.market_embedding.in_features == 13
assert sum(p.numel() for p in model.parameters()) == 4964643

# (a) 嚴格不變性測試：相同市場視窗、不同 position (0 vs 1)，Mamba2 骨幹輸出必須 bit-for-bit 相同
model.eval()
x0 = torch.randn(4, 300, 14, device=device)
x0[:, :, 13] = 0.0
x1 = x0.clone()
x1[:, :, 13] = 1.0
with torch.no_grad():
    h0 = model.temporal_pool(model.mixer(model.feature_embedding(model.market_embedding(x0[:, :, :13])))[0])
    h1 = model.temporal_pool(model.mixer(model.feature_embedding(model.market_embedding(x1[:, :, :13])))[0])
    assert torch.max(torch.abs(h0 - h1)).item() == 0.0
    mu0, _, v0 = model(x0)
    mu1, _, v1 = model(x1)
    assert torch.max(torch.abs(mu0 - mu1)).item() > 0.0, 'Late-fusion portfolio path must still affect Actor'

# (b) 嚴格梯度隔離測試：歷史 0..298 步 position 梯度必須嚴格為 0，僅最後一步 t=299 非零
model.train()
xg = x1.detach().clone().requires_grad_(True)
mu, std, val = model(xg)
loss = mu.sum() + val.sum()
loss.backward()
assert torch.max(torch.abs(xg.grad[:, :-1, 13])).item() == 0.0, 'Historical position steps leaked into Mamba2!'
assert torch.max(torch.abs(xg.grad[:, -1, 13])).item() > 0.0, 'Current-step position gradient missing!'

# (c) grad_diagnostics & optimizer 全參數覆蓋 + 無 NaN/Inf
groups = build_param_groups(model)
gn = group_grad_norms(groups)
assert gn['total'] > 0 and gn['portfolio'] > 0 and gn['backbone'] > 0
opt = build_optimizer(model, lr=float(cfg.LEARNING_RATE), weight_decay=float(cfg.WEIGHT_DECAY))
opt.step()

# (d) 邊界條件：portfolio_dim=0
m0 = MambaContinuousActorCritic(d_model=13, action_dim=1, seq_dim=64, hidden_size=int(cfg.HIDDEN_SIZE), nlayers=1, portfolio_dim=0, ssm_cfg=ssm_cfg).to(device)
assert m0.market_embedding.in_features == 13 and m0.portfolio_encoder is None
_ = m0(torch.randn(2, 64, 13, device=device))
print('ALL MODEL DECOUPLING INVARIANTS PASSED!')
"

# 2. PPO2Engine + PPO2Evaluator 端到端煙霧測試（建構 Agent、檢查維度與推論前向）：
/home/b0457812963/Mamba3RL/bin/python -c "
import numpy as np
from Brain.PPO2.lib.engine import PPO2Engine
engine = PPO2Engine(formal=False, config='configs/training/ppo_v2.yaml')
agent = engine.build_agent()
assert agent.model.market_embedding.in_features == 13
assert agent.ob_space['states'].shape == (300, 14)
obs = {'states': np.zeros((2, 300, 14), dtype=np.float32)}
out = agent.get_action(obs, stochastic=False)
assert out.clipped_action.shape == (2, 1)
print('PPO2Engine + Agent SMOKE TEST PASSED! params =', sum(p.numel() for p in agent.model.parameters()))
"

# 3. 離線歸因診斷腳本 (diagnose_ppo2_feature_attribution.py) 端到端相容性驗證（使用暫存解耦 checkpoint）：
/home/b0457812963/Mamba3RL/bin/python -c "
import os, tempfile, torch
from omegaconf import OmegaConf
from Brain.PPO2.lib.model import MambaContinuousActorCritic
import tests.diagnose_ppo2_feature_attribution as diag

cfg = OmegaConf.load('configs/training/ppo_v2.yaml')
ssm_cfg = OmegaConf.to_container(cfg.SSM_CFG, resolve=True)
model = MambaContinuousActorCritic(
    d_model=14, action_dim=1, seq_dim=300,
    hidden_size=int(cfg.HIDDEN_SIZE), nlayers=int(cfg.NLAYERS),
    portfolio_dim=1, ssm_cfg=ssm_cfg,
)
with tempfile.TemporaryDirectory() as td:
    ckpt_p = os.path.join(td, 'decoupled_test.pt')
    out_p = os.path.join(td, 'attr.json')
    torch.save({'iteration': 0, 'total_steps': 0, 'model_state_dict': model.state_dict()}, ckpt_p)
    report = diag.run_diagnostics(
        checkpoint_path=ckpt_p, config_path='configs/training/ppo_v2.yaml',
        num_symbols=2, samples_per_symbol=4, perm_repeats=1,
        output_path=out_p, save_json=True, verbose=False,
    )
    assert len(report['A1']['states']) == 14
    assert report['checkpoint_meta']['n_parameters'] == 4964643
    assert all(row['weight_l2'] > 0.0 for row in report['A1']['states'])
    print('FEATURE ATTRIBUTION END-TO-END CHECK PASSED!')
"
```

### 4.2 驗收測試查核清單 (Acceptance Verification Checklist)
- [x] **1. 功能正確性與數學不變性 (Functional Correctness & Mathematical Invariants)**:
  - [x] `model.market_embedding.in_features == 13`（等於 `model.market_dim`），總參數量精確為 `4,964,643`。
  - [x] 相同市場視窗下切換 `position = 0.0` vs `1.0`，Mamba2 骨幹與 `temporal_pool` 輸出差異嚴格為 `0.0`，而最終 Actor/Critic 輸出透過晚融合 `portfolio_encoder` 正常響應。
  - [x] 反向傳播後，歷史時間步 `xg.grad[:, :-1, 13]` 嚴格全為 `0.0`，僅當前步 `xg.grad[:, -1, 13]` 非零。
- [x] **2. 邊界與嚴格單變因隔離 (Boundary Conditions & Single-Variable Isolation)**:
  - [x] `portfolio_dim == 0` 與 `portfolio_dim == 1` 邊界建構與前向均通過。
  - [x] `git diff` 確認僅修改 [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py)、[`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)（僅 1 條斷言）與 [`tests/diagnose_ppo2_feature_attribution.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/diagnose_ppo2_feature_attribution.py)；環境、特徵、獎勵與超參數均零非預期改動。
- [x] **3. 實際執行與 GPU 驗證 (Runtime & GPU Verification)**:
  - [x] 三項驗證指令於虛擬環境 `/home/b0457812963/Mamba3RL/bin/python` 及 GPU 上全數通過，無 NaN/Inf 或形狀例外。
  - [ ] 訓練啟動後的實證觀測指標（供訓練完成後檢核）：透過 [`tests/diagnose_ppo2_feature_attribution.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/diagnose_ppo2_feature_attribution.py) 比較 `portfolio_all / market_all` 置換敏感度比值是否由基線 `1.62×` 顯著下降，以及 `>96 bar` 死扛單比例與單幣 MDD 是否同步收斂。
