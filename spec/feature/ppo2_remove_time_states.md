# 規格文件 (Specification): PPO2 特徵消融實驗第二步 — 移除 `time_states` 時間特徵與 Time2Vec Gate 融合層

- **類別 (Category)**: `feature`
- **規格路徑 (Spec Path)**: `spec/feature/ppo2_remove_time_states.md`
- **狀態 (Status)**: `Implemented & Verified`

## 📋 目錄 (Index / Contents)
1. [1. 背景與根本原因分析 (Background & Root Cause Analysis)](#1-背景與根本原因分析-background--root-cause-analysis)
2. [2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)](#2-改動範圍與架構邊界-target-scope--architecture-boundaries)
3. [3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)](#3-實作查核清單與詳細藍圖-implementation-checklist--detailed-blueprint)
4. [4. 驗證與測試計畫 (Validation & Verification Plan)](#4-驗證與測試計畫-validation--verification-plan)

---

## 1. 背景與根本原因分析 (Background & Root Cause Analysis)

### 1.1 問題陳述與情境脈絡 (Problem Statement & Context)
在完成第一步單變因消融（[`spec/feature/ppo2_remove_vol_regime_abs.md`](file:///home/b0457812963/Mamba3RL/SynapseX/spec/feature/ppo2_remove_vol_regime_abs.md)）移除 `vol_regime_abs` 後，模型（[`checkpoint_iter_8200.pt`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/Meta/checkpoint_iter_8200.pt)）確實成功跳脫空手死區（`buy_frac`: `15.0% -> 30.6%`、`action_mean`: `0.105 -> 0.283`），且 13 維市場特徵內部的 Critic 歸因恢復高度均衡（`candle_body 0.271` vs. `ma_dev_30 0.268`，比值 `1.01×`）。

然而，最新特徵歸因報告 [`results/feature_attribution/20261124-190108.json`](file:///home/b0457812963/Mamba3RL/SynapseX/results/feature_attribution/20261124-190108.json) 與 2026 OOS 榜單 [`results/ppo2_orders/leaderboard.csv`](file:///home/b0457812963/Mamba3RL/SynapseX/results/ppo2_orders/leaderboard.csv)（46 個 checkpoint 全數 0 過關，`iter_8200` `iqm_sharpe = -0.58`、`iqm_cum_return = -7.09%`）揭露：**在 `vol_regime_abs` 移除後，6 維 `time_states`（搭配 `SineActivation` + 50/50 `gate_layer`）立刻補位成為新的過擬合捷徑**：

1. **歸因異常暴衝，壓過所有均線與量能特徵**：
   - `time_all` 群組置換擾動自基線 `Perm_Δμ = 1.20 / Perm_ΔV = 0.195` 暴增至 **`Perm_Δμ = 3.41 / Perm_ΔV = 0.316`**（已達全部 13 維市場特徵總和 `5.02 / 0.395` 的 **68% / 80%**）。
   - `hour_sin` 登上全場 Critic 第 1（`Perm_ΔV = 0.2712`、`W_L2 = 6.34`）；`dayofweek_cos/sin` 擠進全場 Actor 第 2、3 名（`Perm_Δμ = 2.086 / 1.884`）；甚至在 30 分鐘線上只是整點 `+1`/半點 `-1` 方波的 `minute_cos` 都高達 `Perm_Δμ = 1.199`，而恆為 `0` 的 `minute_sin`（`x_std = 2.83e-16`）仍白佔 `W_L2 = 5.45`。
2. **實作偏離原版 Time2Vec 與 50/50 Gate 能量劫持**：
   - [`Brain/Common/model_components.py:L20-L31`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/model_components.py#L20-L31) 的 [`SineActivation`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/model_components.py#L20-L31) 對已經 `sin/cos` 化的 6 維週期向量再套一次 `torch.sin`（雙重正弦諧波）且以未縮放 `torch.randn`（std = 1.0）初始化；
   - [`Brain/PPO2/lib/model.py:L428-L435`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L428-L435) 以 `gate * market_emb + (1 - gate) * time_proj` 進行 50/50 通道替換。實測 iter 0 初始化時 `time / market` RMS 能量比即達 **`1.10×`**，iter 8200 時 `gate` 飽和為近二元開關（`mean = 0.564, std = 0.438`），**近半數進入 Mamba2 的主幹通道被純時間訊號取代**（能量比 **`0.97×`**）。
3. **Critic 透過 6.25 天時間相位背誦訓練期時段**：
   - A2 時間歸因顯示 Critic 在 `last_16` 的權重由 `20.3%` 腰斬至 `9.3%`，高達 `67.1%` 落在遠古 `bars_65_300`（`critic_bars_for_50pct = 107` 根 K 棒），利用多幣種高相關 Beta 特性記憶訓練期週內／日內漲跌巧合，一到 2026 OOS 便完全失效。

為進行嚴謹的**第二步單變因消融測試**，本規格將**徹底移除 PPO2 管線中的 `time_states` 觀測通道與 Time2Vec / Gated Fusion 模組**，讓模型純粹輸入 **13 維市場量價特徵 + 1 維倉位特徵（`states` shape `(300, 14)`）**；同時保留 [`Brain/Common/`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/) 中的共用時間特徵工具不動，以維持 legacy DQN 相容性並方便未來視需要再行引入。

### 1.2 當前行為 vs. 預期行為 (Current vs. Expected Behavior)
| 維度 | 當前行為 (Current Behavior) | 預期行為 (Expected Behavior) |
| :--- | :--- | :--- |
| **PPO2 觀測空間 (`observation_space`)** | `Dict({"states": (300, 14), "time_states": (300, 6)})` | **`Dict({"states": (300, 14)})`**（僅保留 13 維量價 + 1 維倉位，完全移除 `"time_states"`） |
| **模型輸入與前置投影 (`MambaContinuousActorCritic`)** | `forward(states, time_states)`；含 `time_embedding` (`SineActivation`)、`time_emb_projection`、`gate_layer` 50/50 融合（參數 `5,104,899`） | **`forward(states)`**；移除 `time_embedding`、`time_emb_projection`、`gate_layer`，改為 `states -> market_embedding -> feature_embedding (Linear+GELU) -> mixer`（參數 **`4,964,899`**，精簡 140,000 參數） |
| **Rollout Buffer & Agent 介面** | `RolloutBuffer` 儲存並 yield 7-tuple（含 `time_states`）；`PPO2Agent.evaluate_actions(states, time_states, actions)` | `RolloutBuffer` 移除 `time_states` 並 yield **6-tuple** `(states, actions, old_log_probs, returns, advantages, old_values)`；`PPO2Agent.evaluate_actions(states, actions)` |
| **YAML 設定 (`ppo_v2.yaml`)** | 包含 `TIME_FEATURES_OUT: 32` | 移除過時鍵 `TIME_FEATURES_OUT`；**其餘所有超參數、獎勵、摩擦排程、SSM 設定 100% 維持不變** |
| **共用層與未來擴充性 (`Brain/Common/`)** | `BasicState.STATIONARY_TIME_FEATURES`、`DataFeature.add_time_feature()`、`SineActivation` 供 PPO2 與 DQN 共用 | **完整保留於 `Brain/Common/` 不刪除**，避免影響 DQN／HopeDQN，且未來若需重新實驗特定時間特徵可直接復用 |

### 1.3 技術根本原因與程式碼參照 (Technical Root Cause & References)
- 時間特徵與 Time2Vec 融合層：[`Brain/PPO2/lib/model.py:L11,L299,L314,L319,L347-L360,L426-L438`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L347-L360)
- PPO2 環境與多進程向量環境觀測封裝：[`Brain/PPO2/lib/environment.py:L206-L255,L286-L293,L488-L492`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L206-L255)
- Rollout 緩衝區時間狀態儲存與批次輸出：[`Brain/PPO2/lib/experience.py:L22-L78,L156-L235`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py#L22-L78)
- 推論特徵適配器：[`Brain/PPO2/lib/data_adapter.py:L171-L212`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py#L171-L212)
- PPO2 Agent 前向與動作評估介面：[`Brain/PPO2/lib/Agent.py:L12-L27,L56-L74,L89-L106,L140-L190`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L56-L106)
- 測試評估器與引擎：[`Brain/PPO2/lib/Evaluator.py:L143-L175,L238-L256`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L143-L175)、[`Brain/PPO2/lib/engine.py:L89,L339-L370,L833-L860`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L339-L370)
- 訓練主程式與 YAML 設定：[`PPO2_rl_train.py:L249-L270,L310-L330,L518-L526,L585-L588`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L249-L330)、[`configs/training/ppo_v2.yaml:L46`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L46)

---

## 2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)

### 2.1 檔案與模組改動矩陣 (File & Module Scope Matrix)
| 編號 | 目標檔案與行號 | 範圍 (Scope) | 規劃動作 (Planned Action) |
| :---: | :--- | :---: | :--- |
| **項目 1** | [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L11) | **In-Scope** | 移除 `SineActivation` 匯入、`time_features_in` / `time_features_out` 參數、`time_embedding`、`time_emb_projection`、`gate_layer`；將 `forward(self, states)` 簡化為 `market_embedding(states) -> feature_embedding -> mixer`。 |
| **項目 2** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L206-L255) | **In-Scope** | `State_time_step.encode()` 僅回傳 `{"states": data_res}`；`BaseTradingEnv.observation_space` 僅包含 `"states"`；`engine_info()` 移除 `"time_input_size"`；`SubprocVecEnv._stack_obs()` 僅堆疊 `"states"`。 |
| **項目 3** | [`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py#L22-L235) | **In-Scope** | 移除 `self.time_states` 之宣告、儲存、`.buffer` 字典鍵、`clear()` 清理，並將 `get_batches()` 改為 yield 6-tuple `(states, actions, old_log_probs, returns, advantages, old_values)`。 |
| **項目 4** | [`Brain/PPO2/lib/data_adapter.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py#L171-L212) | **In-Scope** | `extract_latest_obs()` 移除 `timelist` 與 `time_res`，僅回傳 `{"states": data_res}`，維持與 `State_time_step.encode()` bit-for-bit 一致。`process_df_to_prices()` 保持與共用 `Prices` 結構相容。 |
| **項目 5** | [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L12-L190) | **In-Scope** | 移除 `time_features_out` 參數與 `time_features_in` 讀取；`_validate_spaces()` 僅驗證 `"states"`；`_preprocess_obs()` 回傳 `(states, is_single)`；`get_action()` 呼叫 `self.model(states)`；`evaluate_actions(states, actions, return_stats=False)` 移除 `time_states` 參數。 |
| **項目 6** | [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L143-L256) | **In-Scope** | `evaluate_batch_step()` 與 `evaluate_symbols()` 移除 `time_states_all`、`time_shape`、`batch_time_np`，僅建構 `{"states": ...}` 送入 `self.agent.get_action()`。 |
| **項目 7** | [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L89) | **In-Scope** | 移除 `self.time_features_out`、`ob_space` 中的 `"time_states"`、傳給 `PPO2Agent` 的 `time_features_out`，以及 `_predict_single_model_orders()` / `_predict_ensemble_orders()` 中的 `"time_states"` 堆疊。 |
| **項目 8** | [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L249-L588) | **In-Scope** | `_prepare_agent()` 移除 `time_features_out` 並加入防禦斷言確認模型無 `time_embedding` / `gate_layer`；`update_ppo()` 解包 6-tuple 並呼叫 `evaluate_actions(states_b, actions_b, return_stats=True)`；`train()` 移除 `trunc_obs` 與 `obs` reset 之 `"time_states"`。 |
| **項目 9** | [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L46) | **In-Scope** | 僅移除過時欄位 `TIME_FEATURES_OUT: 32`，其餘全部超參數一字不動。 |
| **項目 10** | [`tests/test_vol_normalized_features_and_no_dain.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_vol_normalized_features_and_no_dain.py) | **In-Scope** | 對齊純 14 維 `states` 觀測與無時間融合層模型斷言（驗證 `obs` 不含 `"time_states"`、模型不含 `time_embedding` / `time_emb_projection` / `gate_layer`、訓練/推論 parity 與 GPU forward/backward/grad_clip）。 |
| **項目 11** | [`tests/diagnose_ppo2_feature_attribution.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/diagnose_ppo2_feature_attribution.py)、[`tests/test_ppo2_feature_attribution.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_ppo2_feature_attribution.py) | **In-Scope** | 對齊純 `states` 模型（維持大寫常數設定、無 `argparse`）：移除 `time_states_np` / `time_embedding` 依賴，JSON 保留 `checkpoint_meta["time_features_in"] = 0` 與 `A1["time_states"] = []` 以便跨實驗對照；同步更新所有單元與端到端測試。 |
| **項目 12** | [`tests/diagnose_ppo2_gradients.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/diagnose_ppo2_gradients.py#L93-L324) | **In-Scope** | 對齊 6-tuple rollout 解包、`trunc_obs`、`obs` reset 與 `evaluate_actions(b["states"], b["actions"])`。 |
| **項目 13** | [`Brain/Common/env_components.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py)、[`Brain/Common/DataFeature.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py)、[`Brain/Common/model_components.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/model_components.py)、[`Brain/Common/grad_diagnostics.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/grad_diagnostics.py)、[`Brain/Common/optimization.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py)、[`PPO2_rl_test.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py)、`Brain/DQN/*`、`Brain/HopeDQN/*`、`AC_train.py`、`HopeModel_traingloop.py`、`test_ppo2_env.py` | **Out-of-Scope** | **明確保留 100% 不更動**：保護 legacy DQN/HopeDQN 零回歸，保留 13 維市場特徵計算公式一字不改，並讓未來若需重新測試時間特徵時可直接從 `Brain/Common/` 取用；`grad_diagnostics.py` 的 `MODULE_GROUPS` 對移除的 prefix 自動回傳 0 參數且 `market_embedding.` / `feature_embedding.` 持續對應至 `"input_fusion"`，無須修改。 |

### 2.2 架構邊界與非目標 (Architectural Boundaries & Non-Goals)
- **範圍內邊界 (In-Scope Boundaries)**:
  - PPO2 觀測字典精簡為單一鍵 `{"states": np.ndarray}`，形狀 `(300, 14)`（13 維 `BasicState.MARKET_FEATURES` + 1 維 `position`）。
  - `MambaContinuousActorCritic` 輸入路徑為：`states (B, L, 14) -> market_embedding Linear(14, 256) -> feature_embedding Linear(256, 256)+GELU -> mixer (Mamba2) -> temporal_pool + portfolio_encoder -> trunk_norm -> fc_actor / fc_val`。
  - `State_time_step.encode()`（訓練）與 `PPO2DataAdapter.extract_latest_obs()`（推論／回測）維持 100% bit-for-bit 一致。
- **非目標 (Out-of-Scope / Non-Goals)**:
  - **不移除** `ret_gap`、`taker_imbalance` 或其餘任何 13 維市場特徵（維持單變因控制）。
  - **不修改** `configs/training/ppo_v2.yaml` 的任何訓練／獎勵／摩擦超參數（除刪除已無效的 `TIME_FEATURES_OUT` 外，`LEARNING_RATE`、`ENT_COEF`、`LOG_STD_MAX`、`FRICTION_ANNEAL_*`、`DEADZONE_THRESHOLD`、`W2_DOWNSIDE_RATIO` 等全數維持原值）。
  - **不刪除** `Brain/Common/` 中的 `STATIONARY_TIME_FEATURES`、`add_time_feature()`、`SineActivation`（方便未來實驗與相容舊模組）。
  - **不回溯相容** 舊含 `time_embedding` / `gate_layer` 之 checkpoint（如 `checkpoint_iter_8200.pt`）：消融實驗將從零重新訓練純 14 維量價模型。

---

## 3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)

### 3.1 步驟清單 (Step-by-Step Task Checklist)
- [x] **步驟 1: 簡化神經網路架構 [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py)**
  - [x] 移除未使用的 `SineActivation` 匯入。
  - [x] 從 `MambaContinuousActorCritic.__init__()` 移除 `time_features_in`、`time_features_out`、`self.time_embedding`、`self.time_emb_projection`、`self.gate_layer`。
  - [x] 將 `MambaContinuousActorCritic.forward(self, states: torch.Tensor)` 改為單輸入 `states`，經 `market_embedding` → `feature_embedding` 直接進入 `self.mixer`。
- [x] **步驟 2: 更新 PPO2 環境、緩衝區與資料適配器**
  - [x] [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)：更新 `State_time_step.encode()`、`BaseTradingEnv.observation_space`、`BaseTradingEnv.engine_info()`、`SubprocVecEnv._stack_obs()`，全面移除 `"time_states"` 與 `"time_input_size"`。
  - [x] [`Brain/PPO2/lib/experience.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py)：從 `RolloutBuffer` 移除 `self.time_states`，使 `get_batches()` yield 6-tuple。
  - [x] [`Brain/PPO2/lib/data_adapter.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py)：更新 `extract_latest_obs()` 僅回傳 `{"states": data_res}`。
- [x] **步驟 3: 更新 Agent、Evaluator、Engine、訓練入口與設定檔**
  - [x] [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py)：更新 `__init__()`、`_validate_spaces()`、`_preprocess_obs()`、`get_action()`、`evaluate_actions()` 移除所有 `time_states` / `time_features_*`。
  - [x] [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py)：更新 `evaluate_batch_step()` 與 `evaluate_symbols()` 僅處理 `"states"`。
  - [x] [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py)：更新 `__init__()`、`build_agent()`、`_predict_single_model_orders()`、`_predict_ensemble_orders()`。
  - [x] [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)：更新 `_prepare_agent()`（含防禦性斷言）、`update_ppo()`、`train()`。
  - [x] [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml)：移除 `TIME_FEATURES_OUT: 32`。
- [x] **步驟 4: 對齊驗證與診斷測試套件**
  - [x] [`tests/test_vol_normalized_features_and_no_dain.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_vol_normalized_features_and_no_dain.py)：更新觀測與模型斷言，確認無 `time_states` / `time_embedding` / `gate_layer` 且訓練/推論與 GPU 前反向全過。
  - [x] [`tests/diagnose_ppo2_feature_attribution.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/diagnose_ppo2_feature_attribution.py) 與 [`tests/test_ppo2_feature_attribution.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_ppo2_feature_attribution.py)：對齊純 `states` 歸因工具與 6 項單元／端到端測試（維持大寫常數、零 `argparse`）。
  - [x] [`tests/diagnose_ppo2_gradients.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/diagnose_ppo2_gradients.py)：對齊 6-tuple rollout 與 `evaluate_actions(states, actions)`。
- [x] **步驟 5: 代碼防禦與清理 (Clean up & Defensiveness)**
  - [x] 全域搜尋確認 `Brain/PPO2/`、`PPO2_rl_train.py`、`PPO2_rl_test.py`、`configs/training/ppo_v2.yaml` 不再殘留任何 `time_states`、`time_embedding`、`time_emb_projection`、`gate_layer`、`TIME_FEATURES_OUT` 或 `SineActivation` 引用。
  - [x] 驗證 `PPO2Engine.build_agent()` 與 `PPO2Evaluator.evaluate_symbols()` 端到端回測路徑無斷裂。

### 3.2 詳細程式碼修改藍圖 (Detailed Code Modifications Blueprint)

#### 3.2.1 [`Brain/PPO2/lib/model.py:L311-L365,L426-L448`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L311-L365)
```python
class MambaContinuousActorCritic(nn.Module):
    def __init__(
        self,
        d_model: int,
        action_dim: int = 1,
        seq_dim: int = 300,
        hidden_size: int = 96,
        nlayers: int = 2,
        init_log_std: float = 0.0,
        portfolio_dim: int = 1,
        ssm_cfg: Optional[dict] = None,
        moe_cfg: Optional[dict] = None,
        d_intermediate: Optional[int] = None,
        short_window: int = 16,
        portfolio_emb_dim: Optional[int] = None,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
    ):
        super().__init__()
        assert d_model > portfolio_dim, (
            f"d_model ({d_model}) must be greater than portfolio_dim ({portfolio_dim})"
        )
        ...
        self.market_embedding = nn.Linear(d_model, hidden_size)
        self.feature_embedding = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.mixer = MixerModel(...)

    def forward(self, states: torch.Tensor):
        # 1. Project pre-normalized market + portfolio features directly (no time_states / gate fusion)
        market_emb = self.market_embedding(states)
        fused = self.feature_embedding(market_emb)

        # 2. Mamba2 temporal backbone
        out, _ = self.mixer(fused)
        ...
        return mu, std, value
```

#### 3.2.2 [`Brain/PPO2/lib/environment.py:L206-L255,L488-L492`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L206-L255)
```python
    def encode(self):
        """
        Encodes observation dict {"states": data_res} (13 market + 1 position = 14 dims)
        using vectorized NumPy slicing without look-ahead bias.
        """
        data_res = np.zeros(shape=self.getStateShape(), dtype=np.float32)

        start = self._offset - self.bars_count
        end = self._offset

        for idx, field in enumerate(self.info_list):
            data_res[:, idx] = getattr(self._prices, field)[start:end]

        data_res[:, len(self.info_list)] = self.position

        return {"states": data_res}

# In BaseTradingEnv.__init__():
        self.observation_space = spaces.Dict(
            {
                "states": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self._state.getStateShape(),
                    dtype=np.float32,
                ),
            }
        )

# In SubprocVecEnv._stack_obs():
    def _stack_obs(self, obs_list):
        return {
            "states": np.stack([o["states"] for o in obs_list], axis=0),
        }
```

#### 3.2.3 [`Brain/PPO2/lib/Agent.py:L56-L190`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L56-L190)
```python
    def _validate_spaces(self):
        ...
        assert isinstance(self.ob_space, Dict), "PPO2Agent 觀測空間必須為 gym.spaces.Dict"
        assert "states" in self.ob_space.spaces, "PPO2Agent 觀測字典必須包含 'states'"

    def _preprocess_obs(self, obs):
        states = torch.as_tensor(obs["states"], dtype=torch.float32, device=self.device)
        is_single = False
        if states.dim() == 2:
            states = states.unsqueeze(0)
            is_single = True
        return states, is_single

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        return_stats: bool = False,
    ):
        self.model.train()
        states = states.to(self.device)
        actions = actions.to(self.device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        mu, std, value = self.model(states)
        ...
```

#### 3.2.4 [`tests/diagnose_ppo2_feature_attribution.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/diagnose_ppo2_feature_attribution.py)
- `load_model_from_checkpoint()`：驗證 `market_embedding.weight.shape[1] == d_model` 並斷言 `state_dict` 不含 `time_embedding.w` 與 `gate_layer.0.weight`；記錄 `ckpt_meta["time_features_in"] = 0`。
- `sample_observation_windows()`：回傳 `(states_np, sample_meta)`（不再回傳 `time_states_np`）。
- `compute_input_gradients(model, states_np, device, batch_size)`：回傳 `g_mu_states`, `g_v_states`, `mu`, `value`。
- `compute_feature_attribution(...)`：針對 `states`（13 market + 1 portfolio）計算 4 指標 + `W_L2`，`group_permutation` 計算 `market_all` 與 `portfolio_all`，保留 `"time_states": []` 以維持 JSON schema 一致性。

---

## 4. 驗證與測試計畫 (Validation & Verification Plan)

### 4.1 驗證環境與執行指令 (Verification Environment & Commands)
> [!IMPORTANT]
> 依據專案規範，所有執行必須使用虛擬環境之 Python 直譯器：
> 絕對路徑: `/home/b0457812963/Mamba3RL/bin/python`
> 相對路徑: `../bin/python`

```bash
# 1. 執行特徵工程、無 DAIN、無時間特徵全套整合測試（含 177 檔真實 CSV、訓練/推論對齊與 GPU 前反向）：
/home/b0457812963/Mamba3RL/bin/python tests/test_vol_normalized_features_and_no_dain.py

# 2. 執行特徵歸因診斷單元與端到端測試（驗證純 14 維 states checkpoint 歸因、SHA-256 唯讀與 JSON 輸出）：
/home/b0457812963/Mamba3RL/bin/python tests/test_ppo2_feature_attribution.py

# 3. 驗證 PPO2Engine + PPO2Evaluator 單/多幣種回測與 formal 推論煙霧測試（純 14 維 states）：
/home/b0457812963/Mamba3RL/bin/python -c "
import tempfile, os, torch
from Brain.PPO2.lib.engine import PPO2Engine
engine = PPO2Engine(formal=False, config='configs/training/ppo_v2.yaml')
agent = engine.build_agent()
assert not hasattr(agent.model, 'time_embedding') and not hasattr(agent.model, 'gate_layer')
assert 'time_states' not in agent.ob_space.spaces
print('PPO2Engine build_agent OK, params =', sum(p.numel() for p in agent.model.parameters()))
"

# 4. 驗證 PPO2 管線殘留引用清理（應回傳空結果，exit code 1）：
grep -rn --exclude-dir=__pycache__ "time_states\|time_embedding\|time_emb_projection\|gate_layer\|TIME_FEATURES_OUT\|SineActivation" Brain/PPO2/ PPO2_rl_train.py PPO2_rl_test.py configs/training/ppo_v2.yaml
```

### 4.2 驗收測試查核清單 (Acceptance Verification Checklist)
- [x] **1. 觀測與架構邊界驗證 (Observation & Model Architecture)**:
  - [x] `State_time_step.encode()` 與 `PPO2DataAdapter.extract_latest_obs()` 僅回傳 `{"states": (300, 14)}` 且不包含 `"time_states"`，兩者數值 bit-for-bit 相等。
  - [x] `MambaContinuousActorCritic` 不含 `time_embedding`、`time_emb_projection`、`gate_layer`、`dean` 屬性與參數，總參數量降為 `4,964,899`。
  - [x] `build_param_groups(agent.model)` 通過全參數覆蓋斷言，且 `input_fusion` 僅包含 `market_embedding` 與 `feature_embedding`。
- [x] **2. 嚴格單變因隔離與零回歸 (Strict Ablation Isolation & Zero Regression)**:
  - [x] `git diff` 確認 `Brain/Common/` 與 `Brain/DQN/` 零更動；`configs/training/ppo_v2.yaml` 除刪除 `TIME_FEATURES_OUT` 外其餘所有超參數零更動。
  - [x] `Brain/PPO2/`、`PPO2_rl_train.py`、`PPO2_rl_test.py`、`configs/training/ppo_v2.yaml` 零殘留 `time_states` / `SineActivation` / `gate_layer` 引用。
- [x] **3. GPU 實際執行與完整管線驗證 (Runtime & GPU Verification)**:
  - [x] `tests/test_vol_normalized_features_and_no_dain.py` 與 `tests/test_ppo2_feature_attribution.py` 於 GPU 上全數通過。
  - [x] `RolloutBuffer` 6-tuple 收集／GAE／mini-batch 抽樣、`PPO2Agent` 訓練與推論、`PPO2Engine` + `PPO2Evaluator` 評估流程皆無形狀或鍵值例外。
