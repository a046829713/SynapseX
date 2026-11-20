# 規格文件 (Specification): PPO2 波動率歸一化平穩特徵重構與移除 DAIN 層

- **類別 (Category)**: `feature`
- **規格路徑 (Spec Path)**: `spec/feature/ppo2_vol_normalized_features_and_remove_dain.md`
- **狀態 (Status)**: `Implemented & Verified`

## 📋 目錄 (Index / Contents)
1. [1. 背景與根本原因分析 (Background & Root Cause Analysis)](#1-背景與根本原因分析-background--root-cause-analysis)
2. [2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)](#2-改動範圍與架構邊界-target-scope--architecture-boundaries)
3. [3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)](#3-實作查核清單與詳細藍圖-implementation-checklist--detailed-blueprint)
4. [4. 驗證與測試計畫 (Validation & Verification Plan)](#4-驗證與測試計畫-validation--verification-plan)

---

## 1. 背景與根本原因分析 (Background & Root Cause Analysis)

### 1.1 問題陳述與情境脈絡 (Problem Statement & Context)
在目前以 30 分鐘 K 棒（`BARS_COUNT=300`、`EPISODE_STEPS=512`、連續倉位 Long-Only `[0, 1]`、含 10 bps 摩擦成本與 60-bar 下行風險懲罰）為核心的多幣種 PPO2 訓練與推論管線中，[`Brain/Common/DataFeature.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py) 與 [`Brain/Common/dain.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/dain.py) 的既有設計經實證檢核存在多項根本缺陷：

1. **`np.log1p(OHLC)` 對低價幣造成嚴重尺度失真與數值壓縮**：
   - 對正價格 $P > 0$ 取 $\ln(1+P)$ 時，相對報酬傳遞率為 $\Delta\ln(1+P) \approx \frac{P}{1+P} \cdot r_t$。
   - 在 pre-2026 可用訓練集（扣除 `USDCUSDT`/`STRCUSDT` 共 83 檔）中，有 **44%（37 檔）中位價 $< 1.0$ USDT**（全 179 檔資料集為 51 檔）。例如 `UBUSDT`（0.018–0.24 USDT）幅度縮水約 **24 倍**；`ADAUSDT`（0.019–3.10 USDT）在同一支幣不同歷史階段的傳遞率差距高達 **40 倍**。
   - 低價幣經 `log1p` 壓縮後，300-bar 視窗變異數可低至 `3.35e-8`（`MINAUSDT`）與 `1.63e-7`（`UBUSDT`），逼近 [`DAIN_Layer.eps = 1e-8`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/dain.py#L31)，不僅扭曲標準差估計，更導致反向傳播經 $1/\sigma$ 產生數千倍梯度放大。

2. **`DAIN_Layer` 實質凍結且破壞 K 棒幾何、均線零點與波動率體制**：
   - 檢核已訓練 10,000 iter（4,096 萬步）之 [`Brain/PPO2/Meta/checkpoint_iter_10000.pt`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/Meta/checkpoint_iter_10000.pt)：因 [`Brain/Common/optimization.py:L46-L56`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py#L46-L56) 將 `base_lr (1e-4)` 與 `mean_lr/scale_lr (1e-5)` 重複相乘使實際學習率低至 **`1e-9`**，導致 `dean.mean_layer.weight` 與 `dean.scaling_layer.weight` 幾乎完全鎖死於單位矩陣 $I$（$\max|W-I| < 4\times 10^{-6}$），Step 3 的 `gating_layer` 亦因去均值後 $\text{mean}(x^{(2)}) \approx 0$ 退化為靜態隨機常數（`0.446 ~ 0.556`）。
   - 實質上 `DAIN_Layer` 僅在執行「逐通道獨立 300-bar Z-Score」，直接造成：
     - **K 棒高低點幾何反轉**：實測 `UBUSDT` / `ETHUSDT` / `ADAUSDT` / `AVNTUSDT` 有 **59.3% ~ 69.0%** 的 K 棒在逐通道標準化後出現 `High_norm < Low_norm`。
     - **均線趨勢零點翻轉**：原本具尺度不變性的 `log_ma_360` 經 300-bar 去均值後，有 **27.8% ~ 29.0%** 的 K 棒發生正負號翻轉（多頭被翻轉成負值）。
     - **波動率盲區**：所有 14 維市場特徵皆被強制歸一為 `std ≈ 1`，使模型無法分辨「波動 0.15% 的低波盤整（進場必虧 10 bps 成本）」與「波動 3% 的高波行情」。

3. **五維原始量能高度共線且埋沒主動買賣力道**：
   - `log_volume` 與 `log_tb_base_av` 相關係數高達 `0.986 ~ 0.994`，`log_quote_av` 與 `log_tb_quote_av` 高達 `0.994 ~ 0.996`，且 `Volume` 受幣別單價影響。真正具微結構價值的「主動買賣失衡（Order Flow Imbalance）」與「相對放量比」未被顯式提取。

4. **資料清洗與推論管線隱患**：
   - [`OriginalDataFeature.cleanData()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L49-L53) 使用 `bfill(axis=0)` 具前段未來函數風險，且 `ffill` 會對缺失 K 棒偽造假成交量，亦無非正價格（`<= 0`）防護。
   - [`get_train_net_work_data_by_pd()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L55-L67) 漏掉呼叫 `cleanData()`；[`load_relative(if_log=False)`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L176-L200) 會靜默回傳 `None`。
   - [`Brain/PPO2/lib/engine.py:L381-L387`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L381-L387) 殘留列印 `dean` 參數與 `time.sleep(100)` 的除錯阻塞碼。

經與使用者完整討論並確認採用 **方案 B（移除 DAIN + 因果波動率歸一化平穩特徵 + 顯式波動率體制通道 + 軟截斷）**，同時維持既有 **6 維時間特徵**、**SMA 均線計算**，且在進入 `market_embedding` 前**不額外加入 LayerNorm / RMSNorm**。

---

### 1.2 當前行為 vs. 預期行為 (Current vs. Expected Behavior)
| 維度 | 當前行為 (Current Behavior) | 預期行為 (Expected Behavior) |
| :--- | :--- | :--- |
| **價格與 K 棒特徵** | `np.log1p(O, H, L, C)` 非平穩價格水準；低價幣失真高達 24–40 倍；逐通道 Z-score 造成 60%+ `H < L` | 4 維嚴格正交 OHLC 分解（`ret_gap`, `candle_body`, `upper_wick`, `lower_wick`），共用同一把因果前一期波動率 $\sigma_{t-1}$ 縮放、**不去均值**、經 `soft_clip(5.0)` 防插針，完全尺度不變且永不反轉 K 棒幾何 |
| **均線偏離特徵** | `log(Close / SMA_p)`（`p ∈ {30,60,120,240,360}`）經 DAIN 300-bar 去均值，約 28% 正負號翻轉 | 維持 `SMA` 計算，改以隨機漫步尺度 $\sigma_{t-1}\sqrt{p/3}$ 無中心化縮放並軟截斷（`ma_dev_30..360`），跨週期天然對齊 $O(1)$ 且 100% 保留零點語義 |
| **波動率體制特徵** | 缺失（`add_ATR` 註解中且具 `shift(-1)` 未來函數與 `log1p` 失真）；DAIN 壓平絕對波動率 | 新增 2 維顯式波動率體制特徵：`vol_regime_abs`（$\ln(\sigma_t / 0.005)$，讓模型感知 10 bps 成本門檻）與 `vol_expansion`（$\ln(\sigma_{\text{short},t}/\sigma_{\text{long},t})$，捕捉波動擴張/收斂） |
| **量能與微結構特徵** | 5 維高度共線（0.99+）之 `log1p` 原始量能；受幣價與零量突刺影響 | 3 維跨幣種平穩微結構特徵：對數空間幾何均值偏移 `rel_quote_vol` 與 `rel_trades`（消除 Jensen 不等式負偏，均值近 `0.00`）、以及有界主動買賣失衡 `taker_imbalance ∈ [-1, 1]`（零量安全填 `0.0`） |
| **時間特徵** | 6 維平穩週期特徵（`hour_sin/cos`, `minute_sin/cos`, `dayofweek_sin/cos`） | **完全維持現有 6 維不變** |
| **模型前處理 (`MambaContinuousActorCritic`)** | 14 維市場特徵經 `self.dean = DAIN_Layer(...)`；`PPO2_rl_train.py` 需分離 DAIN 排除梯度裁剪並配 `BASE_LR` | 徹底移除 `self.dean`，14 維市場特徵 + 1 維倉位直接進入 `market_embedding`（不額外加 LayerNorm/RMSNorm）；恢復全局統一 `clip_grad_norm_` 與精簡梯度監控 |
| **資料清洗 (`cleanData`)** | `ffill` + `bfill`（前段未來函數 + 偽造量能）、無 `<= 0` 價格防護、`get_train_net_work_data_by_pd` 漏清洗 | 移除 `bfill`；僅對 OHLC 做 `ffill` 並丟棄開頭缺失列；非正價格轉 `NaN` 防護；缺失或負量能統一填 `0.0`；訓練/推論/DataFrame 入口完全一致 |

---

### 1.3 技術根本原因與程式碼參照 (Technical Root Cause & References)
- 非平穩 `log1p` 價量與過時 `load_relative`：[`Brain/Common/DataFeature.py:L10-L40`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L10-L40)、[`L49-L67`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L49-L67)、[`L104-L121`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L104-L121)、[`L176-L227`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L176-L227)
- 硬編碼特徵清單分散於兩處：[`Brain/Common/env_components.py:L33-L48`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py#L33-L48) 與 [`Brain/PPO2/lib/data_adapter.py:L198-L202`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py#L198-L202)
- PPO2 模型內嵌凍結之 `DAIN_Layer` 與重複屬性賦值：[`Brain/PPO2/lib/model.py:L11`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L11)、[`L340-L353`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L340-L353)、[`L435-L442`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L435-L442)；[`Brain/PPO2/lib/Agent.py:L21,L69`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L21)
- DAIN 特製梯度裁剪分離與 `BASE_LR`：[`PPO2_rl_train.py:L15-L20`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L15-L20)、[`L267-L290`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L267-L290)、[`L400-L438`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L400-L438)、[`L671-L674`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L671-L674)；[`configs/training/ppo_v2.yaml:L29`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L29)
- 推論引擎殘留除錯阻塞碼：[`Brain/PPO2/lib/engine.py:L381-L387`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L381-L387)

---

## 2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)

### 2.1 檔案與模組改動矩陣 (File & Module Scope Matrix)
| 編號 | 目標檔案與行號 | 範圍 (Scope) | 規劃動作 (Planned Action) |
| :---: | :--- | :---: | :--- |
| **項目 1** | [`Brain/Common/env_components.py:L1-L56`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py#L1-L56) | **In-Scope** | 於 `BasicState` 類別內部新增單一真相來源常數 `MARKET_FEATURES`（14 維新特徵清單），並令 `self.info_list = list(self.MARKET_FEATURES)`；保留 `STATIONARY_TIME_FEATURES`（6 維）不變。 |
| **項目 2** | [`Brain/Common/DataFeature.py:L1-L227`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L1-L227) | **In-Scope** | 重構 `Prices` 為 24 欄（4 原始 OHLC + 14 波動率歸一化市場特徵 + 6 時間特徵）；修復 `cleanData()` 移除 `bfill` 與加入非正價/負量防護；實作 `add_market_features()` 計算全因果 14 維特徵（維持 SMA 均線）；修正 `get_train_net_work_data_by_pd()` 補齊 `cleanData()`；重構 `load_relative(df=None)`；清理 4 個未使用 import 與過時註解。 |
| **項目 3** | [`Brain/PPO2/lib/data_adapter.py:L51-L216`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py#L51-L216) | **In-Scope** | 更新 `process_df_to_prices()` 改呼叫 `feature_loader.add_market_features()` 並以 `BasicState.MARKET_FEATURES` 檢查缺欄；更新 `extract_latest_obs()` 直接引用 `BasicState.MARKET_FEATURES` 與 `BasicState.STATIONARY_TIME_FEATURES`，確保與訓練環境 bit-for-bit 一致。 |
| **項目 4** | [`Brain/PPO2/lib/model.py:L11,L293-L471`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L293-L471) | **In-Scope** | 從 `MambaContinuousActorCritic` 移除 `DAIN_Layer`（`self.dean`）與無用之 `mode` 參數；`states` 直接輸入 `self.market_embedding`（不加額外 LayerNorm/RMSNorm），保留 `portfolio_encoder` 後期跳躍連接（Late Fusion）；順手清除重複之 `self.log_std_min = log_std_min` 行。 |
| **項目 5** | [`Brain/PPO2/lib/Agent.py:L11-L77`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L11-L77) | **In-Scope** | 移除 `PPO2Agent.__init__` 中傳遞給 DAIN 的無用 `mode: str = "full"` 參數。 |
| **項目 6** | [`Brain/PPO2/lib/engine.py:L372-L389`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L372-L389) | **In-Scope** | 移除 `load_checkpoint()` 中殘留之 `if "dean" in name: print(...)` 與 `time.sleep(100)` 除錯阻塞碼。 |
| **項目 7** | [`PPO2_rl_train.py:L15-L20,L267-L291,L400-L452,L671-L674`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L267-L291) | **In-Scope** | 移除 `DAIN_GROUPS` 與 `self._rest_params` 排除邏輯，改以全模型參數統一進行 `clip_grad_norm_`；移除 `BASE_LR` 讀取；調整梯度診斷日誌為 `GN[total/actor/critic/backbone]` 與 `clip_coef(group_norms["total"], max_grad_norm)`。 |
| **項目 8** | [`configs/training/ppo_v2.yaml:L26-L30`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L26-L30) | **In-Scope** | 移除僅供 DAIN 使用的過時參數 `BASE_LR: 0.0001`。 |
| **項目 9** | `tests/test_vol_normalized_features_and_no_dain.py` *(新增)* | **In-Scope** | 建立完整單元與真實資料集整合驗證測試（尺度不變性、嚴格因果性、K 棒幾何不變性、全資料集有限值與 $O(1)$ 分布、訓練/推論 bit-for-bit 對齊、模型無 DAIN 前向/反向驗證）。 |
| **項目 10** | [`Brain/Common/dain.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/dain.py)、[`Brain/Common/optimization.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/optimization.py)、[`Brain/Common/grad_diagnostics.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/grad_diagnostics.py) | **Out-of-Scope** | 明確保留不更動：`optimization.py` 已有 `if hasattr(model, "dean"):` 防護，`grad_diagnostics.py` 遇空 DAIN 群組會安全回傳 `0.0`，不影響既有工具或舊模組匯入。 |
| **項目 11** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)、[`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py)、舊版 `Brain/DQN/` / `HopeDQN/` / `A2C/` / `DDPG/` | **Out-of-Scope** | `State_time_step` 透過 `self.info_list` 與 `self.timelist` 動態編碼觀測張量，不需修改即自動相容新 `Prices` 與 `BasicState.MARKET_FEATURES`；獎勵函數與會計邏輯完全維持不變；舊版 RL 管線不在維護範圍。 |

### 2.2 架構邊界與非目標 (Architectural Boundaries & Non-Goals)
- **範圍內邊界 (In-Scope Boundaries)**:
  - `observation_space["states"].shape` 維持 `(bars_count, 15)`（14 維市場特徵 + 1 維 `position`），`observation_space["time_states"].shape` 嚴格維持 `(bars_count, 6)`。
  - 全部 14 維市場特徵必須嚴格滿足 **因果性（Zero Look-Ahead）**、**價格尺度不變性** 與 **有限有界性（`[-5.0, +5.0]`）**。
  - `State_time_step.encode()` 與 `PPO2DataAdapter.extract_latest_obs()` 共用 `BasicState.MARKET_FEATURES` 與 `BasicState.STATIONARY_TIME_FEATURES`，確保訓練與實盤/回測特徵順序與數值完全一致。
  - **為何採用 4 維正交 OHLC 分解而非同時納入 `ret_close` 與 `candle_body`**：於全 177 檔真實資料實測，`corr(ret_close, candle_body)` 中位數達 **`0.99934`**（因數學恆等式 $\ln(C_t/C_{t-1}) = \ln(O_t/C_{t-1}) + \ln(C_t/O_t)$）。採用 `ret_gap`、`candle_body`、`upper_wick`、`lower_wick` 4 維正交分解可無損表達完整 K 棒四點幾何且零線性冗餘，並直接對應環境 `State_time_step.step()` 的 `gap_simple_ret` 與 `bar_simple_ret`，同時讓市場特徵維持精煉的 14 維。
  - **為何量能相對比採用對數空間幾何均值（Log-EMA）**：若用算術均值 $\ln(q_t / \text{EMA}(q)_t)$，因成交額右偏分布受 Jensen 不等式影響會產生 `-0.57` 的系統性負偏移；改用 $\ln(1 + q_t) - \text{EMA}_{60}(\ln(1 + q))_t$ 在全 177 檔資料集實測均值為 **`+0.0043`**（`rel_trades` 為 **`-0.0028`**），完美零中心化且對零成交量 K 棒天然安全無 `-inf`。
- **非目標 (Out-of-Scope / Non-Goals)**:
  - **不回溯相容舊版 Checkpoint**（如含 `dean.*` 與舊 `log1p` 特徵之 `checkpoint_iter_10000.pt`）：本規格針對新特徵與無 DAIN 架構重新訓練模型。
  - 不更改 6 維時間特徵數學公式（維持 `hour_sin/cos`, `minute_sin/cos`, `dayofweek_sin/cos`）。
  - 不改用 EMA 計算均線偏離（維持 `SMA` 視窗 `[30, 60, 120, 240, 360]`）。
  - 不在 `market_embedding` 前額外加入 `LayerNorm` / `RMSNorm`。
  - 不修改獎勵函數、摩擦退火排程、Mamba2 SSM 內部超參數或 `tests/diagnose_ppo2_gradients.py`。

---

## 3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)

### 3.1 步驟清單 (Step-by-Step Task Checklist)
- [x] **步驟 1: 於 `BasicState` 定義 `MARKET_FEATURES` 單一真相來源（[`Brain/Common/env_components.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py)）**
  - [x] 新增類別屬性 `MARKET_FEATURES`（14 維 tuple，順序固定）。
  - [x] 將 `self.info_list` 改為 `list(self.MARKET_FEATURES)`，保留 `STATIONARY_TIME_FEATURES` 不變。
- [x] **步驟 2: 重構特徵工程管線（[`Brain/Common/DataFeature.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py)）**
  - [x] 更新 `Prices` namedtuple 為 `("open", "high", "low", "close") + BasicState.MARKET_FEATURES + BasicState.STATIONARY_TIME_FEATURES`（共 24 欄）。
  - [x] 新增常數與輔助函式 `soft_clip(x, c=5.0) = c * np.tanh(x / c)`。
  - [x] 重構 `cleanData(df)`：複製 `df`、替換 `±inf -> NaN`、非正 OHLC 轉 `NaN`、先裁切首根完整有效 OHLC 之前的開頭殘缺列再對 OHLC 執行 `ffill()`（移除 `bfill()` 並防止殘缺列外溢）、缺失或負值量能欄位統一填 `0.0`。
  - [x] 實作 `add_market_features(df, periods=(30, 60, 120, 240, 360))`：
    - [x] 防禦性包絡約束：`oc_max = max(O, C)`、`oc_min = min(O, C)`、`h_safe = max(H, oc_max)`、`l_safe = min(L, oc_min)`。
    - [x] 計算因果 EWMA RMS 波動率：$\sigma_{\text{long}, t} = \max(\sqrt{\text{EWMA}_{60}(r_t^2)}, 10^{-4})$（`min_periods=20`）、$\sigma_{\text{short}, t} = \max(\sqrt{\text{EWMA}_{12}(r_t^2)}, 10^{-4})$（`min_periods=6`）、以及前一期基準 $\sigma_{t-1} = \sigma_{\text{long}, t-1}$。
    - [x] 計算 4 維正交 K 棒特徵（共用 $\sigma_{t-1}$、不去均值、`soft_clip(5.0)`）：`ret_gap`, `candle_body`, `upper_wick`, `lower_wick`。
    - [x] 計算 5 維 SMA 均線偏離（`ma_dev_30..360`）：`soft_clip(log(Close / SMA_p) / (sigma_prev * sqrt(p / 3.0)))`。
    - [x] 計算 2 維波動率體制特徵：`vol_regime_abs = soft_clip(log(sigma_long / 0.005))`、`vol_expansion = soft_clip(log(sigma_short / sigma_long))`。
    - [x] 計算 3 維量能微結構特徵：`rel_quote_vol`、`rel_trades`（Log-EMA 偏離並 `soft_clip(5.0)`）與 `taker_imbalance = clip(2 * tb_quote_av / quote_av - 1, -1, 1)`（`quote_av <= 0` 時填 `0.0`）。
    - [x] 僅針對 `list(BasicState.MARKET_FEATURES)` 執行 `dropna(subset=...)` 並回傳 `.copy()`。
  - [x] 修正 `get_train_net_work_data_by_pd()` 補上 `self.cleanData(df)` 並改呼叫 `self.add_market_features()`。
  - [x] 更新 `get_train_net_work_data_by_path()` 改呼叫 `self.add_market_features()`。
  - [x] 重構 `load_relative(self, df: Optional[pd.DataFrame] = None) -> Prices`：移除 `if_log` 死分支與過時 docstring，支援顯式傳入 `df`（預設 fallback `self.df`）。
  - [x] 清除無用 import（`time`, `VolatilityCalculator`, `debug`, `InvalidModeError`）與過時註解碼（`add_ATR` 註解區塊）。
- [x] **步驟 3: 對齊推論適配層（[`Brain/PPO2/lib/data_adapter.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py)）**
  - [x] 更新 `process_df_to_prices()`：改檢查 `BasicState.MARKET_FEATURES` 並呼叫 `feature_loader.add_market_features(cleaned)`；若缺少任何市場特徵欄位則防禦性補 `0.0`；呼叫 `feature_loader.load_relative(cleaned)`。
  - [x] 更新 `extract_latest_obs()`：改以 `info_list = list(BasicState.MARKET_FEATURES)` 取代舊硬編碼清單。
- [x] **步驟 4: 自 PPO2 模型與 Agent 移除 DAIN（[`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py)、[`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py)）**
  - [x] 移除 `model.py` 中 `from Brain.Common.dain import DAIN_Layer`、`self.dean = DAIN_Layer(...)` 與 `mode` 參數；`forward()` 直接將 `states` 送入 `self.market_embedding(states)`（不加額外 norm），並保留 `portfolio_raw[:, -1, :] -> self.portfolio_encoder` 後期融合。
  - [x] 清除 `model.py:L340-L341` 重複之 `self.log_std_min = log_std_min` 與更新類別 docstring。
  - [x] 移除 `Agent.py` `PPO2Agent.__init__` 中無用的 `mode: str = "full"` 參數與傳遞。
- [x] **步驟 5: 清理推論引擎除錯碼與訓練管線 DAIN 殘留（[`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py)、[`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)、[`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml)）**
  - [x] 確認 `engine.py` `load_checkpoint()` 已無列印 `dean` 參數與 `time.sleep(100)` 的除錯阻塞碼。
  - [x] 精簡 `PPO2_rl_train.py`：移除 `DAIN_GROUPS` 匯入與 `self._rest_params` 切分；改對全模型參數執行 `clip_grad_norm_(self.agent.model.parameters(), max_grad_norm)`；`clip_coef` 與監控改報 `total/actor/critic/backbone`；`_prepare_optimizer` 移除 `BASE_LR` 讀取。
  - [x] 從 `configs/training/ppo_v2.yaml` 移除無用之 `BASE_LR: 0.0001`。
- [x] **步驟 6: 代碼防禦、測試撰寫與清理檢核 (Clean up & Verification)**
  - [x] 全域搜尋確認 `Brain/PPO2/` 與 `PPO2_rl_train.py` 不再殘留 `log_open`、`log_ma_30`、`self.dean` 或 `time.sleep(100)`。
  - [x] 新增 `tests/test_vol_normalized_features_and_no_dain.py` 並以虛擬環境執行所有驗證測試。

---

### 3.2 詳細程式碼修改藍圖 (Detailed Code Modifications Blueprint)

#### 3.2.1 [`Brain/Common/env_components.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py)
```python
class BasicState:
    MARKET_FEATURES = (
        # K-bar orthogonal 4-DoF geometry (shared causal sigma_{t-1}, uncentered)
        "ret_gap",
        "candle_body",
        "upper_wick",
        "lower_wick",
        # Multi-horizon SMA trend deviation (horizon-scaled by sigma_{t-1} * sqrt(p / 3))
        "ma_dev_30",
        "ma_dev_60",
        "ma_dev_120",
        "ma_dev_240",
        "ma_dev_360",
        # Explicit volatility regime (preserves absolute scale vs 10 bps friction)
        "vol_regime_abs",
        "vol_expansion",
        # Scale-invariant volume & order flow microstructure
        "rel_quote_vol",
        "rel_trades",
        "taker_imbalance",
    )

    STATIONARY_TIME_FEATURES = (
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
        "dayofweek_sin",
        "dayofweek_cos",
    )
```
並在 `BasicState.__init__` 中設定：
```python
        self.info_list = list(self.MARKET_FEATURES)
        self.timelist = list(self.STATIONARY_TIME_FEATURES)
```

#### 3.2.2 [`Brain/Common/DataFeature.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py)
```python
from typing import Optional, Sequence
import collections
import numpy as np
import pandas as pd
from Brain.Common.env_components import BasicState

OHLC_RAW_FIELDS = ("open", "high", "low", "close")

Prices = collections.namedtuple(
    "Prices",
    field_names=(
        *OHLC_RAW_FIELDS,
        *BasicState.MARKET_FEATURES,
        *BasicState.STATIONARY_TIME_FEATURES,
    ),
)

SOFT_CLIP_LIMIT: float = 5.0
VOL_FLOOR: float = 1e-4
VOL_REF: float = 0.005
VOL_LONG_SPAN: int = 60
VOL_LONG_MIN_PERIODS: int = 20
VOL_SHORT_SPAN: int = 12
VOL_SHORT_MIN_PERIODS: int = 6
VOL_EMA_SPAN: int = 60


def soft_clip(x: np.ndarray, limit: float = SOFT_CLIP_LIMIT) -> np.ndarray:
    """Smoothly bounds values into (-limit, +limit) via limit * tanh(x / limit)."""
    return limit * np.tanh(x / limit)
```
核心特徵計算方法 `OriginalDataFeature.cleanData` 與 `OriginalDataFeature.add_market_features` 藍圖：
```python
    def cleanData(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

        ohlc_cols = [c for c in ("Open", "High", "Low", "Close") if c in cleaned.columns]
        for col in ohlc_cols:
            cleaned.loc[cleaned[col] <= 0.0, col] = np.nan
        if ohlc_cols:
            cleaned[ohlc_cols] = cleaned[ohlc_cols].ffill(axis=0)
            cleaned = cleaned.dropna(subset=ohlc_cols)

        vol_cols = [
            c
            for c in ("Volume", "quote_av", "trades", "tb_base_av", "tb_quote_av")
            if c in cleaned.columns
        ]
        for col in vol_cols:
            cleaned.loc[cleaned[col] < 0.0, col] = 0.0
            cleaned[col] = cleaned[col].fillna(0.0)

        return cleaned.copy()

    def add_market_features(
        self,
        df: pd.DataFrame,
        periods: Sequence[int] = (30, 60, 120, 240, 360),
    ) -> pd.DataFrame:
        out = df.copy()

        o = out["Open"].to_numpy(dtype=np.float64)
        h = out["High"].to_numpy(dtype=np.float64)
        l = out["Low"].to_numpy(dtype=np.float64)
        c = out["Close"].to_numpy(dtype=np.float64)

        oc_max = np.maximum(o, c)
        oc_min = np.minimum(o, c)
        h_safe = np.maximum(h, oc_max)
        l_safe = np.minimum(l, oc_min)

        close_s = pd.Series(c, index=out.index, dtype=np.float64)
        prev_c = close_s.shift(1).to_numpy(dtype=np.float64)
        log_ret = np.log(c / prev_c)
        log_ret_s = pd.Series(log_ret, index=out.index, dtype=np.float64)

        sigma_long_s = np.sqrt(
            log_ret_s.pow(2)
            .ewm(span=VOL_LONG_SPAN, min_periods=VOL_LONG_MIN_PERIODS, adjust=False)
            .mean()
        ).clip(lower=VOL_FLOOR)
        sigma_short_s = np.sqrt(
            log_ret_s.pow(2)
            .ewm(span=VOL_SHORT_SPAN, min_periods=VOL_SHORT_MIN_PERIODS, adjust=False)
            .mean()
        ).clip(lower=VOL_FLOOR)

        sigma_prev = sigma_long_s.shift(1).to_numpy(dtype=np.float64)
        sigma_long = sigma_long_s.to_numpy(dtype=np.float64)
        sigma_short = sigma_short_s.to_numpy(dtype=np.float64)

        # 1. Orthogonal 4-DoF candlestick features (uncentered, shared sigma_{t-1})
        out["ret_gap"] = soft_clip(np.log(o / prev_c) / sigma_prev)
        out["candle_body"] = soft_clip(np.log(c / o) / sigma_prev)
        out["upper_wick"] = soft_clip(np.log(h_safe / oc_max) / sigma_prev)
        out["lower_wick"] = soft_clip(np.log(oc_min / l_safe) / sigma_prev)

        # 2. Multi-horizon SMA trend deviations (horizon-scaled by sigma_{t-1} * sqrt(p/3))
        for p in periods:
            sma = close_s.rolling(window=int(p)).mean().to_numpy(dtype=np.float64)
            scale_p = sigma_prev * np.sqrt(float(p) / 3.0)
            out[f"ma_dev_{int(p)}"] = soft_clip(np.log(c / sma) / scale_p)

        # 3. Explicit volatility regime features
        out["vol_regime_abs"] = soft_clip(np.log(sigma_long / VOL_REF))
        out["vol_expansion"] = soft_clip(np.log(sigma_short / sigma_long))

        # 4. Volume & order flow microstructure features (causal log-EMA)
        q = np.maximum(out["quote_av"].to_numpy(dtype=np.float64), 0.0)
        trades = np.maximum(out["trades"].to_numpy(dtype=np.float64), 0.0)
        tbq = np.maximum(out["tb_quote_av"].to_numpy(dtype=np.float64), 0.0)

        log_q_s = pd.Series(np.log1p(q), index=out.index, dtype=np.float64)
        log_t_s = pd.Series(np.log1p(trades), index=out.index, dtype=np.float64)
        log_q_ema = log_q_s.ewm(span=VOL_EMA_SPAN, adjust=False).mean().to_numpy(dtype=np.float64)
        log_t_ema = log_t_s.ewm(span=VOL_EMA_SPAN, adjust=False).mean().to_numpy(dtype=np.float64)

        out["rel_quote_vol"] = soft_clip(log_q_s.to_numpy(dtype=np.float64) - log_q_ema)
        out["rel_trades"] = soft_clip(log_t_s.to_numpy(dtype=np.float64) - log_t_ema)
        imb = np.where(q > 0.0, 2.0 * (tbq / np.maximum(q, 1e-12)) - 1.0, 0.0)
        out["taker_imbalance"] = np.clip(imb, -1.0, 1.0)

        out = out.dropna(subset=list(BasicState.MARKET_FEATURES)).copy()
        return out
```

#### 3.2.3 [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py)
```python
# MambaContinuousActorCritic.__init__:
#   - 移除 self.dean = DAIN_Layer(...) 與 mode 參數
#   - 清除重複的 self.log_std_min = log_std_min
#   - 保留 self.market_embedding = nn.Linear(d_model, hidden_size)（輸入前不加額外 norm）

# MambaContinuousActorCritic.forward(self, states, time_states):
        time_emb = self.time_embedding(time_states)
        time_proj = self.time_emb_projection(time_emb)

        # Pre-normalized stationary market features + portfolio state directly enter projection
        market_emb = self.market_embedding(states)

        gate = self.gate_layer(torch.cat([market_emb, time_proj], dim=-1))
        fused = self.feature_embedding(gate * market_emb + (1 - gate) * time_proj)

        out, _ = self.mixer(fused)

        temporal_feat = self.temporal_pool(out)
        if self.portfolio_encoder is not None:
            current_portfolio = states[:, -1, self.market_dim :]
            portfolio_feat = self.portfolio_encoder(current_portfolio)
            trunk = torch.cat([temporal_feat, portfolio_feat], dim=-1)
        else:
            trunk = temporal_feat

        trunk = self.trunk_norm(trunk)
        ...
```

#### 3.2.4 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)
```python
# _prepare_agent():
        self._param_groups = build_param_groups(self.agent.model)
        assert not hasattr(self.agent.model, "dean"), "DAIN layer should be removed from PPO2 model"

# _prepare_optimizer():
        lr = self.cfg_training.LEARNING_RATE
        weight_decay = self.cfg_training.WEIGHT_DECAY
        self.optimizer = build_optimizer(
            self.agent.model, lr=lr, weight_decay=weight_decay
        )

# update_ppo():
        group_norms = group_grad_norms(self._param_groups)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.model.parameters(), max_grad_norm
        )
        self.optimizer.step()
        ...
        clip_coefs.append(clip_coef(group_norms["total"], max_grad_norm))
```

---

## 4. 驗證與測試計畫 (Validation & Verification Plan)

### 4.1 驗證環境與執行指令 (Verification Environment & Commands)
> [!IMPORTANT]
> 依據專案規範，所有執行必須使用虛擬環境之 Python 直譯器：
> 絕對路徑: `/home/b0457812963/Mamba3RL/bin/python`
> 相對路徑: `../bin/python`

```bash
# 1. 執行新增之特徵與無 DAIN 架構完整單元與整合測試：
/home/b0457812963/Mamba3RL/bin/python -m pytest tests/test_vol_normalized_features_and_no_dain.py -v

# 2. 全真實資料集掃描（驗證 177 檔 CSV 無 NaN/Inf、尺度對齊、與 PPO2 環境/推論對齊）：
/home/b0457812963/Mamba3RL/bin/python tests/test_vol_normalized_features_and_no_dain.py
```

### 4.2 驗收測試查核清單 (Acceptance Verification Checklist)
- [x] **1. 數學不變性與物理性質 (Mathematical & Physical Properties)**:
  - [x] **嚴格價格尺度不變性**：對同一真實幣種（如 `UBUSDT`）之 OHLC 整體乘以 $10^{-4}$ 與 $10^{+4}$，14 維市場特徵最大絕對誤差 $< 10^{-6}$。
  - [x] **嚴格因果性（Zero Look-Ahead）**：篡改時間點 $t+1$ 之後的所有價格與成交量，驗證 $0 \dots t$ 所有 14 維特徵誤差嚴格為 `0.0`（`np.array_equal`）。
  - [x] **K 棒幾何與零點語義保留**：`upper_wick >= 0`、`lower_wick >= 0` 恆成立；`Close > Open` 時 `candle_body > 0`、`Close > SMA_p` 時 `ma_dev_p > 0`（零點符號 100% 吻合）。
  - [x] **波動率體制感知**：構造低波（`σ = 0.001`）與高波（`σ = 0.02`）合成序列，確認 `vol_regime_abs` 嚴格單調反映絕對波動規模（高波顯著大於低波）。
- [x] **2. 邊界與極值處理 (Boundary & Edge Case Handling)**:
  - [x] 零成交量 K 棒（`Volume = quote_av = trades = tb_quote_av = 0`）不產生 `NaN`/`Inf`，且 `taker_imbalance == 0.0`。
  - [x] 開頭含 `NaN`/`Inf` 或 `<= 0` 異常價格時，`cleanData()` 不執行 `bfill` 洩漏未來價格，且後續特徵計算不崩潰。
  - [x] 單根 $100\sigma$ 極端插針經 `soft_clip` 後嚴格落在 `(-5.0, +5.0)`。
- [x] **3. 訓練／推論一致與模型前向／反向驗證 (Pipeline & Model Verification)**:
  - [x] `State_time_step.encode()`（訓練）與 `PPO2DataAdapter.extract_latest_obs()`（實盤/回測）產出之 `states`（shape `(300, 15)`）與 `time_states`（shape `(300, 6)`）bit-for-bit 完全一致。
  - [x] `OriginalDataFeature.get_train_net_work_data_by_pd()` 與 `get_train_net_work_data_by_path()` 產出完全一致。
  - [x] `MambaContinuousActorCritic` 不含 `dean` 屬性與任何 `dean.*` 參數；於隨機與真實 batch 上執行 forward + PPO loss backward + `clip_grad_norm_` 皆無 `NaN`/`Inf`，且 `group_grad_norms` 正常覆蓋全部可訓練參數。
  - [x] 全真實訓練資料集（177 檔非穩定幣 CSV）載入後全部 14 維市場特徵 100% 有限值、落在 `[-5.0, 5.0]`、且 `candle_body` / `ma_dev_30..360` 跨幣種中位標準差皆對齊至 $O(1)$（`0.7 ~ 1.1`）。
  - [x] `PPO2Engine.load_checkpoint()` 不再包含 `time.sleep(100)` 阻塞碼。
