# 規格文件 (Specification): PPO2 Checkpoint 排行榜計分引擎、統計防過擬合檢定與多模型整合推論

- **類別 (Category)**: `feature`
- **規格路徑 (Spec Path)**: `spec/feature/ppo2_checkpoint_leaderboard_and_selection.md`
- **狀態 (Status)**: `Implemented & Verified`

## 📋 目錄 (Index / Contents)
1. [1. 背景與根本原因分析 (Background & Root Cause Analysis)](#1-背景與根本原因分析-background--root-cause-analysis)
2. [2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)](#2-改動範圍與架構邊界-target-scope--architecture-boundaries)
3. [3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)](#3-實作查核清單與詳細藍圖-implementation-checklist--detailed-blueprint)
4. [4. 驗證與測試計畫 (Validation & Verification Plan)](#4-驗證與測試計畫-validation--verification-plan)

---

## 1. 背景與根本原因分析 (Background & Root Cause Analysis)

### 1.1 問題陳述與情境脈絡 (Problem Statement & Context)
在 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) 的長週期多商品強化學習訓練中，系統每 `CHECKPOINT_EVERY_ITER: 10` 個 iteration 會儲存一個權重檔（如 `checkpoint_iter_24810.pt`），一次完整訓練會產生上千個 `.pt` 檔案。然而，目前的評估與測試管線存在以下關鍵缺口，導致無法科學化決定最終正式交易（`TradingSysteam.py`）應採用的權重：

1. **環境損益帳務在評估時被丟棄**：
   [`BaseTradingEnv.step()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L217-L237) 與 [`State_time_step.step()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L79-L152) 已經精確計算了每一步的複利淨對數報酬（`step_return`）、毛對數報酬（`gross_return`）、交易摩擦成本（`cost`）與換手率（`turnover`），並封裝於 `info` 字典中回傳。但 [`PPO2Evaluator.evaluate_symbol()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L181-L196) 接收 `info` 後未將任何損益資訊寫入紀錄，僅輸出 `[Datetime, Symbol, Open, Close, Action, Position]`。
2. **缺乏 Checkpoint 排行榜（`leaderboard.csv`）與風險調整後評分**：
   [`PPO2Engine.run_test()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L338-L503) 產出的 `multi_model_test_manifest.csv` 僅統計動作分布（`mean_action`, `long_ratio`, `flat_ratio`），完全沒有 Sharpe Ratio、Sortino Ratio、相對於同等平均曝險 Buy & Hold 的超額資訊比率（Information Ratio, $\text{IR}$）、最大回撤（$\text{MDD}$）或成本壓力測試（$2\times$ 成本），無法對上千個 Checkpoint 進行量化排名。
3. **缺乏針對強化學習 Policy Churn 與多重檢定偏差（Selection Bias）的防護**：
   單純挑選單點報酬最高的 Checkpoint 極易落入隨機尖峰（Lucky Spike）與回測過擬合（Backtest Overfitting）。需要引入 Google Brain / DeepMind `rliable` 標準的跨幣種去極端值平均（$\text{IQM}$, Interquartile Mean）、鄰居高原平滑分數（Plateau Score）、Deflated Sharpe Ratio（$\text{DSR}$）與回測過擬合機率（$\text{PBO}$）。
4. **缺乏粗篩抽樣（Stride）與正式交易多模型動作平均（Action Ensembling）**：
   面對上千個 `.pt` 時，[`PPO2_rl_test.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py) 缺少 `--stride` 間隔抽樣參數進行快速粗篩；且正式模式 [`PPO2Engine.get_if_order_map()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L559-L676) 僅能載入單一 `self.model_paths[0]`，無法將排行榜選出的 Top-$K$ 穩健權重進行動作平均（Action Ensemble）推論。
5. **缺乏邊訓練邊驗證的獨立 Sidecar 自動監看與增量累積機制（`--watch` 模式）**：
   目前 [`PPO2Engine.run_test()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L338-L503) 是一次性批次腳本，且每次執行都會從頭重跑所有 `.pt`。使用者希望在訓練進行中，能開啟第二個終端機視窗常駐監看 `saves/` 目錄，每隔一段時間自動偵測新產生的 `checkpoint_iter_XXX.pt`，**只對新增的權重進行驗證**，並即時增量更新與累積同一張 `leaderboard.csv`。

### 1.2 當前行為 vs. 預期行為 (Current vs. Expected Behavior)
| 維度 | 當前行為 (Current Behavior) | 預期行為 (Expected Behavior) |
| :--- | :--- | :--- |
| **單步評估輸出 (`Evaluator.py`)** | 忽略 `env.step()` 回傳之 `info`，僅紀錄 `[Datetime, Symbol, Open, Close, Action, Position]`。 | 保留原有 6 個欄位順序，並新增 `StepReturn`, `GrossReturn`, `Cost`, `Turnover`, `MarketReturn` 5 個精確帳務欄位。 |
| **單一商品指標計算** | 僅計算 `mean_action`, `std_action`, `long_ratio`, `short_ratio`, `flat_ratio`。 | 將 30 分鐘對數報酬聚合為**日報酬**（消除日內自相關），計算 `sharpe`, `sortino`, `ir_1x`（相對同曝險 B&H 之超額夏普）、`ir_2x`（2 倍摩擦成本壓力測試下之超額夏普）、`max_drawdown_pct`, `calmar`, `cum_return_pct`, `bh_return_pct`, `daily_turnover`。 |
| **跨商品匯總與排行榜 (`leaderboard.csv`)** | 無排行榜；僅輸出 (Model × Symbol) 明細表 `multi_model_test_manifest.csv`。 | 自動產出以 Checkpoint 為單位排序的 `leaderboard.csv` 與 `selection_summary.json`，採用跨商品 **IQM（中間 50% 平均）** 匯總指標，結合硬性淘汰門檻（Hard Gates）、鄰居高原平滑（`plateau_score`）、$\text{DSR}$ 與 $\text{PBO}$ 進行綜合排名，並標記推薦入選 Ensemble 的 Top-$K$ 模型。 |
| **增量快取與雙視窗自動監看 (`--watch`)** | 無增量快取（重跑會重複評估所有舊 `.pt`），無常駐監看模式。 | 1. **增量評估（Incremental Resume）**：自動讀取已完成的 `<output_dir>/<model_name>/manifest.json` 快取，跳過已評估過的 `.pt`，只計算新增的 `.pt` 並合併重排 `leaderboard.csv`。<br/>2. **Sidecar 監看模式（`--watch`）**：在另一視窗每隔 `--poll_interval` 秒自動掃描最新 `saves/` 資料夾（具備檔案寫入完成保護，防止讀取正在存檔中的半成品 `.pt`），在訓練過程中即時累積 `leaderboard.csv`。 |
| **批次粗篩與區間配置 (`PPO2_rl_test.py`)** | 只能一次跑全部 `.pt` 或手動列出每個檔案路徑。 | 支援 `--stride N`（例如每 10 個 Checkpoint 抽 1 個粗篩）、`--top_k K`（指定推薦 Ensemble 數量），以及 YAML / CLI 驗證與測試時間區間切分。 |
| **正式交易推論 (`get_if_order_map`)** | 僅載入單一模型 `self.model_paths[0]`。 | 若傳入多個 Checkpoint 路徑（Ensemble 模式），自動對各模型輸出的連續倉位取算術平均後再套用 `deadzone_threshold` 與 `rebalance_threshold`；單一模型時行為與原本 100% 相同。 |

### 1.3 技術根本原因與程式碼參照 (Technical Root Cause & References)
- 環境單步帳務已計算但未導出：[`Brain/PPO2/lib/environment.py:L120-L140`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L120-L140)、[`L226-L235`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L226-L235)
- 評估器忽略 `info` 字典：[`Brain/PPO2/lib/Evaluator.py:L181-L196`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L181-L196)
- 測試引擎缺乏財務指標聚合、Leaderboard 排名與多模型推論：[`Brain/PPO2/lib/engine.py:L338-L503`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L338-L503)、[`L559-L676`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L559-L676)
- CLI 執行入口缺少 `--stride` 與排行榜摘要輸出：[`PPO2_rl_test.py:L7-L111`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py#L7-L111)

---

## 2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)

### 2.1 檔案與模組改動矩陣 (File & Module Scope Matrix)
| 編號 | 目標檔案與行號 | 範圍 (Scope) | 規劃動作 (Planned Action) |
| :---: | :--- | :---: | :--- |
| **項目 1** | [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py) | **In-Scope** | 1. 在 `evaluate_symbol()` 中擷取 `env.step()` 的 `info` 帳務與標的單步對數報酬 `MarketReturn`。<br/>2. 新增靜態/輔助方法 `compute_symbol_metrics()`，計算日頻聚合之 Sharpe、Sortino、$\text{IR}_{1\times}$、$\text{IR}_{2\times}$、MDD、Calmar、換手率與每日淨報酬序列。 |
| **項目 2** | [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) | **In-Scope** | 1. `prepare()` 支援 `stride: int = 1` 抽樣過濾，以及自動定位最新 `saves/` 訓練目錄。<br/>2. 擴充 `run_test()` 支援**增量快取續跑（Incremental Resume）**：已評估過的 `.pt` 直接從 `<output_dir>/<model_name>/manifest.json` 載入指標，僅對新增的 `.pt` 進行推論，並合併計算跨商品 IQM、Hard Gates、Plateau Score、$\text{DSR}$ 與 $\text{PBO}$，動態更新 `leaderboard.csv` 與 `selection_summary.json`。<br/>3. 新增 `watch_and_evaluate()` 常駐監看迴圈（具備寫入完成延遲保護，防止讀到訓練存檔一半的 `.pt`）。<br/>4. 擴充 `get_if_order_map()` 支援多 Checkpoint 權重之動作平均（Action Ensembling）。 |
| **項目 3** | [`PPO2_rl_test.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py) | **In-Scope** | 新增 `--watch`（開啟獨立視窗常駐自動監看）、`--poll_interval`（預設 `30` 秒掃描一次）、`--stride`（預設 `1`）與 `--top_k`（預設 `5`）CLI 參數，並在每次評估更新後印出最新 Top-$K$ 排行榜摘要。 |
| **項目 4** | [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) | **In-Scope** | 新增選填的驗證與篩選配置參數（`VAL_START_DATE`, `VAL_END_DATE`, `LEADERBOARD_TOP_K`, `PLATEAU_WINDOW`, `MAX_MDD_GATE_PCT`, `MIN_BEAT_BH_RATIO`），完全向後相容現有設定。 |
| **項目 5** | `tests/test_ppo2_leaderboard.py` | **In-Scope [NEW]** | 新增自動化單元測試，驗證單步損益帳務一致性、$2\times$ 成本壓力計算、IQM 匯總、Plateau Score 平滑、DSR/PBO、**增量快取續跑（不重算舊 `.pt`）**、**`--watch` 動態新增 `.pt` 自動累積 `leaderboard.csv`** 以及多模型 Ensemble 推論。 |
| **項目 6** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py), [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py), [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py), [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) | **Out-of-Scope** | 訓練環境、模型架構與訓練迴圈完全不更動，確保已訓練出的所有 `.pt` 權重 100% 相容。 |

### 2.2 架構邊界與非目標 (Architectural Boundaries & Non-Goals)
- **範圍內邊界 (In-Scope Boundaries)**:
  - 所有績效指標完全基於 NumPy / Pandas / SciPy（或內建數學函式）向量化計算，不引入任何繪圖套件（如 Matplotlib）或外部回測框架依賴，維持 $O(1)$ 串流記憶體釋放特性。
  - 原有的 `multi_model_test_manifest.csv` 與各模型下的 `manifest.json`、`<symbol>_orders.csv` 全部保留並擴充指標欄位，不破壞任何既有下游讀取流程。
- **非目標 (Out-of-Scope / Non-Goals)**:
  - 本次改動不修改 PPO2 訓練邏輯與神經網路結構。
  - 不直接更動 `TradingSysteam.py` 的實盤下單邏輯（`get_if_order_map()` 輸出的 `formal_if_order_map.json` 格式維持 `{symbol: float}` 完全不變）。

---

## 3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)

### 3.1 步驟清單 (Step-by-Step Task Checklist)
- [x] **步驟 1: 擴充 [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py) 帳務紀錄與單商品績效計算**
  - [x] 在 `evaluate_symbol()` 迴圈中，紀錄每一步的 `StepReturn`（`info["step_return"]`）、`GrossReturn`（`info["gross_return"]`）、`Cost`（`info["cost"]`）、`Turnover`（`info["turnover"]`），以及標的本身單步對數報酬 `MarketReturn` $=\ln(\max(C_t, 10^{-8}) / \max(C_{t-1}, 10^{-8}))$。
  - [x] 實作 `compute_symbol_metrics(order_df, commission_perc, default_slippage)`：
    - 計算 $2\times$ 摩擦成本下之單步精確淨對數報酬：
      $$r^{(2\times)}_t = \text{GrossReturn}_t + \ln\big(\max(1.0 - 2.0 \cdot \text{Turnover}_t \cdot c, 10^{-8})\big), \quad c = \text{commission\_perc} + \text{default\_slippage}$$
    - 按日期（`Datetime` 取日曆日 $d$）將 30 分鐘對數報酬加總為每日對數報酬序列：$R_d = \sum_{t \in d} r_t$、$R^{(2\times)}_d = \sum_{t \in d} r^{(2\times)}_t$、$R^{\text{mkt}}_d = \sum_{t \in d} r^{\text{mkt}}_t$。
    - 計算期間平均持倉曝險 $\bar{w} = \text{mean}(\text{Position}_t)$，並定義同等曝險之 Buy & Hold 基準日報酬 $R^{\text{bench}}_d = \bar{w} \cdot R^{\text{mkt}}_d$，得出每日超額主動報酬（Active Return）：
      $$a^{(1\times)}_d = R_d - R^{\text{bench}}_d, \qquad a^{(2\times)}_d = R^{(2\times)}_d - R^{\text{bench}}_d$$
    - 計算年化指標（乘以 $\sqrt{365}$）：`sharpe`, `sortino`, `ir_1x`, `ir_2x`，以及資金曲線最大回撤 `max_drawdown_pct`、累積淨報酬 `cum_return_pct`、大盤累積報酬 `bh_return_pct`、`calmar`、`daily_turnover`、`trade_count`。
- [x] **步驟 2: 在 [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) 實作增量快取、跨商品 IQM 匯總、Plateau 平滑、DSR/PBO 與 `leaderboard.csv`**
  - [x] 新增數學輔助函式：
    - `_compute_iqm(values)`：計算 25% 至 75% 分位數區間之去頭去尾平均值（Interquartile Mean）。
    - `_compute_deflated_sharpe(daily_returns_matrix)`：依據 Bailey & López de Prado (2014) 計算考慮有效試驗次數 $N_{\text{eff}}$、偏態與峰度的 Deflated Sharpe Ratio 機率值（$\text{DSR} \in [0, 1]$）。
    - `_compute_cscv_pbo(daily_returns_matrix, n_partitions=8)`：依據 Bailey et al. (2017) 組合對稱交叉驗證（CSCV）計算回測過擬合機率（$\text{PBO} \in [0, 1]$）。
  - [x] 在 `prepare()` 中加入 `stride: int = 1` 參數，當 `stride > 1` 時對排序後的 `self.model_paths` 每隔 `stride` 取樣（保留最後一個 Checkpoint）。
  - [x] **實作增量快取續跑（Incremental Cache Resume）**：
    - 在 `run_test()` 中檢查 `<output_dir>/<model_name>/manifest.json` 是否已存在且包含完整評估區間與目標商品指標。
    - 若已存在快取，直接載入該 Checkpoint 的各商品指標與跨商品等權日報酬序列（耗時 $< 1\text{ ms}$，免重跑神經網路推論）。
    - 只對尚未評估過的新 `.pt` 執行 `PPO2Evaluator.evaluate_symbol()`，並將結果與快取的模型合併（依 `_natural_sort_key` 迭代順序排列）。
  - [x] 在 `run_test()` 中為每個 Checkpoint 匯總所有評估商品的績效，套用淘汰門檻（`gate_pass` / `gate_reason`），計算鄰居高原平滑分數：
    $$\text{PlateauScore}_k = \operatorname{median}_{j \in [k-w, k+w]}\big(\text{IQM\_IR}_{2\times, j}\big) - 0.5 \cdot \operatorname{MAD}_{j \in [k-w, k+w]}\big(\text{IQM\_IR}_{2\times, j}\big)$$
  - [x] 依 `(gate_pass, plateau_score, iqm_ir_2x)` 由高至低排序，產生 `leaderboard.csv` 與 `selection_summary.json`（內含推薦 Top-$K$ Checkpoint 路徑清單與整體 $\text{PBO}$ 指標）。
- [x] **步驟 3: 在 [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) 實作獨立視窗自動監看模式 (`watch_and_evaluate`)**
  - [x] 新增 `_resolve_watch_model_dir(model_dir)`：若使用者未指定 `--model_dir`，自動定位 `saves/` 下最新建立的訓練資料夾（如 `saves/20261108-xxxx-300k-`）。
  - [x] 實作檔案寫入保護：過濾掉最後修改時間距離當下不足 `min_file_age_sec`（預設 `2.0` 秒）的 `.pt` 檔案；若 `torch.load` 發生暫時性讀寫例外（如訓練程序剛寫到一半），將該檔留待下一輪輪詢重試，不會誤判為永久損壞。
  - [x] 實作 `watch_and_evaluate(output_dir, poll_interval=30.0, stride=1, top_k=5, max_idle_polls=None)`：
    - 每隔 `poll_interval` 秒掃描目標資料夾。
    - 當偵測到尚未完成 `manifest.json` 快取的新 `.pt` 時，觸發 `run_test()` 增量評估新權重並重新計算與累積 `leaderboard.csv`。
    - 支援 `Ctrl+C` (`KeyboardInterrupt`) 安全中斷退出，保證已寫入的 CSV / JSON 完整不損毀。
- [x] **步驟 4: 在 [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) 支援多模型 Ensemble 正式推論**
  - [x] 擴充 `get_if_order_map()`：當 `len(self.model_paths) > 1` 時，依序載入每個 Checkpoint 對各商品最新觀測視窗進行無梯度確定性推論（`apply_deadzone=False` 取得原始 `[0.0, 1.0]` 目標倉位），取算術平均後再統一套用 `deadzone_threshold` 與 `rebalance_threshold`，輸出最終 `formal_if_order_map.json`。
- [x] **步驟 5: 更新 [`PPO2_rl_test.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py) 與 [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml)**
  - [x] 在 `PPO2_rl_test.py` 增加 `--watch`、`--poll_interval`（預設 `30` 秒）、`--stride` 與 `--top_k` 參數，並在每次評估完成後於終端機印出 `leaderboard.csv` 前 $K$ 名表格與對應的 `--formal --models ...` 執行指令提示。
- [x] **步驟 6: 代碼防禦與清理 (Clean up & Defensiveness)**
  - [x] 確保當商品無交易、標準差為 0、評估天數不足（$< 2$ 天）或只評估單一 Checkpoint（$N=1$）時，所有統計指標皆有安全的預設防禦（不產生 `NaN` 或 `ZeroDivisionError`）。

### 3.2 詳細程式碼修改藍圖 (Detailed Code Modifications Blueprint)

#### 3.2.1 [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py)
在 `evaluate_symbol()` 中保留既有欄位並新增 5 個損益帳務欄位：
```python
        columns = [
            "Datetime",
            "Symbol",
            "Open",
            "Close",
            "Action",
            "Position",
            "StepReturn",
            "GrossReturn",
            "Cost",
            "Turnover",
            "MarketReturn",
        ]
```
在逐步推論迴圈中記錄精確單步數值：
```python
        with torch.no_grad():
            while not done and env._state._offset < end_offset:
                curr_offset = env._state._offset
                dt = datetime_series[curr_offset]
                open_val = float(open_series[curr_offset])
                close_val = float(close_series[curr_offset])
                prev_close_val = max(float(close_series[curr_offset - 1]), 1e-8)
                mkt_log_ret = float(np.log(max(close_val, 1e-8) / prev_close_val))

                out = self.agent.get_action(obs, stochastic=False)
                action_val = float(np.clip(float(out.clipped_action.item()), 0.0, 1.0))

                obs, reward, terminated, truncated, info = env.step(action_val)
                done = bool(terminated or truncated)
                pos_val = float(env._state.position)

                records.append({
                    "Datetime": dt,
                    "Symbol": symbol,
                    "Open": open_val,
                    "Close": close_val,
                    "Action": action_val,
                    "Position": pos_val,
                    "StepReturn": float(info.get("step_return", 0.0)),
                    "GrossReturn": float(info.get("gross_return", 0.0)),
                    "Cost": float(info.get("cost", 0.0)),
                    "Turnover": float(info.get("turnover", 0.0)),
                    "MarketReturn": mkt_log_ret,
                })
```

並新增 `compute_symbol_metrics()` 方法：
```python
    @staticmethod
    def compute_symbol_metrics(
        order_df: pd.DataFrame,
        commission_perc: float = 0.0001,
        default_slippage: float = 0.0001,
    ) -> Dict[str, Any]:
        """
        根據單一商品之逐筆評估明細計算日頻聚合風險調整後指標與成本壓力測試。
        """
```

#### 3.2.2 [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py)
`leaderboard.csv` 輸出的核心欄位結構與常駐監看介面：
```python
        leaderboard_columns = [
            "rank",
            "model_name",
            "model_path",
            "gate_pass",
            "gate_reason",
            "ensemble_selected",
            "plateau_score",
            "iqm_ir_2x",
            "iqm_ir_1x",
            "iqm_sharpe",
            "median_sharpe",
            "iqm_sortino",
            "median_mdd_pct",
            "p90_mdd_pct",
            "iqm_cum_return_pct",
            "beat_bh_ratio",
            "positive_return_ratio",
            "mean_exposure",
            "mean_daily_turnover",
            "dsr_prob",
            "evaluated_symbols",
        ]
```
常駐監看與增量累積介面 `watch_and_evaluate()`：
```python
    def watch_and_evaluate(
        self,
        output_dir: str = "results/ppo2_orders",
        model_dir: Optional[str] = None,
        target_symbols: Optional[List[str]] = None,
        poll_interval: float = 30.0,
        stride: int = 1,
        top_k: int = 5,
        min_file_age_sec: float = 2.0,
        max_idle_polls: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Sidecar 常駐監看模式：定期掃描 saves/（或指定 model_dir）中的新 .pt 權重，
        略過已具備 manifest.json 快取的舊權重，僅對新增權重進行評估並即時更新 leaderboard.csv。
        """
```

---

## 4. 驗證與測試計畫 (Validation & Verification Plan)

### 4.1 驗證環境與執行指令 (Verification Environment & Commands)
> [!IMPORTANT]
> 依據專案規範，所有執行必須使用虛擬環境之 Python 直譯器：
> 絕對路徑: `/home/b0457812963/Mamba3RL/bin/python`
> 相對路徑: `../bin/python`

```bash
# 1. 執行排行榜、增量快取續跑與 --watch 自動監看之單元測試:
../bin/python -m unittest tests/test_ppo2_leaderboard.py -v

# 2. 單次批次測試與產出 leaderboard.csv 驗證:
../bin/python PPO2_rl_test.py --model_dir Brain/PPO2/Meta --symbols AAVEUSDT AGLDUSDT --output_dir results/ppo2_leaderboard_verify

# 3. 訓練期間開啟第二個視窗自動監看並即時累積 leaderboard.csv 之實際指令:
../bin/python PPO2_rl_test.py --watch --poll_interval 30 --output_dir results/ppo2_live_leaderboard
```

### 4.2 驗收測試查核清單 (Acceptance Verification Checklist)
- [x] **1. 功能正確性 (Functional Correctness)**:
  - [x] `order_df` 保留原有 6 個欄位並正確附加 `StepReturn`, `GrossReturn`, `Cost`, `Turnover`, `MarketReturn`，且 `StepReturn` 與 `State_time_step` 複利公式誤差 $< 10^{-7}$。
  - [x] `leaderboard.csv` 與 `selection_summary.json` 順利寫入目標輸出資料夾，且所有指標欄位無 `NaN` 或 `inf`。
  - [x] **增量快取與 `--watch` 自動監看**：當新 `.pt` 檔案加入監看目錄時，只會評估新增的 `.pt`（舊 `.pt` 直接由 `manifest.json` 秒級讀取快取），並將新舊模型合併重排更新至同一張 `leaderboard.csv`。
  - [x] `get_if_order_map()` 在傳入單一或多個 `.pt` 權重時皆能輸出合法 `[0.0, 1.0]` 範圍內的 `formal_if_order_map.json`。
- [x] **2. 邊界與極值處理 (Boundary & Edge Case Handling)**:
  - [x] 當訓練端正在寫入 `.pt`（檔案剛建立未滿 `min_file_age_sec` 或暫時無法讀取）時，`--watch` 能安全等待下一輪輪詢而不崩潰、也不會誤將該檔永久跳過。
  - [x] 當模型全程空手（`Position == 0`）或全程滿倉不交易（`Turnover == 0`）時，不會發生除以零例外，且能被 Hard Gate 正確識別。
  - [x] 當只測試 1 個 Checkpoint（$N=1$）時，`plateau_score` 自動退化為該模型自身的 `iqm_ir_2x`，不發生索引越界。
- [x] **3. 實際執行與經驗驗證 (Runtime & Empirical Verification)**:
  - [x] 單元測試 `tests/test_ppo2_leaderboard.py` 全數通過。
  - [x] 實際載入 `Brain/PPO2/Meta/checkpoint_iter_24810.pt` 執行 `PPO2_rl_test.py` 成功產出 `leaderboard.csv`。
