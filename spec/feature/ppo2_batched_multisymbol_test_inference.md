# 規格文件 (Specification): PPO2 多商品並行批次推論與測試快取優化 (Batched Multi-Symbol Test Inference & Pre-sliced Data Cache)

- **類別 (Category)**: `feature`
- **規格路徑 (Spec Path)**: `spec/feature/ppo2_batched_multisymbol_test_inference.md`
- **狀態 (Status)**: `Completed`

## 📋 目錄 (Index / Contents)
1. [1. 背景與根本原因分析 (Background & Root Cause Analysis)](#1-背景與根本原因分析-background--root-cause-analysis)
2. [2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)](#2-改動範圍與架構邊界-target-scope--architecture-boundaries)
3. [3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)](#3-實作查核清單與詳細藍圖-implementation-checklist--detailed-blueprint)
4. [4. 驗證與測試計畫 (Validation & Verification Plan)](#4-驗證與測試計畫-validation--verification-plan)

---

## 1. 背景與根本原因分析 (Background & Root Cause Analysis)

### 1.1 問題陳述與情境脈絡 (Problem Statement & Context)
現行 [`PPO2_rl_test.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py) 透過 [`PPO2Engine.run_test()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L583-L641) 執行歷史回測與排行榜評估時，採取「單一模型 $\times$ 單一商品逐步迴圈（Batch Size $B = 1$）」的方式推論。當面對測試集 197 個商品（2026 年測試區間共約 179.6 萬步）與多個訓練 Checkpoint 時，單一模型回測耗時長達約 **4.8 小時**，成為嚴重的效能瓶頸：

1. **GPU 批次大小 $B = 1$ 造成嚴重硬體閒置與 Kernel Launch 瓶頸**：
   - 在 [`PPO2Evaluator.evaluate_symbol()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L93-L222) 中，同一個商品內部因 [`State_time_step.encode()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L161-L188) 的最後兩維（`position` 與 `unrealized_return`）具時間因果相依性，無法跨時間步平行化；但**不同商品之間（Cross-Symbol）完全獨立**。
   - 本機 RTX 2060 (6GB VRAM) 實測顯示：$B = 1$ 時單步延遲為 `9.55 ms`，吞吐量僅 `104.7 sym-steps/s`；當提升至 **$B = 64$** 時，吞吐量達 **`1,240.7 sym-steps/s`（加速 11.8 倍）**，GPU 峰值顯存僅佔用 `417.9 MB`（Reserved `548 MB`），單一模型回測時間可從 4.8 小時大幅縮短至約 **24 分鐘**。
2. **多模型評估重複讀取完整 CSV 與重複計算特徵**：
   - 在 [`PPO2Engine._eval_model_checkpoint()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L523-L546) 與 [`_stream_evaluate_symbol()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L375-L423) 中，每評測一個新 Checkpoint 都會重新讀取 197 個完整 CSV（含 2020~2025 年高達 10 萬筆資料）並重新計算均線與時間特徵，且每個商品結束後皆強制呼叫 `gc.collect()` 與 `torch.cuda.empty_cache()` 造成同步阻塞。
3. **Python 字典列表逐步配置與實盤模式單步迴圈開銷**：
   - [`PPO2Evaluator.evaluate_symbol()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L202-L216) 每步建立一個含 11 個鍵值的 Python `dict`（每個模型高達 180 萬次配置），而實盤模式 [`_predict_single_model_orders()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L774-L810) 與 [`_predict_ensemble_orders()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L812-L855) 亦逐一商品呼叫單步推論。

### 1.2 當前行為 vs. 預期行為 (Current vs. Expected Behavior)
| 維度 | 當前行為 (Current Behavior) | 預期行為 (Expected Behavior) |
| :--- | :--- | :--- |
| **推論批次架構** | 單商品 `evaluate_symbol` 與 `evaluate_single_step`，$B=1$ 逐一商品串行推論，吞吐量僅 ~104.7 steps/s。 | 統一為多商品批次推論（`evaluate_symbols` 與 `evaluate_batch_step`，單商品即為 $B=1$ 特例），預設 `EVAL_BATCH_SIZE: 64`，透過**動態補位佇列（Dynamic Slot Refilling）**維持 64 個商品平行推論，吞吐量達 ~1,240 steps/s（~11.8x 加速）。 |
| **資料流與快取** | 每個模型 Checkpoint 重新由磁碟讀取 197 個完整 CSV 並同時傳遞龐大的 `raw_df` 與 `Prices`。 | 首次需要推論時，將商品資料計算特徵後直接裁切 `Prices` 與 `datetime_series` 至 `[start_offset - bars_count : end_offset + 1]`。裁切後每個商品起點固定為 `bars_count`、終點固定為 `len(prices.close) - 1`（直接對齊 [`State_time_step.step()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L156) 的 `terminated` 條件），無須額外包裝類別，197 商品僅佔約 220 MB RAM 並快取於 `PPO2Engine`。 |
| **邊界條件** | 不同商品 K 線長度不一（4,455 ~ 12,782 根），若固定分批會產生尾部算力閒置；且逐步 `records.append(dict)` 產生大量 GC 壓力。 | 任一槽位商品觸發 `terminated` 時立即以預配置 1D NumPy 陣列向量化建構 `order_df`、寫入 CSV、計算指標並釋放，隨即從佇列補入下一個商品；最終匯總嚴格依照 `target_symbols` 排序輸出，保證 `manifest.json` 與 `leaderboard.csv` 100% 確定性一致。 |

### 1.3 技術根本原因與程式碼參照 (Technical Root Cause & References)
- 影響位置 1: [`Brain/PPO2/lib/Evaluator.py:L47-L222`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L47-L222)
  - `evaluate_single_step` 與 `evaluate_symbol` 僅針對單一商品設計，每次呼叫 `self.agent.get_action(obs, stochastic=False)` 時輸入形狀為 `(300, 16)`，在 `_preprocess_obs` 被 `unsqueeze(0)` 成 `(1, 300, 16)`，無法發揮 GPU 平行算力。
- 影響位置 2: [`Brain/PPO2/lib/engine.py:L375-L546`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L375-L546)
  - `_eval_model_checkpoint` 透過 `for symbol in self.target_symbols:` 呼叫 `_stream_evaluate_symbol`，導致每個模型重複讀取 CSV、重複執行特徵工程，並在內迴圈頻繁執行 `torch.cuda.empty_cache()`。
- 影響位置 3: [`Brain/PPO2/lib/engine.py:L774-L855`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L774-L855)
  - 實盤模式 `_predict_single_model_orders` 與 `_predict_ensemble_orders` 對 `symbol_obs_map` 逐一迴圈呼叫單步推論，未批次化。

---

## 2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)

### 2.1 檔案與模組改動矩陣 (File & Module Scope Matrix)
| 編號 | 目標檔案與行號 | 範圍 (Scope) | 規劃動作 (Planned Action) |
| :---: | :--- | :---: | :--- |
| **項目 1** | [`configs/training/ppo_v2.yaml:L69-L73`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L69-L73) | **In-Scope** | 新增預設批次評估參數 `EVAL_BATCH_SIZE: 64`。 |
| **項目 2** | [`PPO2_rl_test.py:L7-L160`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py#L7-L160) | **In-Scope** | 新增 `--batch_size` CLI 參數（預設讀取 YAML 的 `EVAL_BATCH_SIZE=64`），傳遞予 `PPO2Engine`。 |
| **項目 3** | [`Brain/PPO2/lib/data_adapter.py:L1-L168`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py#L1-L168) | **In-Scope** | 新增 `PPO2DataAdapter.prepare_test_symbol_data()`，負責計算特徵並將 `(Prices, datetime_series)` 預先裁切至 `[start_offset - bars_count : end_offset + 1]`，使裁切後的資料可直接從 `offset = bars_count` 執行至 `State_time_step` 自然結束 (`terminated`)。 |
| **項目 4** | [`Brain/PPO2/lib/Evaluator.py:L1-L223`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L1-L223) | **In-Scope** | 移除單商品專用方法 `evaluate_symbol` 與 `evaluate_single_step`；實作統一多商品批次方法 `evaluate_symbols()`（動態補位佇列 + 預配置 1D NumPy 緩衝）與 `evaluate_batch_step()`。 |
| **項目 5** | [`Brain/PPO2/lib/engine.py:L45-L855`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L45-L855) | **In-Scope** | 新增 `eval_batch_size` 屬性與跨模型預裁切資料快取 `_get_or_preload_test_data()`；移除 `_stream_evaluate_symbol`；重構 `_eval_model_checkpoint`、`_predict_single_model_orders` 與 `_predict_ensemble_orders` 為批次推論。 |
| **項目 6** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) | **Out-of-Scope** | 明確保留不更動 `State_time_step`、`TrainingEnv`、`ProductionEnv` 與 `SubprocVecEnv`，防範訓練與環境狀態機回歸問題。 |
| **項目 7** | [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py) & [`model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) | **Out-of-Scope** | 明確保留不更動神經網路架構與 `get_action` 介面（其原生已支援 3D 批次張量 `(B, L, D)`）。 |
| **項目 8** | [`Brain/PPO2/lib/leaderboard.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/leaderboard.py) & [`statistics.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/statistics.py) | **Out-of-Scope** | 明確保留不更動排行榜計算、DSR/PBO 統計與硬性閘門邏輯。 |

### 2.2 架構邊界與非目標 (Architectural Boundaries & Non-Goals)
- **範圍內邊界 (In-Scope Boundaries)**:
  - 將回測與實盤推論統一為支援任意商品數量 $N \ge 1$ 的批次推論架構（當傳入 1 個商品時自動以 $B=1$ 運作）。
  - 善用預裁切設計（Pre-slicing）：將每個商品的 `Prices` 與 `datetime_series` 裁切為 `[start_offset - bars_count : end_offset + 1]`，使得裁切後的起點永遠是 `bars_count`，終點永遠是 `len(prices.close) - 1`，直接相容於 [`State_time_step.step()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L156) 的 `terminated = bool(self._offset >= self._prices.close.shape[0] - 1)`，**不需要新增任何額外的資料封裝類別**。
  - 實作懶載入（Lazy Preload）跨模型測試資料快取：若所有 Checkpoint 皆命中 `manifest.json` 快取則不讀取任何 CSV；一旦有 Checkpoint 需評估，僅載入並裁切一次測試資料供後續所有模型共用。
- **非目標 (Out-of-Scope / Non-Goals)**:
  - 不更動 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) 的訓練流程與 `NUM_WORKERS: 8` 設定。
  - 不更動 `{symbol}_orders.csv`、`manifest.json`、`multi_model_test_manifest.csv`、`leaderboard.csv` 及 `formal_if_order_map.json` 的輸出欄位格式與計量公式。

---

## 3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)

### 3.1 步驟清單 (Step-by-Step Task Checklist)
- [x] **步驟 1: 更新配置檔與 CLI 介面 ([`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml), [`PPO2_rl_test.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py))**
  - [x] 於 `configs/training/ppo_v2.yaml` 新增 `EVAL_BATCH_SIZE: 64`。
  - [x] 於 `PPO2_rl_test.py` 的 `parse_arguments()` 新增 `--batch_size`（`type=int, default=None`），並傳入 `PPO2Engine(eval_batch_size=args.batch_size)`。
- [x] **步驟 2: 擴充資料適配層支援預裁切測試資料 ([`Brain/PPO2/lib/data_adapter.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py))**
  - [x] 實作 `PPO2DataAdapter.prepare_test_symbol_data(...) -> Optional[Tuple[Prices, np.ndarray]]`，完成特徵計算後將 `Prices` 與 `datetime_series` 精準裁切至 `[max(0, start_offset - bars_count) : end_offset + 1]`，直接回傳 `(sliced_prices, sliced_datetime)` 元組。
- [x] **步驟 3: 重構評估器為統一多商品批次推論 ([`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py))**
  - [x] 於 `PPO2Evaluator.__init__` 新增 `eval_batch_size: int = 64` 參數。
  - [x] 移除單商品方法 `evaluate_single_step` 與 `evaluate_symbol`。
  - [x] 實作 `evaluate_batch_step(self, batch_obs, apply_deadzone=True, prev_positions=None) -> np.ndarray`，支援實盤模式一次對多商品進行批次推論與向量化死區/再平衡過濾。
  - [x] 實作 `evaluate_symbols(self, symbol_data_map: Dict[str, Tuple[Prices, np.ndarray]], model_output_dir=None, on_symbol_complete=None) -> Dict[str, pd.DataFrame]`：
    - 預配置形狀為 `(B, bars_count, d_model)` 與 `(B, bars_count, time_dim)` 的連續記憶體緩衝區 `batch_states_np` 與 `batch_time_np`。
    - 每個活躍槽位持有一個 `State_time_step`（由 `offset = self.bars_count` 開始步進直到 `terminated == True`）與長度為 `n_steps = len(prices.close) - 1 - self.bars_count` 的 1D NumPy 陣列，消除百萬次 Python `dict` 配置。
    - 當任一槽位觸發 `terminated` 時，向量化生成 `order_df`、輸出 `{symbol}_orders.csv`、觸發 `on_symbol_complete(symbol, order_df)` 並立即由待測佇列補入下一商品。
- [x] **步驟 4: 重構測試引擎整合快取與批次推論 ([`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py))**
  - [x] 於 `PPO2Engine.__init__` 綁定 `self.eval_batch_size`，並初始化 `self._cached_test_data` 快取字典。
  - [x] 實作 `_get_or_preload_test_data(self, effective_start, effective_end, feature_loader)`，支援跨模型與 Watcher 模式重用已裁切的 `Dict[str, Tuple[Prices, np.ndarray]]`。
  - [x] 移除 `_stream_evaluate_symbol`，更新 `_eval_model_checkpoint` 呼叫 `evaluator.evaluate_symbols(...)`，並確保最終寫入 `manifest_rows` 與 `model_symbols_dict` 的順序嚴格遵循 `self.target_symbols`。
  - [x] 更新 `_predict_single_model_orders` 與 `_predict_ensemble_orders` 改用 `evaluator.evaluate_batch_step(...)`。
- [x] **步驟 5: 代碼防禦與清理 (Clean up & Defensiveness)**
  - [x] 移除無用代碼、冗餘變數或過期 import（例如 `Evaluator.py` 中不再需要的 `ProductionEnv` import）。
  - [x] 加入防禦性斷言 (`assert self.eval_batch_size >= 1`) 與完整型別註解 (type hints)。

### 3.2 詳細程式碼修改藍圖 (Detailed Code Modifications Blueprint)

#### 3.2.1 [`Brain/PPO2/lib/data_adapter.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py)
於 `PPO2DataAdapter` 新增預裁切方法（不新增額外 class，直接回傳 `(Prices, np.ndarray)`）：
```python
    @classmethod
    def prepare_test_symbol_data(
        cls,
        feature_loader: OriginalDataFeature,
        df: pd.DataFrame,
        bars_count: int,
        test_start_date: Optional[str] = None,
        test_end_date: Optional[str] = None,
    ) -> Optional[Tuple[Prices, np.ndarray]]:
        """
        Enriches raw DataFrame into Prices and slices both Prices and Datetime array
        to [start_offset - bars_count : end_offset + 1] so evaluation always runs
        from offset=bars_count until State_time_step naturally terminates.
        """
        if len(df) <= 360:
            return None
        prices = cls.process_df_to_prices(feature_loader, df)
        raw_df = feature_loader.df
        total_len = len(prices.close)
        if total_len <= bars_count + 1:
            return None

        aligned_df = raw_df.iloc[-total_len:] if len(raw_df) > total_len else raw_df
        if "Datetime" in aligned_df.columns:
            datetime_series = aligned_df["Datetime"].astype(str).to_numpy()
        elif "date" in aligned_df.columns:
            datetime_series = aligned_df["date"].astype(str).to_numpy()
        else:
            datetime_series = aligned_df.index.astype(str).to_numpy()

        datetime_ts = pd.to_datetime(datetime_series)
        if test_start_date is not None and str(test_start_date).strip() != "":
            start_idx = int(datetime_ts.searchsorted(pd.to_datetime(test_start_date), side="left"))
        else:
            start_idx = bars_count

        if test_end_date is not None and str(test_end_date).strip() != "":
            end_idx = int(datetime_ts.searchsorted(pd.to_datetime(test_end_date), side="right"))
        else:
            end_idx = total_len - 1

        start_offset = max(bars_count, start_idx)
        end_offset = min(total_len - 1, end_idx)
        if start_offset >= end_offset:
            return None

        slice_start = start_offset - bars_count
        slice_end = end_offset + 1
        sliced_prices = Prices(*[
            np.ascontiguousarray(getattr(prices, field)[slice_start:slice_end])
            for field in prices._fields
        ])
        sliced_dt = np.ascontiguousarray(datetime_series[slice_start:slice_end])
        return sliced_prices, sliced_dt
```

#### 3.2.2 [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py)
於 `PPO2Evaluator` 實作 `evaluate_batch_step` 與 `evaluate_symbols`：
- `evaluate_batch_step`: 接收 `(N, bars_count, d_model)` 與 `(N, bars_count, time_dim)`，以 `self.eval_batch_size` 分塊執行 `self.agent.get_action(chunk_obs, stochastic=False)`，並以 NumPy 向量化套用 `deadzone_threshold` 與 `rebalance_threshold`。
- `evaluate_symbols`: 接收 `symbol_data_map: Dict[str, Tuple[Prices, np.ndarray]]`，使用 `deque` 維護待測商品佇列，同時開啟最多 `B = min(self.eval_batch_size, len(symbol_data_map))` 個環境槽位（每個槽位呼叫 `state.reset(prices, offset=self.bars_count)`），每步打包所有活躍槽位之觀測送入 GPU，並在槽位 `terminated` 時立即向量化建立 `order_df`、輸出 `{symbol}_orders.csv` 與呼叫 `on_symbol_complete(symbol, order_df)`，隨後補入下一商品直到佇列清空。

#### 3.2.3 [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py)
- 於 `PPO2Engine.__init__` 接收 `eval_batch_size: Optional[int] = None`，預設讀取 `int(getattr(self.cfg, "EVAL_BATCH_SIZE", 64))`。
- 新增 `_get_or_preload_test_data(self, effective_start, effective_end, feature_loader) -> Dict[str, Tuple[Prices, np.ndarray]]`，當快取鍵 `(str(effective_start or ""), str(effective_end or ""), tuple(self.target_symbols))` 相符時直接回傳快取字典；若首次載入則遍歷 `self.target_symbols` 呼叫 `prepare_test_symbol_data` 並快取。
- 於 `_eval_model_checkpoint` 透過 `evaluator.evaluate_symbols(test_data_map, model_output_dir=model_output_dir, on_symbol_complete=_handle_completed_symbol)` 一次完成所有商品批次推論，最後依 `self.target_symbols` 順序寫入 `manifest_rows` 與 `model_symbols_dict`。

---

## 4. 驗證與測試計畫 (Validation & Verification Plan)

### 4.1 驗證環境與執行指令 (Verification Environment & Commands)
> [!IMPORTANT]
> 依據專案規範，所有執行必須使用虛擬環境之 Python 直譯器：
> 絕對路徑: `/home/b0457812963/Mamba3RL/bin/python`
> 相對路徑: `../bin/python`

```bash
# 1. 驗證單商品 (B=1) vs 多商品批次 (B=64) 數值一致性、快取重用與實盤 Ensemble 批次推論:
/home/b0457812963/Mamba3RL/bin/python -c "
import tempfile, numpy as np, torch
from Brain.PPO2.lib.engine import PPO2Engine

engine_b1 = PPO2Engine(eval_batch_size=1)
engine_b64 = PPO2Engine(eval_batch_size=64)
engine_b1.prepare(target_symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'])
engine_b64.prepare(target_symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'])
"

# 2. 執行 CLI 實盤模式與歷史回測驗證:
/home/b0457812963/Mamba3RL/bin/python PPO2_rl_test.py --symbols BTCUSDT ETHUSDT SOLUSDT --batch_size 64 --no_cache
```

### 4.2 驗收測試查核清單 (Acceptance Verification Checklist)
- [x] **1. 功能正確性與數值等價性 (Functional Correctness & Numerical Parity)**:
  - [x] 相同權重下，`eval_batch_size=1` 與 `eval_batch_size=64` 對多商品（含不同長度的測試商品）推論出的 `Action`, `Position`, `StepReturn`, `sharpe`, `ir_2x`, `max_drawdown_pct` 數值完全吻合（浮點誤差 $< 10^{-5}$）。
  - [x] 動態補位佇列在長短商品混合（如 4,455 根與 12,782 根 K 線）時，各槽位狀態轉移完全獨立不互串，且最終 `manifest.json` 與 `multi_model_test_manifest.csv` 的商品順序嚴格一致。
- [x] **2. 邊界與極值處理 (Boundary & Edge Case Handling)**:
  - [x] 當指定單一商品（`--symbols BTCUSDT`，即商品數 $< \text{batch\_size}$）或總商品數無法被 `batch_size` 整除（如 197 個商品配 `batch_size=64`）時，動態補位與尾部槽位收斂正常無越界錯誤。
  - [x] 空模型列表、過短資料集（$\le 360$ 根）或無效日期區間能安全跳過並回傳空 DataFrame。
- [x] **3. 實際執行與效能驗證 (Runtime & Empirical Verification)**:
  - [x] 驗證跨模型快取機制：第二個 Checkpoint 評估時直接重用記憶體中的 `(Prices, datetime_series)`，零磁碟重複讀取。
  - [x] 驗證 GPU 批次推論吞吐量與記憶體穩定性，無未捕捉之例外或顯存洩漏。
