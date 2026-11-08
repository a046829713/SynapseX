# 規格文件 (Specification): PPO2 Engine 模組化架構重構 (Modular Refactoring)

- **類別 (Category)**: `feature`
- **規格路徑 (Spec Path)**: `spec/feature/ppo2_engine_modular_refactoring.md`
- **狀態 (Status)**: `Completed`

## 📋 目錄 (Index / Contents)
1. [1. 背景與根本原因分析 (Background & Root Cause Analysis)](#1-背景與根本原因分析-background--root-cause-analysis)
2. [2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)](#2-改動範圍與架構邊界-target-scope--architecture-boundaries)
3. [3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)](#3-實作查核清單與詳細藍圖-implementation-checklist--detailed-blueprint)
4. [4. 驗證與測試計畫 (Validation & Verification Plan)](#4-驗證與測試計畫-validation--verification-plan)

---

## 1. 背景與根本原因分析 (Background & Root Cause Analysis)

### 1.1 問題陳述與情境脈絡 (Problem Statement & Context)
現行 [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) 代碼量已達 1,546 行，承擔了過多不同層次的系統職責，違反單一職責原則（Single Responsibility Principle, SRP），形成典型的「上帝類別（God Class）」與「巨型函數（Long Method）」：
1. **單一函數過於龐大，閱讀與除錯成本極高**：
   - [`run_test()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L914-L1224)（310 行）：同時處理快取命中檢查、權重未寫完保護（File Age Guard）、Agent 動態建構、巢狀雙重遍歷（Model × Symbol）、調用 Evaluator、日收益率聚合、原子寫入 Manifest、強制顯存/內存垃圾回收（Garbage Collection），並在最後觸發 Leaderboard。
   - [`_build_and_save_leaderboard()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L646-L823)（177 行）：同時負責時序自然排序、Plateau 鄰域中位數與 MAD 平滑評分、跨檢查點 $D \times N$ 矩陣重組、DSR 與 PBO 計算、硬性閘門檢驗、Top-K 集成挑選、CSV/JSON 雙格式原子寫入。
   - [`get_if_order_map()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L1389-L1546)（157 行）：同時處理 DataFrame / Dict / CSV 三種資料來源適配、單模型推論 vs 多模型集成（Action Ensemble）推論、死區與再平衡閾值應用、寫入實盤委託 JSON。
2. **底層計量統計演算法（約 210 行）污染核心引擎**：
   - Acklam 逆累積分佈函數演算法（[`_norm_ppf`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L39-L93) 的大段多項式常數）、標準常態分佈累計（[`_norm_cdf`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L32-L37)）、IQM（[`_compute_iqm`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L95-L113)）、Bailey & López de Prado 的 Deflated Sharpe Ratio（[`_compute_deflated_sharpe`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L115-L180)）、Bailey 等人的 CSCV PBO（[`_compute_cscv_pbo`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L182-L233)）本質為通用量化統計與風險度量方法，硬編碼在 `engine.py` 內嚴重干擾主流程閱讀。
3. **特徵清洗與觀測截取適配器缺乏抽象**：
   - [`_process_df_to_prices`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L1317-L1370) 與 [`_extract_latest_obs`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L494-L550) 屬於特徵工程與觀測適配層，與測試引擎生命週期無直接相依。
4. **代碼重複（違反 DRY 原則）**：
   - [`_apply_deadzone_and_rebalance`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L1371-L1388) 與 [`PPO2Evaluator.evaluate_single_step`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L71-L91) 重複實作了相同的死區與再平衡邏輯。

### 1.2 當前行為 vs. 預期行為 (Current vs. Expected Behavior)
| 維度 | 當前行為 (Current Behavior) | 預期行為 (Expected Behavior) |
| :--- | :--- | :--- |
| **檔案結構** | 單一 1,546 行的 [`engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) 包含計量統計、資料前處理、觀測提取、排行榜過濾、多模型測試、委託輸出。 | 職責清晰分離：<br>1. `statistics.py`：純量化統計計算<br>2. `data_adapter.py`：資料清洗、特徵與觀測適配<br>3. `leaderboard.py`：排行榜管理、閘門檢驗與選拔<br>4. `engine.py`：高階調度 Facade（瘦身至約 350 行）。 |
| **函數長度** | `run_test` (310 行), `_build_and_save_leaderboard` (177 行), `get_if_order_map` (157 行)。 | 拆解為單一關注點的私有子方法，單一函數長度皆小於 40~50 行。 |
| **資料載體** | 散落在多個 Dict 中，缺乏型別約束與明確欄位定義。 | 透過 `@dataclass` 建立清晰的資料載體（`GateConfig`, `ModelSummaryRecord`, `PortfolioState`）。 |
| **向後相容性** | 測試與外部工具直接引用 `engine.py`。 | `engine.py` 保持對外公開介面與統計函式 alias import，[`tests/test_ppo2_leaderboard.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_ppo2_leaderboard.py) 與 [`PPO2_rl_test.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py) 100% 零修改相容。 |

---

## 2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)

### 2.1 檔案與模組改動矩陣 (File & Module Scope Matrix)
| 編號 | 目標檔案 | 範圍 (Scope) | 規劃動作 (Planned Action) |
| :---: | :--- | :---: | :--- |
| **項目 1** | [`Brain/PPO2/lib/statistics.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/statistics.py) | **In-Scope (New)** | 新增計量統計模組：包含 `_natural_sort_key`, `norm_cdf`, `norm_ppf`, `compute_iqm`, `compute_deflated_sharpe`, `compute_cscv_pbo`。 |
| **項目 2** | [`Brain/PPO2/lib/data_adapter.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py) | **In-Scope (New)** | 新增資料適配模組：包含 `PortfolioState` dataclass、`PPO2DataAdapter` 類別，封裝 DataFrame 清洗至 Prices 及最新觀測截取。 |
| **項目 3** | [`Brain/PPO2/lib/leaderboard.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/leaderboard.py) | **In-Scope (New)** | 新增排行榜與篩選管理模組：包含 `GateConfig`、`ModelSummaryRecord` dataclass，以及 `LeaderboardManager` 類別。 |
| **項目 4** | [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) | **In-Scope** | 瘦身為 Facade 調度器，委託專職模組，拆解巨型函數（`run_test`, `get_if_order_map` 等），並保留相容 alias import。 |
| **項目 5** | [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py) | **Out-of-Scope** | 維持單一標的滾動評估與指標計算不變。 |
| **項目 6** | [`PPO2_rl_test.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py) | **Out-of-Scope** | 既有 CLI 工具保持完整相容。 |
| **項目 7** | [`tests/test_ppo2_leaderboard.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_ppo2_leaderboard.py) | **Out-of-Scope** | 既有單元測試完全相容且必須無損全數通過。 |

---

## 3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)

### 3.1 步驟清單 (Step-by-Step Task Checklist)
- [x] **步驟 1: 建立量化統計模組 [`Brain/PPO2/lib/statistics.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/statistics.py)**
  - [x] 搬移並實作 `_natural_sort_key`。
  - [x] 搬移並實作標準常態累積與逆累積（Acklam algorithm）`norm_cdf`, `norm_ppf`。
  - [x] 搬移並實作 IQM（Interquartile Mean）`compute_iqm`。
  - [x] 搬移並實作 Deflated Sharpe Ratio（DSR）`compute_deflated_sharpe`。
  - [x] 搬移並實作 CSCV PBO `compute_cscv_pbo`。
- [x] **步驟 2: 建立資料與觀測適配模組 [`Brain/PPO2/lib/data_adapter.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py)**
  - [x] 建立 `PortfolioState` dataclass。
  - [x] 實作 `PPO2DataAdapter.resolve_portfolio_state`。
  - [x] 實作 `PPO2DataAdapter.process_df_to_prices`（欄位標準化、技術指標補齊、時間特徵生成）。
  - [x] 實作 `PPO2DataAdapter.extract_latest_obs`（對齊 `State_time_step` 的 6 維循環時間特徵與帳戶損益狀態）。
- [x] **步驟 3: 建立排行榜與模型篩選模組 [`Brain/PPO2/lib/leaderboard.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/leaderboard.py)**
  - [x] 建立 `GateConfig` 與 `ModelSummaryRecord` dataclass。
  - [x] 實作 `LeaderboardManager.evaluate_hard_gate`。
  - [x] 實作 `LeaderboardManager.summarize_model_manifest`。
  - [x] 實作 `LeaderboardManager.build_and_save_leaderboard`（包含 Plateau smoothing、DSR/PBO 矩陣計算、Top-K 篩選、原子儲存）。
  - [x] 實作 `LeaderboardManager.is_valid_manifest_cache` 快取驗證。
- [x] **步驟 4: 重構 [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py)**
  - [x] 引入 `statistics.py`, `data_adapter.py`, `leaderboard.py`，並設置 alias import 保障既有導入路徑相容。
  - [x] 重構 `PPO2Engine.__init__`，初始化 `data_adapter` 與 `leaderboard_manager`。
  - [x] 拆解巨型函數 `run_test`（310 行）：
    - [x] `_eval_model_checkpoint()`：單一權重評估流程。
    - [x] `_stream_evaluate_symbol()`：單一標的評估與指標萃取。
  - [x] 拆解巨型函數 `get_if_order_map`（157 行）：
    - [x] `_collect_symbol_observations()`：由各格式資料源組裝 observation。
    - [x] `_predict_single_model_orders()`：單權重推論。
    - [x] `_predict_ensemble_orders()`：多權重集成推論。
  - [x] 複用 `PPO2Evaluator` 的死區與再平衡邏輯，消除重複代碼。
  - [x] 程式碼總行數由 1,546 行大幅瘦身，巨型函數拆解完畢。
- [x] **步驟 5: 驗證與回歸測試**
  - [x] 執行 `tests/test_ppo2_leaderboard.py` 驗證所有單元測試（包含統計計算、快取機制、Leaderboard 生成與 Ensembling 模式）。
  - [x] 驗證實盤模式 `get_if_order_map` 輸出輸出格式相容性。

---

### 3.2 詳細程式碼架構藍圖 (Detailed Code Architecture Blueprint)

#### 3.2.1 `Brain/PPO2/lib/statistics.py` 介面藍圖
```python
import math
import re
from itertools import combinations
from typing import Optional, List, Union, Tuple
import numpy as np

def natural_sort_key(s: str) -> List[Union[int, str]]: ...
def norm_cdf(x: float) -> float: ...
def norm_ppf(p: float) -> float: ...
def compute_iqm(values: Union[List[float], np.ndarray]) -> float: ...
def compute_deflated_sharpe(daily_returns_matrix: np.ndarray) -> Tuple[np.ndarray, float]: ...
def compute_cscv_pbo(daily_returns_matrix: np.ndarray, n_partitions: int = 8) -> Optional[float]: ...
```

#### 3.2.2 `Brain/PPO2/lib/data_adapter.py` 介面藍圖
```python
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from Brain.Common.DataFeature import OriginalDataFeature, Prices
from Brain.PPO2.lib.environment import STATIONARY_TIME_FEATURES

@dataclass
class PortfolioState:
    position: float = 0.0
    open_price: float = 0.0

class PPO2DataAdapter:
    @staticmethod
    def resolve_portfolio_state(symbol: str, portfolio_state: Optional[Dict[str, Any]]) -> PortfolioState: ...
    @staticmethod
    def process_df_to_prices(feature_loader: OriginalDataFeature, df: pd.DataFrame) -> Prices: ...
    @staticmethod
    def extract_latest_obs(prices: Prices, bars_count: int, position: float = 0.0, open_price: float = 0.0) -> Dict[str, np.ndarray]: ...
```

#### 3.2.3 `Brain/PPO2/lib/leaderboard.py` 介面藍圖
```python
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

@dataclass
class GateConfig:
    max_mdd_gate_pct: float = 35.0
    min_beat_bh_ratio: float = 0.50
    plateau_window: int = 2
    top_k: int = 5

class LeaderboardManager:
    def __init__(self, gate_config: GateConfig):
        self.config = gate_config

    def evaluate_hard_gate(self, evaluated_symbols: int, mean_exposure: float, mean_daily_turnover: float, median_mdd_pct: float, beat_bh_ratio: float, iqm_ir_2x: float) -> Tuple[bool, str]: ...
    def summarize_model_manifest(self, model_name: str, model_path: str, sym_dict: Dict[str, Dict[str, Any]], portfolio_daily_returns: Dict[str, float]) -> Dict[str, Any]: ...
    def build_and_save_leaderboard(self, model_records: List[Dict[str, Any]], output_dir: str, top_k: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]: ...
    def is_valid_manifest_cache(self, manifest_path: str, effective_start: Optional[str], effective_end: Optional[str], requested_symbols: List[str]) -> Optional[Dict[str, Any]]: ...
```

#### 3.2.4 `Brain/PPO2/lib/engine.py` 瘦身調度器藍圖
```python
from Brain.PPO2.lib.statistics import (
    natural_sort_key as _natural_sort_key,
    norm_cdf as _norm_cdf,
    norm_ppf as _norm_ppf,
    compute_iqm as _compute_iqm,
    compute_deflated_sharpe as _compute_deflated_sharpe,
    compute_cscv_pbo as _compute_cscv_pbo,
)
from Brain.PPO2.lib.data_adapter import PPO2DataAdapter, PortfolioState
from Brain.PPO2.lib.leaderboard import LeaderboardManager, GateConfig

class PPO2Engine:
    # 僅負責生命週期調度、協同子模組，代碼行數大幅縮減
    ...
```

---

## 4. 驗證與測試計畫 (Validation & Verification Plan)

### 4.1 驗證環境與執行指令 (Verification Environment & Commands)
> [!IMPORTANT]
> 依據專案規範，所有測試與執行必須使用指定虛擬環境之 Python 直譯器：
> 絕對路徑: `/home/b0457812963/Mamba3RL/bin/python`
> 相對路徑: `../bin/python`

```bash
# 1. 執行 Leaderboard 與選拔機制單元測試 (包含所有統計與集成推論測試)
../bin/python -m unittest tests/test_ppo2_leaderboard.py

# 2. 驗證 PPO2Engine 模組載入與介面相容性
../bin/python -c "
from Brain.PPO2.lib.engine import PPO2Engine, _compute_iqm, _compute_deflated_sharpe, _compute_cscv_pbo
from Brain.PPO2.lib.statistics import compute_iqm, compute_deflated_sharpe, compute_cscv_pbo
from Brain.PPO2.lib.data_adapter import PPO2DataAdapter, PortfolioState
from Brain.PPO2.lib.leaderboard import LeaderboardManager, GateConfig
print('All modular imports and backwards-compatible engine exports loaded successfully!')
"
```

### 4.2 驗收測試查核清單 (Acceptance Verification Checklist)
- [x] **1. 功能與數值等價性 (Numerical & Functional Parity)**:
  - [x] 統計函數（IQM, DSR, CSCV PBO）數值輸出與原實作 100% 精確一致。
  - [x] `_extract_latest_obs` 與 `State_time_step.encode()` 輸出之張量完全一致。
- [x] **2. 代碼整潔度與結構約束 (Code Cleanliness & Constraints)**:
  - [x] `Brain/PPO2/lib/engine.py` 巨型函數（`run_test`, `_build_and_save_leaderboard`, `get_if_order_map`）皆已拆解解耦。
  - [x] 單一子函數職責單一、結構清晰，無冗餘過度嵌套。
- [x] **3. 回歸測試通過 (Regression Test Pass)**:
  - [x] `tests/test_ppo2_leaderboard.py` 內所有 3 項完整測試（`test_iqm_and_overfitting_metrics`, `test_engine_run_and_cache`, `test_formal_if_order_map`）全數通過（9.567s）。
  - [x] 跨模組導出與 `PPO2_rl_test.py` 命令行工具相容性驗證通過。
