# 規格文件 (Specification): 移除舊版 14 維日曆時間特徵並統一收斂為 6 維平穩週期時間特徵

- **類別 (Category)**: `feature`
- **規格路徑 (Spec Path)**: `spec/feature/remove_legacy_14dim_time_features.md`
- **狀態 (Status)**: `Implemented & Verified`

## 📋 目錄 (Index / Contents)
1. [1. 背景與根本原因分析 (Background & Root Cause Analysis)](#1-背景與根本原因分析-background--root-cause-analysis)
2. [2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)](#2-改動範圍與架構邊界-target-scope--architecture-boundaries)
3. [3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)](#3-實作查核清單與詳細藍圖-implementation-checklist--detailed-blueprint)
4. [4. 驗證與測試計畫 (Validation & Verification Plan)](#4-驗證與測試計畫-validation--verification-plan)

---

## 1. 背景與根本原因分析 (Background & Root Cause Analysis)

### 1.1 問題陳述與情境脈絡 (Problem Statement & Context)
在目前以 30 分鐘 K 棒（`BARS_COUNT=300`、`EPISODE_STEPS=512`、2020–2025 訓練 / 2026+ 樣本外測試）為主的 PPO2 訓練與推論管線中，專案先前為了相容舊版 DQN 而同時保留了 **14 維時間特徵** 與 **6 維平穩週期時間特徵**，造成以下問題：
1. **冗餘定義與重複覆寫**：[`Brain/Common/env_components.py:L41-L65`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py#L41-L65) 的 `BasicState.__init__` 連續賦值兩次 `self.timelist`（先設 14 維再覆寫為 6 維），且未提供跨模組共用的單一真相來源常數。
2. **廢棄特徵持續計算與記憶體浪費**：[`Brain/Common/DataFeature.py:L11-L49`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L11-L49)、[`L170-L212`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L170-L212)、[`L214-L281`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L214-L281) 仍計算並於 `Prices` namedtuple 儲存 8 個已廢棄之非平穩／低頻日曆特徵（`age_log_minutes`, `age_years`, `month_sin`, `month_cos`, `day_sin`, `day_cos`, `week_sin`, `week_cos`）。這些特徵不僅在 2026 樣本外會產生嚴格 OOD 漂移、歷史日期記憶洩漏，且無端佔用約 25% 預載記憶體。
3. **跨模組匯入斷裂 (`ImportError`)**：[`Brain/PPO2/lib/data_adapter.py:L6`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py#L6) 仍從 `Brain.PPO2.lib.environment` 匯入已移除的 `STATIONARY_TIME_FEATURES`，且 [`data_adapter.py:L96-L105`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py#L96-L105) 仍以已廢棄的 `"age_log_minutes"` 判斷時間特徵是否存在並對 14 個欄位補零。
4. **殘留除錯阻塞碼**：工作區在 [`Brain/PPO2/lib/environment.py:L358-L374`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L358-L374)、[`L505`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L505) 與 [`Brain/PPO2/lib/engine.py:L357-L358`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L357-L358) 留有除錯用 `print` 與 `time.sleep(100)`，將阻塞訓練與推論流程。

經與使用者確認：舊版 DQN 管線無須再維護，現階段統一保留 6 維平穩週期時間特徵（`hour_sin`, `hour_cos`, `minute_sin`, `minute_cos`, `dayofweek_sin`, `dayofweek_cos`），並一併移除舊 14 維殘留代碼與上述除錯阻塞碼。

### 1.2 當前行為 vs. 預期行為 (Current vs. Expected Behavior)
| 維度 | 當前行為 (Current Behavior) | 預期行為 (Expected Behavior) |
| :--- | :--- | :--- |
| **時間特徵定義 (`timelist`)** | `BasicState.__init__` 先寫 14 維再覆寫為 6 維；`STATIONARY_TIME_FEATURES` 遺失導致 `data_adapter.py` 匯入失敗 | 直接於 `BasicState` 類別內部定義類別屬性 `STATIONARY_TIME_FEATURES`（6 維），`BasicState.timelist` 與 `PPO2DataAdapter` 共同引用 `BasicState.STATIONARY_TIME_FEATURES` |
| **資料結構與前處理 (`DataFeature.py`)** | `Prices` 儲存 14 個時間欄位；`add_time_feature()` 計算 `age_log_minutes` / `age_years` / `month_*` / `day_*` / `week_*` | `Prices` 與 `add_time_feature()` / `load_relative()` 僅保留並計算 6 個平穩週期欄位（`hour_sin/cos`, `minute_sin/cos`, `dayofweek_sin/cos`） |
| **推論適配層 (`data_adapter.py`)** | 匯入失敗；以 `"age_log_minutes"` 判斷時間特徵並針對 14 維補零 | 正常由 `Brain.Common.env_components` 匯入 `BasicState` 並引用 `BasicState.STATIONARY_TIME_FEATURES`；改以 6 維清單檢查是否缺欄並執行防禦性補零 |
| **訓練與推論環境 (`environment.py`, `engine.py`)** | 呼叫 `TrainingEnv.reset()` / `SubprocVecEnv.reset()` / `PPO2Engine.build_agent()` 時觸發除錯 `print` 與 `time.sleep(100)` | 移除除錯 `print`、`time.sleep(100)` 及無用之 `import time`，恢復正常非阻塞執行 |

### 1.3 技術根本原因與程式碼參照 (Technical Root Cause & References)
- 重複定義 `self.timelist`：[`Brain/Common/env_components.py:L41-L65`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py#L41-L65)
- `Prices` 欄位、舊特徵計算與 `load_relative()`：[`Brain/Common/DataFeature.py:L11-L49`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L11-L49)、[`L170-L212`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L170-L212)、[`L214-L281`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L214-L281)
- 失效之 `STATIONARY_TIME_FEATURES` 匯入與 14 維補零邏輯：[`Brain/PPO2/lib/data_adapter.py:L6`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py#L6)、[`L96-L105`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py#L96-L105)、[`L204`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py#L204)
- 殘留除錯列印與睡眠阻塞：[`Brain/PPO2/lib/environment.py:L10`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L10)、[`L358-L375`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L358-L375)、[`L505`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L505)；[`Brain/PPO2/lib/engine.py:L357-L358`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L357-L358)

---

## 2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)

### 2.1 檔案與模組改動矩陣 (File & Module Scope Matrix)
| 編號 | 目標檔案與行號 | 範圍 (Scope) | 規劃動作 (Planned Action) |
| :---: | :--- | :---: | :--- |
| **項目 1** | [`Brain/Common/env_components.py:L1-L71`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py#L1-L71) | **In-Scope** | 直接於 `BasicState` 類別內部定義類別屬性 `STATIONARY_TIME_FEATURES`（6 維）；移除舊 14 維 `self.timelist`，改為 `self.timelist = list(self.STATIONARY_TIME_FEATURES)`。 |
| **項目 2** | [`Brain/Common/DataFeature.py:L1-L281`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L1-L281) | **In-Scope** | 從 `Prices`、`add_time_feature()`、`load_relative()` 移除 8 個已廢棄時間特徵與無用的 `timedelta` 匯入；保留 `first_date=None` 選填參數以維持介面安全。 |
| **項目 3** | [`Brain/PPO2/lib/data_adapter.py:L6,L96-L105,L204`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py#L6) | **In-Scope** | 改由 `Brain.Common.env_components` 匯入 `BasicState` 並引用 `BasicState.STATIONARY_TIME_FEATURES`；將 `process_df_to_prices()` 的時間特徵檢查與缺欄補零邏輯收斂為該 6 維清單。 |
| **項目 4** | [`Brain/PPO2/lib/environment.py:L10,L358-L375,L505`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L358-L375) | **In-Scope** | 移除除錯用 `print`、`time.sleep(100)` 與未使用的 `import time`。 |
| **項目 5** | [`Brain/PPO2/lib/engine.py:L357-L358`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L357-L358) | **In-Scope** | 移除 `build_agent()` 內除錯用 `print(ob_space)` 與 `time.sleep(100)`。 |
| **項目 6** | [`Brain/DQN/`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN)、[`Brain/HopeDQN/`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/HopeDQN)、[`TradingSysteam.py`](file:///home/b0457812963/Mamba3RL/SynapseX/TradingSysteam.py) | **Out-of-Scope** | 舊版 DQN 相關模組不維護，明確排除於本次改動範圍外。 |
| **項目 7** | [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py)、[`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) | **Out-of-Scope** | 維持 `time_features_in = 6` 與既有模型結構、訓練超參數不更動，確保現有 6 維 Checkpoint 完全相容。 |

### 2.2 架構邊界與非目標 (Architectural Boundaries & Non-Goals)
- **範圍內邊界 (In-Scope Boundaries)**:
  - 確保 `observation_space["time_states"].shape` 嚴格維持 `(bars_count, 6)`，特徵順序固定為：
    `("hour_sin", "hour_cos", "minute_sin", "minute_cos", "dayofweek_sin", "dayofweek_cos")`。
  - `State_time_step.encode()` 與 `PPO2DataAdapter.extract_latest_obs()` 必須共用同一份 `BasicState.STATIONARY_TIME_FEATURES` 定義，確保訓練與推論特徵順序與數值完全一致。
  - `Prices` namedtuple 結構精簡為 24 欄（4 原始 OHLC + 9 log 價量 + 5 log MA + 6 平穩時間特徵）。
- **非目標 (Out-of-Scope / Non-Goals)**:
  - 不更改既有 6 維特徵之數學公式（暫不改為連續相位或新增 8h 資金費率特徵），以維持與既有 PPO2 Checkpoint（如 [`Brain/PPO2/Meta/checkpoint_iter_10000.pt`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/Meta/checkpoint_iter_10000.pt)）之直接相容性。
  - 不修改 [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) 中使用者暫設之 `NUM_WORKERS: 1`。
  - 不維護舊版 DQN / HopeDQN / 過時測試檔 [`test_ppo2_env.py`](file:///home/b0457812963/Mamba3RL/SynapseX/test_ppo2_env.py)。

---

## 3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)

### 3.1 步驟清單 (Step-by-Step Task Checklist)
- [x] **步驟 1: 於 `BasicState` 內定義 `STATIONARY_TIME_FEATURES` 並清理 [`Brain/Common/env_components.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py)**
  - [x] 直接於 `BasicState` 類別內部定義類別屬性 `STATIONARY_TIME_FEATURES`（6 維 tuple，不抽到模組層級）。
  - [x] 刪除 `BasicState.__init__` 中舊 14 維 `self.timelist`，改為 `self.timelist = list(self.STATIONARY_TIME_FEATURES)`。
- [x] **步驟 2: 清理 [`Brain/Common/DataFeature.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py) 舊 8 維時間特徵**
  - [x] 移除未使用的 `from datetime import timedelta`。
  - [x] 從 `Prices` namedtuple 的 `field_names` 移除 8 個廢棄時間欄位。
  - [x] 精簡 `OriginalDataFeature.add_time_feature(self, df: pd.DataFrame, first_date=None)` 僅計算 `hour_sin/cos`, `minute_sin/cos`, `dayofweek_sin/cos`。
  - [x] 更新 `OriginalDataFeature.load_relative()` 的 `Prices(...)` 實例化參數與 docstring。
- [x] **步驟 3: 修復並對齊 [`Brain/PPO2/lib/data_adapter.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py)**
  - [x] 將第 6 行匯入改為 `from Brain.Common.env_components import BasicState`。
  - [x] 將 `process_df_to_prices()` 內時間特徵缺失判斷與缺欄補零迴圈改為使用 `BasicState.STATIONARY_TIME_FEATURES`，並更新 `extract_latest_obs()` 使用 `list(BasicState.STATIONARY_TIME_FEATURES)`。
- [x] **步驟 4: 移除工作區殘留除錯阻塞碼**
  - [x] 清除 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) 之除錯 `print`、`time.sleep(100)` 與未使用的 `import time`。
  - [x] 清除 [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) `build_agent()` 內之 `print(ob_space)` 與 `time.sleep(100)`。
- [x] **步驟 5: 代碼防禦與清理檢核 (Clean up & Defensiveness)**
  - [x] 全域搜尋確認 `Brain/Common/` 與 `Brain/PPO2/` 不再殘留對 `age_log_minutes`、`age_years`、`month_sin/cos`、`day_sin/cos`、`week_sin/cos` 之引用。

### 3.2 詳細程式碼修改藍圖 (Detailed Code Modifications Blueprint)

#### 3.2.1 [`Brain/Common/env_components.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py)
```python
class BasicState:
    STATIONARY_TIME_FEATURES = (
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
        "dayofweek_sin",
        "dayofweek_cos",
    )

    def __init__(
        self,
        bars_count,
        commission_perc,
        model_train,
        default_slippage,
        N_steps,
    ):
        ...
        self.timelist = list(self.STATIONARY_TIME_FEATURES)
```

#### 3.2.2 [`Brain/Common/DataFeature.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py)
```python
Prices = collections.namedtuple(
    "Prices",
    field_names=[
        "open",
        "high",
        "low",
        "close",
        "log_open",
        "log_high",
        "log_low",
        "log_close",
        "log_volume",
        "log_quote_av",
        "log_trades",
        "log_tb_base_av",
        "log_tb_quote_av",
        # 新增均線特徵
        "log_ma_30",
        "log_ma_60",
        "log_ma_120",
        "log_ma_240",
        "log_ma_360",
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
        "dayofweek_sin",
        "dayofweek_cos",
    ],
)
```
```python
    def add_time_feature(self, df: pd.DataFrame, first_date=None):
        datetime_index = pd.to_datetime(df.index)

        # --- 平穩週期性時間特徵 (天然標準化在 [-1, 1]) ---
        df["hour_sin"] = np.sin(2 * np.pi * datetime_index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * datetime_index.hour / 24)

        df["minute_sin"] = np.sin(2 * np.pi * datetime_index.minute / 60)
        df["minute_cos"] = np.cos(2 * np.pi * datetime_index.minute / 60)

        df["dayofweek_sin"] = np.sin(2 * np.pi * datetime_index.dayofweek / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * datetime_index.dayofweek / 7)

        return df
```
`load_relative()` 同步僅回傳上述 6 個時間欄位。

#### 3.2.3 [`Brain/PPO2/lib/data_adapter.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py)
```python
from Brain.Common.env_components import BasicState
```
在 `process_df_to_prices()` 中：
```python
        if (
            not all(col in cleaned.columns for col in BasicState.STATIONARY_TIME_FEATURES)
            and len(cleaned) > 0
        ):
            cleaned = feature_loader.add_time_feature(cleaned)
        for t_col in BasicState.STATIONARY_TIME_FEATURES:
            if t_col not in cleaned.columns:
                cleaned[t_col] = 0.0
```
在 `extract_latest_obs()` 中：
```python
        timelist = list(BasicState.STATIONARY_TIME_FEATURES)
```

#### 3.2.4 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) 與 [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py)
- `Brain/PPO2/lib/environment.py`：移除第 10 行未使用的 `import time`、移除 `TrainingEnv._load_data_for_instrument()` 的除錯 `print` 與 `time.sleep(100)`、移除 `TrainingEnv.reset()` 與 `SubprocVecEnv.reset()` 的除錯 `print`。
- `Brain/PPO2/lib/engine.py`：移除 `PPO2Engine.build_agent()` 中的 `print(ob_space)` 與 `time.sleep(100)`。

---

## 4. 驗證與測試計畫 (Validation & Verification Plan)

### 4.1 驗證環境與執行指令 (Verification Environment & Commands)
> [!IMPORTANT]
> 依據專案規範，所有執行必須使用虛擬環境之 Python 直譯器：
> 絕對路徑: `/home/b0457812963/Mamba3RL/bin/python`
> 相對路徑: `../bin/python`

```bash
# 1. 驗證 Import、Prices 欄位、6 維時間特徵、Training/Inference Parity 與 Checkpoint 載入推論
/home/b0457812963/Mamba3RL/bin/python -c "
import numpy as np, pandas as pd, torch
from omegaconf import OmegaConf
from Brain.Common.env_components import BasicState
from Brain.Common.DataFeature import OriginalDataFeature, Prices
from Brain.PPO2.lib.environment import State_time_step, ProductionEnv
from Brain.PPO2.lib.data_adapter import PPO2DataAdapter
from Brain.PPO2.lib.engine import PPO2Engine

assert len(BasicState.STATIONARY_TIME_FEATURES) == 6
obsolete = {'age_log_minutes','age_years','month_sin','month_cos','day_sin','day_cos','week_sin','week_cos'}
assert obsolete.isdisjoint(set(Prices._fields)), f'Found obsolete fields: {obsolete & set(Prices._fields)}'

# 驗證真實資料載入與 ProductionEnv vs PPO2DataAdapter bit-for-bit 一致性
df = pd.read_csv('Brain/simulation/test_data/AAVEUSDT-F-30-Min.csv').tail(1000)
loader = OriginalDataFeature()
prices = PPO2DataAdapter.process_df_to_prices(loader, df)
state = State_time_step(300, 0.0005, False, 0.0005, 512, 0.03, 0.05, 1.0, 60, 0.25, 100.0)
assert state.getTimeShape() == (300, 6)
state.reset(prices, len(prices.close))
obs_env = state.encode()
obs_adapter = PPO2DataAdapter.extract_latest_obs(prices, bars_count=300, position=0.0)
assert np.array_equal(obs_env['states'], obs_adapter['states'])
assert np.array_equal(obs_env['time_states'], obs_adapter['time_states'])
assert obs_env['time_states'].shape == (300, 6)
assert np.isfinite(obs_env['time_states']).all()
assert obs_env['time_states'].min() >= -1.0 - 1e-6 and obs_env['time_states'].max() <= 1.0 + 1e-6

# 驗證既有 6 維 PPO2 Checkpoint 可無阻塞建構並推論
cfg = OmegaConf.load('configs/training/ppo_v2.yaml')
engine = PPO2Engine(config=cfg, formal=False)
agent = engine.build_agent()
engine.load_checkpoint(agent, 'Brain/PPO2/Meta/checkpoint_iter_10000.pt')
with torch.no_grad():
    out = agent.get_action(obs_adapter, stochastic=False)
assert np.isfinite(out.clipped_action.cpu().numpy()).all()
print('ALL VERIFICATION CHECKS PASSED')
"
```

### 4.2 驗收測試查核清單 (Acceptance Verification Checklist)
- [x] **1. 功能正確性 (Functional Correctness)**:
  - [x] `BasicState.timelist`、`State_time_step.getTimeShape()` 嚴格對應 6 維 `BasicState.STATIONARY_TIME_FEATURES`。
  - [x] `Prices._fields` 完全移除 8 個舊日曆欄位，僅保留 24 欄。
  - [x] `PPO2DataAdapter` 可正常匯入並通過 `process_df_to_prices()`、`prepare_test_symbol_data()`、`extract_latest_obs()`。
- [x] **2. 數值與推論一致性 (Numerical & Inference Parity)**:
  - [x] `State_time_step.encode()["time_states"]` 與 `PPO2DataAdapter.extract_latest_obs()["time_states"]` 數值 **bit-for-bit 完全相同**。
  - [x] 6 維時間特徵無 `NaN`/`Inf`，且嚴格落於 `[-1.0, 1.0]` 區間。
- [x] **3. 實時無阻塞與 Checkpoint 相容性 (Non-Blocking & Checkpoint Compatibility)**:
  - [x] `PPO2Engine.build_agent()`、`TrainingEnv.reset()`、`SubprocVecEnv.reset()` 無 `time.sleep(100)` 阻塞或冗餘除錯印出。
  - [x] [`Brain/PPO2/Meta/checkpoint_iter_10000.pt`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/Meta/checkpoint_iter_10000.pt) 可正常載入並完成前向推論。
- [x] **4. 殘留引用檢查 (Dead-Code Scan)**:
  - [x] `Brain/Common/` 與 `Brain/PPO2/` 不存在任何對已移除之 8 個時間欄位的引用。
