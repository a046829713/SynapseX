# 規格文件 (Specification): PPO2 特徵消融實驗第一步 — 移除 `vol_regime_abs` 絕對波動率特徵

- **類別 (Category)**: `feature`
- **規格路徑 (Spec Path)**: `spec/feature/ppo2_remove_vol_regime_abs.md`
- **狀態 (Status)**: `Implemented & Verified`

## 📋 目錄 (Index / Contents)
1. [1. 背景與根本原因分析 (Background & Root Cause Analysis)](#1-背景與根本原因分析-background--root-cause-analysis)
2. [2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)](#2-改動範圍與架構邊界-target-scope--architecture-boundaries)
3. [3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)](#3-實作查核清單與詳細藍圖-implementation-checklist--detailed-blueprint)
4. [4. 驗證與測試計畫 (Validation & Verification Plan)](#4-驗證與測試計畫-validation--verification-plan)

---

## 1. 背景與根本原因分析 (Background & Root Cause Analysis)

### 1.1 問題陳述與情境脈絡 (Problem Statement & Context)
在 commit `72291f4` 引入 14 維因果波動率歸一化市場特徵並移除 `DAIN_Layer` 後，實證檢核（含 [`tests/diagnose_ppo2_feature_attribution.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/diagnose_ppo2_feature_attribution.py) 對 [`Brain/PPO2/Meta/checkpoint_iter_9000.pt`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/Meta/checkpoint_iter_9000.pt) 之四指標歸因分析與 2026 OOS 監督消融）顯示，`vol_regime_abs` 存在三大結構性問題：

1. **跨幣種常數偏移（淪為幣種身分證）**：
   - [`Brain/Common/DataFeature.py:L198`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L198) 以固定常數 `VOL_REF = 0.005` 計算 `soft_clip(log(sigma_long / 0.005))`。
   - 跨 83 檔 pre-2026 訓練幣種實測，各幣種均值標準差達 `0.316`（如 `BNBUSDT` 中位數 `-0.163`、`ETHUSDT` `-0.069`、`SOLUSDT` `+0.309`、`UBUSDT` `+0.866`），非平穩地洩漏各幣種固有波動規模。
2. **Pre-2026 與 2026 OOS 跨期分佈嚴重漂移**：
   - Pre-2026 訓練集均值為 `+0.381`（std `0.490`），2026 OOS 測試集驟降至 `+0.149`（std `0.623`），均值偏移高達 **`-0.473 std`**（其餘價格與均線特徵偏移均小於 `0.05 std`）。
3. **劫持共享 Mamba2 骨幹與 Critic 價值估計**：
   - 在 `checkpoint_iter_9000.pt` 的特徵歸因審計中（[`results/feature_attribution/20261122-124323.json`](file:///home/b0457812963/Mamba3RL/SynapseX/results/feature_attribution/20261122-124323.json)），`vol_regime_abs` 在 Critic $V(s)$ 上全面排名第 1（`W_L2 = 3.85`、`|∂V/∂x| = 0.00507`、`Perm_ΔV = 0.2391`，大幅高於第 2 名 `candle_body` 的 `0.1496`）。
   - 在 Mamba2 16-bar 監督消融中，僅移除波動率體制通道即將 2026 OOS IC 由 `+0.0005` 提升至 `+0.0217`。

為進行嚴謹的**單變因逐步消融測試（Controlled Single-Variable Ablation）**，本規格第一步**僅移除 `vol_regime_abs` 單一特徵**，其餘 13 維市場特徵、6 維時間特徵、1 維倉位特徵、模型架構、訓練超參數、環境獎勵與摩擦力排程**全數維持完全不變**。

### 1.2 當前行為 vs. 預期行為 (Current vs. Expected Behavior)
| 維度 | 當前行為 (Current Behavior) | 預期行為 (Expected Behavior) |
| :--- | :--- | :--- |
| **市場特徵清單 (`BasicState.MARKET_FEATURES`)** | 14 維，包含 `vol_regime_abs` 與 `vol_expansion` | **13 維**：僅移除 `vol_regime_abs`，其餘 13 維特徵與順序 100% 保留不變（保留 `vol_expansion`） |
| **`Prices` 資料結構欄位數** | 24 欄（4 OHLC + 14 市場 + 6 時間） | **23 欄**（4 OHLC + 13 市場 + 6 時間） |
| **觀測空間形狀 (`states` / `time_states`)** | `states`: `(300, 15)`（14 市場 + 1 倉位）；`time_states`: `(300, 6)` | `states`: **`(300, 14)`**（13 市場 + 1 倉位）；`time_states`: `(300, 6)` 不變 |
| **特徵計算 (`DataFeature.py`)** | 計算 `out["vol_regime_abs"] = soft_clip(np.log(sigma_long / VOL_REF))` 並定義常數 `VOL_REF = 0.005` | 移除 `VOL_REF` 常數與 `out["vol_regime_abs"]` 計算；`sigma_long`、`sigma_short`、`sigma_prev` 與 `vol_expansion` 計算公式完全不變 |
| **其餘訓練／環境／模型／設定** | 現行 `HEAD`（commit `92d9e90`） | **100% 維持不變**（嚴格單變因控制） |

### 1.3 技術根本原因與程式碼參照 (Technical Root Cause & References)
- 市場特徵單一真相來源定義：[`Brain/Common/env_components.py:L4-L23`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py#L4-L23)（[`L17`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py#L17) 定義 `"vol_regime_abs"`）。
- `VOL_REF` 常數與 `vol_regime_abs` 計算邏輯：[`Brain/Common/DataFeature.py:L20`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L20)、[`L131-L137`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L131-L137)、[`L197-L201`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L197-L201)、[`L243-L246`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L243-L246)。
- 硬編碼 14/15/24 維與 `vol_regime_abs` 斷言之驗證測試：[`tests/test_vol_normalized_features_and_no_dain.py:L22`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_vol_normalized_features_and_no_dain.py#L22)、[`L73-L101`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_vol_normalized_features_and_no_dain.py#L73-L101)、[`L180-L191`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_vol_normalized_features_and_no_dain.py#L180-L191)、[`L277-L293`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_vol_normalized_features_and_no_dain.py#L277-L293)。
- 綁定舊 15 維 `checkpoint_iter_9000.pt` 之歸因整合測試：[`tests/test_ppo2_feature_attribution.py:L259-L307`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_ppo2_feature_attribution.py#L259-L307)。

---

## 2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)

### 2.1 檔案與模組改動矩陣 (File & Module Scope Matrix)
| 編號 | 目標檔案與行號 | 範圍 (Scope) | 規劃動作 (Planned Action) |
| :---: | :--- | :---: | :--- |
| **項目 1** | [`Brain/Common/env_components.py:L4-L23`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py#L4-L23) | **In-Scope** | 從 `BasicState.MARKET_FEATURES` 移除 `"vol_regime_abs"`（14 維 → 13 維），並更新該分組註解為 `# Relative volatility expansion`。 |
| **項目 2** | [`Brain/Common/DataFeature.py:L20,L131-L137,L197-L201,L243-L246`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L197-L201) | **In-Scope** | 移除過時常數 `VOL_REF` 與 `out["vol_regime_abs"] = ...` 計算行；保留 `sigma_long`、`sigma_short`、`sigma_prev` 與 `out["vol_expansion"]` 不變；更新 docstring 中的維度說明（13 維市場特徵、23 欄 `Prices`）。 |
| **項目 3** | [`tests/test_vol_normalized_features_and_no_dain.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_vol_normalized_features_and_no_dain.py) | **In-Scope** | 移除 `VOL_REF` 匯入；更新維度斷言為 13 市場特徵 / 23 欄 `Prices` / `(300, 14)` `states`；將 `test_volatility_regime_awareness()` 改為驗證 `vol_regime_abs` 已徹底移除且 `vol_expansion` 正常計算。 |
| **項目 4** | [`tests/test_ppo2_feature_attribution.py:L259-L307`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_ppo2_feature_attribution.py#L259-L307) | **In-Scope** | 將 `test_end_to_end_json_roundtrip_and_checkpoint_immutability()` 改為於暫存目錄建立符合當前 `BasicState.MARKET_FEATURES` 維度（13+1=14）之測試用 checkpoint，避免因舊 15 維 `checkpoint_iter_9000.pt` 維度不相容而誤報失敗，同時確保未來每一步消融測試皆可持續通過。 |
| **項目 5** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)、[`Brain/PPO2/lib/data_adapter.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py)、[`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py)、[`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py)、[`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py)、[`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py)、[`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)、[`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml)、[`tests/diagnose_ppo2_feature_attribution.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/diagnose_ppo2_feature_attribution.py) | **Out-of-Scope** | **明確保留 100% 不更動**：上述模組皆動態讀取 `BasicState.MARKET_FEATURES` 與 `ob_space["states"].shape[1]`，自動對齊 13+1=14 維；所有超參數、獎勵、摩擦退火、時間特徵與模型結構皆維持不動以確保嚴格單變因消融。 |

### 2.2 架構邊界與非目標 (Architectural Boundaries & Non-Goals)
- **範圍內邊界 (In-Scope Boundaries)**:
  - `BasicState.MARKET_FEATURES` 精確調整為 13 維（順序完全沿用現狀，僅移除 `"vol_regime_abs"`）：
    `("ret_gap", "candle_body", "upper_wick", "lower_wick", "ma_dev_30", "ma_dev_60", "ma_dev_120", "ma_dev_240", "ma_dev_360", "vol_expansion", "rel_quote_vol", "rel_trades", "taker_imbalance")`。
  - `observation_space["states"].shape` 自動更新為 `(bars_count, 14)`（13 市場特徵 + 1 倉位），`observation_space["time_states"].shape` 嚴格維持 `(bars_count, 6)`。
  - `State_time_step.encode()`（訓練）與 `PPO2DataAdapter.extract_latest_obs()`（推論／回測）維持 100% bit-for-bit 一致。
- **非目標 (Out-of-Scope / Non-Goals)**:
  - **不移除** `ret_gap`、`taker_imbalance`、`minute_sin`、`dayofweek_sin/cos` 或其他任何特徵（留待後續消融步驟逐一測試）。
  - **不修改** [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) 的任何超參數（`FRICTION_ANNEAL_*`、`DEADZONE_THRESHOLD`、`ENT_COEF`、`WEIGHT_DECAY`、`W2_DOWNSIDE_RATIO`、`WEIGHT_SCALE` 全數維持原值）。
  - **不修改** [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) 的 `v_clipped` 或任何訓練迴圈邏輯。
  - **不回溯相容** 舊 15 維 checkpoint（如 `checkpoint_iter_9000.pt`）：消融實驗將從零重新訓練 14 維模型。

---

## 3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)

### 3.1 步驟清單 (Step-by-Step Task Checklist)
- [x] **步驟 1: 更新 [`Brain/Common/env_components.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py)**
  - [x] 從 `BasicState.MARKET_FEATURES` 移除 `"vol_regime_abs"`，並更新鄰近註解。
- [x] **步驟 2: 更新 [`Brain/Common/DataFeature.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py)**
  - [x] 移除未再使用的常數 `VOL_REF: float = 0.005`。
  - [x] 移除 `add_market_features()` 中的 `out["vol_regime_abs"] = soft_clip(np.log(sigma_long / VOL_REF))`（保留 `sigma_long`、`sigma_short`、`sigma_prev` 與 `out["vol_expansion"]`）。
  - [x] 更新 `add_market_features()` 與 `load_relative()` docstring 維度數字（13 維市場特徵、23 欄 `Prices`）與移除多餘空白行。
- [x] **步驟 3: 對齊驗證測試套件**
  - [x] 更新 [`tests/test_vol_normalized_features_and_no_dain.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_vol_normalized_features_and_no_dain.py)：移除 `VOL_REF` 匯入、將維度斷言更新為 `13` / `23` / `(300, 14)`、更新波動率測試函式以驗證 `vol_regime_abs` 不存在且 `vol_expansion` 正常有限。
  - [x] 更新 [`tests/test_ppo2_feature_attribution.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_ppo2_feature_attribution.py)：將 `test_end_to_end_json_roundtrip_and_checkpoint_immutability()` 改為在暫存目錄動態產生符合當前 `BasicState.MARKET_FEATURES` 維度之測試 checkpoint 執行端到端歸因、SHA-256 唯讀驗證與 JSON 序列化驗證。
- [x] **步驟 4: 代碼防禦與清理 (Clean up & Defensiveness)**
  - [x] 全域搜尋確認 `Brain/Common/`、`Brain/PPO2/` 與 `PPO2_rl_train.py` 不再殘留對 `vol_regime_abs` 或 `VOL_REF` 之引用。

### 3.2 詳細程式碼修改藍圖 (Detailed Code Modifications Blueprint)

#### 3.2.1 [`Brain/Common/env_components.py:L4-L23`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py#L4-L23)
```python
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
        # Relative volatility expansion (short vs long causal EWMA RMS)
        "vol_expansion",
        # Scale-invariant volume & order flow microstructure
        "rel_quote_vol",
        "rel_trades",
        "taker_imbalance",
    )
```

#### 3.2.2 [`Brain/Common/DataFeature.py:L18-L25,L197-L202`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/DataFeature.py#L18-L25)
```python
SOFT_CLIP_LIMIT: float = 5.0
VOL_FLOOR: float = 1e-4
VOL_LONG_SPAN: int = 60
VOL_LONG_MIN_PERIODS: int = 20
VOL_SHORT_SPAN: int = 12
VOL_SHORT_MIN_PERIODS: int = 6
VOL_EMA_SPAN: int = 60

# In OriginalDataFeature.add_market_features():
        # 3. Relative volatility expansion feature
        out["vol_expansion"] = soft_clip(np.log(sigma_short / sigma_long))
```

#### 3.2.3 [`tests/test_vol_normalized_features_and_no_dain.py:L180-L191`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_vol_normalized_features_and_no_dain.py#L180-L191)
```python
def test_vol_regime_abs_removed_and_vol_expansion_valid():
    """vol_regime_abs must be removed while vol_expansion remains finite and responsive."""
    assert "vol_regime_abs" not in BasicState.MARKET_FEATURES
    assert "vol_regime_abs" not in Prices._fields

    df_syn = _make_synthetic_ohlcv(n=700, base_price=100.0, sigma=0.01, seed=101)
    loader = OriginalDataFeature()
    cleaned = loader.cleanData(df_syn)
    enriched = loader.add_market_features(cleaned)
    assert "vol_regime_abs" not in enriched.columns
    assert "vol_expansion" in enriched.columns
    assert np.all(np.isfinite(enriched["vol_expansion"].to_numpy()))
    assert np.all(np.abs(enriched["vol_expansion"].to_numpy()) <= SOFT_CLIP_LIMIT + 1e-6)
```

---

## 4. 驗證與測試計畫 (Validation & Verification Plan)

### 4.1 驗證環境與執行指令 (Verification Environment & Commands)
> [!IMPORTANT]
> 依據專案規範，所有執行必須使用虛擬環境之 Python 直譯器：
> 絕對路徑: `/home/b0457812963/Mamba3RL/bin/python`
> 相對路徑: `../bin/python`

```bash
# 1. 執行特徵工程與無 DAIN 全套整合驗證（含全 177 檔真實 CSV 掃描與訓練／推論對齊）：
/home/b0457812963/Mamba3RL/bin/python tests/test_vol_normalized_features_and_no_dain.py

# 2. 執行特徵歸因診斷單元與端到端測試（驗證 13+1=14 維相容）：
/home/b0457812963/Mamba3RL/bin/python tests/test_ppo2_feature_attribution.py

# 3. 驗證殘留引用清理（應回傳空結果）：
grep -rn "vol_regime_abs\|VOL_REF" Brain/ PPO2_rl_train.py PPO2_rl_test.py configs/
```

### 4.2 驗收測試查核清單 (Acceptance Verification Checklist)
- [x] **1. 維度與單一真相來源對齊 (Shape & Schema Alignment)**:
  - [x] `len(BasicState.MARKET_FEATURES) == 13` 且 `"vol_regime_abs" not in BasicState.MARKET_FEATURES`。
  - [x] `len(Prices._fields) == 23` 且 `"vol_regime_abs" not in Prices._fields`。
  - [x] `State_time_step.getStateShape() == (300, 14)`、`getTimeShape() == (300, 6)`。
  - [x] `State_time_step.encode()` 與 `PPO2DataAdapter.extract_latest_obs()` 輸出 shape 皆為 `(300, 14)` 且數值 bit-for-bit 相等。
- [x] **2. 嚴格單變因隔離 (Strict Single-Variable Isolation)**:
  - [x] 其餘 13 維市場特徵的數學公式、因果性、價格尺度不變性與全 177 檔有限性皆 100% 通過驗證。
  - [x] `git diff` 確認 [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml)、[`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)、[`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) 與 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) 零變更。
- [x] **3. 實際執行與 GPU 驗證 (Runtime & GPU Verification)**:
  - [x] `tests/test_vol_normalized_features_and_no_dain.py` 與 `tests/test_ppo2_feature_attribution.py` 於 GPU 上全數通過，且 `MambaContinuousActorCritic(d_model=14)` 前向／反向與梯度裁剪無 NaN/Inf。
