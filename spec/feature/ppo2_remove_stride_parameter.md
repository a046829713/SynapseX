# 規格文件 (Specification): 移除 PPO2 測試環境中的 stride 間隔抽樣參數以實施全量模型評估

- **類別 (Category)**: `feature`
- **規格路徑 (Spec Path)**: `spec/feature/ppo2_remove_stride_parameter.md`
- **狀態 (Status)**: `Approved`

## 📋 目錄 (Index / Contents)
1. [1. 背景與根本原因分析 (Background & Root Cause Analysis)](#1-背景與根本原因分析-background--root-cause-analysis)
2. [2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)](#2-改動範圍與架構邊界-target-scope--architecture-boundaries)
3. [3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)](#3-實作查核清單與詳細藍圖-implementation-checklist--detailed-blueprint)
4. [4. 驗證與測試計畫 (Validation & Verification Plan)](#4-驗證與測試計畫-validation--verification-plan)

---

## 1. 背景與根本原因分析 (Background & Root Cause Analysis)

### 1.1 問題陳述與情境脈絡 (Problem Statement & Context)
在先前 [`spec/feature/ppo2_checkpoint_leaderboard_and_selection.md`](file:///home/b0457812963/Mamba3RL/SynapseX/spec/feature/ppo2_checkpoint_leaderboard_and_selection.md) 中，為了因應權重檔數量龐大時的快速粗篩需求，於 [`PPO2_rl_test.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py) 與 [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) 中引入了 `--stride` 抽樣參數。

然而，在量化交易模型篩選的實際需求中：
1. **高原平滑（Plateau Score）依賴連續序列**：鄰近 Checkpoint 的績效平滑評估需要連續的權重評估結果，若進行間隔抽樣會造成鄰居權重斷裂，無法客觀評估模型是否處於穩健高原或隨機突刺。
2. **已具備增量快取機制（Incremental Resume）**：測試引擎已支援基於 `manifest.json` 的增量快取，評估過的權重不會重複計算，因此全量回測所有 Checkpoint 已具備高效率，抽樣需求已不存在。
3. **消除冗餘參數**：使用者明確要求對所有 Checkpoint 進行無遺漏測試，`--stride` 參數已屬非必要冗餘設計，應從 CLI 介面及底層引擎中乾淨移除。

### 1.2 當前行為 vs. 預期行為 (Current vs. Expected Behavior)
| 維度 | 當前行為 (Current Behavior) | 預期行為 (Expected Behavior) |
| :--- | :--- | :--- |
| **CLI 介面** | `PPO2_rl_test.py` 包含 `--stride` 參數，預設為 `1`。 | 移除 `--stride`，使用者無須指定且不會看到該選項。 |
| **資料流 (Engine Prepare)** | `prepare()` 接收 `stride: int = 1`，並在 `stride > 1` 時執行 `discovered[::stride_val]` 切片抽樣。 | `prepare()` 不再接收 `stride`，直接完整納入所有發現的 Checkpoint。 |
| **資料流 (Watch Mode)** | `watch_and_evaluate()` 接收並轉傳 `stride` 至 `prepare()`。 | `watch_and_evaluate()` 不再接收 `stride`，常駐監看時對所有新產生的 `.pt` 進行全量評估。 |
| **終端日誌 (Logging)** | 啟動與準備模型時印出 `(stride=...)`。 | 日誌簡化，直接輸出符號與模型總數。 |

### 1.3 技術根本原因與程式碼參照 (Technical Root Cause & References)
- CLI 參數定義與轉傳：
  - [`PPO2_rl_test.py:L60-L64`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py#L60-L64)：`parser.add_argument("--stride", ...)`
  - [`PPO2_rl_test.py:L163`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py#L163)：`stride=args.stride` 傳入 `watch_and_evaluate()`
  - [`PPO2_rl_test.py:L177`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py#L177)：`stride=args.stride` 傳入 `prepare()`
- 引擎底層抽樣邏輯：
  - [`Brain/PPO2/lib/engine.py:L285`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L285)：`prepare()` 函式簽章包含 `stride: int = 1`
  - [`Brain/PPO2/lib/engine.py:L313-L325`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L313-L325)：`stride_val` 切片取樣邏輯
  - [`Brain/PPO2/lib/engine.py:L660, L674, L682`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L660)：`watch_and_evaluate()` 之 `stride` 簽章、日誌與轉傳

---

## 2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)

### 2.1 檔案與模組改動矩陣 (File & Module Scope Matrix)
| 編號 | 目標檔案與行號 | 範圍 (Scope) | 規劃動作 (Planned Action) |
| :---: | :--- | :---: | :--- |
| **項目 1** | [`PPO2_rl_test.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py) | **In-Scope** | 移除 `--stride` CLI 參數及其在 `prepare()` 與 `watch_and_evaluate()` 的呼叫傳遞。 |
| **項目 2** | [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) | **In-Scope** | 移除 `prepare()` 與 `watch_and_evaluate()` 的 `stride` 參數與切片邏輯，確保所有 discovered checkpoints 全量納入測試。 |
| **項目 3** | [`Brain/PPO2/lib/leaderboard.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/leaderboard.py) | **Out-of-Scope** | 保留既有排行榜聚合與高原平滑計算，不受影響。 |
| **項目 4** | [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py) | **Out-of-Scope** | 單 Checkpoint 評估邏輯保持不變。 |

### 2.2 架構邊界與非目標 (Architectural Boundaries & Non-Goals)
- **範圍內邊界 (In-Scope Boundaries)**:
  - 徹底移除 `stride` 相關的入參、邏輯切片與日誌，還原乾淨的全量評估流程。
- **非目標 (Out-of-Scope / Non-Goals)**:
  - 不更動 `configs/training/ppo_v2.yaml`。
  - 不更動正式下單模式 (`--formal`) 之動作推論與 Ensemble 流程。

---

## 3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)

### 3.1 步驟清單 (Step-by-Step Task Checklist)
- [x] **步驟 1: 更新 [`PPO2_rl_test.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_test.py)**
  - [x] 移除 `parse_arguments()` 中的 `--stride` 參數區塊。
  - [x] 移除常駐監看呼叫處的 `stride=args.stride`。
  - [x] 移除批次準備呼叫處的 `stride=args.stride`。
- [x] **步驟 2: 更新 [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py)**
  - [x] 調整 `prepare()` 簽章，移除 `stride: int = 1`。
  - [x] 移除 `stride_val` 切片邏輯，直接將 `discovered` 指定給 `self.model_paths`。
  - [x] 移除日誌中的 `(stride=...)`。
  - [x] 調整 `watch_and_evaluate()` 簽章，移除 `stride: int = 1`、日誌與轉傳參數。
- [x] **步驟 3: 驗證與回測確認**
  - [x] 驗證 `--help` CLI 介面無 `--stride`。
  - [x] 執行單元檢查，驗證全量 checkpoints 載入無誤。

---

## 4. 驗證與測試計畫 (Validation & Verification Plan)

### 4.1 驗證環境與執行指令 (Verification Environment & Commands)
> [!IMPORTANT]
> 依據專案規範，所有執行必須使用虛擬環境之 Python 直譯器：
> 絕對路徑: `/home/b0457812963/Mamba3RL/bin/python`
> 相對路徑: `../bin/python`

```bash
# 1. 驗證 CLI 介面（確認 --stride 已不存在）
/home/b0457812963/Mamba3RL/bin/python PPO2_rl_test.py --help

# 2. 驗證 engine.prepare() 正確載入目錄下所有 Checkpoints
/home/b0457812963/Mamba3RL/bin/python -c "
from Brain.PPO2.lib.engine import PPO2Engine
e = PPO2Engine()
e.prepare(model_dir='Brain/PPO2/Meta')
print('Discovered models count:', len(e.model_paths))
assert len(e.model_paths) > 0, 'No models discovered'
"

# 3. 執行批次測試小驗證
/home/b0457812963/Mamba3RL/bin/python PPO2_rl_test.py --model_dir Brain/PPO2/Meta --symbols AAVEUSDT --output_dir results/test_remove_stride
```

### 4.2 驗收測試查核清單 (Acceptance Verification Checklist)
- [x] **1. CLI 介面**: `--help` 中無 `--stride`，非預期傳入 `--stride` 時 argparse 正常報錯。
- [x] **2. 引擎行為**: `prepare()` 與 `watch_and_evaluate()` 乾淨接收參數，完整載入所有目標 Checkpoints（日誌無 `(stride=...)`）。
- [x] **3. 回測評估**: 實際執行批次評估正常完成，產出 `manifest.json` 與 `leaderboard.csv`。
