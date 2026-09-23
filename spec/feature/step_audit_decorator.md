# Specification: DQN 環境單步審計裝飾器設計規範 (Step Audit Decorator)

## 📋 Index / 目錄
1. [Overview & Motivation / 概述與動機](#1-overview--motivation--概述與動機)
2. [Target Scope & Architectural Choice / 目標範疇與架構決策](#2-target-scope--architectural-choice--目標範疇與架構決策)
3. [Core Technical Specifications / 核心技術規格](#3-core-technical-specifications--核心技術規格)
   - [3.1 低耗能批次緩衝機制 (Batch Flush Buffer)](#31-低耗能批次緩衝機制-batch-flush-buffer)
   - [3.2 資料採集欄位架構 (Pre-step Input & Post-step Output)](#32-資料採集欄位架構-pre-step-input--post-step-output)
   - [3.3 序列化相容性與異常安全 (Serialization & Exception Safety)](#33-序列化相容性與異常安全-serialization--exception-safety)
   - [3.4 放空機制語意化校驗標籤 (Short-Selling Semantic Verification)](#34-放空機制語意化校驗標籤-short-selling-semantic-verification)
4. [Implementation Blueprint / 實作藍圖](#4-implementation-blueprint--實作藍圖)
   - [4.1 裝飾器模組設計 (`Brain/DQN/lib/audit_logger.py`)](#41-裝飾器模組設計-braindqnlibaudit_loggerpy)
   - [4.2 環境掛載示範 (`Brain/DQN/lib/environment.py`)](#42-環境掛載示範-braindqnlibenvironmentpy)
5. [Verification Checklist / 人工驗證查核清單](#5-verification-checklist--人工驗證查核清單)

---

## 1. Overview & Motivation / 概述與動機

在引入放空機制（參考 [issue_short_selling_mechanism.md](file:///home/b0457812963/Mamba3RL/SynapseX/spec/issue_short_selling_mechanism.md)）後，DQN 交易環境（`Brain/DQN/lib/environment.py`）具備了三元離散部位（`position ∈ {-1, 0, 1}`）、雙向滑價撮合、反向浮動與已實現損益、持倉 K 棒累計以及違規交易懲罰等複雜財務計算。

為確保環境在不同行情與隨機動作推進下的行為完全符合預期，需要一套**輕量、無侵入式、可即插即用（Plug-and-Play）**的審計監控工具，其核心目標為：
1. **人工驗證邏輯**：完整截取單步進入前的環境狀態（Input）與完成計算後的狀態轉變及輸出（Output）。
2. **極低耗能與高 I/O 效率**：每 50 筆批次寫入本地端點 JSON Lines（`.jsonl`），避免頻繁磁碟 I/O 拖慢模擬速度。
3. **完全解耦**：以 Python Decorator（裝飾器）形式封裝，需要除錯或驗證時一鍵掛載，驗證完成後一鍵拔除或關閉，不污染交易環境主體代碼。

---

## 2. Target Scope & Architectural Choice / 目標範疇與架構決策

經過方案評估，確認選用 **【選項 A：掛載於 `State_time_step.step(self, action: Actions)`】**：

| 評估項目 | 選項 A：`State_time_step.step` (選定) | 選項 B：`BaseTradingEnv.step` |
| :--- | :--- | :--- |
| **資訊細緻度** | **極高**：可直接提取 `open_price`、`closecash`、`cost_sum`、`TotalPortfolioPercent` 等底層運算變數。 | **一般**：僅能獲得 Gym 的 `(obs, reward, done, info)`，細部指標需透過 `_state` 再次查詢。 |
| **邏輯相關性** | **核心邏輯所在地**：[issue_short_selling_mechanism.md](file:///home/b0457812963/Mamba3RL/SynapseX/spec/issue_short_selling_mechanism.md) 中的所有公式均在此實作。 | 僅為外層 Gym API 包裝轉發。 |
| **性能開銷** | 直接存取實例屬性，無額外層層封裝開銷。 | 多一層函數調用包裝。 |

---

## 3. Core Technical Specifications / 核心技術規格

### 3.1 低耗能批次緩衝機制 (Batch Flush Buffer)
- **記憶體緩衝區**：內部維護 `List[Dict[str, Any]]` 佇列。
- **批次寫入閾值**：緩衝區累積達 **50 筆** 時觸發磁碟寫入並清空（`buffer.clear()`）。
- **儲存格式與 I/O 模式**：
  - 採用 **JSON Lines (`.jsonl`)** 格式。
  - 使用追加寫入模式（`mode="a"`），避免重寫整個 JSON 檔案導致隨時間呈 $O(N)$ 增長的寫入負擔。
- **退場保底 (Graceful Shutdown)**：
  - 註冊 Python 內建 `atexit.register(self.flush)`。
  - 當環境結束、模型提早中止或使用者手動發送中斷訊號（`Ctrl+C`）時，自動將未滿 50 筆的剩餘記錄刷入硬碟，防止數據丟失。

### 3.2 資料採集欄位架構 (Pre-step Input & Post-step Output)

每筆審計記錄均包含以下四大結構：

```json
{
  "step_index": 50,
  "offset": 350,
  "input": {
    "action": "Actions.Sell (2)",
    "pre_position": 0,
    "pre_trade_bar": 0,
    "pre_open_price": 0.0,
    "current_bar_close": 102.5,
    "pre_benchmark_return": 0.0012,
    "pre_total_equity": 1.0,
    "pre_cost_sum": 0.0,
    "pre_closecash": 0.0
  },
  "output": {
    "reward": -0.0005,
    "done": false,
    "post_position": -1,
    "post_trade_bar": 1,
    "post_open_price": 102.3975,
    "post_total_equity": 0.9995,
    "step_equity_diff": -0.0005,
    "cost_diff": 0.0005,
    "realized_diff": 0.0
  },
  "short_selling_audit": {
    "transition": "FLAT (0) -> SHORT (-1)",
    "operation_type": "OPEN_SHORT",
    "is_wrong_trade": false,
    "entry_slippage_check": "PASS (102.5 * (1 - 0.001) = 102.3975)",
    "commission_charged": true
  }
}
```

### 3.3 序列化相容性與異常安全 (Serialization & Exception Safety)
1. **Numpy 類型適配**：自定義 `_json_serializer`，將 `np.float32`, `np.int64`, `np.ndarray` 轉換為 Python 原生 `float`, `int`, `list`，數值預設取 6 位小數以保持可讀性。
2. **列舉相容**：將 `Actions` 列舉序列化為 `Actions.Name(Value)`（如 `Actions.Sell(2)`）。
3. **容錯機制**：日誌寫入異常均由 `try...except` 捕捉，確保日誌紀錄器即便發生非預期檔案鎖定或磁碟已滿時，絕不影響 RL 訓練與環境步進。

### 3.4 放空機制語意化校驗標籤 (Short-Selling Semantic Verification)
針對放空機制的 4 種核心操作狀態轉移，自動產出驗證標籤：
- **`OPEN_SHORT`（$0 \to -1$）**：空手開空單，驗證向下扣除滑價 $P_{\text{open}} = P_{\text{close}} \times (1 - \text{slip})$ 與開倉佣金扣除。
- **`HOLD_SHORT`（$-1 \to -1$）**：續抱空單，驗證 `open_price` 恆定、`trade_bar` 遞增，且價格下跌時淨值提升。
- **`COVER_SHORT`（$-1 \to 0$）**：買進平空回補，驗證向上包含滑價 $P_{\text{exec}} = P_{\text{close}} \times (1 + \text{slip})$、實現損益計算及部位歸零。
- **`WRONG_TRADE`（$-1 \to -1$ 重複下賣單，或 $1 \to 1$ 重複下買單）**：驗證部位不變，並產出違規懲罰扣分。

---

## 4. Implementation Blueprint / 實作藍圖

### 4.1 裝飾器模組設計 (`Brain/DQN/lib/audit_logger.py`)

```python
import os
import json
import atexit
from functools import wraps
from typing import Any, Dict, List
import numpy as np

from Brain.DQN.lib.reward import Actions


class StepAuditLogger:
    """
    DQN 環境 State_time_step.step 專用審計裝飾器。
    緩衝 50 筆批次寫入本地端點 JSONL，專門用於驗證部位轉移、多空損益與滑價。
    """
    def __init__(self, log_path: str = "audit_logs/step_audit.jsonl", batch_size: int = 50, enabled: bool = True):
        self.log_path = log_path
        self.batch_size = batch_size
        self.enabled = enabled
        self.buffer: List[Dict[str, Any]] = []

        if self.enabled:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            atexit.register(self.flush)

    def flush(self):
        """將記憶體中累積的審計數據批次刷入硬碟"""
        if not self.buffer:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                for record in self.buffer:
                    f.write(json.dumps(record, ensure_ascii=False, default=self._json_serializer) + "\n")
            self.buffer.clear()
        except Exception as e:
            print(f"[StepAuditLogger Warning] Flush failed: {e}")

    @staticmethod
    def _json_serializer(obj):
        """轉換 numpy / Enum 資料型態為 JSON 標準型態"""
        if isinstance(obj, (np.floating, float)):
            return round(float(obj), 6)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Actions):
            return f"{obj.name}({obj.value})"
        return str(obj)

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)

            # 鎖定 State_time_step 實例 (選項 A)
            state = args[0]
            action = args[1] if len(args) > 1 else kwargs.get("action")
            action_enum = Actions(action) if isinstance(action, int) else action

            # 1. 擷取進入前的狀態 (Input Snapshot)
            pre_pos = state.position
            pre_open = state.open_price
            pre_bar = state.trade_bar
            pre_equity = state.TotalPortfolioPercent
            pre_cost = state.cost_sum
            pre_closecash = state.closecash
            cur_close = float(state._prices.close[state._offset]) if hasattr(state, "_prices") else 0.0

            # 2. 執行原函數
            reward, done = func(*args, **kwargs)

            # 3. 擷取執行後的狀態 (Output Snapshot)
            post_pos = state.position
            post_open = state.open_price
            post_bar = state.trade_bar
            post_equity = state.TotalPortfolioPercent
            post_cost = state.cost_sum
            post_closecash = state.closecash

            # 4. 放空機制自動語意分析
            op_type = "UNKNOWN"
            if pre_pos == 0 and post_pos == -1:
                op_type = "OPEN_SHORT"
            elif pre_pos == -1 and post_pos == 0:
                op_type = "COVER_SHORT"
            elif pre_pos == 0 and post_pos == 1:
                op_type = "OPEN_LONG"
            elif pre_pos == 1 and post_pos == 0:
                op_type = "CLOSE_LONG"
            elif pre_pos == post_pos and action_enum == Actions.Hold:
                op_type = "HOLD"
            elif (pre_pos == 1 and action_enum == Actions.Buy) or (pre_pos == -1 and action_enum == Actions.Sell):
                op_type = "WRONG_TRADE"

            record = {
                "step": state.game_steps,
                "offset": state._offset,
                "input": {
                    "action": action_enum,
                    "pre_position": pre_pos,
                    "pre_open_price": pre_open,
                    "current_close": cur_close,
                    "pre_trade_bar": pre_bar,
                    "pre_equity": pre_equity,
                    "pre_cost_sum": pre_cost,
                    "pre_closecash": pre_closecash
                },
                "output": {
                    "reward": reward,
                    "done": done,
                    "post_position": post_pos,
                    "post_open_price": post_open,
                    "post_trade_bar": post_bar,
                    "post_equity": post_equity,
                    "step_equity_diff": post_equity - pre_equity,
                    "cost_diff": post_cost - pre_cost,
                    "realized_pnl_diff": post_closecash - pre_closecash
                },
                "audit": {
                    "operation_type": op_type,
                    "transition": f"{pre_pos} -> {post_pos}"
                }
            }

            self.buffer.append(record)
            if len(self.buffer) >= self.batch_size:
                self.flush()

            return reward, done
        return wrapper


# 預設單例裝飾器
step_audit_logger = StepAuditLogger(
    log_path="audit_logs/dqn_step_audit.jsonl",
    batch_size=50,
    enabled=True
)
```

### 4.2 環境掛載示範 (`Brain/DQN/lib/environment.py`)

掛載於 `State_time_step.step` 上：

```python
from Brain.DQN.lib.audit_logger import step_audit_logger

class State_time_step(State_time_step_template):
    ...
    @step_audit_logger  # <--- 無侵入式掛載
    def step(self, action: Actions):
        assert isinstance(action, Actions)
        ...
        return reward, done
```

---

## 5. Verification Checklist / 人工驗證查核清單

當審計日誌生成至 `audit_logs/dqn_step_audit.jsonl` 後，人工審閱只需檢視對應行次是否符合下述準則：

- [x] **1. 開立空頭倉位驗證（`OPEN_SHORT`）**
  - [x] `input.action` 為 `Actions.Sell` 且 `input.pre_position == 0`。
  - [x] `output.post_position == -1`。
  - [x] `output.post_trade_bar == 1`。
  - [x] `output.post_open_price` 必須等於 `current_close * (1 - default_slippage)`（向下滑價）。
  - [x] `output.cost_diff` 必須等於單步手續費佣金。
- [x] **2. 續抱空單驗證（`HOLD`）**
  - [x] `input.action` 為 `Actions.Hold` 且 `input.pre_position == -1`。
  - [x] `output.post_position == -1`。
  - [x] `output.post_open_price` 保持不變。
  - [x] `output.post_trade_bar == input.pre_trade_bar + 1`。
  - [x] `output.cost_diff == 0.0`。
  - [x] 當 K 棒價格下跌時，`output.step_equity_diff > 0`（做空獲利）。
- [x] **3. 回補空單驗證（`COVER_SHORT`）**
  - [x] `input.action` 為 `Actions.Buy` 且 `input.pre_position == -1`。
  - [x] `output.post_position == 0`。
  - [x] `output.post_open_price == 0.0`（重置）。
  - [x] `output.post_trade_bar == 0`（重置）。
  - [x] `output.cost_diff` 必須扣除平倉手續費。
  - [x] `output.realized_pnl_diff` 損益實現，計算公式符合 $(P_{\text{open}} - P_{\text{exec}}) / P_{\text{open}}$。
- [x] **4. 違規交易判定驗證（`WRONG_TRADE`）**
  - [x] 持有多單（`pre_position == 1`）再下 `Buy`，或持有空單（`pre_position == -1`）再下 `Sell`。
  - [x] `output.post_position` 與前一步完全一致（部位未被修改）。
  - [x] `output.reward` 包含負項違規懲罰扣分。
  - [x] 不扣除手續費（`output.cost_diff == 0.0`）。
