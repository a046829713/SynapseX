# Specification: `RewardHelp` 與 `Reward` 類別代碼品質重構規範 (Refactoring Plan)

## 📋 目錄 (Index)
1. [Overview & Design Philosophy / 概述與設計哲學](#1-overview--design-philosophy--概述與設計哲學)
2. [Current Issues & Quality Analysis / 現存問題與品質分析](#2-current-issues--quality-analysis--現存問題與品質分析)
   - [2.1 `RewardHelp` 現存問題與清理清單](#21-rewardhelp-現存問題與清理清單)
   - [2.2 `Reward` 現存問題與清理清單](#22-reward-現存問題與清理清單)
3. [Detailed Architecture & Interface Design / 重構介面與極簡代碼設計](#3-detailed-architecture--interface-design--重構介面與極簡代碼設計)
   - [3.1 `RewardHelp` 極簡無狀態工具類 (僅保留 6 個核心方法)](#31-rewardhelp-極簡無狀態工具類-僅保留-6-個核心方法)
   - [3.2 `Reward` 單一職責類別 (僅保留 `wrong_trade`)](#32-reward-單一職責類別-僅保留-wrong_trade)
4. [Method Mapping & Migration Table / 方法對照與遷移清單](#4-method-mapping--migration-table--方法對照與遷移清單)
5. [Impact Analysis / 外部依賴與後續呼叫點修正計畫](#5-impact-analysis--外部依賴與後續呼叫點修正計畫)
6. [Implementation Checklist / 實作改動查核清單](#6-implementation-checklist--實作改動查核清單)
7. [Verification Checklist / 驗證查核清單](#7-verification-checklist--驗證查核清單)

---

## 1. Overview & Design Philosophy / 概述與設計哲學

本規範聚焦於對 [`Brain/DQN/lib/reward.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/reward.py) 中的 `RewardHelp` 與 `Reward` 兩個類別進行**徹底的代碼品質清理與重構**。

### 核心設計哲學 (Design Philosophy)：
- 🚫 **不提供向後相容層 (No Backward Compatibility Layer)**：
  捨棄一切舊方法別名（Aliases）、相容性包裝器（Wrappers）與 `@property` 舊屬性對映，**徹底斬斷歷史包袱**。
- 🎯 **保持接口單一 (Single Point of Truth)**：
  每個功能僅保留唯一標準、符合 PEP 8 命名規範的介面，杜絕同一個功能有多種呼叫名稱的混亂現象。
- 🧹 **先求代碼簡潔與正確 (Simplicity & Correctness First)**：
  所有全專案無人使用的「死代碼」一律徹底刪除，參數中掩蓋錯字的雙重容錯補丁（如 `closePrcie`, `OpenPrice`）一律拔除。
- 🔧 **後續接口調適 (Post-Refactoring Caller Migration)**：
  本階段先確保 `reward.py` 內部代碼達到最高品質；重構後若外部環境（如 `environment.py`）或測試出現接口不相容，再按標準接口進行相應修正。

---

## 2. Current Issues & Quality Analysis / 現存問題與品質分析

### 2.1 `RewardHelp` 現存問題與清理清單

經全專案檢索，`RewardHelp` 原有 11 個方法中，實際上只有 6 個被交易環境所使用，其餘皆為死代碼或語意名不副實的方法：

| 方法名稱 | 呼叫現狀 | 處置方案 | 說明 |
| :--- | :---: | :---: | :--- |
| `CaculateOpenPrcie` | ✅ 被呼叫 | ✏️ **重新命名** ➡️ `calculate_open_price` | 修正雙重錯字（`Caculate` 與 `Prcie`），移除 `closePrcie` 補丁參數。 |
| `CaculatePostion` | ✅ 被呼叫 | ✏️ **重新命名** ➡️ `calculate_position` | 修正雙重錯字（補齊 `i`）。 |
| `Caculatetrade_bar` | ✅ 被呼叫 | ✏️ **重新命名** ➡️ `calculate_trade_bar` | 規範為標準 `snake_case`。 |
| `CaculateCost` | ✅ 被呼叫 | ✏️ **重新命名** ➡️ `calculate_cost` | 規範為標準 `snake_case`。 |
| `CaculateCloseProfit` | ✅ 被呼叫 | ✏️ **重新命名** ➡️ `calculate_close_profit` | 規範為標準 `snake_case`，移除 `closePrcie` 補丁參數。 |
| `CaculateOpenProfit` | ✅ 被呼叫 | ✏️ **重新命名** ➡️ `calculate_open_profit` | 規範為標準 `snake_case`，移除無用參數 `action` 與 `OpenPrice` 補丁參數。 |
| `clip` | ❌ 無人呼叫 | 🗑️ **直接刪除** | 名不副實（內部調用 `np.tanh`），且全庫無任何調用。 |
| `CaculateGameDoneInfo` | ❌ 無人呼叫 | 🗑️ **直接刪除** | 死代碼，無任何呼叫者。 |
| `CaculateEquity_peak_before` | ❌ 無人呼叫 | 🗑️ **直接刪除** | 死代碼，環境已改用滑動視窗下行風險。 |
| `CaculateEquity_peak_after` | ❌ 無人呼叫 | 🗑️ **直接刪除** | 死代碼，環境已改用滑動視窗下行風險。 |
| `Caculate_max_profit_this_trade` | ❌ 無人呼叫 | 🗑️ **直接刪除** | 死代碼，無任何呼叫者。 |

### 2.2 `Reward` 現存問題與清理清單

`Reward` 類別原有 6 個方法與 6 個權重屬性，實際上外部交易環境僅使用了違規交易懲罰：

| 方法 / 屬性 | 呼叫現狀 | 處置方案 | 說明 |
| :--- | :---: | :---: | :--- |
| `wrongTrade(...)` | ✅ 被呼叫 | ✏️ **重新命名** ➡️ `wrong_trade` | 唯一被環境呼叫的懲罰邏輯，規範命名為 `snake_case`。 |
| `wrongTrade_weight` | ✅ 內部使用 | ✏️ **重新命名** ➡️ `wrong_trade_weight` | 唯一使用的權重屬性。 |
| `tradeReturn(...)` | ❌ 無人呼叫 | 🗑️ **直接刪除** | 死代碼。 |
| `OpenReturn(...)` | ❌ 無人呼叫 | 🗑️ **直接刪除** | 死代碼。 |
| `closeReturn(...)` | ❌ 無人呼叫 | 🗑️ **直接刪除** | 死代碼。 |
| `drawdown_penalty(...)` | ❌ 無人呼叫 | 🗑️ **直接刪除** | 死代碼。 |
| `trendTrade(...)` | ❌ 無人呼叫 | 🗑️ **直接刪除** | 死代碼。 |
| 其餘 5 項 weights 屬性 | ❌ 無人呼叫 | 🗑️ **直接刪除** | 刪除 `tradeReturn_weight`, `OpenReturn_weight`, `closeReturn_weight`, `trendTrade_weight`, `drawdown_penalty_weight`。 |

---

## 3. Detailed Architecture & Interface Design / 重構介面與極簡代碼設計

### 3.1 `RewardHelp` 極簡無狀態工具類 (僅保留 6 個核心方法)

所有方法重構為純 `@staticmethod`，沒有任何 alias 殘留，參數嚴格精簡：

```python
class RewardHelp:
    """交易損益、開平倉滑價、手續費與部位轉移之輔助計算工具類別 (無狀態靜態方法集合)"""

    @staticmethod
    def calculate_cost(position: int, action: Actions, cost: float) -> float:
        """
        計算動作產生的交易手續費。

        - 空手開倉 (0 -> 1 或 0 -> -1) 產生手續費
        - 持倉平倉 (1 -> 0 或 -1 -> 0) 產生手續費
        - 續抱或違規操作不產生手續費
        """
        if isinstance(position, bool):
            position = 1 if position else 0

        # 開多 (0 -> 1) 或 開空 (0 -> -1)
        if position == 0 and (action == Actions.Buy or action == Actions.Sell):
            return cost
        # 平多 (1 -> 0)
        elif position == 1 and action == Actions.Sell:
            return cost
        # 平空 (-1 -> 0)
        elif position == -1 and action == Actions.Buy:
            return cost
        return 0.0

    @staticmethod
    def calculate_position(position: int, action: Actions) -> int:
        """
        計算下一個時間步的持倉狀態。

        Returns:
            int: -1 (持有空單), 0 (空手), 1 (持有多單)
        """
        if isinstance(position, bool):
            position = 1 if position else 0

        if position == 0:
            if action == Actions.Buy:
                return 1
            elif action == Actions.Sell:
                return -1
            return 0
        elif position == 1:
            return 0 if action == Actions.Sell else 1
        elif position == -1:
            return 0 if action == Actions.Buy else -1
        return position

    @staticmethod
    def calculate_open_price(
        open_price: float,
        position: int,
        action: Actions,
        default_slippage: float,
        close_price: Optional[float] = None,
    ) -> float:
        """
        計算持倉開倉價格。

        - 空手開多 (0 -> 1): 向上滑價 (close * (1 + slippage))
        - 空手開空 (0 -> -1): 向下滑價 (close * (1 - slippage))
        - 平倉: 重置為 0.0
        - 續抱或無效動作: 維持原 open_price
        """
        if isinstance(position, bool):
            position = 1 if position else 0

        if close_price is not None:
            if position == 0 and action == Actions.Buy:
                return close_price * (1.0 + default_slippage)
            if position == 0 and action == Actions.Sell:
                return close_price * (1.0 - default_slippage)

        # 平多 (1 -> 0) 或 平空 (-1 -> 0): 重置為 0.0
        if (position == 1 and action == Actions.Sell) or (
            position == -1 and action == Actions.Buy
        ):
            return 0.0

        return open_price

    @staticmethod
    def calculate_close_profit(
        position: int,
        action: Actions,
        open_price: float,
        default_slippage: float,
        close_price: Optional[float] = None,
    ) -> float:
        """
        計算平倉時的已實現損益率。

        - 平多單: (close * (1 - slippage) - open) / open
        - 平空單: (open - close * (1 + slippage)) / open
        - 非平倉狀態回傳 0.0
        """
        if isinstance(position, bool):
            position = 1 if position else 0

        closecash_diff = 0.0
        if open_price > 0.0 and close_price is not None:
            # 平多單 (1 -> 0): 賣出平倉向下滑價
            if position == 1 and action == Actions.Sell:
                closecash_diff = (
                    close_price * (1.0 - default_slippage) - open_price
                ) / open_price
            # 平空單 (-1 -> 0): 買進回補向上滑價
            elif position == -1 and action == Actions.Buy:
                closecash_diff = (
                    open_price - close_price * (1.0 + default_slippage)
                ) / open_price

        return closecash_diff

    @staticmethod
    def calculate_open_profit(
        next_position: int,
        close_price: float,
        open_price: Optional[float] = None,
    ) -> float:
        """
        計算當前持倉的未實現浮動損益率。

        - 多頭: (close - open) / open
        - 空頭: (open - close) / open
        - 空手: 0.0
        """
        if isinstance(next_position, bool):
            next_position = 1 if next_position else 0

        opencash_diff = 0.0
        if open_price is not None and open_price > 0.0:
            if next_position == 1:
                opencash_diff = (close_price - open_price) / open_price
            elif next_position == -1:
                opencash_diff = (open_price - close_price) / open_price

        return opencash_diff

    @staticmethod
    def calculate_trade_bar(
        trade_bar: int,
        position: int,
        action: Actions,
    ) -> int:
        """
        計算當前持倉已經過的時間步數 (K棒數)。

        - 開倉步: 回傳 1
        - 續抱步: trade_bar + 1
        - 平倉或空手: 回傳 0
        """
        if isinstance(position, bool):
            position = 1 if position else 0

        # 空手開倉 (0 -> 1 或 0 -> -1)
        if position == 0 and (action == Actions.Buy or action == Actions.Sell):
            return 1
        # 持有多單續抱或重複買
        if position == 1 and (action == Actions.Hold or action == Actions.Buy):
            return trade_bar + 1
        # 持有空單續抱或重複賣
        if position == -1 and (action == Actions.Hold or action == Actions.Sell):
            return trade_bar + 1
        # 平倉或空手維持
        return 0
```

---

### 3.2 `Reward` 單一職責類別 (僅保留 `wrong_trade`)

無任何無用方法、無相容別名，直截了當：

```python
class Reward:
    """強化學習違規交易行為懲罰計算器 (專注於約束智能體合理操作)"""

    def __init__(self, wrong_trade_weight: float = 1.0):
        """
        Args:
            wrong_trade_weight (float): 違規操作的懲罰權重乘數，預設為 1.0。
        """
        self.wrong_trade_weight = wrong_trade_weight

    def wrong_trade(self, position: int, action: Actions) -> float:
        """
        計算違規交易操作之懲罰：
        - 持有多單 (position=1) 時再次發出買入訊號 (action=Buy)，給予懲罰 (-0.001 * weight)
        - 持有空單 (position=-1) 時再次發出賣出訊號 (action=Sell)，給予懲罰 (-0.001 * weight)
        - 空手狀態下的正常開倉、持倉狀態下的平倉或續抱，均不予懲罰 (0.0)

        Args:
            position (int): 當前持倉狀態 (-1: 空頭, 0: 空手, 1: 多頭)
            action (Actions): 智能體採取的動作

        Returns:
            float: 懲罰值（0.0 或負數）
        """
        if isinstance(position, bool):
            position = 1 if position else 0

        reward = 0.0
        if position == 1 and action == Actions.Buy:
            reward = 0.001
        elif position == -1 and action == Actions.Sell:
            reward = 0.001

        return float(self.wrong_trade_weight * reward * -1)
```

---

## 4. Method Mapping & Migration Table / 方法對照與遷移清單

| 原名稱 (Legacy) | 新標準名稱 (Refactored) | 狀態 | 變更說明 |
| :--- | :--- | :---: | :--- |
| **`RewardHelp`** | | | |
| `CaculateCost` | `calculate_cost` | ⚠️ **重大變更 (Breaking)** | 修正錯字，升級為 `@staticmethod` |
| `CaculatePostion` | `calculate_position` | ⚠️ **重大變更 (Breaking)** | 修正雙重錯字（補齊 `i`），升級為 `@staticmethod` |
| `Caculatetrade_bar` | `calculate_trade_bar` | ⚠️ **重大變更 (Breaking)** | 修正命名為 `snake_case`，升級為 `@staticmethod` |
| `CaculateOpenPrcie` | `calculate_open_price` | ⚠️ **重大變更 (Breaking)** | 修正雙重錯字，移除 `closePrcie` 補丁參數 |
| `CaculateCloseProfit` | `calculate_close_profit` | ⚠️ **重大變更 (Breaking)** | 修正錯字，移除 `closePrcie` 補丁參數 |
| `CaculateOpenProfit` | `calculate_open_profit` | ⚠️ **重大變更 (Breaking)** | 修正錯字，移除無用參數 `action` 與 `OpenPrice` 補丁 |
| `clip` | - | 🗑️ **徹底刪除** | 死代碼 + 語意誤導 |
| `CaculateGameDoneInfo` | - | 🗑️ **徹底刪除** | 死代碼 |
| `CaculateEquity_peak_before` | - | 🗑️ **徹底刪除** | 死代碼 |
| `CaculateEquity_peak_after` | - | 🗑️ **徹底刪除** | 死代碼 |
| `Caculate_max_profit_this_trade` | - | 🗑️ **徹底刪除** | 死代碼 |
| **`Reward`** | | | |
| `wrongTrade(...)` | `wrong_trade(...)` | ⚠️ **重大變更 (Breaking)** | 規範為標準 `snake_case`，**不留別名** |
| `wrongTrade_weight` | `wrong_trade_weight` | ⚠️ **重大變更 (Breaking)** | 規範為標準 `snake_case`，**不留相容 property** |
| `tradeReturn(...)` | - | 🗑️ **徹底刪除** | 死代碼 |
| `OpenReturn(...)` | - | 🗑️ **徹底刪除** | 死代碼 |
| `closeReturn(...)` | - | 🗑️ **徹底刪除** | 死代碼 |
| `drawdown_penalty(...)` | - | 🗑️ **徹底刪除** | 死代碼 |
| `trendTrade(...)` | - | 🗑️ **徹底刪除** | 死代碼 |
| 其餘 5 項權重屬性 | - | 🗑️ **徹底刪除** | 刪除所有未使用權重變數 |

---

## 5. Impact Analysis / 外部依賴與後續呼叫點修正計畫

由於本方案採取**不保留向後相容層**的乾淨重構策略，完成 `reward.py` 的重構後，以下依賴檔案在未修改前會觸發 `AttributeError`，需在後續進行呼叫點的同步更新：

1. **[`Brain/DQN/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py)**：
   - 呼叫點更新：
     - L198: `self.reward_function.wrongTrade` ➡️ `self.reward_function.wrong_trade`
     - L203: `self.reward_help.CaculateCloseProfit` ➡️ `RewardHelp.calculate_close_profit`（或靜態呼叫）
     - L212: `self.reward_help.CaculateOpenPrcie` ➡️ `RewardHelp.calculate_open_price`
     - L221: `self.reward_help.CaculatePostion` ➡️ `RewardHelp.calculate_position`
     - L226: `self.reward_help.CaculateCost` ➡️ `RewardHelp.calculate_cost`
     - L234: `self.reward_help.CaculateOpenProfit` ➡️ `RewardHelp.calculate_open_profit`
     - L242: `self.reward_help.Caculatetrade_bar` ➡️ `RewardHelp.calculate_trade_bar`
2. **[`Brain/HopeDQN/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/HopeDQN/lib/environment.py)**：
   - 與上述相同之 7 個呼叫點更新。
3. **[`tests/test_dqn_short_selling.py`](file:///home/b0457812963/Mamba3RL/SynapseX/tests/test_dqn_short_selling.py)**：
   - 將測試用例中呼叫的舊方法名全部同步更名為新方法名。

---

## 6. Implementation Checklist / 實作改動查核清單

- [X] **1. `Brain/DQN/lib/reward.py` 實作重構**
  - [X] 1.1 刪除 `RewardHelp` 中 5 個未使用的函數（`clip`, `CaculateGameDoneInfo`, `CaculateEquity_peak_*`, `Caculate_max_profit_this_trade`）。
  - [X] 1.2 將剩餘 6 個核心方法重寫為標準 `@staticmethod` 與 `snake_case`，清理補丁參數與無用參數。
  - [X] 1.3 刪除 `Reward` 中 5 個未使用的函數與無用屬性，僅保留 `wrong_trade` 與 `wrong_trade_weight`。
  - [X] 1.4 **不加入任何向後相容層或別名**。
- [X] **2. 外部呼叫點與測試同步更新**
  - [X] 2.1 更新 `Brain/DQN/lib/environment.py` 中的呼叫點。
  - [X] 2.2 更新 `Brain/HopeDQN/lib/environment.py` 中的呼叫點。
  - [X] 2.3 更新 `tests/test_dqn_short_selling.py` 中的單元測試。
- [X] **3. 測試驗證**
  - [X] 3.1 執行單元測試確保重構後 100% 綠燈通過。

---

## 7. Verification Checklist / 驗證查核清單

完成重構與呼叫點更新後，執行以下指令驗證：

```bash
# 1. 執行 DQN 放空與 Reward 單元測試
/home/b0457812963/Mamba3RL/bin/python -m unittest tests/test_dqn_short_selling.py

# 2. 執行全庫自動化測試
/home/b0457812963/Mamba3RL/bin/python -m unittest discover -s tests
```
- 驗證預期：所有測試通過（OK），無任何 `AttributeError` 或型別報錯。
