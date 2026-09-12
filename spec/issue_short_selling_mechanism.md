# Specification: DQN 環境放空機制設計規範 (Short-Selling Mechanism)

## 📋 Index / 目錄
1. [Overview & Motivation / 概述與動機](#1-overview--motivation--概述與動機)
2. [Current State & Limitations / 現狀與瓶頸分析](#2-current-state--limitations--現狀與瓶頸分析)
3. [Core Architecture Design / 核心架構設計](#3-core-architecture-design--核心架構設計)
   - [3.1 部位狀態表示 (Position Representation)](#31-部位狀態表示-position-representation)
   - [3.2 動作空間與狀態轉移表 (Action Space & State Transitions)](#32-動作空間與狀態轉移表-action-space--state-transitions)
   - [3.3 財務數學與損益計算公式 (Financial Math & PnL Formula)](#33-財務數學與損益計算公式-financial-math--pnl-formula)
   - [3.4 特徵編碼覆寫 (Observation Encoding - `encode()`)](#34-特徵編碼覆寫-observation-encoding--encode)
   - [3.5 獎勵函數與風險指標相容性 (Reward & Risk Metrics)](#35-獎勵函數與風險指標相容性-reward--risk-metrics)
4. [Module Design & RewardHelp Architecture / 模組權責與 RewardHelp 重構設計](#4-module-design--rewardhelp-architecture--模組權責與-rewardhelp-重構設計)
5. [Implementation Checklist / 實作改動查核清單](#5-implementation-checklist--實作改動查核清單)
6. [Verification Checklist / 驗證查核清單](#6-verification-checklist--驗證查核清單)

---

## 1. Overview & Motivation / 概述與動機

本規範旨在為 `Brain/DQN` 的核心交易環境新增**放空（Short-Selling）交易機制**。

### 背景與目的：
目前 SynapseX 的 DQN 交易環境僅支援「做多（Buy）」與「空手（Hold）」之單向交易。在加密貨幣與期貨市場中，雙向交易是基本且關鍵的獲利與避險工具。在熊市或劇烈下行行情中，單向交易系統只能被迫空手防禦；引入放空機制後，智能體（Agent）將具備在下行趨勢中主動放空獲利的能力，進一步擴展策略適應性與夏普比率（Sharpe Ratio）。

---

## 2. Current State & Limitations / 現狀與瓶頸分析

經檢視現行代碼（[environment.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py#L19-L274) 及 [reward.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/reward.py#L408-L608)），現有實作存在以下限制：

| 模組 / 項目 | 現行實作 | 限制與問題 |
| :--- | :--- | :--- |
| **部位狀態** | `self.have_position: bool` | 僅有 `True`/`False` 兩種狀態，無法表達空頭部位。 |
| **動作語意** | `Actions.Hold=0, Buy=1, Sell=2` | 空手時執行 `Sell` 會被直接判定為違規交易（觸發 `wrongTrade` 懲罰）。 |
| **開平倉滑價** | 開倉皆 `(1 + slippage)`、平倉皆 `(1 - slippage)` | 放空開倉應向下滑價 `(1 - slippage)`，平倉回補應向上滑價 `(1 + slippage)`。 |
| **浮動與平倉損益** | `(close - open) / open` | 僅考慮價格上漲為正收益，若為做空則下跌才應是獲利。 |
| **輔助類別 `RewardHelp`** | 多處判斷基於 `have_position: bool` | 無法處理 `position == -1`（空頭）時的回補、浮動損益與手續費計算。 |
| **觀測編碼 (`encode`)** | `data_res[:, idx] = 1.0` | 僅向神經網絡傳遞「有倉/無倉」，智能體無法從 Observation 感知當前是多頭還是空頭。 |

---

## 3. Core Architecture Design / 核心架構設計

### 3.1 部位狀態表示 (Position Representation)

將二元變數 `have_position: bool` 升級為三元離散部位值 **`self.position: int`**：

$$
\text{position} \in \{ -1, 0, 1 \}
$$

- `+1`：持有多單（Long）
- `0`：空手（Flat）
- `-1`：持有空單（Short）

> **簡化架構設計說明**：
> DQN 內部已全面升級並統一採用 `self.position`（取值 `-1, 0, 1`），原向下相容之 `have_position` 經評估已屬多餘並直接刪除，持倉判斷直接統一使用 `self.position != 0`。

---

### 3.2 動作空間與狀態轉移表 (Action Space & State Transitions)

**決策：採用【方案 A：訂單操作制（維持 3 個動作）】**
維持既有神經網絡的輸出空間大小 `action_space.n = 3`（`Hold=0, Buy=1, Sell=2`），無需調整模型架構或輸出層權重。

狀態轉移規則如下表所示：

| 當前部位 (`position`) | 智能體動作 (`action`) | 下一步部位 (`next_position`) | 交易動作意義 | 產生費用與懲罰 |
| :---: | :---: | :---: | :---: | :---: |
| **`0` (空手)** | `Actions.Hold (0)` | `0` (空手) | 維持空手 | 無手續費，無懲罰 |
| **`0` (空手)** | `Actions.Buy (1)` | `1` (多單) | **買進開多 (Open Long)** | 扣除開倉手續費，按買方滑價開倉 |
| **`0` (空手)** | `Actions.Sell (2)` | `-1` (空單) | **賣出放空 (Open Short)** | 扣除開倉手續費，按賣方滑價開倉 |
| **`1` (多單)** | `Actions.Hold (0)` | `1` (多單) | 續抱多單 | 無手續費，累計持倉 K 棒 |
| **`1` (多單)** | `Actions.Sell (2)` | `0` (空手) | **賣出平多 (Close Long)** | 扣除平倉手續費，結算多單損益 |
| **`1` (多單)** | `Actions.Buy (1)` | `1` (多單) | 違規重複買進 | 觸發 `wrongTrade` 懲罰，部位維持不變 |
| **`-1` (空單)** | `Actions.Hold (0)` | `-1` (空單) | 續抱空單 | 無手續費，累計持倉 K 棒 |
| **`-1` (空單)** | `Actions.Buy (1)` | `0` (空手) | **買進平空 (Cover Short)** | 扣除平倉手續費，結算空單損益 |
| **`-1` (空單)** | `Actions.Sell (2)` | `-1` (空單) | 違規重複放空 | 觸發 `wrongTrade` 懲罰，部位維持不變 |

*註：不採用「單步直接反手」（如多單按 Sell 直接轉為空單），而是必須遵循「開倉 $\to$ 平倉 $\to$ 開倉」的兩步走邏輯，確保滑價、成交量與手續費計算具備清晰的物理與會計語意。*

---

### 3.3 財務數學與損益計算公式 (Financial Math & PnL Formula)

#### 1. 開倉價格與滑價（Entry Price with Slippage）
- **開多單 ($0 \to 1$)**：以較高價買入
  $$P_{\text{open}} = P_{\text{close}} \times (1 + \delta_{\text{slip}})$$
- **開空單 ($0 \to -1$)**：以較低價賣出
  $$P_{\text{open}} = P_{\text{close}} \times (1 - \delta_{\text{slip}})$$

#### 2. 平倉價格與滑價（Exit Price with Slippage）
- **平多單 ($1 \to 0$)**：賣出平倉，以較低價成交
  $$P_{\text{exec}} = P_{\text{close}} \times (1 - \delta_{\text{slip}})$$
- **平空單 ($-1 \to 0$)**：買入回補，以較高價成交
  $$P_{\text{exec}} = P_{\text{close}} \times (1 + \delta_{\text{slip}})$$

#### 3. 平倉已實現損益（Realized Close Profit Diff）
- **平多單損益**：
  $$\Delta_{\text{close\_profit}} = \frac{P_{\text{exec}} - P_{\text{open}}}{P_{\text{open}}}$$
- **平空單損益**：
  $$\Delta_{\text{close\_profit}} = \frac{P_{\text{open}} - P_{\text{exec}}}{P_{\text{open}}}$$

#### 4. 未平倉浮動損益（Unrealized Open Profit Diff）
使用部位乘數進行統一簡潔表達：
$$\Delta_{\text{open\_profit}} = \text{position} \times \frac{P_{\text{close}} - P_{\text{open}}}{P_{\text{open}}}$$
- 當 $\text{position} = 1$：價格上漲獲利、下跌虧損。
- 當 $\text{position} = -1$：價格下跌獲利、上漲虧損。
- 當 $\text{position} = 0$：損益為 $0.0$。

#### 5. 總淨值與單步報酬（Portfolio Equity & Step Return）
$$\text{TotalPortfolioPercent} = 1.0 - \text{cost\_sum} + \text{closecash} + \Delta_{\text{open\_profit}}$$
$$R_{p, t} = \text{TotalPortfolioPercent}_t - \text{TotalPortfolioPercent}_{t-1}$$

---

### 3.4 特徵編碼覆寫 (Observation Encoding - `encode()`)

在 `State_time_step` 中直接覆寫繼承自 `State_time_step_template` 的 `encode()` 方法：

```python
def encode(self):
    data_res = np.zeros(shape=self.getStateShape(), dtype=np.float32)
    time_res = np.zeros(shape=self.getTimeShape(), dtype=np.float32)

    ofs = self.bars_count
    for bar_idx in range(self.bars_count):
        for idx, field in enumerate(self.info_list):
            data_res[bar_idx][idx] = getattr(self._prices, field)[
                self._offset - ofs + bar_idx
            ]

    # --- 部位與交易特徵更新 ---
    if self.position != 0:
        # 特徵 1: 部位方向與大小 (-1.0 代表空頭，1.0 代表多頭)
        data_res[:, len(self.info_list)] = float(self.position)
        
        # 特徵 2: 浮動損益率 (依多空方向正確計算，做空跌為正)
        unrealized_return = self.position * (
            self._prices.close[self._offset - 1] - self.open_price
        ) / self.open_price
        data_res[:, len(self.info_list) + 1] = float(unrealized_return)
        
        # 特徵 3: 持倉時間累計
        data_res[:, len(self.info_list) + 2] = float(self.trade_bar)

    for bar_idx in range(self.bars_count):
        for idx, field in enumerate(self.timelist):
            time_res[bar_idx][idx] = getattr(self._prices, field)[
                self._offset - ofs + bar_idx
            ]

    return data_res, time_res
```

---

### 3.5 獎勵函數與風險指標相容性 (Reward & Risk Metrics)

1. **下行風險懲罰（Downside Risk Penalty）**：
   - 由於放空操作虧損時，$R_{p, t} < 0$，現有 `calculate_step_downside_penalty()` 採用的半變異數（Semi-Variance）公式會自動統計此虧損並施加懲罰，完全無縫相容。
2. **大盤超額回報與相對指標**：
   - 當大盤下跌（$R_{b, t} < 0$）而做空策略上漲（$R_{p, t} > 0$）時，相對超額回報 $R_{p, t} - R_{b, t}$ 會顯著為正，有效激勵模型在熊市尋找做空機會。

---

## 4. Module Design & RewardHelp Architecture / 模組權責與 RewardHelp 重構設計

### 4.1 設計定位確認：`Brain/DQN/lib/reward.py` 為 DQN 專屬模組
- 原則確認：`Brain/DQN/lib/reward.py` 本身即專屬於 DQN 架構。
- 策略方針：**直接重構 `Brain/DQN/lib/reward.py` 中的 `RewardHelp` 與 `Reward`**，讓其第一公民支援 `position: int`（$-1, 0, 1$），保持 `State_time_step` 與 `RewardHelp` 之間的職責劃分與良好封裝。

### 4.2 `RewardHelp` 升級重構規範
在 [Brain/DQN/lib/reward.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/reward.py#L408-L538) 中升級以下方法：

1. **`CaculatePostion(self, position: int, action: Actions) -> int`**：
   - `position == 0`: `Buy -> 1`, `Sell -> -1`, `Hold -> 0`
   - `position == 1`: `Sell -> 0`, `Hold/Buy -> 1`
   - `position == -1`: `Buy -> 0`, `Hold/Sell -> -1`
2. **`CaculateCost(self, position: int, action: Actions, cost: float) -> float`**：
   - 任何發生開倉（$0 \to 1$ 或 $0 \to -1$）或平倉（$1 \to 0$ 或 $-1 \to 0$）時回傳 `cost`，否則回傳 `0.0`。
3. **`CaculateOpenPrcie(self, openPrice: float, position: int, action: Actions, default_slippage: float, closePrice: float) -> float`**：
   - 開多單：`closePrice * (1 + default_slippage)`
   - 開空單：`closePrice * (1 - default_slippage)`
   - 平多/平空：重置為 `0.0`
   - 續抱：維持原 `openPrice`
4. **`CaculateCloseProfit(self, position: int, action: Actions, openPrice: float, default_slippage: float, closePrice: float) -> float`**：
   - 平多：`((closePrice * (1 - default_slippage)) - openPrice) / openPrice`
   - 平空：`(openPrice - (closePrice * (1 + default_slippage))) / openPrice`
   - 無平倉：`0.0`
5. **`CaculateOpenProfit(self, next_position: int, action: Actions, closePrice: float, openPrice: float) -> float`**：
   - `next_position == 1`: `(closePrice - openPrice) / openPrice`
   - `next_position == -1`: `(openPrice - closePrice) / openPrice`
   - `next_position == 0`: `0.0`
6. **`Caculatetrade_bar(self, trade_bar: int, position: int, action: Actions) -> int`**：
   - 發生開倉時設為 `1`
   - 續抱時 `trade_bar + 1`
   - 平倉或空手時重置為 `0`

### 4.3 `Reward.wrongTrade` 升級重構規範
在 [Brain/DQN/lib/reward.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/reward.py#L590-L602) 中：
- `position == 1 and action == Actions.Buy`: 懲罰 `0.001`（多單重複買）
- `position == -1 and action == Actions.Sell`: 懲罰 `0.001`（空單重複賣）
- 空手（`position == 0`）時的 `Buy` 與 `Sell` 均為合法開倉，**不予懲罰**。

---

## 5. Implementation Checklist / 實作改動查核清單

### 階段一：`Brain/DQN/lib/reward.py` 修改查核清單
- [x] **5.1.1 重構 `RewardHelp.CaculatePostion`**
  - 輸入參數由 `havePostion: bool` 改為 `position: int`（取值 `-1, 0, 1`）。
  - 空手（0）執行 `Buy` 回傳 `1`，執行 `Sell` 回傳 `-1`，執行 `Hold` 回傳 `0`。
  - 多單（1）執行 `Sell` 回傳 `0`，其他維持 `1`。
  - 空單（-1）執行 `Buy` 回傳 `0`，其他維持 `-1`。
- [x] **5.1.2 重構 `RewardHelp.CaculateCost`**
  - 參數適配 `position: int`。
  - 僅在發生實質開倉（$0 \to 1, 0 \to -1$）或實質平倉（$1 \to 0, -1 \to 0$）時回傳手續費 `cost`，其餘狀態（續抱、空手維持、違規動作）回傳 `0.0`。
- [x] **5.1.3 重構 `RewardHelp.CaculateOpenPrcie`**
  - 參數適配 `position: int`。
  - 空手開多（`position == 0 and action == Actions.Buy`）：`closePrice * (1 + default_slippage)`。
  - 空手開空（`position == 0 and action == Actions.Sell`）：`closePrice * (1 - default_slippage)`。
  - 平倉（多單 Sell 或空單 Buy）：重置為 `0.0`。
  - 續抱或違規操作：保持原 `openPrice`。
- [x] **5.1.4 重構 `RewardHelp.CaculateCloseProfit`**
  - 參數適配 `position: int`。
  - 平多單（`position == 1 and action == Actions.Sell`）：`((closePrice * (1 - default_slippage)) - openPrice) / openPrice`。
  - 平空單（`position == -1 and action == Actions.Buy`）：`(openPrice - (closePrice * (1 + default_slippage))) / openPrice`。
  - 非平倉情境一律回傳 `0.0`。
- [x] **5.1.5 重構 `RewardHelp.CaculateOpenProfit`**
  - 參數適配 `next_position: int`。
  - 持多單（`next_position == 1`）：`(closePrice - openPrice) / openPrice`。
  - 持空單（`next_position == -1`）：`(openPrice - closePrice) / openPrice`。
  - 空手（`next_position == 0`）：回傳 `0.0`。
- [x] **5.1.6 重構 `RewardHelp.Caculatetrade_bar`**
  - 參數適配 `position: int`。
  - 剛開倉時設為 `1`。
  - 續抱中（多單按 Buy/Hold，空單按 Sell/Hold）累加 `1`。
  - 平倉或空手維持設為 `0`。
- [x] **5.1.7 更新其餘輔助方法（`CaculateEquity_peak_*` 等）**
  - 檢查並更新 `RewardHelp` 中其餘引用 `havePostion` 之函數，改以 `position != 0` 或多空雙向平倉判斷。
- [x] **5.1.8 重構 `Reward.wrongTrade`**
  - 參數由 `havePostion: bool` 改為 `position: int`。
  - 多單時重複買（`position == 1 and action == Actions.Buy`）：給予負懲罰。
  - 空單時重複賣（`position == -1 and action == Actions.Sell`）：給予負懲罰。
  - 空手開倉（`position == 0` 的 Buy/Sell）與正常平倉/續抱：懲罰為 `0.0`。

### 階段二：`Brain/DQN/lib/environment.py` 修改查核清單
- [x] **5.2.1 `State_time_step.reset()` 重構**
  - 將 `self.have_position = False` 改為 `self.position = 0`。
  - 確認移除多餘之 `have_position` 屬性，統一採用 `self.position`。
  - 確認初始值 `self.open_price = 0.0`、`self.closecash = 0.0`、`self.cost_sum = 0.0`、`self.trade_bar = 0` 正確重置。
- [x] **5.2.2 `State_time_step.step()` 重構**
  - 呼叫 `reward_function.wrongTrade(self.position, action=action)`。
  - 呼叫 `reward_help.CaculateCloseProfit(self.position, action, self.open_price, self.max_default_slippage, _close_price)`。
  - 呼叫 `reward_help.CaculateOpenPrcie(self.open_price, self.position, action, self.max_default_slippage, _close_price)`。
  - 呼叫 `reward_help.CaculatePostion(self.position, action=action)` 取得 `next_position`。
  - 呼叫 `reward_help.CaculateCost(self.position, action=action, cost=self.max_commission)` 計算單步手續費。
  - 呼叫 `reward_help.CaculateOpenProfit(next_position, action, _close_price, self.open_price)` 計算浮動損益。
  - 呼叫 `reward_help.Caculatetrade_bar(self.trade_bar, self.position, action=action)`。
  - 更新 `self.position = next_position`。
  - 累計 `self.TotalPortfolioPercent`，並正確計算 `current_p_return` 與下行風險懲罰。
- [x] **5.2.3 `State_time_step.encode()` 覆寫**
  - 直接在 `State_time_step` 中覆寫繼承自 `State_time_step_template` 之 `encode`。
  - 特徵槽 `len(info_list)`：寫入 `float(self.position)`（`-1.0`, `0.0`, `1.0`）。
  - 特徵槽 `len(info_list) + 1`：寫入依多空方向正確計算之浮動損益率。
  - 特徵槽 `len(info_list) + 2`：寫入當前 `self.trade_bar`。
- [x] **5.2.4 `BaseTradingEnv.step()` info 字典適配**
  - 將 `"postion": float(self._state.have_position)` 改為 `"postion": float(self._state.position)`，精確輸出 `-1.0, 0.0, 1.0`。

---

## 6. Verification Checklist / 驗證查核清單

- [x] **6.1 單元測試腳本編寫與執行 (`tests/test_dqn_short_selling.py`)**
  - [x] **6.1.1 完整多單交易週期驗證 (Long Cycle)**
    - [x] 空手 (0) 執行 `Buy`：部位變為 `1`，扣除手續費，開倉價向上滑價 `close * (1 + slip)`，`trade_bar == 1`。
    - [x] 多單 (1) 執行 `Hold`：部位維持 `1`，手續費不增加，`trade_bar == 2`，價格上漲時淨值增加。
    - [x] 多單 (1) 執行 `Sell`：部位變為 `0`，扣除手續費，平倉價向下滑價 `close * (1 - slip)`，實現盈虧結算正確，`open_price == 0`，`trade_bar == 0`。
  - [x] **6.1.2 完整空單交易週期驗證 (Short Cycle)**
    - [x] 空手 (0) 執行 `Sell`：部位變為 `-1`，扣除手續費，開倉價向下滑價 `close * (1 - slip)`，`trade_bar == 1`。
    - [x] 空單 (-1) 執行 `Hold`：部位維持 `-1`，手續費不增加，`trade_bar == 2`，價格下跌時浮動盈虧為正、價格上漲時浮動盈虧為負。
    - [x] 空單 (-1) 執行 `Buy`：部位變為 `0`，扣除手續費，平倉回補價向上滑價 `close * (1 + slip)`，實現盈虧結算正確，`open_price == 0`，`trade_bar == 0`。
  - [x] **6.1.3 違規交易與無效操作懲罰驗證 (Wrong Trade Penalty)**
    - [x] 空手 (0) 執行 `Buy` 或 `Sell`：確認為合法開倉，違規懲罰為 `0.0`。
    - [x] 多單 (1) 重複執行 `Buy`：部位維持 `1`，觸發 `wrongTrade` 負懲罰。
    - [x] 空單 (-1) 重複執行 `Sell`：部位維持 `-1`，觸發 `wrongTrade` 負懲罰。
  - [x] **6.1.4 觀測矩陣 Observation 編碼驗證 (`encode()`)**
    - [x] 空手狀態：部位維度值為 `0.0`，損益值為 `0.0`，持倉時間為 `0.0`。
    - [x] 多頭狀態：部位維度值為 `1.0`，浮動損益為正向 `(close - open) / open`。
    - [x] 空頭狀態：部位維度值為 `-1.0`，浮動損益為逆向 `(open - close) / open`。
    - [x] 確認整體矩陣 Shape 完全相容既有模型（`bars_count, len(info_list) + 3`）。
  - [x] **6.1.5 環境隨機推進與數值穩定性驗證 (End-to-End Stepping)**
    - [x] 執行 `TrainingEnv.reset()` 檢查初始狀態各指標無誤。
    - [x] 連續執行至少 200 步隨機動作（包含隨機 Hold/Buy/Sell），確認無 `NaN`、`Inf`，淨值與回報連續平滑。
