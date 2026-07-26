# Specification: Trading Environment Step Logic & Convergence Fixes

## 📋 Index / 目錄
1. [Issue Overview & Index / 問題概述與索引](#1-issue-overview--index--問題概述與索引)
2. [Issue 1: 未平倉損益特徵未來價格洩漏 (Look-Ahead Bias)](#2-issue-1-未平倉損益特徵未來價格洩漏-look-ahead-bias)
3. [Issue 2: TrainingEnv.reset() 隨機 Offset 範圍優化 (Explicit Boundary)](#3-issue-2-trainingenvreset-隨機-offset-範圍優化-explicit-boundary)
4. [Issue 3: 違規懲罰權重過大 (Reward Scale Mismatch)](#4-issue-3-違規懲罰權重過大-reward-scale-mismatch)
5. [Issue 5: 變數覆蓋與殘留程式碼清理 (Dead Code & Reward Variable Overwrite)](#5-issue-5-變數覆蓋與殘留程式碼清理-dead-code--reward-variable-overwrite)
6. [Proposed Code Modifications Summary / 修改方案彙整](#6-proposed-code-modifications-summary--修改方案彙整)
7. [Verification Plan / 驗證計畫](#7-verification-plan--驗證計畫)

---

## 1. Issue Overview & Index / 問題概述與索引

本規範針對 [environment.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py) 及相關環境組件進行系統性邏輯與強化學習收斂問題修正。經過討論與審查，確認以下 4 項關鍵問題進行修正與結構優化：

| Index / 編號 | 問題分類 | 影響檔案與行號 | 問題說明 |
| :---: | :--- | :--- | :--- |
| **Item 1** | 邏輯 Bug (Data Leak) | [env_components.py: L86](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py#L86) | 觀測特徵讀取到未來 $t+1$ 期的價格（Look-Ahead Bias）。 |
| **Item 2** | 代碼優化 (Readability) | [environment.py: L368](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py#L368) | 原公式以 10 倍 `bars_count` 緩衝留白，改為以 `N_steps` 計算更直觀明確。 |
| **Item 3** | RL 訓練問題 (Scale) | [reward.py: L590](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/reward.py#L590) | 違規交易懲罰（`-0.01`）遠大於單步報酬（`~0.001`），主導梯度更新。 |
| **Item 5** | 代碼品質 (Dead Code) | [environment.py: L182, L241](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py#L182) | `reward` 變數在中途被賦值後又於尾端被重新覆蓋，且存在未使用的殘留權重。 |

---

## 2. Issue 1: 未平倉損益特徵未來價格洩漏 (Look-Ahead Bias)

### 2.1 原因分析
在 [BaseTradingEnv.step()](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py#L296-L310) 中：
1. 呼叫 `self._state.step(action)`，其中執行了 `self._offset += 1`（Offset 由當前完成期 $t$ 增至 $t+1$）。
2. 隨後呼叫 `self._state.encode()` 產生對 Agent 的 Observation。
3. `encode()` 中使用 `self._prices.close[self._offset]` 計算持倉浮動損益：
   ```python
   data_res[:, len(self.info_list) + 1] = (
       self._prices.close[self._offset] - self.open_price
   ) / self.open_price
   ```
   由於 `self._offset` 已變為 $t+1$，該處存取到了 $t+1$ 的收盤價，但 K 棒特徵矩陣僅包含到 $t$ 期，造成了觀測狀態的未來價格洩漏。

### 2.2 修正方案
在 [env_components.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py#L86) 中將 `self._offset` 調整為 `self._offset - 1`，精確存取剛完成的 $t$ 期收盤價。

---

## 3. Issue 2: TrainingEnv.reset() 隨機 Offset 範圍優化 (Explicit Boundary)

### 3.1 討論與原因分析
在 [environment.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py#L368-L370) 中，原本 `TrainingEnv.reset()` 的寫法為：
```python
offset = (
    np.random.choice(prices.high.shape[0] - self._state.bars_count * 10)
    + self._state.bars_count
)
```
原作者寫法係透過扣除 10 倍 `bars_count`（例如 $30 \times 10 = 300$）作為尾端緩衝留白。在資料長度足夠時該寫法能正常運作，並非必然導致越界。

然而，以 `bars_count * 10` 作為緩衝屬於經驗數值，與回合實際執行的總步數 `N_steps` 無直接語意關聯。

### 3.2 修正方案
將上限計算顯式調整為與 Episode 最大步數 `N_steps` 直接綁定：
```python
max_offset = prices.high.shape[0] - self._state.N_steps - 1
min_offset = self._state.bars_count
assert max_offset > min_offset, f"Dataset length {prices.high.shape[0]} too short for N_steps={self._state.N_steps}"
offset = np.random.randint(min_offset, max_offset)
```
**效益**：讓邊界約束邏輯更為直觀、語意明確且提高可維護性。

---

## 4. Issue 3: 違規懲罰權重過大 (Reward Scale Mismatch)

### 4.1 原因分析
在 [reward.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/reward.py#L590) 中，當持倉時執行 Buy 或空倉時執行 Sell，會觸發 `wrongTrade` 懲罰：
```python
def wrongTrade(self, havePostion: bool, action: Actions):
    _reward = 0
    if havePostion and action == Actions.Buy:
        _reward = 0.01  # -1.0% 的懲罰
    elif not (havePostion) and action == Actions.Sell:
        _reward = 0.01
    return self.wrongTrade_weight * _reward * -1
```
單步真實價格報酬 `current_p_return` 數量級約為 `0.0005 ~ 0.002`（0.05% ~ 0.2%），違規懲罰為其 10~20 倍，導致 DQN 的 Loss 與 Q 值更新完全被「防範違規」主導，無法有效學習買賣時機。

### 4.2 修正方案
1. 將 `wrongTrade` 基底懲罰從 `0.01` 降至 `0.001`（0.1%）。
2. 在 `State_time_step.weights` 增設 `w5_wrong_trade` 配置項，便於微調與管理。

---

## 5. Issue 5: 變數覆蓋與殘留程式碼清理 (Dead Code & Reward Variable Overwrite)

### 5.1 原因分析
在 [environment.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py#L182, L241) 中：
```python
182: reward += wrongTrade_reward 
...
241: reward = (self.weights['w1_step_return'] * current_p_return  + wrongTrade_reward)
```
第 182 行對 `reward` 的累加會被第 241 行的直接賦值 `=` 完全覆蓋，屬於死碼（Dead Code）。

### 5.2 修正方案
1. 移除第 182 行無意義的累加，統一在第 241 行集中計算 `reward`。
2. 清理未採用的下行風險計算及註解殘留，提升程式碼 readability 與可維護性。

---

## 6. Proposed Code Modifications Summary / 修改方案彙整

### 6.1 Target File: [Brain/Common/env_components.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py)
```diff
        if self.have_position:
            data_res[:, len(self.info_list)] = 1.0
            data_res[:, len(self.info_list) + 1] = (
-               self._prices.close[self._offset] - self.open_price
+               self._prices.close[self._offset - 1] - self.open_price
            ) / self.open_price
            data_res[:, len(self.info_list) + 2] = self.trade_bar
```

### 6.2 Target File: [Brain/DQN/lib/reward.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/reward.py)
```diff
    def wrongTrade(self, havePostion: bool, action: Actions):
        _reward = 0
        if havePostion and action == Actions.Buy:
-           _reward = 0.01
+           _reward = 0.001
        elif not (havePostion) and action == Actions.Sell:
-           _reward = 0.01
+           _reward = 0.001

        return self.wrongTrade_weight * _reward * -1
```

### 6.3 Target File: [Brain/DQN/lib/environment.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py)
```diff
        self.weights = {
            'w1_step_return': 1,  # 年化報酬權重
            'w2_downside_ratio': 30,   # 下行風險權重 (懲罰項)
            'w3_diff_return': 0.2, # 差異報酬權重 
            'w4_treynor': 0.15,     # 崔諾指標權重
+           'w5_wrong_trade': 1.0   # 違規交易懲罰權重
        }
...
-       reward += wrongTrade_reward 
...
-       reward = (self.weights['w1_step_return'] * current_p_return  + wrongTrade_reward)
+       reward = (self.weights['w1_step_return'] * current_p_return + self.weights['w5_wrong_trade'] * wrongTrade_reward)
...
    def reset(self, symbol: str = None):
...
-       offset = (
-           np.random.choice(prices.high.shape[0] - self._state.bars_count * 10)
-           + self._state.bars_count
-       )
+       max_offset = prices.high.shape[0] - self._state.N_steps - 1
+       min_offset = self._state.bars_count
+       assert max_offset > min_offset, f"Dataset length {prices.high.shape[0]} too short for N_steps={self._state.N_steps}"
+       offset = np.random.randint(min_offset, max_offset)
```

---

## 7. Verification Plan / 驗證計畫

### 7.1 環境測試指令
使用專案指定 Python 環境執行語法與單元測試：
```bash
/home/b0457812963/Mamba3RL/bin/python -c "
from Brain.DQN.lib.environment import State_time_step, TrainingEnv
print('SUCCESS: Environment imports without error.')
"
```

### 7.2 邏輯與邊界驗證項目
1. **無未來價格洩漏**：驗證 `encode()` 輸出的浮動損益確實等於 `(close[t] - open_price) / open_price`。
2. **Offset 隨機抽樣直觀邊界**：確認 `reset()` 顯式使用 `N_steps` 約束。
3. **Reward Scale 檢查**：確認單步違規懲罰值降為 `-0.001`，與 `current_p_return` 的比重平衡。
