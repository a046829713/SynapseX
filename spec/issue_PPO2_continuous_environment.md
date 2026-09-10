# Specification: PPO2 Continuous Action Trading Environment Refactoring

## 📋 Index / 目錄
1. [Background & Objectives / 背景與目標](#1-background--motivation--背景與目標)
2. [Comparison: DQN (Discrete) vs PPO2 (Continuous) / DQN 離散與 PPO2 連續環境差異比較](#2-comparison-dqn-discrete-vs-ppo2-continuous--dqn-離散與-ppo2-連續環境差異比較)
3. [Continuous Action Definition & Deadzone Analysis / 連續動作定義與平倉死區深度分析](#3-continuous-action-definition--deadzone-analysis--連續動作定義與平倉死區深度分析)
4. [Portfolio Accounting & State Dynamics / 部位轉移與記帳數學邏輯](#4-portfolio-accounting--state-dynamics--部位轉移與記帳數學邏輯)
5. [Observation Space & Feature Encoding / 觀測空間 (spaces.Dict) 與特徵編碼對齊](#5-observation-space--feature-encoding--觀測空間-spacesdict-與特徵編碼對齊)
6. [Reward Function Formulation / 獎勵函數設計](#6-reward-function-formulation--獎勵函數設計)
7. [Environment Architecture & Class Design / 環境架構與類別設計](#7-environment-architecture--class-design--環境架構與類別設計)
8. [Proposed Code Modifications / 預計代碼改動細節](#8-proposed-code-modifications--預計代碼改動細節)
9. [Discussion Conclusions / 討論結論彙整](#9-discussion-conclusions--討論結論彙整)
10. [Implementation Checklist / 實作檢核清單](#10-implementation-checklist--實作檢核清單)
11. [Verification Plan / 驗證計畫](#11-verification-plan--驗證計畫)

---

## 1. Background & Motivation / 背景與目標

在 [Brain/PPO2/lib/environment.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) 中，目前的實作處於半成品狀態：
1. `step()` 函式僅有骨架，固定回傳 `reward = 0.0, done = False`，完全未實現交易撮合、淨值計算與回報統計。
2. 原先動作空間殘留了混合動作空間設計 `spaces.Tuple(spaces.Discrete(3), spaces.Box(0.0, 1.0, (1,)))`，與純連續動作的 PPO 演算法目標不符。
3. 未平倉損益、開倉價維護、換手成本（手續費與滑價）與下行風險懲罰皆未實作。

### 核心目標
參考已成熟穩定的 [Brain/DQN/lib/environment.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py) 架構，重構並修復 [Brain/PPO2/lib/environment.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)，全面支援 **連續動作空間 $[-1.0, 1.0]$**，使其具備：
- 精確的多空雙向部位管理（做多、做空、平倉、同向調倉、反手翻倉）。
- 嚴謹的交易摩擦成本（手續費與滑價）計算，無需額外人為換手懲罰。
- 支援滾動視窗下行風險懲罰（Rolling Window Downside Risk Penalty），防止大幅回撤。
- 採用標準 `spaces.Dict` 格式提供 `states` 與 `time_states`，與 Transformer/Mamba 時序模型無縫對接。

---

## 2. Comparison: DQN (Discrete) vs PPO2 (Continuous) / DQN 離散與 PPO2 連續環境差異比較

| 維度 / 特性 | DQN 環境 (`Brain/DQN/lib/environment.py`) | PPO2 連續環境 (`Brain/PPO2/lib/environment.py`) |
| :--- | :--- | :--- |
| **動作空間 (Action Space)** | `Discrete(3)`: `0: Hold`, `1: Buy`, `2: Sell` | `Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)` |
| **部位狀態 (Position State)** | `have_position: bool` (僅持有 100% 多頭 或 0% 空倉) | `position: float` $\in [-1.0, 1.0]$ (多頭比例、空頭比例、完全平倉) |
| **放空機制 (Shorting)** | 不支援放空（僅做多） | 完整支援多空雙向 ($a < 0$ 為放空) |
| **調倉粒度 (Rebalancing)** | 全買或全賣 (All-in / All-out) | 連續比例調倉（例如 $0.3 \to 0.8$ 或 $+0.5 \to -0.5$） |
| **違規懲罰 (Wrong Trade)** | 非法轉移懲罰（如持倉時 Buy 或空倉時 Sell 給予 $w_5 \times R_{\text{wrong}}$） | 連續動作無「非法」動作，由真實手續費與滑價自然提供經濟約束 |
| **開倉成本價 (Open Price)** | 單一買入價，平倉時重置為 0 | 支援同向加倉加權平均成本、部分平倉維持成本、反向開倉重置成本 |
| **觀測空間格式 (Obs Space)** | `Tuple(states, time_states)` | `spaces.Dict({"states": ..., "time_states": ...})` |
| **觀測特徵編碼 (Observation)** | `data_res[:, len(info_list)] = 1.0 if have_position else 0.0` | `data_res[:, len(info_list)] = self.position` $\in [-1.0, 1.0]$ |

---

## 3. Continuous Action Definition & Deadzone Analysis / 連續動作定義與平倉死區深度分析

### 3.1 動作取值範圍與對應部位
Agent 輸出的動作 $a_t \in [-1.0, 1.0]$ 代表**目標投資組合部位權重（Target Position Ratio）**：
- $a_t > 0$：做多（Long），部位比例為 $+a_t$（上限 100% 多頭）。
- $a_t < 0$：放空（Short），部位比例為 $-|a_t|$（上限 100% 空頭）。
- $a_t = 0$：完全平倉（Flat / Neutral），部位為 0，持有 100% 現金。

### 3.2 平倉死區（Epsilon Deadzone）深度探討：5% 是否會影響訓練？

在連續動作空間中，PPO 通常使用高斯分佈（Gaussian Policy）輸出動作 $a_t \sim \mathcal{N}(\mu, \sigma)$。神經網路採樣輸出剛好為精確數值 `0.000000` 的機率幾乎為 0。因此需要探討「死區閾值」對強化學習訓練的影響：

#### 🔴 5% 階躍死區（Hard Step 0.05）的潛在隱患：
1. **策略梯度斷層（Gradient Discontinuity / Jump）**：
   若設定「小於 0.05 為 0，大於 0.05 為原值」：
   - 當 $a = 0.049$ 時部位為 $0$；當 $a = 0.051$ 時部位瞬間跳到 $0.051$。
   - 這在 $0.05$ 邊界處造成了階躍突變，對於以平滑預期回報為前提的連續策略網路，容易在閾值邊界產生不穩定的策略更新。
2. **探勘壓制（Exploration Barrier）**：
   在訓練初期，網路輸出常集中在 0 附近做微小探索（例如試探性建立 2%~3% 的試倉部位）。若 5% 以下全部被強制斬斷為 0，Agent 將失去對微小倉位的回饋感知。
3. **5% 金融規模顯著性**：
   在實際量化交易中，5% 已經是具備實質損益的部位大小，不宜輕易將其視為「零」。

#### 🟢 推薦解決方案：
* **方案 A：微死區閾值（Small Deadzone: $\epsilon = 0.01$，即 1%）**：
  - 1% 的資金曝險在損益上幾乎可忽略，但能有效過濾掉浮點數微小噪聲（如 $0.0003$），確保 Agent 在想要平倉時能乾淨利落地將部位歸零、重置持倉時間與開倉價。
  - 進入門檻極低（只需超過 0.01），不阻礙網路的小額建倉探索。
* **方案 B：連續平滑過渡（Linear Continuous Deadband，無階躍斷層）**：
  若希望數學上完全連續無跳躍，可採用線性收縮：
  $$p_t = \begin{cases} 
  0.0, & |a_t| \le \epsilon \\
  \text{sign}(a_t) \cdot \frac{|a_t| - \epsilon}{1 - \epsilon}, & |a_t| > \epsilon
  \end{cases}$$
  當 $a_t = \epsilon$ 時，$p_t = 0$；隨著 $|a_t|$ 增加，$p_t$ 平滑從 $0$ 增長到 $1$。完全消除了邊界處的斷層。

> **結論建議**：預設採用 **$\epsilon = 0.01$（1% 微死區）**，並開放作為 `config` 參數可配置，既維護了訓練平滑度，又確保了能精準平倉。

---

## 4. Portfolio Accounting & State Dynamics / 部位轉移與記帳數學邏輯

設當前時間步為 $t$：
- 剛結束時段的資產收盤價：$S_t = \text{prices.close}[offset]$
- 上一期收盤價：$S_{t-1} = \text{prices.close}[offset - 1]$
- 上一期持有部位：$p_{t-1} \in [-1.0, 1.0]$
- 依據動作 $a_t$ 決定之新目標部位：$p_t \in [-1.0, 1.0]$

### 4.1 換倉量（Turnover）與交易摩擦成本（Transaction Cost & Slippage）
換倉變化量：
$$\Delta p_t = p_t - p_{t-1}$$

交易手續費與滑價總摩擦成本：
$$\text{Cost}_t = |\Delta p_t| \times (\text{commission\_perc} + \text{default\_slippage})$$
此項成本直接自淨值扣除，並累計至 `cost_sum += Cost_t`。
> 備註：依據討論，**不另設人為 Turnover Penalty**，手續費與滑價即為自然懲罰項。

### 4.2 部位切換時開倉成本價（Average Open Price）與平倉損益（Realized PnL）
為了準確計算未平倉浮動損益與勝率統計，維護 `open_price`：

1. **同向持倉加倉 ($p_{t-1} \cdot p_t \ge 0$ 且 $|p_t| > |p_{t-1}|$)**：
   - 增加持倉比例：$\delta = |p_t| - |p_{t-1}|$
   - 開倉均價更新（加權平均）：
     $$\text{open\_price}_t = \frac{|p_{t-1}| \times \text{open\_price}_{t-1} + \delta \times S_t}{|p_t|}$$
   - 無已平倉損益產生。

2. **同向持倉減倉 / 部分平倉 ($p_{t-1} \cdot p_t \ge 0$ 且 $|p_t| < |p_{t-1}|$)**：
   - 平倉比例：$\delta = |p_{t-1}| - |p_t|$
   - 開倉均價保持不變：$\text{open\_price}_t = \text{open\_price}_{t-1}$
   - 結算該減倉部分之已平倉損益：
     $$\Delta \text{CloseCash}_t = \begin{cases} 
     \delta \times \frac{S_t - \text{open\_price}}{\text{open\_price}}, & p_{t-1} > 0 \text{ (多頭減倉)} \\
     \delta \times \frac{\text{open\_price} - S_t}{\text{open\_price}}, & p_{t-1} < 0 \text{ (空頭減倉)}
     \end{cases}$$
   - 累計 `closecash += \Delta CloseCash_t`。

3. **反手翻倉（Flip Position，例如由多轉空 $p_{t-1} > 0 \to p_t < 0$ 或由空轉多）**：
   - 先完全平掉上一期部位 $p_{t-1}$，結算全部已平倉損益（平倉比例為 $|p_{t-1}|$）。
   - 隨後以當前價格 $S_t$ 建立新方向部位 $p_t$，重置開倉價：$\text{open\_price}_t = S_t$。

4. **完全平倉 ($p_t = 0$)**：
   - 結算剩餘全部持倉之已平倉損益。
   - 重置開倉價：$\text{open\_price}_t = 0.0$。

### 4.3 浮動未平倉損益（Unrealized PnL）
當步新部位 $p_t$ 的未平倉損益：
$$\text{OpenCash}_t = \begin{cases} 
p_t \times \frac{S_t - \text{open\_price}_t}{\text{open\_price}_t}, & p_t > 0 \\
|p_t| \times \frac{\text{open\_price}_t - S_t}{\text{open\_price}_t}, & p_t < 0 \\
0.0, & p_t = 0
\end{cases}$$

### 4.4 投資組合總淨值（Total Portfolio Equity）與單步報酬
為避免同向加倉時因開倉均價分母跳變導致累積浮盈稀釋（幽靈虧損），淨值演進與單步報酬採用標準盯市（Mark-to-Market）計算：
$$R_{p, t} = p_{t-1} \times \frac{S_t - S_{t-1}}{S_{t-1}} - \text{Cost}_t$$
$$\text{TotalPortfolioPercent}_t = \text{TotalPortfolioPercent}_{t-1} + R_{p, t}$$
同時持續維護 `open_price`、`closecash`、`trade_bar` 提供特徵編碼與統計指標使用。

---

## 5. Observation Space & Feature Encoding / 觀測空間 (spaces.Dict) 與特徵編碼對齊

### 5.1 Observation Space 定義
依據用戶確認，環境標準輸出採用 `gymnasium.spaces.Dict`：
```python
self.observation_space = spaces.Dict({
    "states": spaces.Box(
        low=-np.inf, high=np.inf, shape=self._state.getStateShape(), dtype=np.float32
    ),
    "time_states": spaces.Box(
        low=-np.inf, high=np.inf, shape=self._state.getTimeShape(), dtype=np.float32
    )
})
```
`reset()` 與 `step()` 回傳之 `obs` 字典鍵值對為：
`{"states": data_res, "time_states": time_res}`。

### 5.2 特徵欄位編碼 (`encode()`)
在 `Brain/PPO2/lib/environment.py` 的 `State_time_step` 中覆寫 `encode()`：
1. **部位狀態特徵 (Feature `len(info_list)`)**：
   填入當前連續部位比例 $p \in [-1.0, 1.0]$。正值代表多頭、負值代表空頭、0 代表平倉。
2. **未平倉損益特徵 (Feature `len(info_list) + 1`)**：
   若 $p \neq 0$：
   $$\text{unrealized\_return} = \text{sign}(p) \times \frac{S_{t} - \text{open\_price}}{\text{open\_price}} \times |p|$$
   無部位時填入 `0.0`。
   > [!IMPORTANT]
   > 嚴格遵循 [spec/issue_adjustEnvironment.md](file:///home/b0457812963/Mamba3RL/SynapseX/spec/issue_adjustEnvironment.md) 規範，使用剛結束期的價格 `self._prices.close[self._offset - 1]`，徹底杜絕 Look-Ahead Bias 未來價格洩漏！
3. **持倉持續時間特徵 (Feature `len(info_list) + 2`)**：
   若 $p \neq 0$ 填入 `self.trade_bar`，若平倉則為 `0`。

---

## 6. Reward Function Formulation / 獎勵函數設計

在連續動作環境中，獎勵函數設計如下：

### 6.1 核心公式
$$R_t = w_1 \cdot R_{p, t} - \text{DownsidePenalty}_t$$

其中：
1. **單步淨資產報酬項 ($w_1 \cdot R_{p, t}$)**：
   直接反應用戶投資組合淨值的單步百分比漲跌（扣除摩擦成本後）。$w_1$ 預設為 `1.0`。
2. **滑動視窗下行風險懲罰項 ($\text{DownsidePenalty}_t$)**：
   承襲 [Brain/DQN/lib/environment.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py) 與 [spec/issue_downside_risk.md](file:///home/b0457812963/Mamba3RL/SynapseX/spec/issue_downside_risk.md) 之設計：
   - 視窗大小：$K = 60$ 根 K 棒。
   - 滾動下行風險：$\sigma_{\text{down}, t} = \sqrt{\frac{1}{|W_t|} \sum_{\tau \in W_t} \max(0, -R_{p, \tau})^2}$。
   - 勢能增量懲罰：$\text{DownsidePenalty}_t = w_2 \cdot \max(0, \sigma_{\text{down}, t} - \sigma_{\text{down}, t-1})$。

---

## 7. Environment Architecture & Class Design / 環境架構與類別設計

```
Brain/PPO2/lib/
├── environment.py          <-- 本次核心修復檔案
│   ├── class State_time_step(State_time_step_template)
│   ├── class BaseTradingEnv(gym.Env, ABC)
│   ├── class TrainingEnv(BaseTradingEnv)
│   └── class ProductionEnv(BaseTradingEnv)
```

### 7.1 `State_time_step`
- 狀態變數：
  - `self.position: float` (當前部位比例，$-1.0 \le \text{position} \le 1.0$)
  - `self.open_price: float` (開倉均價)
  - `self.trade_bar: int` (當前部位持續 K 棒數)
  - `self.TotalPortfolioPercent: float` (投資組合總淨值)
  - `self.cost_sum: float`, `self.closecash: float`
  - `self.return_history: deque(maxlen=N_steps)`
  - `self.prev_downside_risk: float`
  - `self.deadzone_threshold: float` (預設 0.01)
- 核心方法：
  - `step(action: float) -> Tuple[float, bool]`
  - `calculate_step_downside_penalty() -> float`
  - `encode() -> Dict[str, np.ndarray]`

### 7.2 `BaseTradingEnv`
- `action_space`: `gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)`
- `observation_space`: `spaces.Dict({"states": ..., "time_states": ...})`
- `step(action)`: 接收連續動作值，執行狀態演進並回傳 `(obs, reward, done, info)`。

### 7.3 `TrainingEnv`
- 繼承 `BaseTradingEnv`。
- `reset()` 修復為與 [spec/issue_adjustEnvironment.md](file:///home/b0457812963/Mamba3RL/SynapseX/spec/issue_adjustEnvironment.md) 一致的邊界約束：
  ```python
  max_offset = prices.high.shape[0] - self._state.N_steps - 1
  min_offset = self._state.bars_count
  assert max_offset > min_offset
  offset = np.random.randint(min_offset, max_offset)
  ```

---

## 8. Proposed Code Modifications / 預計代碼改動細節

### 目標檔案：[Brain/PPO2/lib/environment.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)
1. 移除無效殘留的 `class Actions(enum.Enum)` 與無用 `Reward` 舊類別。
2. 重寫 `State_time_step.step(action)`，支援連續目標部位、多空轉移、平均開倉價維護與滑價手續費結算。
3. 覆寫 `State_time_step.encode()`，使回傳字典 `{"states": data_res, "time_states": time_res}`，且 `data_res` 正確反映連續部位比例 $p$ 與多空浮動盈虧。
4. 修正 `TrainingEnv.reset()` 隨機起點公式，修復越界隱患。
5. 修改 `BaseTradingEnv.action_space` 為 `gym.spaces.Box(-1.0, 1.0, shape=(1,))`，`observation_space` 為 `spaces.Dict`。

---

## 9. Discussion Conclusions / 討論結論彙整

經過與用戶深入討論，確認以下實作標準：
1. **平倉死區建議**：
   - 避免過大的 5% 斷層對訓練造成探勘障礙。
   - 採用 **$\epsilon = 0.01$（1% 微死區）** 或線性連續過渡，保留空倉平倉功能同時兼顧梯度的平滑性。
2. **換手懲罰**：
   - **不額外添加人為換手懲罰**，讓手續費與滑價自然約束 Agent 的換倉行為。
3. **觀測空間規格**：
   - 確認採用 **`spaces.Dict`** 格式（包含 `"states"` 與 `"time_states"`），回傳符合該結構的 observation dictionary。

---

## 10. Implementation Checklist / 實作檢核清單

- [x] **Phase 1: 舊程式碼清理與空間規格重構 (Cleanup & Spaces)**
  - [x] 移除無用的離散動作列舉 `class Actions(enum.Enum)`
  - [x] 移除舊有非連續架構的 `class Reward`
  - [x] 設定 `action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)`
  - [x] 設定 `observation_space = spaces.Dict({"states": Box(...), "time_states": Box(...)})`

- [x] **Phase 2: 狀態變數與部位記帳核心 (State Management & Accounting)**
  - [x] 在 `State_time_step.__init__` 初始化部位變數 (`position`, `open_price`, `cost_sum`, `closecash`, `trade_bar`, `TotalPortfolioPercent`, `return_history`, `prev_downside_risk`, `deadzone_threshold=0.01`)
  - [x] 在 `State_time_step.reset()` 完整重置部位狀態、資金淨值與歷史佇列
  - [x] 實作動作死區處理：$|a_t| \le 0.01 \implies p_t = 0.0$，否則 $p_t = a_t$
  - [x] 實作換手量與交易摩擦成本：$\Delta p_t = p_t - p_{t-1}$，$\text{cost} = |\Delta p_t| \times (\text{commission} + \text{slippage})$
  - [x] 實作開倉均價維護：
    - [x] 同向加倉：依權重更新平均成本價
    - [x] 同向減倉：開倉均價維持不變，結算減倉比例的已實現損益並累加至 `closecash`
    - [x] 反手翻倉：全平舊倉結算損益，並以當前價格重置新方向開倉價
    - [x] 完全平倉：結算剩餘全部損益，均價歸零
  - [x] 實作多空雙向未平倉損益 (`OpenCash`)
  - [x] 實作淨資產計算 $\text{TotalPortfolioPercent} = 1.0 - \text{cost\_sum} + \text{closecash} + \text{OpenCash}$ 及單步回報 $R_{p, t}$
  - [x] 實作持倉時間 `trade_bar` 計數更新

- [x] **Phase 3: 滾動下行風險懲罰計算 (Downside Risk Penalty)**
  - [x] 實作 `calculate_downside_risk_numpy(returns)`
  - [x] 實作 `calculate_step_downside_penalty()` (滑動視窗 60 根 K 棒勢能增量)
  - [x] 組合單步總獎勵：$\text{reward} = w_1 \cdot R_{p, t} - \text{downside\_penalty}$

- [x] **Phase 4: 觀測特徵編碼覆寫 (Observation Encoding)**
  - [x] 覆寫 `State_time_step.encode()`，回傳 `{"states": data_res, "time_states": time_res}`
  - [x] 特徵矩陣第 `len(info_list)` 填入連續部位 $p_t \in [-1.0, 1.0]$
  - [x] 特徵矩陣第 `len(info_list) + 1` 填入多空浮動損益（使用 $t-1$ 收盤價，杜絕 Look-Ahead Bias）
  - [x] 特徵矩陣第 `len(info_list) + 2` 填入持倉時間 `trade_bar`

- [x] **Phase 5: 環境介面與防護機制 (Gym Environment Wrappers)**
  - [x] 實作 `BaseTradingEnv.step(action)`：支援 scalar/tensor/ndarray 輸入並裁剪到 $[-1.0, 1.0]$
  - [x] 實作 `BaseTradingEnv.engine_info()` 提供特徵與動作維度
  - [x] 修復 `TrainingEnv.reset()` 隨機 Offset 計算：`max_offset = len - N_steps - 1`
  - [x] 實作 `ProductionEnv` 支援固定數據推論

- [x] **Phase 6: 驗證測試 (Verification)**
  - [x] 建立測試腳本 `test_ppo2_env.py`
  - [x] 執行做多、做空、平倉死區、翻倉損益、特徵無洩漏等單元測試
  - [x] 確保測試全部 PASS

---

## 11. Verification Plan / 驗證計畫

### 11.1 單元測試 (Unit Tests)
撰寫獨立測試腳本 `test_ppo2_env.py`，驗證下列場景：
1. **多頭與空頭開倉**：輸入 $a = 1.0$ 與 $a = -0.5$，檢查 `position`、`open_price` 與資產回報的正負號邏輯。
2. **平倉與死區測試**：輸入 $a = 0.0$ 與 $a = 0.005$，確認部位成功歸零且開倉價重置。
3. **反手翻倉 (Flip)**：自 $a = +1.0$ 瞬間轉為 $a = -1.0$，確認平倉損益有正確結算、換倉量計算為 $2.0$、手續費正確扣除、新開倉價正確設定。
4. **特徵矩陣無洩漏**：驗證 `encode()` 輸出的 Dict 結構與特徵欄位數值正確，且收盤價不存取未來 $t+1$ 價格。

### 11.2 與 PPO2 訓練迴圈對接測試
執行一個小規模 Rollout 測試，確保 `train_env.step(action)` 輸出的 shape 與 dtype 能順暢送入 `RolloutBuffer`。
