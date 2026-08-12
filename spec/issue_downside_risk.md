# Specification: Downside Risk Reward Redesign via Rolling Window Potential Shaping (方案 B)

## 📋 Index / 目錄
1. [Background & Motivation / 背景與動機](#1-background--motivation--背景與動機)
2. [Problem Analysis of Original Implementation / 原實作問題分析](#2-problem-analysis-of-original-implementation--原實作問題分析)
3. [Mathematical Formulation (方案 B) / 數學公式設計](#3-mathematical-formulation-方案-b--數學公式設計)
4. [Architecture & Modular Function Design / 模組化函式封裝設計](#4-architecture--modular-function-design--模組化函式封裝設計)
5. [Proposed Code Modifications / 預計代碼改動細節](#5-proposed-code-modifications--預計代碼改動細節)
6. [Hyperparameter Configuration & Scale Balancing / 權重與尺度平衡](#6-hyperparameter-configuration--scale-balancing--權重與尺度平衡)
7. [Verification Plan / 驗證計畫](#7-verification-plan--驗證計畫)

---

## 1. Background & Motivation / 背景與動機

在金融強化學習交易環境 [environment.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py) 中，為了落實論文 *《Risk-Aware Reinforcement Learning Reward for Financial Trading》*（Srivastava et al., arXiv:2506.04358, 參見 [Risk-Aware.pdf](file:///home/b0457812963/Mamba3RL/SynapseX/Risk-Aware.pdf)）的精神，需要在報酬函數中引入**下行風險（Downside Risk / Semi-Deviation）**懲罰，促使 Agent 在追求高報酬的同時具備資本保護（Capital Preservation）能力。

經過評估，決定採用 **方案 B：滑動視窗滾動下行風險（Rolling Window Potential Shaping）**，並透過**獨立函式封裝（Encapsulated Helper Function）**讓主流程 `step()` 保持最精簡、高可讀性的架構。

---

## 2. Problem Analysis of Original Implementation / 原實作問題分析

原先 [environment.py#L248-L254](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py#L248-L254) 採用回合結束時一次性結算下行風險：
```python
if done:
    final_downside_risk = self.calculate_downside_risk_numpy(list(self.return_history))
    final_penalty = np.clip(final_downside_risk, 0, 0.5)
    reward -= self.weights['w2_downside_ratio'] * final_penalty
```

### 存在之核心缺陷：
1. **信用分配失效（Credit Assignment Failure）與替罪羊效應**：
   在 DQN 的 Replay Buffer 中，該筆巨大的負懲罰只會加在第 1000 步的 Transition 上，第 1000 步的動作（如平倉或單純 Hold）會承受整局所有的下行懲罰，而前期真正造成大虧損的決策無法有效接收到梯度。
2. **獎勵尺度爆炸（Scale Mismatch）**：
   單步報酬 $r_t \approx \pm 0.001$，而最後一步懲罰 $30 \times 0.5 = 15.0$，造成 TD-Error 瞬間暴增數千倍，破壞 Q 網路訓練穩定性。
3. **時間稀釋漏洞（Time-Dilution Exploit）**：
   若 Agent 造成單次大虧損後，後續 990 步選擇全程空倉（return = 0），分母 1000 步會將該次下行風險開根號稀釋 $\sqrt{10/1000} \approx 31.6$ 倍。

---

## 3. Mathematical Formulation (方案 B) / 數學公式設計

### 3.1 滾動下行風險定義（Rolling Downside Deviation）
依據論文 Eq. (3) 與 Eq. (10)，在時間步 $t$，取近 $K$ 步（預設 $K = 60$ 根 K 棒）歷史收益集合 $W_t$：
$$\sigma_{\text{down}, t} = \sqrt{\frac{1}{|W_t|} \sum_{\tau \in W_t} \max(0, -R_{p, \tau})^2}$$

### 3.2 勢能塑形獎勵增量（Potential-based Reward Shaping Delta）
定義下行風險勢能函數 $\Phi(s_t) = - \sigma_{\text{down}, t}$。
單步下行風險回饋為其勢能變化量（差分）：
$$r_{\text{down}, t} = - \max(0, \sigma_{\text{down}, t} - \sigma_{\text{down}, t-1})$$

* **單步複合總獎勵**：
  $$R_t = w_1 \cdot R_{p, t} - w_2 \cdot \max(0, \sigma_{\text{down}, t} - \sigma_{\text{down}, t-1}) + w_5 \cdot R_{\text{wrong}}$$

---

## 4. Architecture & Modular Function Design / 模組化函式封裝設計

為保持 `step()` 主流程的高度精簡與單一職責原則（SRP），將滾動下行風險計算與狀態更新封裝為獨立方法：
* **`calculate_step_downside_penalty(self) -> float`**：
  負責切片 `self.return_history`、呼叫 `calculate_downside_risk_numpy`、計算差分增量、更新 `self.prev_downside_risk` 並乘上權重回傳最終懲罰值。
* **`step()` 主流程**：
  僅需單行呼叫 `downside_penalty = self.calculate_step_downside_penalty()`。

---

## 5. Proposed Code Modifications / 預計代碼改動細節

### 修改 [Brain/DQN/lib/environment.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py)

#### 1. `State_time_step.__init__`
```python
self.downside_window_size = 60  # 滾動下行風險視窗大小 (60 根 30m K 棒)
self.prev_downside_risk = 0.0

self.weights = {
    "w1_step_return": 1.0,      # 單步報酬權重
    "w2_downside_ratio": 1.0,    # 下行風險增量懲罰權重 (與 w1 同數量級)
    "w3_diff_return": 0.2,       # 差異報酬權重
    "w4_treynor": 0.15,          # 崔諾指標權重
    "w5_wrong_trade": 1.0,       # 違規交易懲罰權重
}
```

#### 2. `State_time_step.reset`
```python
self.prev_downside_risk = 0.0
```

#### 3. 新增封裝函式 `State_time_step.calculate_step_downside_penalty`
```python
def calculate_step_downside_penalty(self) -> float:
    """
    計算單步滑動視窗下行風險增量懲罰 (Rolling Window Downside Risk Penalty)
    
    回傳:
        float: 當步應扣除的下行風險懲罰值
    """
    if len(self.return_history) < 2:
        return 0.0

    rolling_slice = list(self.return_history)[-self.downside_window_size:]
    current_downside_risk = self.calculate_downside_risk_numpy(rolling_slice)
    downside_delta = max(0.0, current_downside_risk - self.prev_downside_risk)
    self.prev_downside_risk = current_downside_risk

    return float(self.weights["w2_downside_ratio"] * downside_delta)
```

#### 4. 精簡後的 `State_time_step.step`
```python
# 10. 計算當前的總淨值 (Equity) 與單步報酬
self.TotalPortfolioPercent = (
    1.0 - self.cost_sum + self.closecash + opencash_diff
)
current_p_return = self.TotalPortfolioPercent - previous_PortfolioPercent
self.return_history.append(current_p_return)

# 計算單步下行風險懲罰
downside_penalty = self.calculate_step_downside_penalty()

# 計算單步總獎勵
reward = (
    self.weights["w1_step_return"] * current_p_return
    - downside_penalty
    + self.weights["w5_wrong_trade"] * wrongTrade_reward
)

# --- 11. 更新步數與結束判斷 ---
self._offset += 1
self.game_steps += 1
done |= self._offset >= self._prices.close.shape[0] - 1

if self.game_steps == self.N_steps and self.model_train:
    done = True

# 移除原先在 if done: 下方的一次回溯扣除邏輯
```

---

## 6. Hyperparameter Configuration & Scale Balancing / 權重與尺度平衡

| 參數名 | 建議初始值 | 數值範圍 | 說明 |
| :--- | :---: | :---: | :--- |
| `downside_window_size` | `60` | `30 ~ 120` | 滾動計算下行風險的 K 棒數量（60 根 30m K 棒 = 30 小時）。 |
| `w1_step_return` | `1.0` | `0.5 ~ 2.0` | 單步真實收益率權重。 |
| `w2_downside_ratio` | `1.0` | `0.5 ~ 3.0` | 下行風險增量懲罰權重，保持與單步收益同一數量級。 |
| `w5_wrong_trade` | `1.0` | `0.5 ~ 1.0` | 違規交易懲罰權重。 |

---

## 7. Verification Plan / 驗證計畫

### 7.1 單元測試與即時數值驗證（Unit Sanity Checks）
1. **零交易測試（All-Cash Test）**：
   連續 100 步採取 `Hold`（空倉），確認 `current_p_return = 0.0`，`calculate_step_downside_penalty() = 0.0`，`reward = 0.0`，無異常懲罰產生。
2. **單步虧損測試（Loss Step Test）**：
   在持倉下跌的情況下，確認 `calculate_step_downside_penalty() > 0` 且 `reward` 正確扣除該懲罰。
3. **獎勵尺度檢查（Scale Check）**：
   在 1000 步模擬中，記錄單步 `reward` 的最大值、最小值與標準差，確保全域獎勵落在 $[-0.05, 0.05]$ 區間內，無劇烈突波。

### 7.2 訓練收斂監控（Training Convergence）
1. 觀察 DQN Loss 曲線，確認不再出現最後一步引發的梯度跳躍（Gradient Spike）。
2. 比對有/無下行風險懲罰下的策略最大回撤（Max Drawdown）與夏普/索提諾比率。
