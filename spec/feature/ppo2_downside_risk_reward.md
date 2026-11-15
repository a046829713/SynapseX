# 規格文件 (Specification): PPO2 下行風險懲罰機制重構與回撤控制 (Downside Risk Penalty & Drawdown Control)

- **類別 (Category)**: `feature`
- **規格路徑 (Spec Path)**: `spec/feature/ppo2_downside_risk_reward.md`
- **狀態 (Status)**: `Implemented & Verified`

## 📋 目錄 (Index / Contents)
1. [1. 背景與根本原因分析 (Background & Root Cause Analysis)](#1-背景與根本原因分析-background--root-cause-analysis)
2. [2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)](#2-改動範圍與架構邊界-target-scope--architecture-boundaries)
3. [3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)](#3-實作查核清單與詳細藍圖-implementation-checklist--detailed-blueprint)
4. [4. 驗證與測試計畫 (Validation & Verification Plan)](#4-驗證與測試計畫-validation--verification-plan)

---

## 1. 背景與根本原因分析 (Background & Root Cause Analysis)

### 1.1 問題陳述與 `checkpoint_iter_8200` 實證診斷 (Problem Statement & Empirical Diagnosis)
根據目前最新的多模型回測排行榜 ([`results/ppo2_orders/leaderboard.csv`](file:///home/b0457812963/Mamba3RL/SynapseX/results/ppo2_orders/leaderboard.csv)) 與 197 個加密貨幣標的明細 ([`results/ppo2_orders/multi_model_test_manifest.csv`](file:///home/b0457812963/Mamba3RL/SynapseX/results/ppo2_orders/multi_model_test_manifest.csv))，[`checkpoint_iter_8200.pt`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/Meta/checkpoint_iter_8200.pt) 在整體指標上排名第 1（`gate_pass = True`、`iqm_ir_2x = 0.2926`、`iqm_ir_1x = 0.6221`、`beat_bh_ratio = 67.01%`、`positive_return_ratio = 60.41%`），展現出良好的基礎擇時能力。然而，實盤與回測檢視暴露出兩個核心痛點：**「整體賺得很少」**且**「最大回撤 (MDD) 滿大的」**：

1. **整體獲利偏低 (`iqm_cum_return_pct = +4.08%`, `median = +3.77%`)**：
   - 在大牛市標的上（如 `BRUSDT` 大盤 B&H 漲幅 `+1739.12%`，策略僅獲利 `+108.32%`；`BTWUSDT` 大盤漲幅 `+2137.45%`，策略僅 `+83.00%`），模型在帳面浮盈達到 `+5% ~ +10%` 時便急於平倉獲利了結，無法抱住主升段趨勢。
2. **最大回撤過大 (`median_mdd_pct = 26.68%`, `mean_mdd_pct = 32.46%`, `p90_mdd_pct = 64.46%`)**：
   - 將 [`checkpoint_iter_8200`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/Meta/checkpoint_iter_8200.pt) 測試的 197 個標的依據平均持倉 (`mean_action`) 進行分組統計，呈現極端的兩極化行為：

| `mean_action` 持倉區間 | 標的數 (`count`) | 策略平均報酬 (`mean`) | 策略中位數報酬 (`median`) | 大盤 B&H 中位數 (`median`) | 平均 MDD (`mean`) | 中位數 MDD (`median`) | 日均換手率 (`daily_turnover`) | 中位數 Sharpe |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`(0.0, 0.2]` (靈活擇時組)** | 63 | **`+28.37%`** | **`+11.16%`** | `-2.35%` | **`12.53%`** | **`9.70%`** | `0.33` | **`1.21`** |
| **`(0.2, 0.5]` (波段操作組)** | 54 | **`+13.69%`** | **`+11.78%`** | `-11.15%` | `32.15%` | `27.42%` | `0.25` | **`0.49`** |
| **`(0.5, 0.8]` (偏高持倉組)** | 32 | `-1.94%` | `+0.14%` | `-10.33%` | `38.42%` | `38.28%` | `0.15` | `0.02` |
| **`(0.8, 1.0]` (深套死扛組)** | 48 | **`-22.62%`** | **`-24.52%`** | `-29.82%` | **`54.99%`** | **`57.71%`** | **`0.04`** | **`-0.58`** |

### 1.2 歷史版本回顧與三大技術根本原因 (Historical Commits Review & Technical Root Causes)

#### 根因一：純對稱報酬缺乏下行風險不對稱約束，誘發嚴重「處置效應 (Disposition Effect：賺一點就跑、賠錢死扛)」
對 [`checkpoint_iter_8200.pt`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/Meta/checkpoint_iter_8200.pt) 進行神經網路特徵敏感度剖析（固定市場特徵，掃描部位狀態 `position` 與未實現對數損益 `unrealized_return`）：
- 當 `position = 1.0` 時：
  - 若 `unrealized_return <= 0`（如 `-0.01, -0.05, -0.20, -0.50` 處於套牢狀態），Actor 輸出恆為 **`0.999`**（完全鎖死滿倉，拒絕停損）。
  - 若 `unrealized_return >= +0.10`（處於獲利狀態），Actor 輸出驟降至 **`0.025`**（立即砍倉止盈）。
- **數學成因**：自 Commit `ded85f0` ([`spec/bugfix/ppo2_unbiased_return_normalization.md`](file:///home/b0457812963/Mamba3RL/SynapseX/spec/bugfix/ppo2_unbiased_return_normalization.md)) 移除下行風險懲罰後，[`Brain/PPO2/lib/environment.py:L151`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L151) 的獎勵函數為純對稱線性對數報酬：
  $$R_t = w_1 \cdot r_{p,t}$$
  在對稱報酬下，停損平倉（$\Delta p = -1.0$）會立即遭受**確定性的交易摩擦懲罰**（$-(\text{comm} + \text{slip}) \times 100 = -0.10$），而繼續死扛套牢部位在環境中**不會受到任何額外的下行波動或回撤擴張懲罰**。加上加密貨幣歷史訓練集中常見超跌反彈，Critic 學到「套牢時死扛等待解套可省下摩擦成本，小賺時落袋為安」的處置效應策略；一旦在測試集遇到單邊熊市標的（如 `RIVERUSDT` 跌幅 `-86.2%`、`SANTOSUSDT` 跌幅 `-71.0%`、`BTCUSDT` 階段回撤 `-35.1%`），策略一旦跌破 `open_price` 便鎖死 `position ≈ 1.0` 連續超過 11,000 根 K 棒零交易，造成巨大 MDD。

#### 根因二：過去兩代下行風險獎勵機制為何訓練不佳？(Historical Commits Comparison)
檢視 `git log` 與 [`Brain/DQN/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py)，本專案曾先後採用過兩代下行風險機制：

1. **第一代：滑動視窗下行風險增量 `calculate_step_downside_penalty`**（現存於 [`Brain/DQN/lib/environment.py:L70-L110`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py#L70-L110)，以及 PPO2 Commit `d067d7e` 以前之 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)）：
   ```python
   rolling_slice = list(self.return_history)[-self.downside_window_size:]
   current_downside_risk = np.sqrt(np.mean(np.maximum(0.0, -np.asarray(rolling_slice)) ** 2))
   downside_delta = max(0.0, current_downside_risk - self.prev_downside_risk)
   self.prev_downside_risk = current_downside_risk
   ```
   此設計的**量綱與單步報酬 $r_{p,t}$ 完全一致（皆為無因次對數報酬率）**，不會像 Z-score 平方項那樣隨時間步尺度的平方根發散；但其原始實作存在**三個致命的數學缺陷**導致早期訓練崩潰：
   - **缺陷 1（冷啟動分母暴衝 Cold-Start Spike）**：`np.mean()` 以當前切片長度 $N = \text{len}(\text{rolling\_slice})$ 作為分母。在 Episode 第 2 步（$N=2$）發生單步虧損 $r_2 < 0$ 時，$\Delta \text{DR}_2 = |r_2| / \sqrt{2} \approx 0.707 |r_2|$；而在第 60 步（$N=60$）發生同樣虧損時，$\Delta \text{DR}_{60} \le |r_{60}| / \sqrt{60} \approx 0.129 |r_{60}|$。**Episode 初期建倉的懲罰被異常放大了 $5.5$ 倍**！
   - **缺陷 2（滑動視窗遺忘棘輪陷阱 Sliding-Window Ratchet Trap）**：視窗是**每一步（Every Step）逐根滾動**的。在 512 步的 Episode 中，自第 61 步起的**全部 452 個時間步**，每一步視窗都會踢出 60 步前的舊報酬 $r_{t-60}$ 並推入當步報酬 $r_t$（視窗內負報酬平方和的變化為 $\Delta S_t = \max(0, -r_t)^2 - \max(0, -r_{t-60})^2$，整場 Episode 相當於將 60 根 K 棒的視窗容量完整換血了 $512 / 60 \approx 8.53$ 輪）。只要當步獲利或持平（$r_t \ge 0$）且剛好踢出 60 步前的一根虧損 K 棒（$r_{t-60} < 0$），`current_downside_risk` 就會下降；此時 `max(0.0, ...)` 不給予任何正向回補，卻在下一行將 `self.prev_downside_risk` 覆寫調低！導致後續任何新虧損進來時，又必須從被調低的基準線**重複計罰**。整場 512 步下來會發生上百次「舊虧損滑出降基準、新虧損進來再扣分」的棘輪效應，使累積懲罰高達真實下行標準差的 8～15 倍，嚴重侵蝕正常波段策略的正報酬。
   - **缺陷 3（平方根凹函數在深跌時的邊際鈍化 $\sqrt{\cdot}$ Saturation）**：因為 $\frac{\partial \sqrt{S}}{\partial S} = \frac{1}{2\sqrt{S}}$，當部位已經深套、視窗內累積虧損平方和 $S$ 很大時，繼續下跌的邊際懲罰 $\Delta \text{DR}_t$ 反而急遽縮小（第 20 根連跌 K 棒的懲罰僅為第 1 根的 $1/9$；連跌滿 60 根後，第 61 根同幅下跌的 $\Delta \text{DR}_{61} = 0$）。這導致**「剛進場小幅震盪罰最重，真正深套暴跌時卻失去懲罰力道」**。

2. **第二代：事前波動率標準化二次方懲罰 `DownsidePenaltyReward`**（Commit `266ac49` 引入，Commit `ded85f0` 移除）：
   - 公式為 $R_t = z_t - \frac{1}{2}\lambda \min(0, z_t)^2$，其中 $z_t = r_{p,t} / \sigma_t$。由於 $\sigma_t \propto \sqrt{\Delta t}$，二次項 $\mathbb{E}[\min(0, z_t)^2] \approx 0.5 p^2$ 為 $O(1)$ 常數級別的每步扣分，在 512 步內累積扣除約 $-64.0$，導致 74.7% 獲利 Episode 獲得負總獎勵而退化為空手 (`position -> 0`)。

#### 根因三：動作死區 (`DEADZONE_THRESHOLD = 0.01`) 與 Sigmoid 尾端失配，導致平倉後 `open_price` 殘留污染
- 在 [`Brain/PPO2/lib/environment.py:L92-L125`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L92-L125) 中，僅當 `clipped_action <= self.deadzone_threshold` (`0.01`) 時，`target_pos` 才會歸零並將 `self.open_price` 重置為 `0.0`。
- 實測顯示 [`checkpoint_iter_8200.pt`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/Meta/checkpoint_iter_8200.pt) 與 `checkpoint_iter_9800.pt` 在發出平倉訊號時，Sigmoid 輸出落在 `0.014 ~ 0.025`（對應 $\mu \approx -4.2 \sim -3.7$，因 Sigmoid 導數 $a(1-a) < 0.02$ 產生梯度飽和）。
- 由於 `0.017 > 0.01`，環境判定未完全平倉（`flat_ratio = 0.0000`），**`self.open_price` 永遠不會重置為 `0.0`**！當標的價格隨後跌破數千根 K 棒前留下的舊 `open_price` 時，特徵 `unrealized_return` 轉為負值，直接觸發「根因一」的套牢死扛鎖死機制（`action -> 0.999`）！

### 1.3 當前行為 vs. 預期行為 (Current vs. Expected Behavior)
| 維度 | 當前行為 (Current Behavior) | 預期行為 (Expected Behavior) |
| :--- | :--- | :--- |
| **單步獎勵函數 (`step`)** | 僅有純對稱線性報酬 `w1_step_return * current_p_return`，對下行波動與深套回撤無任何不對稱懲罰。 | 結合無偏單步報酬與改良版 `calculate_step_downside_penalty()`（固定視窗分母無棘輪之邊際下行風險 + 浮虧回撤增量懲罰），與 `w1_step_return` 同量綱縮放。 |
| **下行風險計算 (`calculate_step_downside_penalty`)** | 已於早期 Commit 移除；而舊版 DQN 實作存在 $N=2$ 冷啟動暴衝、60 步遺忘棘輪重複計罰、以及深跌時 $\sqrt{\cdot}$ 邊際鈍化三大問題。 | 採用固定視窗分母 $W$ 計算當步真實邊際增量（消除冷啟動暴衝與視窗滑出棘輪），並結合線性下行回撤項防止深套時懲罰鈍化。 |
| **平倉死區與成本重置 (`deadzone_threshold`)** | `DEADZONE_THRESHOLD: 0.01` 小於 Sigmoid 實務平倉尾端 (`0.015 ~ 0.025`)，導致 `flat_ratio = 0.0` 且 `open_price` 永不重置。 | 對齊死區門檻與再平衡門檻（或確保低部位平倉時正確歸零重置 `open_price`），徹底切斷歷史殘留 `open_price` 誘發的假套牢鎖死。 |
| **診斷與可觀測性 (`info`)** | `info` 字典僅記錄 `step_return`, `gross_return`, `cost`, `turnover`, `traded`。 | 新增 `downside_penalty` 診斷欄位，便於訓練與回測監控每步實際扣除的下行風險懲罰。 |

---

## 2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)

### 2.1 檔案與模組改動矩陣 (File & Module Scope Matrix)
| 編號 | 目標檔案與行號 | 範圍 (Scope) | 規劃動作 (Planned Action) |
| :---: | :--- | :---: | :--- |
| **項目 1** | [`Brain/PPO2/lib/environment.py:L20-L160`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L20-L160), [`L203-L212`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L203-L212), [`L348-L361`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L348-L361) | **In-Scope** | 於 [`State_time_step`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L21-L249) 實作改良版 `calculate_downside_risk_numpy()` 與 `calculate_step_downside_penalty()`，支援 `downside_window_size`、`w2_downside_ratio` 與 `weight_scale` 參數，並於 [`TrainingEnv`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L331-L399) 綁定配置。 |
| **項目 2** | [`configs/training/ppo_v2.yaml:L64-L70`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L64-L70) | **In-Scope** | 設定 `W1_STEP_RETURN: 1.0`、`DOWNSIDE_WINDOW_SIZE: 60`、`W2_DOWNSIDE_RATIO: 0.5` 與 `WEIGHT_SCALE: 100.0`，並調整 `DEADZONE_THRESHOLD: 0.03`、`REBALANCE_THRESHOLD: 0.05`。 |
| **項目 3** | [`Brain/PPO2/lib/engine.py:L99-L116`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L99-L116), [`L344-L357`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L344-L357), [`L536-L552`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L536-L552), [`L833-L901`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L833-L901) | **In-Scope** | 讀取 `DOWNSIDE_WINDOW_SIZE`、`W2_DOWNSIDE_RATIO` 與 `WEIGHT_SCALE` 並傳遞予 `State_time_step` 與 `PPO2Evaluator`。 |
| **項目 4** | [`Brain/PPO2/lib/Evaluator.py:L90-L140`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L90-L140) | **In-Scope** | 於 [`PPO2Evaluator.__init__`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L90-L125) 與 [`_make_state`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L126-L140) 轉發 `downside_window_size`、`w2_downside_ratio` 與 `weight_scale`。 |
| **項目 5** | `tests/test_ppo2_downside_penalty.py` | **In-Scope [NEW]** | 新增單元測試驗證固定視窗邊際下行風險計算、無遺忘棘輪效應、無冷啟動暴衝及配置綁定。 |
| **項目 6** | [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py), [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py), [`Brain/DQN/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/environment.py) | **Out-of-Scope** | 明確保留不更動，確保已訓練之權重格式（如 `checkpoint_iter_8200.pt`）100% 向後相容且不影響 DQN 模組。 |

### 2.2 架構邊界與非目標 (Architectural Boundaries & Non-Goals)
- **範圍內邊界 (In-Scope Boundaries)**:
  - 觀測空間維度 (`getStateShape() = (300, 16)`, `getTimeShape() = (300, 6)`) 與動作空間 (`Box(0.0, 1.0, shape=(1,))`) 保持 100% 不變，既有 Checkpoint 可直接載入評估或繼續訓練。
  - 下行風險懲罰與單步對數報酬維持相同量綱（皆乘以 `w1_step_return` 或以 `w1_step_return * w2_downside_ratio` 縮放），確保當 `w2_downside_ratio = 0.0` 時完全退化為現行無偏對數報酬。
- **非目標 (Out-of-Scope / Non-Goals)**:
  - 不重新引入會破壞時間尺度 ($\Delta t$) 的 Z-Score 二次方懲罰 (`DownsidePenaltyReward`)。
  - 不修改 Mamba2 模型骨幹、DAIN 正規化層或 `RolloutBuffer` GAE 計算邏輯。

---

## 3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)

### 3.1 步驟清單 (Step-by-Step Task Checklist)
- [x] **步驟 1: 於 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) 實作改良版下行風險懲罰機制**
  - [x] 引入 `from collections import deque`。
  - [x] 於 [`State_time_step.__init__`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L21-L61) 加入 `downside_window_size: int`、`w2_downside_ratio: float` 與 `weight_scale: float`，初始化 `self.return_history = deque(maxlen=downside_window_size)` 與 `self.last_downside_penalty = 0.0`。
  - [x] 於 [`State_time_step.reset`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L84-L99) 清空 `self.return_history.clear()` 並重置 `self.last_downside_penalty = 0.0`。
  - [x] 實作固定視窗分母之 `calculate_downside_risk_numpy(self, returns)` 與無遺忘棘輪之 `calculate_step_downside_penalty(self, current_p_return: float) -> float`（回傳純邊際擴張量 `downside_delta`）。
  - [x] 於 [`State_time_step.step`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L139-L220) 計算 `downside_penalty` 並以 `weight_scale * w1_step_return * current_p_return - weight_scale * w2_downside_ratio * downside_penalty` 合成單步獎勵，同時暴露於 [`BaseTradingEnv.step`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L284-L310) 的 `info["downside_penalty"]`。
  - [x] 於 [`TrainingEnv.__init__`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L337-L363) 綁定 `downside_window_size`、`w2_downside_ratio` 與 `weight_scale`。
- [x] **步驟 2: 同步更新配置與評估引擎 ([`ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml), [`engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py), [`Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py))**
  - [x] 於 [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) 設定 `DEADZONE_THRESHOLD`、`REBALANCE_THRESHOLD`、`W1_STEP_RETURN`、`DOWNSIDE_WINDOW_SIZE`、`W2_DOWNSIDE_RATIO` 與 `WEIGHT_SCALE`。
  - [x] 於 [`PPO2Engine`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) 與 [`PPO2Evaluator`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py) 綁定並轉發上述參數。
- [x] **步驟 3: 代碼防禦與清理 (Clean up & Defensiveness)**
  - [x] 加入 `assert downside_window_size >= 1`、`assert w2_downside_ratio >= 0.0` 與 `assert weight_scale > 0.0` 防禦性斷言。
  - [x] 確保當 `current_p_return >= 0.0` 時，`calculate_step_downside_penalty` 零額外陣列配置、直接快速回傳 `0.0`。

### 3.2 詳細程式碼修改藍圖 (Detailed Code Modifications Blueprint)

#### 3.2.1 改良版 `calculate_step_downside_penalty` 數學設計 ([`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py))
為解決舊版 DQN `calculate_step_downside_penalty` 的 $N=2$ 冷啟動分母暴衝與 60 步視窗遺忘棘輪，並將獎勵倍率 `weight_scale` 與相對權重 `w1_step_return` / `w2_downside_ratio` 清晰解耦，設計如下：

```python
    def calculate_downside_risk_numpy(self, returns) -> float:
        """
        Computes fixed-window semi-deviation (downside risk) normalized by
        self.downside_window_size to eliminate early-episode small-N denominator spikes.
        """
        if len(returns) == 0:
            return 0.0
        downside_diff = np.maximum(0.0, -np.asarray(returns, dtype=np.float64))
        squared_downside = downside_diff ** 2
        return float(np.sqrt(np.sum(squared_downside) / float(self.downside_window_size)))

    def calculate_step_downside_penalty(self, current_p_return: float) -> float:
        """
        Computes the exact single-step marginal downside risk expansion caused by
        current_p_return over the active rolling window.
        Eliminates sliding-window drop-off ratchet artifacts by comparing the same prior window
        before and after appending current_p_return.
        """
        w2 = self.weights["w2_downside_ratio"]
        if w2 <= 0.0 or current_p_return >= 0.0:
            self.return_history.append(float(current_p_return))
            return 0.0

        # Prior window of at most (W - 1) completed step returns
        prior_slice = (
            list(self.return_history)[-(self.downside_window_size - 1) :]
            if self.downside_window_size > 1
            else []
        )
        self.return_history.append(float(current_p_return))

        prev_dr = self.calculate_downside_risk_numpy(prior_slice)
        curr_dr = self.calculate_downside_risk_numpy(
            prior_slice + [float(current_p_return)]
        )
        downside_delta = max(0.0, curr_dr - prev_dr)

        return downside_delta
```

並且於 `State_time_step.step()` 中：
```python
        downside_penalty = self.calculate_step_downside_penalty(current_p_return)
        self.last_downside_penalty = downside_penalty

        # Risk-adjusted single-step net portfolio reward
        reward = float(
            self.weights["weight_scale"] * self.weights["w1_step_return"] * current_p_return
            - self.weights["weight_scale"] * self.weights["w2_downside_ratio"] * downside_penalty
        )
```

---

## 4. 驗證與測試計畫 (Validation & Verification Plan)

### 4.1 驗證環境與執行指令 (Verification Environment & Commands)
> [!IMPORTANT]
> 依據專案規範，所有執行必須使用虛擬環境之 Python 直譯器：
> 絕對路徑: `/home/b0457812963/Mamba3RL/bin/python`
> 相對路徑: `../bin/python`

```bash
# 驗證執行指令:
/home/b0457812963/Mamba3RL/bin/python -m unittest tests/test_ppo2_downside_penalty.py
```

### 4.2 驗收測試查核清單 (Acceptance Verification Checklist)
- [x] **1. 功能正確性 (Functional Correctness)**:
  - [x] 當 `current_p_return >= 0.0` 時，`calculate_step_downside_penalty()` 嚴格回傳 `0.0`，不懲罰任何正向獲利與上漲波動。
  - [x] 當 `w2_downside_ratio == 0.0` 時，`reward == w1_step_return * current_p_return` 保持 $10^{-12}$ 內完全相等。
- [x] **2. 邊界與極值處理 (Boundary & Edge Case Handling)**:
  - [x] **無冷啟動暴衝**：在 Episode 第 1 步與第 60 步（空手後首次發生相同虧損 $-1\%$）時，邊際懲罰值完全一致，不會因 $N=2$ 產生 $5.5$ 倍暴衝。
  - [x] **無滑動視窗棘輪**：當 60 步前的舊虧損滑出視窗時，不會降低基準線導致後續小虧損被重複超額計罰。
- [x] **3. 實際執行與經驗驗證 (Runtime & Empirical Verification)**:
  - [x] `TrainingEnv`、`SubprocVecEnv`、`PPO2Engine` 與 `PPO2Evaluator` 均可透過 [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) 正常初始化並執行推論與訓練。
