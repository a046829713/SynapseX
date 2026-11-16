# 規格文件 (Specification): PPO2 環境複利淨損益 (Compounded NetPnL%) 狀態追蹤、訓練日誌對接與評估指標去裁剪

- **類別 (Category)**: `bugfix`
- **規格路徑 (Spec Path)**: `spec/bugfix/ppo2_env_compounded_net_pnl.md`
- **狀態 (Status)**: `Implemented & Verified`

## 📋 目錄 (Index / Contents)
1. [1. 背景與根本原因分析 (Background & Root Cause Analysis)](#1-背景與根本原因分析-background--root-cause-analysis)
2. [2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)](#2-改動範圍與架構邊界-target-scope--architecture-boundaries)
3. [3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)](#3-實作查核清單與詳細藍圖-implementation-checklist--detailed-blueprint)
4. [4. 驗證與測試計畫 (Validation & Verification Plan)](#4-驗證與測試計畫-validation--verification-plan)

---

## 1. 背景與根本原因分析 (Background & Root Cause Analysis)

### 1.1 問題陳述與情境脈絡 (Problem Statement & Context)
在最近的獎勵函數與環境重構（[`spec/feature/ppo2_downside_risk_reward.md`](file:///home/b0457812963/Mamba3RL/SynapseX/spec/feature/ppo2_downside_risk_reward.md) 與 [`spec/bugfix/ppo2_env_accounting_and_temporal_split.md`](file:///home/b0457812963/Mamba3RL/SynapseX/spec/bugfix/ppo2_env_accounting_and_temporal_split.md)）之後，[`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)、[`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) 與 [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py) 存在以下兩項需要同步修正的問題：

1. **訓練腳本 `NetPnL` 誤將「累加對數報酬率」當作「真實複利資金報酬率 (%)」，且未由 `environment.py` 直接提供**：
   - 在引入下行風險懲罰 (`downside_penalty`) 與獎勵倍率 (`weight_scale = 100.0`) 後，[`PPO2_rl_train.py:L594`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L594) 顯示的 `Return` 是含下行風險懲罰的強化學習總回報，不再與投資組合淨報酬率 1:1 相等。
   - 在 [`Brain/PPO2/lib/environment.py:L193-L197`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L193-L197) 中，`self.last_step_return = current_p_return = float(np.log(m_net))` 為單步**對數報酬率 (Log-Return)** $\ln(E_t / E_{t-1})$。
   - 然而在 [`PPO2_rl_train.py:L561`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L561) 與 [`L578`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L578) 中，訓練迴圈在外部手動累加 `step_return` 並直接乘以 `100.0`：
     ```python
     episode_net_pnl[w_idx] += float(info_w.get("step_return", 0.0))
     net_pnl_pct_i = float(episode_net_pnl[i] * 100.0)
     ```
     這計算出的是**累積對數報酬** $\left(\sum_{t=1}^T \ln(m_{\text{net}, t})\right) \times 100$，而非真實的複利百分比淨損益 $\left(\exp\left(\sum_{t=1}^T \ln(m_{\text{net}, t})\right) - 1\right) \times 100\%$。
   - 投資組合的淨值 (`equity`) 與複利淨損益百分比 (`net_pnl_pct`) 應由 [`State_time_step`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L21-L220) 在環境內部直接維護，並透過 [`BaseTradingEnv.step()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L285-L306) 的 `info["net_pnl_pct"]`（與 `info["equity"]`）直接接出給 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)，避免外部腳本自行拼湊計算而產生判斷錯誤。

2. **[`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py) 殘留早期雙向做空時代的人工 `np.clip` 裁剪**：
   - 在 [`Evaluator.py:L322`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L322)、[`L338`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L338)、[`L419`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L419)、[`L423-L431`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L423-L431) 與 [`L469-L471`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L469-L471) 中，殘留了多處早期為了防範做空爆倉或線性對數近似失真而加上的人工裁剪（例如 `np.clip(..., -50.0, 50.0)`、`np.clip(np.mean(positions), 0.0, 1.0)`、`np.clip(np.max(drawdowns) * 100.0, 0.0, 100.0)`）。
   - 在現行 Long-Only (`[0.0, 1.0]`) 精確淨值複利機制下，`positions` 本身已在 `[0.0, 1.0]` 內、`drawdowns` 本身已在 `[0.0, 1.0)` 內，且 `cum_log_ret`、`bh_log_ret`、`sharpe`、`sortino`、`calmar` 不應受到人工 `[-50.0, 50.0]` 截斷，應全數移除以反映真實未截斷的評估指標。

### 1.2 當前行為 vs. 預期行為 (Current vs. Expected Behavior)
| 維度 | 當前行為 (Current Behavior) | 預期行為 (Expected Behavior) |
| :--- | :--- | :--- |
| **環境與訓練 `NetPnL%`** | [`PPO2_rl_train.py:L578`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L578) 將 `sum(step_return) * 100.0`（累積對數報酬）直接作為 `NetPnL%` 印出。 | 於 [`State_time_step`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L21-L220) 內部維護累積對數報酬 `cum_log_return`、真實複利淨值 `equity = float(np.exp(self.cum_log_return))` 與 `net_pnl_pct = float((self.equity - 1.0) * 100.0)`，並由 `info["net_pnl_pct"]` 接出至 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)。 |
| **評估器報酬與淨值曲線 (`Evaluator.py`)** | `cum_return_pct`、`bh_return_pct`、`equity_curve` 對累積對數報酬執行 `np.clip(..., -50.0, 50.0)`；`mean_exposure` 與 `max_drawdown_pct` 亦做多餘的 `np.clip`。 | 移除上述 `np.clip`，直接計算 `np.exp(cum_log_ret)`、`np.exp(bh_log_ret)`、`np.exp(cum_curve)`、`float(np.mean(positions))` 與 `float(np.max(drawdowns) * 100.0)`。 |
| **評估器風險調整比率 (`Evaluator.py`)** | `_safe_annualized_ratio` (`sharpe`, `ir_1x`, `ir_2x`)、`_safe_annualized_sortino` (`sortino`) 與 `calmar` 皆被 `np.clip(..., -50.0, 50.0)` 截斷。 | 移除 `[-50.0, 50.0]` 裁剪，保留真實比率數值（僅維持 `np.isfinite` 與分母 `1e-8` / `1e-4` 除零保護）。 |

### 1.3 技術根本原因與程式碼參照 (Technical Root Cause & References)
- 環境單步對數報酬與狀態定義: [`Brain/PPO2/lib/environment.py:L67-L74`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L67-L74), [`L85-L99`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L85-L99), [`L181-L212`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L181-L212), [`L294-L306`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L294-L306)
- 訓練腳本 `episode_net_pnl` 累加與 `NetPnL` 格式化輸出: [`PPO2_rl_train.py:L478`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L478), [`L560-L567`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L560-L567), [`L578-L612`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L578-L612)
- 評估器早期遺留 `np.clip`: [`Brain/PPO2/lib/Evaluator.py:L322`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L322), [`L338`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L338), [`L419-L431`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L419-L431), [`L469-L471`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L469-L471)

---

## 2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)

### 2.1 檔案與模組改動矩陣 (File & Module Scope Matrix)
| 編號 | 目標檔案與行號 | 範圍 (Scope) | 規劃動作 (Planned Action) |
| :---: | :--- | :---: | :--- |
| **項目 1** | [`Brain/PPO2/lib/environment.py:L67-L99`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L67-L99), [`L192-L205`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L192-L205), [`L294-L306`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L294-L306) | **In-Scope** | 於 [`State_time_step`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L21-L220) 追蹤 `cum_log_return`、`equity = float(np.exp(self.cum_log_return))` 與 `net_pnl_pct = float((self.equity - 1.0) * 100.0)`，在 `reset()` 時重置，在 `step()` 時更新，並於 [`BaseTradingEnv.step()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L285-L306) 的 `info` 傳出 `"equity"` 與 `"net_pnl_pct"`。 |
| **項目 2** | [`PPO2_rl_train.py:L478`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L478), [`L560-L612`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L560-L612) | **In-Scope** | 移除外部手動將對數報酬相加乘 100 的 `episode_net_pnl` 舊邏輯，直接從環境 `infos[i]["net_pnl_pct"]` 接出真實複利 `net_pnl_pct_i` 用於 `[Ep #xxxx]` 印出與 `recent_episode_net_pnl_pct` (`NetPnL(100)`, `Win(100)`) 統計。 |
| **項目 3** | [`Brain/PPO2/lib/Evaluator.py:L322`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L322), [`L338`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L338), [`L419-L431`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L419-L431), [`L469-L471`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L469-L471) | **In-Scope** | 移除 `_safe_annualized_ratio`、`_safe_annualized_sortino`、`mean_exposure`、`cum_return_pct`、`bh_return_pct`、`equity_curve`、`max_drawdown_pct` 與 `calmar` 的早期遺留 `np.clip`。 |
| **項目 4** | [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py), [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py), [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py) | **Out-of-Scope** | 明確保留不更動，防範回歸問題。 |

### 2.2 架構邊界與非目標 (Architectural Boundaries & Non-Goals)
- **範圍內邊界 (In-Scope Boundaries)**:
  - `State_time_step` 內部新增 `self.cum_log_return = 0.0`、`self.equity = 1.0`、`self.net_pnl_pct = 0.0` 診斷狀態，在每次 `reset()` 時重置、每次 `step()` 時更新（無 `np.clip`）。
  - `BaseTradingEnv.step()` 的 `info` 新增 `"equity"` 與 `"net_pnl_pct"` 欄位，保留既有欄位以維持 100% 向後相容。
  - `PPO2_rl_train.py` 的 `net_pnl_pct_i` 直接由 `infos[i]["net_pnl_pct"]` 取得。
  - `PPO2Evaluator` 移除所有針對報酬、淨值曲線、回撤、曝險與風險調整比率的 `np.clip`。
- **非目標 (Out-of-Scope / Non-Goals)**:
  - 不修改 PPO 核心演算法必要的裁剪（如 `clip_eps`、`max_grad_norm`、`log_std_min/max`）或動作空間 `[0.0, 1.0]` 邊界保護。

---

## 3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)

### 3.1 步驟清單 (Step-by-Step Task Checklist)
- [x] **步驟 1: 於 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) 實作環境內複利淨值與 `net_pnl_pct` 追蹤並由 `info` 匯出**
  - [x] 在 [`State_time_step.__init__`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L67-L74) 初始化 `self.cum_log_return = 0.0`、`self.equity = 1.0` 與 `self.net_pnl_pct = 0.0`。
  - [x] 在 [`State_time_step.reset`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L85-L99) 重置 `self.cum_log_return = 0.0`、`self.equity = 1.0` 與 `self.net_pnl_pct = 0.0`。
  - [x] 在 [`State_time_step.step`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L192-L205) 累加 `self.cum_log_return += current_p_return`，計算 `self.equity = float(np.exp(self.cum_log_return))` 與 `self.net_pnl_pct = float((self.equity - 1.0) * 100.0)`。
  - [x] 在 [`BaseTradingEnv.step`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L294-L306) 的 `info` 字典加入 `"equity": float(self._state.equity)` 與 `"net_pnl_pct": float(self._state.net_pnl_pct)`。
- [x] **步驟 2: 於 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) 直接對接環境 `info["net_pnl_pct"]`**
  - [x] 於 [`PPO2.train`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L478) 移除冗餘的外部對數累加陣列 `episode_net_pnl`。
  - [x] 於 Episode 結束時 ([`PPO2_rl_train.py:L578`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L578)) 直接讀取 `net_pnl_pct_i = float(infos[i].get("net_pnl_pct", 0.0))`，供 L585 `recent_episode_net_pnl_pct` 與 L595 `f"NetPnL: {net_pnl_pct_i:+.2f}%"` 使用。
- [x] **步驟 3: 移除 [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py) 中早期遺留的指標與報酬 `np.clip`**
  - [x] 於 [`_safe_annualized_ratio`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L312-L322) 與 [`_safe_annualized_sortino`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L325-L338) 移除 `np.clip(val, -50.0, 50.0)`，直接回傳 `float(val) if np.isfinite(val) else 0.0`。
  - [x] 於 [`compute_symbol_metrics`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L419-L431) 移除 `mean_exposure`、`cum_return_pct`、`bh_return_pct`、`equity_curve`、`max_drawdown_pct` 的 `np.clip`。
  - [x] 於 [`compute_symbol_metrics`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L469-L471) 移除 `calmar` 的 `np.clip(..., -50.0, 50.0)`。
- [x] **步驟 4: 代碼防禦與清理 (Clean up & Defensiveness)**
  - [x] 清理 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) 中不再使用的 `episode_net_pnl` 變數。

### 3.2 詳細程式碼修改藍圖 (Detailed Code Modifications Blueprint)

#### 3.2.1 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)
於 `State_time_step.__init__` 與 `reset`:
```python
        # Per-step & episode diagnostic accounting attributes
        self.last_step_return = 0.0
        self.last_gross_return = 0.0
        self.last_cost = 0.0
        self.last_turnover = 0.0
        self.last_traded = False
        self.last_downside_penalty = 0.0
        self.cum_log_return = 0.0
        self.equity = 1.0
        self.net_pnl_pct = 0.0
```
於 `State_time_step.step`:
```python
        self.last_step_return = current_p_return
        self.last_gross_return = gross_p_return
        self.last_cost = effective_cost
        self.last_turnover = float(abs(delta_p))
        self.last_traded = bool(abs(delta_p) > 1e-8)
        self.cum_log_return += current_p_return
        self.equity = float(np.exp(self.cum_log_return))
        self.net_pnl_pct = float((self.equity - 1.0) * 100.0)
```
於 `BaseTradingEnv.step`:
```python
        info = {
            "instrument": self._instrument,
            "offset": self._state._offset,
            "position": float(self._state.position),
            "step_return": float(self._state.last_step_return),
            "gross_return": float(self._state.last_gross_return),
            "cost": float(self._state.last_cost),
            "turnover": float(self._state.last_turnover),
            "traded": bool(self._state.last_traded),
            "downside_penalty": float(self._state.last_downside_penalty),
            "equity": float(self._state.equity),
            "net_pnl_pct": float(self._state.net_pnl_pct),
        }
```

#### 3.2.2 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)
於 `PPO2.train()` 中：
```python
                    for w_idx, info_w in enumerate(infos):
                        episode_cost[w_idx] += float(info_w.get("cost", 0.0))
                        episode_trades[w_idx] += int(bool(info_w.get("traded", False)))
                        episode_turnover[w_idx] += float(info_w.get("turnover", 0.0))
                        episode_abs_pos_sum[w_idx] += abs(
                            float(info_w.get("position", 0.0))
                        )

                    episode_steps += 1
                    total_steps += num_workers
                    obs = next_obs

                    for i, done in enumerate(dones):
                        if done:
                            episode_count += 1
                            steps_i = int(max(episode_steps[i], 1))
                            ret_i = float(episode_rewards[i])
                            net_pnl_pct_i = float(infos[i].get("net_pnl_pct", 0.0))
                            cost_pct_i = float(episode_cost[i] * 100.0)
                            trades_i = int(episode_trades[i])
                            turnover_i = float(episode_turnover[i])
                            avg_abs_pos_i = float(episode_abs_pos_sum[i] / steps_i)
```

#### 3.2.3 [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py)
於 `_safe_annualized_ratio` 與 `_safe_annualized_sortino`:
```python
        val = (mean_val / std_val) * annual_factor
        return float(val) if np.isfinite(val) else 0.0
```
```python
        val = (mean_val / downside_std) * annual_factor
        return float(val) if np.isfinite(val) else 0.0
```
於 `compute_symbol_metrics`:
```python
        mean_exposure = float(np.mean(positions))

        cum_log_ret = float(np.sum(step_ret))
        bh_log_ret = float(np.sum(mkt_ret))
        cum_return_pct = float((np.exp(cum_log_ret) - 1.0) * 100.0)
        bh_return_pct = float((np.exp(bh_log_ret) - 1.0) * 100.0)

        # Bar-level compounded equity curve for exact Max Drawdown
        cum_curve = np.concatenate([[0.0], np.cumsum(step_ret)])
        equity_curve = np.exp(cum_curve)
        running_peak = np.maximum.accumulate(equity_curve)
        drawdowns = 1.0 - (equity_curve / np.maximum(running_peak, 1e-12))
        max_drawdown_pct = float(np.max(drawdowns) * 100.0)
```
以及 `calmar`:
```python
        ann_log_ret = (cum_log_ret / max(eval_days, 1)) * 365.0
        mdd_frac = max_drawdown_pct / 100.0
        if mdd_frac > 1e-4:
            calmar = float(ann_log_ret / mdd_frac)
        else:
            calmar = float(ann_log_ret * 10.0) if ann_log_ret > 0 else 0.0
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
/home/b0457812963/Mamba3RL/bin/python -m unittest tests/test_ppo2_downside_penalty.py tests/test_ppo2_leaderboard.py
```

### 4.2 驗收測試查核清單 (Acceptance Verification Checklist)
- [x] **1. 功能正確性 (Functional Correctness)**:
  - [x] 驗證多步交易後，`info["net_pnl_pct"]` 嚴格等於 `(np.exp(np.sum(step_returns)) - 1.0) * 100.0`，且與 `PPO2Evaluator.compute_symbol_metrics` 的 `cum_return_pct` 誤差 `< 1e-12`。
  - [x] 驗證 `reset()` 後 `state.cum_log_return == 0.0`、`state.equity == 1.0`、`state.net_pnl_pct == 0.0`。
  - [x] 驗證 `Evaluator.py` 的 `compute_symbol_metrics` 已移除報酬、淨值曲線、回撤、曝險與風險調整比率的 `np.clip`。
- [x] **2. 邊界與極值處理 (Boundary & Edge Case Handling)**:
  - [x] 空手 (`action = 0.0`) 時，`info["equity"] == 1.0` 且 `info["net_pnl_pct"] == 0.0`。
  - [x] 連續大幅虧損時，Long-Only `info["net_pnl_pct"]` 嚴格 `> -100.0%`，不會發生線性對數加總跌破 `-100%` 的假象。
- [x] **3. 實際執行與經驗驗證 (Runtime & Empirical Verification)**:
  - [x] `SubprocVecEnv` 多進程環境回傳之 `infos` 正確包含 `"equity"` 與 `"net_pnl_pct"`，且在 Worker `reset_at` 後新 Episode 重新從 `0.0%` 起算。
