# 規格文件 (Specification): 移除 PPO2 觀測空間之未實現損益特徵與 `open_price` 狀態追蹤 (`PORTFOLIO_DIM: 2 -> 1`)

- **類別 (Category)**: `feature`
- **規格路徑 (Spec Path)**: `spec/feature/ppo2_remove_unrealized_pnl_observation.md`
- **狀態 (Status)**: `Approved / Completed`

## 📋 目錄 (Index / Contents)
1. [1. 背景與根本原因分析 (Background & Root Cause Analysis)](#1-背景與根本原因分析-background--root-cause-analysis)
2. [2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)](#2-改動範圍與架構邊界-target-scope--architecture-boundaries)
3. [3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)](#3-實作查核清單與詳細藍圖-implementation-checklist--detailed-blueprint)
4. [4. 驗證與測試計畫 (Validation & Verification Plan)](#4-驗證與測試計畫-validation--verification-plan)

---

## 1. 背景與根本原因分析 (Background & Root Cause Analysis)

### 1.1 問題陳述與情境脈絡 (Problem Statement & Context)
在 PPO2 連續動作空間（Continuous Long-Only Rebalancing, $a_t \in [0.0, 1.0]$）架構下，環境於每一步執行逐日盯市（Mark-to-Market, MtM）對數權益結算。然而，目前 [`State_time_step`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L21-L258) 仍保留了源自舊版離散 Ticket 交易機制的加權平均進場價 `self.open_price` 以及未實現損益觀測特徵 `data_res[:, len(self.info_list) + 1]`。經嚴密評估，該特徵存在以下四大結構性問題：

1. **馬可夫決策過程（MDP）中之沉沒成本冗餘**：
   - 在 [`State_time_step.step()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L187-L220) 中，單步淨對數報酬 $r_t$（`current_p_return`）與獎勵 $R_t$ 完全取決於當步價格變化、上一刻持倉權重 `self.position`（$p_{t-1}$）以及調倉量 $|\Delta p_t|$ 的摩擦成本。
   - 歷史進場均價 `self.open_price` 對未來報酬、交易成本與下行風險懲罰（[`calculate_step_downside_penalty`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L118-L144)）皆無任何數學影響，屬於純粹的沉沒成本（Sunk Cost）。唯一具備馬可夫轉移意義的帳戶狀態僅有當前持倉權重 `self.position`。
2. **連續調倉下之虛擬會計失真与非平穩漂移（Non-Stationary Drift）**：
   - 在 [`State_time_step.step()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L173-L184) 中，加倉時以加權平均更新 `self.open_price`，減倉（部分平倉）時則保持 `self.open_price` 不變；只要倉位未完全歸零（`target_pos == 0.0`），`self.open_price` 便永遠不會重置。
   - 在 [`State_time_step.encode()`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L246-L252) 中，特徵計算式 `1.0 + self.position * ((prev_close - safe_open) / safe_open)` 將「當前最新倉位比例」乘上「遠古進場均價至今的累積漲跌幅」，在連續加減倉時會產生劇烈跳變，且在長週期測試集（數千根 K 棒）中產生遠超訓練期（`EPISODE_STEPS = 512`）的 Out-of-Distribution (OOD) 漂移。
3. **誘發處置效應（Disposition Effect）與干擾 Critic 價值估計**：
   - 將歷史浮動損益輸入 Actor 與 Critic，容易使策略過擬合於「浮盈鎖利、浮虧死扛」的處置效應，並以無關未來期望回報的歷史損益干擾 $V(s_t)$ 的回歸收斂。
4. **實盤推論（Formal Live Mode）狀態對齊複雜度**：
   - 移除 `open_price` 與損益特徵後，實盤推論僅需傳入單一標量 `position` $\in [0.0, 1.0]$，徹底消除實盤重啟或部分成交時歷史均價難以精確重建的問題。

因此，本規格將 `data_res` 中的未實現損益特徵與所有關聯的 `open_price` 帳戶狀態追蹤完全移除，使 `PORTFOLIO_DIM` 由 `2` 降為 `1`（僅保留 `position`）。

### 1.2 當前行為 vs. 預期行為 (Current vs. Expected Behavior)
| 維度 | 當前行為 (Current Behavior) | 預期行為 (Expected Behavior) |
| :--- | :--- | :--- |
| **觀測空間形狀 (`getStateShape`)** | `(bars_count, len(info_list) + 2)`，即 `(300, 16)` | `(bars_count, len(info_list) + 1)`，即 `(300, 15)` |
| **Portfolio 特徵維度 (`PORTFOLIO_DIM`)** | `2`：`[position, unrealized_return]` | `1`：`[position]` |
| **帳戶進場均價追蹤 (`self.open_price`)** | 於 `State_time_step`、`PPO2DataAdapter`、`PPO2Engine` 中維護並傳遞 `open_price` | 完全移除 `self.open_price` 屬性、加權均價計算與函式參數 |
| **Evaluator 批次倉位回退索引** | `states_all[:, -1, -2]`（假設倒數第 2 欄為 `position`） | `states_all[:, -1, -1]`（最後 1 欄即為 `position`） |

### 1.3 技術根本原因與程式碼參照 (Technical Root Cause & References)
- 環境狀態形狀與 `open_price` 維護：[`Brain/PPO2/lib/environment.py:L63`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L63)、[`L85-L86`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L85-L86)、[`L93`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L93)、[`L173-L185`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L173-L185)
- 環境觀測編碼 `encode()`：[`Brain/PPO2/lib/environment.py:L244-L253`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L244-L253)
- 訓練設定檔 `PORTFOLIO_DIM`：[`configs/training/ppo_v2.yaml:L48`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L48)
- 模型與 Agent 預設 `portfolio_dim`：[`Brain/PPO2/lib/model.py:L298-L323`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L298-L323)、[`Brain/PPO2/lib/Agent.py:L23`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L23)
- 資料轉接器 `PortfolioState` 與 `extract_latest_obs`：[`Brain/PPO2/lib/data_adapter.py:L9-L51`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py#L9-L51)、[`L175-L230`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py#L175-L230)
- 測試與實盤引擎 `PPO2Engine`：[`Brain/PPO2/lib/engine.py:L221-L245`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L221-L245)、[`L767-L813`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L767-L813)
- 評估器 `evaluate_batch_step` 倉位索引：[`Brain/PPO2/lib/Evaluator.py:L193-L195`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L193-L195)

---

## 2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)

### 2.1 檔案與模組改動矩陣 (File & Module Scope Matrix)
| 編號 | 目標檔案與行號 | 範圍 (Scope) | 規劃動作 (Planned Action) |
| :---: | :--- | :---: | :--- |
| **項目 1** | [`Brain/PPO2/lib/environment.py:L61-L257`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L61-L257) | **In-Scope** | 1. 移除 `__init__`、`reset`、`step` 中的 `self.open_price` 狀態變數與加權均價維護邏輯。<br/>2. 將 `getStateShape()` 改為回傳 `(self.bars_count, len(self.info_list) + 1)`。<br/>3. 於 `encode()` 移除 `data_res[:, len(self.info_list) + 1]` 損益計算。 |
| **項目 2** | [`configs/training/ppo_v2.yaml:L48`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L48) | **In-Scope** | 將 `PORTFOLIO_DIM: 2` 更新為 `PORTFOLIO_DIM: 1`。 |
| **項目 3** | [`Brain/PPO2/lib/model.py:L298-L323`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py#L298-L323) | **In-Scope** | 將 `MambaContinuousActorCritic.__init__` 預設參數 `portfolio_dim: int = 2` 改為 `1`，並更新 docstring。 |
| **項目 4** | [`Brain/PPO2/lib/Agent.py:L23`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py#L23) | **In-Scope** | 將 `PPO2Agent.__init__` 預設參數 `portfolio_dim: int = 2` 改為 `1`。 |
| **項目 5** | [`Brain/PPO2/lib/data_adapter.py:L9-L230`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py#L9-L230) | **In-Scope** | 1. 移除 `PortfolioState.open_price` 與 `resolve_portfolio_state` 中的 `open_price` 解析。<br/>2. 移除 `extract_latest_obs` 的 `open_price` 參數與損益編碼，`data_res` 形狀改為 `(bars_count, len(info_list) + 1)`。 |
| **項目 6** | [`Brain/PPO2/lib/engine.py:L221-L245, L767-L813`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py#L221-L245) | **In-Scope** | 1. 移除 `_extract_latest_obs` 的 `open_price` 參數。<br/>2. 將 `_resolve_symbol_portfolio_state` 改為直接回傳 `float`（`state.position`）。<br/>3. 更新 `_collect_symbol_observations` 內所有呼叫端。 |
| **項目 7** | [`Brain/PPO2/lib/Evaluator.py:L193-L195`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py#L193-L195) | **In-Scope** | 將 `evaluate_batch_step` 中未提供 `prev_positions` 時的 fallback 索引由 `states_all[:, -1, -2]` 修正為 `states_all[:, -1, -1]`。 |
| **項目 8** | [`Brain/Common/env_components.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/Common/env_components.py) | **Out-of-Scope** | 保持共用模板不變，避免影響其他模型（如 DQN）。 |

### 2.2 架構邊界與非目標 (Architectural Boundaries & Non-Goals)
- **範圍內邊界 (In-Scope Boundaries)**:
  - `states` 觀測張量形狀嚴格對齊為 `(bars_count, len(info_list) + 1)`（即 `(300, 15)`）。
  - `State_time_step.encode()` 與 `PPO2DataAdapter.extract_latest_obs()` 保持 100% bit-for-bit 輸出一致。
  - `MambaContinuousActorCritic` 的雙流解耦架構維持不變：前 `market_dim = 14` 維市場特徵進入 `DAIN_Layer`，最後 `portfolio_dim = 1` 維（`position`）繞過 DAIN 並進入 `portfolio_encoder`。
- **非目標 (Out-of-Scope / Non-Goals)**:
  - 不更動 `State_time_step.step()` 中用於計算當步 K 棒開盤跳空與日內報酬的區域變數 `open_price = max(float(self._prices.open[self._offset]), 1e-8)`。
  - 不遷移舊有 `PORTFOLIO_DIM = 2` 或 `3` 的歷史 Checkpoint 權重檔（舊權重在載入時會由 `PPO2Engine` 的異常捕捉機制安全跳過）。

---

## 3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)

### 3.1 步驟清單 (Step-by-Step Task Checklist)
- [x] **步驟 1: 重構環境類別 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)**
  - [x] 移除 `__init__` 與 `reset` 中的 `self.open_price = 0.0`。
  - [x] 將 `getStateShape(self)` 改為 `return self.bars_count, len(self.info_list) + 1`。
  - [x] 移除 `step()` 中維護 `self.open_price` 的加權平均區塊，僅保留 `self.position = target_pos`。
  - [x] 移除 `encode()` 中 `data_res[:, len(self.info_list) + 1]` 的計算與賦值。
- [x] **步驟 2: 更新設定檔與模型預設值**
  - [x] 修改 [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) 之 `PORTFOLIO_DIM: 1`。
  - [x] 修改 [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) 與 [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py) 之 `portfolio_dim: int = 1`。
- [x] **步驟 3: 清理 [`Brain/PPO2/lib/data_adapter.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py)、[`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) 與 [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py)**
  - [x] 於 `PPO2DataAdapter` 完全移除 `open_price` 欄位、參數與損益計算邏輯。
  - [x] 於 `PPO2Engine` 移除 `open_price` 傳遞並簡化 `_resolve_symbol_portfolio_state`。
  - [x] 於 `PPO2Evaluator.evaluate_batch_step` 修正 fallback 倉位索引為 `states_all[:, -1, -1]`。
- [x] **步驟 4: 代碼防禦與清理 (Clean up & Defensiveness)**
  - [x] 確認 `Brain/PPO2/` 目錄下無任何殘留的 `self.open_price` 或 `open_price` 帳戶狀態引用。

### 3.2 詳細程式碼修改藍圖 (Detailed Code Modifications Blueprint)

#### 3.2.1 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)
```python
        # 狀態變數初始化
        self.position = 0.0
        self.game_steps = 0
        self._prev_close = 0.0
...
    def getStateShape(self):
        return self.bars_count, len(self.info_list) + 1

    def reset(self, prices: Prices, offset: int):
        assert offset >= self.bars_count - 1
        self._prices = prices
        self._offset = offset
        self.position = 0.0
        self.game_steps = 0
        self._prev_close = float(self._prices.close[self._offset - 1])
...
        delta_p = target_pos - prev_pos
        friction_rate = self.commission_perc + self.default_slippage

        self.position = target_pos
...
    def encode(self):
        """
        Encodes observation dict {"states": data_res, "time_states": time_res}
        using vectorized NumPy slicing without look-ahead bias.
        """
        data_res = np.zeros(shape=self.getStateShape(), dtype=np.float32)
        time_res = np.zeros(shape=self.getTimeShape(), dtype=np.float32)

        start = self._offset - self.bars_count
        end = self._offset

        for idx, field in enumerate(self.info_list):
            data_res[:, idx] = getattr(self._prices, field)[start:end]

        data_res[:, len(self.info_list)] = self.position

        for idx, field in enumerate(self.timelist):
            time_res[:, idx] = getattr(self._prices, field)[start:end]

        return {"states": data_res, "time_states": time_res}
```

#### 3.2.2 [`Brain/PPO2/lib/data_adapter.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/data_adapter.py)
```python
@dataclass
class PortfolioState:
    """Dataclass representing portfolio position for a symbol."""
    position: float = 0.0


class PPO2DataAdapter:
    @staticmethod
    def resolve_portfolio_state(
        symbol: str, portfolio_state: Optional[Dict[str, Any]]
    ) -> PortfolioState:
        """
        Resolves position for a given symbol from portfolio_state dictionary.
        """
        if not portfolio_state:
            return PortfolioState(0.0)

        entry = None
        if symbol in portfolio_state:
            entry = portfolio_state[symbol]
        elif "position" in portfolio_state:
            entry = portfolio_state

        if entry is None:
            return PortfolioState(0.0)
        if isinstance(entry, dict):
            pos = float(np.clip(float(entry.get("position", 0.0)), 0.0, 1.0))
            return PortfolioState(pos)
        if isinstance(entry, (tuple, list)):
            pos = float(np.clip(float(entry[0]) if len(entry) > 0 else 0.0, 0.0, 1.0))
            return PortfolioState(pos)
        if isinstance(entry, (int, float, np.floating, np.integer)):
            pos = float(np.clip(float(entry), 0.0, 1.0))
            return PortfolioState(pos)
        return PortfolioState(0.0)
...
    @staticmethod
    def extract_latest_obs(
        prices: Prices,
        bars_count: int,
        position: float = 0.0,
        portfolio_state: Optional[Dict[str, Any]] = None,
        current_position: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extracts the latest observation window of length bars_count for formal live mode,
        matching State_time_step.encode() bit-for-bit across market, portfolio, and 6-dim time features.
        """
        if portfolio_state is not None:
            position = float(portfolio_state.get("position", position))
        elif current_position is not None:
            position = float(current_position)

        pos = float(np.clip(position, 0.0, 1.0))

        n_bars = len(prices.close)
        if n_bars < bars_count:
            raise ValueError(
                f"Data length {n_bars} is shorter than required window {bars_count}"
            )

        start = n_bars - bars_count
        end = n_bars

        info_list = [
            "log_open", "log_high", "log_low", "log_close", "log_volume",
            "log_quote_av", "log_trades", "log_tb_base_av", "log_tb_quote_av",
            "log_ma_30", "log_ma_60", "log_ma_120", "log_ma_240", "log_ma_360",
        ]
        timelist = list(STATIONARY_TIME_FEATURES)

        data_res = np.zeros(shape=(bars_count, len(info_list) + 1), dtype=np.float32)
        time_res = np.zeros(shape=(bars_count, len(timelist)), dtype=np.float32)

        for idx, field in enumerate(info_list):
            data_res[:, idx] = getattr(prices, field)[start:end]

        data_res[:, len(info_list)] = pos

        for idx, field in enumerate(timelist):
            time_res[:, idx] = getattr(prices, field)[start:end]

        return {"states": data_res, "time_states": time_res}
```

#### 3.2.3 [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) & [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py)
```python
# Brain/PPO2/lib/engine.py
    def _extract_latest_obs(
        self,
        prices: Prices,
        position: float = 0.0,
        portfolio_state: Optional[Dict[str, Any]] = None,
        current_position: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        return self.data_adapter.extract_latest_obs(
            prices=prices,
            bars_count=self.bars_count,
            position=position,
            portfolio_state=portfolio_state,
            current_position=current_position,
        )

    def _resolve_symbol_portfolio_state(
        self,
        symbol: str,
        portfolio_state: Optional[Dict[str, Any]],
    ) -> float:
        state = self.data_adapter.resolve_portfolio_state(symbol, portfolio_state)
        return state.position
```
```python
# Brain/PPO2/lib/Evaluator.py
        if prev_positions is not None:
            prev_pos = np.clip(
                np.asarray(prev_positions, dtype=np.float64).reshape(-1), 0.0, 1.0
            )
        elif states_all.shape[-1] >= 1:
            prev_pos = np.clip(states_all[:, -1, -1].astype(np.float64), 0.0, 1.0)
        else:
            prev_pos = np.zeros(n_items, dtype=np.float64)
```

---

## 4. 驗證與測試計畫 (Validation & Verification Plan)

### 4.1 驗證環境與執行指令 (Verification Environment & Commands)
> [!IMPORTANT]
> 依據專案規範，所有執行必須使用虛擬環境之 Python 直譯器：
> 絕對路徑: `/home/b0457812963/Mamba3RL/bin/python`
> 相對路徑: `../bin/python`

```bash
# 1. 驗證 State_time_step 形狀、step 狀態轉移與 PPO2DataAdapter bit-for-bit 對齊
../bin/python -c "
import numpy as np
from Brain.PPO2.lib.environment import State_time_step
from Brain.PPO2.lib.data_adapter import PPO2DataAdapter
from Brain.Common.DataFeature import Prices

N = 320
fields = {f: np.ones(N, dtype=np.float32) * 100.0 for f in Prices._fields}
dummy_prices = Prices(**fields)

state = State_time_step(
    bars_count=300, commission_perc=0.0005, model_train=True,
    default_slippage=0.0005, N_steps=512, deadzone_threshold=0.03,
    rebalance_threshold=0.05, w1_step_return=1.0, downside_window_size=60,
    w2_downside_ratio=0.25, weight_scale=100.0
)
state.reset(dummy_prices, offset=300)
reward, term, trunc = state.step(0.65)
obs_env = state.encode()
assert obs_env['states'].shape == (300, 15), f'Expected (300, 15), got {obs_env[\"states\"].shape}'
assert not hasattr(state, 'open_price'), 'state.open_price should be completely removed'
assert np.allclose(obs_env['states'][:, -1], 0.65), 'Last column must equal current position 0.65'

sliced_prices = Prices(*[getattr(dummy_prices, f)[:301] for f in Prices._fields])
obs_adapter = PPO2DataAdapter.extract_latest_obs(sliced_prices, bars_count=300, position=0.65)
assert np.array_equal(obs_env['states'], obs_adapter['states']), 'Bit-for-bit mismatch in states'
assert np.array_equal(obs_env['time_states'], obs_adapter['time_states']), 'Bit-for-bit mismatch in time_states'
print('Validation 1 Passed!')
"

# 2. 驗證 PPO2Engine、PPO2Agent、MambaContinuousActorCritic 與 PPO2Evaluator 前向與批次推論
../bin/python -c "
import torch
import numpy as np
from omegaconf import OmegaConf
from Brain.PPO2.lib.engine import PPO2Engine
from Brain.PPO2.lib.Evaluator import PPO2Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = OmegaConf.load('configs/training/ppo_v2.yaml')
assert int(cfg.PORTFOLIO_DIM) == 1, f'Expected PORTFOLIO_DIM=1, got {cfg.PORTFOLIO_DIM}'
engine = PPO2Engine(config=cfg, device=device)
agent = engine.build_agent()
assert agent.model.portfolio_dim == 1
assert agent.model.market_dim == 14

evaluator = PPO2Evaluator(agent=agent, bars_count=300, device=device)
dummy_obs = {
    'states': np.zeros((4, 300, 15), dtype=np.float32),
    'time_states': np.zeros((4, 300, 6), dtype=np.float32),
}
dummy_obs['states'][:, :, -1] = 0.5
acts = evaluator.evaluate_batch_step(dummy_obs, apply_deadzone=True, prev_positions=None)
assert acts.shape == (4,), f'Expected (4,), got {acts.shape}'
print('Validation 2 Passed!')
"
```

### 4.2 驗收測試查核清單 (Acceptance Verification Checklist)
- [x] **1. 功能正確性 (Functional Correctness)**:
  - [x] `State_time_step.getStateShape()` 回傳 `(300, 15)`，且 `State_time_step` 不再帶有 `open_price` 屬性。
  - [x] `PPO2DataAdapter.extract_latest_obs()` 與 `State_time_step.encode()` 輸出完全一致。
  - [x] `MambaContinuousActorCritic` 的 `market_dim == 14`、`portfolio_dim == 1`。
- [x] **2. 邊界與極值處理 (Boundary & Edge Case Handling)**:
  - [x] `PPO2Evaluator.evaluate_batch_step` 在 `prev_positions=None` 時正確從最後一欄 `states_all[:, -1, -1]` 讀取 `prev_pos`。
- [x] **3. 實際執行與經驗驗證 (Runtime & Empirical Verification)**:
  - [x] 執行上述兩組驗證指令與現有相關單元測試皆以 exit code 0 通過。
