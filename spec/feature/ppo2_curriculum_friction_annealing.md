# 規格文件 (Specification): PPO2 交易手續費與滑點課程學習動態退火排程 (Curriculum Friction Annealing)

- **類別 (Category)**: `feature`
- **規格路徑 (Spec Path)**: `spec/feature/ppo2_curriculum_friction_annealing.md`
- **狀態 (Status)**: `Implemented & Verified`

## 📋 目錄 (Index / Contents)
1. [1. 背景與根本原因分析 (Background & Root Cause Analysis)](#1-背景與根本原因分析-background--root-cause-analysis)
2. [2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)](#2-改動範圍與架構邊界-target-scope--architecture-boundaries)
3. [3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)](#3-實作查核清單與詳細藍圖-implementation-checklist--detailed-blueprint)
4. [4. 驗證與測試計畫 (Validation & Verification Plan)](#4-驗證與測試計畫-validation--verification-plan)

---

## 1. 背景與根本原因分析 (Background & Root Cause Analysis)

### 1.1 問題陳述與情境脈絡 (Problem Statement & Context)
在強化學習連續動作空間交易任務中，PPO2 策略模型以往使用較低的手續費與滑點參數（`MODEL_DEFAULT_COMMISSION_PERC_TRAING: 0.0001`，`DEFAULT_SLIPPAGE: 0.0001`，單趟摩擦 0.02%，來回 0.04%）。為了貼近真實加密貨幣衍生品或高流動性現貨市場的真實交易成本，使用者將參數調整為真實水準：
- `MODEL_DEFAULT_COMMISSION_PERC_TRAING: 0.0005`（手續費 0.05%）
- `DEFAULT_SLIPPAGE: 0.0005`（滑點 0.05%）
- 單趟摩擦合計為 $0.0010$ (10 bps)，來回調倉摩擦達到 $0.0020$ (20 bps)。

在此設定下，模型於訓練開始後迅速產生「**策略完全不交易 / 策略凍結**」（MeanMu / MeanAct / Avg|Pos| 迅速趨近於 0）的現象：
1. **探索期的高頻噪聲調倉損耗**：在訓練初期，神經網路尚未學會價格趨勢特徵，動作輸出帶有較大隨機性（$\sigma \approx 1.0$）。在 30 分鐘級別 K 線上，動作值每根隨機波動導致頻繁換倉。隨機行情的預期報酬為 0，但每輪 512 步的 Episode 中，頻繁換倉摩擦損耗高達 15%～25%，產生顯著的負回報。
2. **零交易成為強大的局部最優解（Lazy Agent Trap）**：若模型輸出持倉為 0（落入 Deadzone $\le 0.01$），回報為 `0.0000`。在 PPO 梯度更新下，「完全不交易」遠優於「隨機探索虧損」，策略迅速被推向完全不交易的死角。
3. **訓練週期與退火節奏校準**：歷史經驗顯示模型通常在 **Iteration 14,000** 附近取得最佳性能（約 5,734 萬步）。若將退火時間拉得過長（例如 Iteration 100,000），當模型於 Iter 14,000 達到最佳權重時，手續費才約 0.00015，未能達到在真實摩擦力下充分適應收斂的要求；反之，若退火過快，則失去保護探索的效果。

因此，需要建立一個**三階段課程學習（Curriculum Learning）動態摩擦力退火排程**：
- **階段一（溫室探索期，Iter 0 ～ 1,500）**：維持低摩擦（0.0001），保護策略探索並學會市場趨勢方向感。
- **階段二（摩擦退火期，Iter 1,500 ～ 6,000）**：手續費與滑點平滑遞增（支援 Cosine 平滑餘弦或 Linear 線性）至 0.0005，逼迫模型淘汰薄利微調動作，拉長持倉週期。
- **階段三（實戰固化期，Iter 6,000 ～ 14,000+）**：鎖定於 0.0005 真實水準，提供超過 55% 的訓練時間讓策略在真實目標分佈下完全收斂。

### 1.2 當前行為 vs. 預期行為 (Current vs. Expected Behavior)
| 維度 | 當前行為 (Current Behavior) | 預期行為 (Expected Behavior) |
| :--- | :--- | :--- |
| **手續費與滑點設定** | 靜態常數（整個訓練固定為 `0.0005`，初期探索即面臨高摩擦懲罰） | 動態課程學習排程：初期低摩擦（0.0001），中段平滑退火，後段維持真實摩擦（0.0005） |
| **多進程環境通訊** | [`SubprocVecEnv`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L382) 僅支援 `step`, `reset`, `get_spaces`, `close`，無法動態同步參數 | 新增 `set_friction` IPC 通訊指令，可於訓練迭代無縫廣播最新費率至所有子進程 Worker |
| **排程平滑度** | 無退火機制 | 支援平滑餘弦（Cosine Annealing）與線性（Linear）遞增，避免價值函數目標突變造成 Critic 不穩定 |
| **訓練日誌指標** | 僅顯示常規指標，無當前手續費/滑點資訊 | 在 `[Iter #XXXX]` 日誌中即時顯示當前費率（例如 `Fee: 1.0->5.0bps | Slip: 1.0->5.0bps`） |
| **策略活躍度** | 初期迅速塌縮為零交易，`Avg|Pos|` 鎖死在 `0.00` | 前期保持活躍探索，在手續費爬升過程中逐步轉化為低換手率的中長波段持倉 |

### 1.3 技術根本原因與程式碼參照 (Technical Root Cause & References)
1. **多進程環境無法動態更新手續費**：
   - [`Brain/PPO2/lib/environment.py:L351-L372`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L351-L372)：`_subproc_worker` 的訊息迴圈未實作動態更新狀態屬性的管道。
   - [`Brain/PPO2/lib/environment.py:L20-L40`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py#L20-L40)：`State_time_step` 中 `self.commission_perc` 與 `self.default_slippage` 於建構時一次性寫入，無安全設值方法。
2. **訓練主迴圈無排程更新機制**：
   - [`PPO2_rl_train.py:L471-L640`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py#L471-L640)：訓練主迴圈每輪直接執行 Rollout，未結合 Iteration 進度計算摩擦力並推送到環境。
3. **YAML 缺少課程學習排程參數**：
   - [`configs/training/ppo_v2.yaml:L10-L11`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml#L10-L11)：僅有靜態常數定義。

---

## 2. 改動範圍與架構邊界 (Target Scope & Architecture Boundaries)

### 2.1 檔案與模組改動矩陣 (File & Module Scope Matrix)
| 編號 | 目標檔案與行號 | 範圍 (Scope) | 規劃動作 (Planned Action) |
| :---: | :--- | :---: | :--- |
| **項目 1** | [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml) | **In-Scope** | 新增摩擦力退火配置開關、起始/目標迭代、起始/目標費率與排程型態。 |
| **項目 2** | [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) | **In-Scope** | 為 `State_time_step`、`TrainingEnv` 新增 `set_friction()` 方法；於 `_subproc_worker` 及 `SubprocVecEnv` 實作 `"set_friction"` IPC 指令廣播。 |
| **項目 3** | [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) | **In-Scope** | 新增 `compute_friction_rates(iteration)` 方法，於每輪 Iteration 起始同步費率至 `vec_env`，並於 Iteration 輸出日誌中顯示當前摩擦力。 |
| **項目 4** | `tests/test_ppo2_friction_annealing.py` | **In-Scope [NEW]** | 新增獨立單元測試，驗證排程計算精準度（溫室期、退火期、固化期、Cosine 與 Linear）、SubprocVecEnv 動態 IPC 同步及邊界條件。 |
| **項目 5** | [`Brain/PPO2/lib/Agent.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py) & [`Brain/PPO2/lib/model.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) | **Out-of-Scope** | 核心神經網路架構、動作 Squashing、損失計算及反向傳播完全不變。 |
| **項目 6** | [`Brain/PPO2/lib/engine.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/engine.py) & [`Brain/PPO2/lib/Evaluator.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Evaluator.py) | **Out-of-Scope** | 測試與評估端固定使用真實環境目標費率（0.0005），不受訓練退火影響。 |

### 2.2 架構邊界與非目標 (Architectural Boundaries & Non-Goals)
- **範圍內邊界 (In-Scope Boundaries)**:
  - 摩擦力退火僅作用於訓練期（`TrainingEnv`），回測與驗證維持嚴格真實環境費率。
  - 支援以 Iteration 為基準的動態調節，保持向後相容（若關閉開關 `ENABLE_FRICTION_ANNEALING: false`，完全維持原有行為）。
- **非目標 (Out-of-Scope / Non-Goals)**:
  - 不修改動作空間定義（仍為連續 $[0, 1]$ Long-Only）。
  - 不引入額外的自適應手續費控制（嚴格依據 Iteration 排程驅動，避免獎勵非平穩性）。

---

## 3. 實作查核清單與詳細藍圖 (Implementation Checklist & Detailed Blueprint)

### 3.1 步驟清單 (Step-by-Step Task Checklist)
- [x] **步驟 1: 配置參數更新 (`configs/training/ppo_v2.yaml`)**
  - [x] 加入 `ENABLE_FRICTION_ANNEALING: true`
  - [x] 加入 `FRICTION_ANNEAL_START_ITER: 1500`
  - [x] 加入 `FRICTION_ANNEAL_END_ITER: 6000`
  - [x] 加入 `COMMISSION_START: 0.0001`、`COMMISSION_TARGET: 0.0005`
  - [x] 加入 `SLIPPAGE_START: 0.0001`、`SLIPPAGE_TARGET: 0.0005`
  - [x] 加入 `FRICTION_SCHEDULE: "cosine"`
- [x] **步驟 2: 環境狀態與多進程 IPC 同步實作 (`Brain/PPO2/lib/environment.py`)**
  - [x] 在 `State_time_step` 中加入 `set_friction(commission_perc: float, default_slippage: float)`。
  - [x] 在 `TrainingEnv` 中加入 `set_friction(commission_perc: float, default_slippage: float)` 轉發。
  - [x] 在 `_subproc_worker` 的訊息迴圈中加入 `elif cmd == "set_friction":`，執行後回傳確認。
  - [x] 在 `SubprocVecEnv` 中加入 `set_friction(commission_perc: float, default_slippage: float)` 廣播方法。
- [x] **步驟 3: 主訓練迴圈退火排程與日誌整合 (`PPO2_rl_train.py`)**
  - [x] 實作 `compute_friction_rates(iteration: int) -> Tuple[float, float]` 方法，支援 Cosine 與 Linear 兩種模式。
  - [x] 在 `PPO2.train()` 迴圈起始呼叫 `self.vec_env.set_friction(comm, slip)` 更新全體 Worker。
  - [x] 在 Iteration 終端輸出日誌加入 `Friction: {comm*10000:.1f}/{slip*10000:.1f}bps` 監控資訊。
- [x] **步驟 4: 代碼防禦與清理 (Clean up & Defensiveness)**
  - [x] 保留類型提示與浮點數安全性防護（`float(...)`, `max(1, end_iter - start_iter)` 防止除以零）。
  - [x] 確保在 `ENABLE_FRICTION_ANNEALING: false` 時平滑回退至原先的靜態設定。
- [x] **步驟 5: 驗證測試 (`tests/test_ppo2_friction_annealing.py`)**
  - [x] 撰寫排程函數之數值驗證（邊界值檢查、平滑曲線斷言）。
  - [x] 撰寫 `SubprocVecEnv` 透過多進程管線動態更新 `commission_perc` 與 `default_slippage` 之端到端測試。

### 3.2 詳細程式碼修改藍圖 (Detailed Code Modifications Blueprint)

#### 3.2.1 [`configs/training/ppo_v2.yaml`](file:///home/b0457812963/Mamba3RL/SynapseX/configs/training/ppo_v2.yaml)
```yaml
# Friction Annealing / Curriculum Learning Configuration
ENABLE_FRICTION_ANNEALING: true
FRICTION_ANNEAL_START_ITER: 1500
FRICTION_ANNEAL_END_ITER: 6000
COMMISSION_START: 0.0001
COMMISSION_TARGET: 0.0005
SLIPPAGE_START: 0.0001
SLIPPAGE_TARGET: 0.0005
FRICTION_SCHEDULE: "cosine"  # "cosine" | "linear"
```

#### 3.2.2 [`Brain/PPO2/lib/environment.py`](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py)
```python
# 1. State_time_step 新增設值方法
class State_time_step(State_time_step_template):
    ...
    def set_friction(self, commission_perc: float, default_slippage: float):
        self.commission_perc = float(commission_perc)
        self.default_slippage = float(default_slippage)

# 2. TrainingEnv 新增轉發方法
class TrainingEnv(BaseTradingEnv):
    ...
    def set_friction(self, commission_perc: float, default_slippage: float):
        self.state.set_friction(commission_perc, default_slippage)

# 3. _subproc_worker IPC 訊息處理
def _subproc_worker(remote, parent_remote, env_fn):
    ...
    while True:
        cmd, data = remote.recv()
        ...
        elif cmd == "set_friction":
            commission_perc, default_slippage = data
            env.set_friction(commission_perc, default_slippage)
            remote.send(True)

# 4. SubprocVecEnv 廣播方法
class SubprocVecEnv:
    ...
    def set_friction(self, commission_perc: float, default_slippage: float):
        for remote in self.remotes:
            remote.send(("set_friction", (float(commission_perc), float(default_slippage))))
        return [remote.recv() for remote in self.remotes]
```

#### 3.2.3 [`PPO2_rl_train.py`](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py)
```python
def compute_friction_rates(self, iteration: int) -> Tuple[float, float]:
    if not getattr(self.cfg_training, "ENABLE_FRICTION_ANNEALING", False):
        return (
            float(self.cfg_training.MODEL_DEFAULT_COMMISSION_PERC_TRAING),
            float(self.cfg_training.DEFAULT_SLIPPAGE),
        )

    start_iter = int(getattr(self.cfg_training, "FRICTION_ANNEAL_START_ITER", 1500))
    end_iter = int(getattr(self.cfg_training, "FRICTION_ANNEAL_END_ITER", 6000))
    comm_start = float(getattr(self.cfg_training, "COMMISSION_START", 0.0001))
    comm_target = float(getattr(self.cfg_training, "COMMISSION_TARGET", 0.0005))
    slip_start = float(getattr(self.cfg_training, "SLIPPAGE_START", 0.0001))
    slip_target = float(getattr(self.cfg_training, "SLIPPAGE_TARGET", 0.0005))
    sched = str(getattr(self.cfg_training, "FRICTION_SCHEDULE", "cosine")).lower()

    if iteration <= start_iter:
        prog = 0.0
    elif iteration >= end_iter:
        prog = 1.0
    else:
        span = max(1, end_iter - start_iter)
        if sched == "linear":
            prog = (iteration - start_iter) / span
        else:  # cosine
            prog = 0.5 * (1.0 - np.cos(np.pi * (iteration - start_iter) / span))

    cur_comm = float(comm_start + (comm_target - comm_start) * prog)
    cur_slip = float(slip_start + (slip_target - slip_start) * prog)
    return cur_comm, cur_slip
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
../bin/python -m unittest tests/test_ppo2_friction_annealing.py
```

### 4.2 驗收測試查核清單 (Acceptance Verification Checklist)
- [x] **1. 排程數學正確性 (Mathematical Correctness)**:
  - [x] $t \le T_{\text{start}}$ 時，費率恆等於 `COMMISSION_START` / `SLIPPAGE_START` (0.0001)。
  - [x] $t \ge T_{\text{end}}$ 時，費率恆等於 `COMMISSION_TARGET` / `SLIPPAGE_TARGET` (0.0005)。
  - [x] 中間階段平滑單調遞增，且在中點處 Cosine 排程正好到達 $\frac{f_{\text{start}} + f_{\text{target}}}{2} = 0.0003$。
- [x] **2. 多進程通訊正確性 (IPC & Multi-Process Consistency)**:
  - [x] 呼叫 `SubprocVecEnv.set_friction()` 後，所有 Worker 進程的 `env.state.commission_perc` 與 `env.state.default_slippage` 均正確更新。
- [x] **3. 回退與相容性 (Backward Compatibility)**:
  - [x] 當 `ENABLE_FRICTION_ANNEALING: false` 時，回傳值保持為靜態配置，不拋出任何異常。
- [x] **4. 訓練迴圈無崩潰運行 (Runtime Smoke Test)**:
  - [x] 模擬迭代更新與 `train()` 流程正常，日誌輸出無例外發生。
