# Specification: PPO2 Continuous Action Trading Architecture & Training Refactoring (Mamba-based Actor-Critic)

## 📋 Index / 目錄
1. [Background & Motivation / 背景與重構目標](#1-background--motivation--背景與重構目標)
2. [Reference Analysis: mambaDuelingModel / 參考 DQN mambaDuelingModel 架構深度分析](#2-reference-analysis-mambaduelingmodel--參考-dqn-mambaduelingmodel-架構深度分析)
3. [Architecture & Modular Division / 系統模組化架構與職責切分](#3-architecture--modular-division--系統模組化架構與職責切分)
4. [Model Layer: MambaContinuousActorCritic / 模型層設計 (Brain/PPO2/lib/model.py)](#4-model-layer-mambacontinuousactorcritic--模型層設計-brainppo2libmodelpy)
5. [Agent Layer: PPO2Agent / 代理層重構 (Brain/PPO2/lib/Agent.py)](#5-agent-layer-ppo2agent--代理層重構-brainppo2libagentpy)
6. [Experience Layer: RolloutBuffer & Mini-batch / 經驗回放層 (Brain/PPO2/lib/experience.py)](#6-experience-layer-rolloutbuffer--mini-batch--經驗回放層-brainppo2libexperiencepy)
7. [Training Script & Optimization / 訓練排程與優化器 (PPO2_rl_train.py)](#7-training-script--optimization--訓練排程與優化器-ppo2_rl_trainpy)
8. [Proposed File Modifications / 預計改動檔案清單](#8-proposed-file-modifications--預計改動檔案清單)
9. [Implementation Checklist / 實作檢核清單](#9-implementation-checklist--實作檢核清單)
10. [Verification & Test Plan / 驗證與測試計畫](#10-verification--test-plan--驗證與測試計畫)

---

## 1. Background & Motivation / 背景與重構目標

在先前的重構中，環境端 [Brain/PPO2/lib/environment.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/environment.py) 已全面升級為**純連續動作空間 $[-1.0, 1.0]$**（參考 [spec/issue_PPO2_continuous_environment.md](file:///home/b0457812963/Mamba3RL/SynapseX/spec/issue_PPO2_continuous_environment.md)）：
- 動作 $a_t \in [-1.0, 1.0]$ 代表目標投資組合部位權重（正做多、負放空、0 平倉）。
- 觀測空間為 `spaces.Dict({"states": Box(...), "time_states": Box(...)})`。
- `step(action)` 接收單一連續浮點數，回傳 4 元組 `(obs, reward, done, info)`。

用戶釐清之核心原則：
> [!NOTE]
> **「先不要改動代碼」** 是指在方案討論與規劃確認完成前，**絕不提前擅自修改工作區代碼**，確保設計思路 100% 對齊。
> 實作時應遵循良好的工程架構進行**合理的模組化分工**（模型在 `model.py`、代理在 `Agent.py`、經驗在 `experience.py`、訓練在 `PPO2_rl_train.py`），維持專案的高內聚與低耦合。

---

## 2. Reference Analysis: mambaDuelingModel / 參考 DQN mambaDuelingModel 架構深度分析

在 [Brain/DQN/lib/model.py:L407-509](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/DQN/lib/model.py#L407-L509) 中，`mambaDuelingModel` 具備專門針對量化時序資料設計的優秀雙流門控架構：

### 2.1 雙流特徵編碼與門控融合 (Dual-stream Embedding & Gating)
```
  市場數據 states [B, L, d_model]         時間特徵 time_states [B, L, time_in]
           │                                          │
    DAIN_Layer (自適應歸一化)                  SineActivation (Time2Vec)
           │                                          │
    Linear (market_embedding)                 Linear (time_emb_projection)
           │                                          │
           └──────────────────┬───────────────────────┘
                              │
               Concat + Linear + Sigmoid (Gate)
                              │
               Fused = Gate * Market + (1 - Gate) * Time
                              │
                     Linear + GELU (Feature)
                              │
                     MixerModel (Mamba/Mamba2)
                              │
                     Flatten [B, L * hidden_size]
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
      Critic Head (fc_val)            Actor Head (fc_actor)
              │                               │
        Linear + LN + ReLU              Linear + LN + ReLU
        Linear + LN + ReLU              Linear + LN + ReLU
        Linear(..., 1)                  Linear(..., 1) + Tanh() -> mu in [-1, 1]
              │                               │
       State Value V(s)             + Learnable log_std -> Normal(mu, std)
```

1. **市場數據流 (Market Stream)**：
   - 透過 `DAIN_Layer` (Deep Adaptive Input Normalization) 進行通道維度的自適應縮放、均值偏移與門控加權，解決金融數據非平穩性問題。
   - 經線性層投影為 `hidden_size`。
2. **時間特徵流 (Time Stream)**：
   - 透過 `SineActivation` (Time2Vec) 提取多尺度週期性時間相位特徵。
   - 經線性投影至 `hidden_size`。
3. **自適應門控融合 (Adaptive Gated Fusion)**：
   - 計算門控係數 $g = \text{Sigmoid}(W [\text{market}; \text{time}])$。
   - 凸組合融合：$e = g \odot \text{market} + (1 - g) \odot \text{time}$。
4. **時序骨幹 (MixerModel / SSM)**：
   - 調用 `Brain.Common.ssm_tool.MixerModel`，支援 Mamba 1 / Mamba 2 狀態空間模型，高效捕捉長度達 $L=300$ 根 K 棒的長程依賴。

### 2.2 DQN Dueling vs PPO Continuous Actor-Critic 映射關係

| 組件 | DQN `mambaDuelingModel` | PPO `MambaContinuousActorCritic` (預計設計) |
| :--- | :--- | :--- |
| **輸入格式** | `src` (市場特徵), `time_tau` (時間特徵) | `states` (市場特徵), `time_states` (時間特徵) |
| **特徵前處理** | `DAIN_Layer` + `SineActivation` + 門控融合 | 相同保留，保證特徵處理能力一致 |
| **時序骨幹** | `MixerModel` (Mamba / Mamba2) | 相同保留，共享骨幹表徵 |
| **Critic 頭部** | `fc_val` 輸出狀態基準價值 $V(s) \in \mathbb{R}^1$ | **保留並對齊**：`fc_val` 輸出狀態期望價值 $V(s)$ |
| **Actor 頭部** | `fc_adv` 輸出各離散動作優勢值 $A(s, a) \in \mathbb{R}^{|A|}$ | **重構為連續策略頭**：輸出連續均值 $\mu \in [-1, 1]$ 與可學習 $\log \sigma$ |
| **輸出內容** | $Q(s, a) = V(s) + (A(s, a) - \bar{A}(s))$ | 動作分佈 $\mathcal{N}(\mu, \sigma)$、採樣動作 $a$、$\log \pi(a|s)$、狀態價值 $V(s)$ |

---

## 3. Architecture & Modular Division / 系統模組化架構與職責切分

為了符合專案整體風格與易維護性，採用模組化架構分工：

```
Brain/PPO2/
├── lib/
│   ├── environment.py    (已完成：連續動作交易環境，action_space=Box(-1, 1), Dict Obs)
│   ├── model.py          (待更新：新增 MambaContinuousActorCritic 連續策略模型)
│   ├── Agent.py          (待重構：PPO2Agent 支援 Box 動作空間、Dict Obs 與動作採樣)
│   └── experience.py     (待強化：RolloutBuffer 新增 mini-batch 生成器)
└── PPO2_rl_train.py      (待重構：主訓練排程器、GAE 計算、PPO 剪裁損失優化、FPS 與 Checkpoint)
```

---

## 4. Model Layer: MambaContinuousActorCritic / 模型層設計 (`Brain/PPO2/lib/model.py`)

在 `Brain/PPO2/lib/model.py` 中新增/重構 `MambaContinuousActorCritic`：

### 4.1 網路類別介面
```python
class MambaContinuousActorCritic(nn.Module):
    def __init__(
        self,
        d_model: int,             # states 特徵維度 (e.g., 22)
        time_features_in: int,    # time_states 特徵維度 (e.g., 8)
        action_dim: int = 1,      # 連續動作維度 (預設 1, 範圍 [-1.0, 1.0])
        seq_dim: int = 300,       # 序列長度 (BARS_COUNT)
        hidden_size: int = 96,    # 隱藏層特徵維度
        nlayers: int = 2,         # Mamba 層數
        time_features_out: int = 32,
        dropout: float = 0.1,
        mode: str = "full",       # DAIN 模式
        init_log_std: float = -0.5, # 初始標準差 (exp(-0.5) ≈ 0.606)
        ssm_cfg: Optional[dict] = None,
        moe_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.time_embedding = SineActivation(in_features=time_features_in, out_features=time_features_out)
        self.dean = DAIN_Layer(mode=mode, input_dim=d_model)
        self.market_embedding = nn.Linear(d_model, hidden_size)
        self.time_emb_projection = nn.Linear(time_features_out, hidden_size)
        
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        self.feature_embedding = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )
        
        self.mixer = MixerModel(
            d_model=hidden_size,
            n_layer=nlayers,
            d_intermediate=256,
            dropout=dropout,
            ssm_cfg=ssm_cfg,
            moe_cfg=moe_cfg,
        )
        
        flat_dim = seq_dim * hidden_size
        
        # Critic 頭 (估計狀態價值 V(s))
        self.fc_val = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # Actor 頭 (輸出連續動作均值 mu，經 Tanh 限制在 [-1.0, 1.0])
        self.fc_actor = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
        # 連續動作 log_std (可學習參數，獨立於狀態)
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

    def forward(self, states: torch.Tensor, time_states: torch.Tensor):
        # 1. 提取時間特徵與市場特徵
        time_emb = self.time_embedding(time_states)
        time_proj = self.time_emb_projection(time_emb)
        
        market_data = states.transpose(1, 2)
        market_data = self.dean(market_data)
        market_data = market_data.transpose(1, 2)
        market_emb = self.market_embedding(market_data)
        
        # 2. 門控融合
        gate = self.gate_layer(torch.cat([market_emb, time_proj], dim=-1))
        fused = self.feature_embedding(gate * market_emb + (1 - gate) * time_proj)
        
        # 3. Mamba 時序提煉
        out, _ = self.mixer(fused)
        flat = out.view(out.size(0), -1)
        
        # 4. 頭部輸出
        value = self.fc_val(flat).squeeze(-1)       # [B]
        mu = self.fc_actor(flat)                    # [B, action_dim]
        std = self.log_std.exp().expand_as(mu)      # [B, action_dim]
        
        return mu, std, value
```

---

## 5. Agent Layer: PPO2Agent / 代理層重構 (`Brain/PPO2/lib/Agent.py`)

移除舊有混合動作驗證，全面適配連續動作空間：

```python
class PPO2Agent:
    def __init__(
        self,
        ob_space: gym.Space,
        ac_space: gym.Space,
        device: torch.device,
        seq_dim: int = 300,
        hidden_size: int = 96,
        nlayers: int = 2,
    ):
        self.device = device
        self.ob_space = ob_space
        self.ac_space = ac_space
        
        self._validate_spaces()
        
        d_model = self.ob_space["states"].shape[1]
        time_features_in = self.ob_space["time_states"].shape[1]
        action_dim = self.ac_space.shape[0]
        
        self.model = MambaContinuousActorCritic(
            d_model=d_model,
            time_features_in=time_features_in,
            action_dim=action_dim,
            seq_dim=seq_dim,
            hidden_size=hidden_size,
            nlayers=nlayers,
        ).to(self.device)

    def _validate_spaces(self):
        assert isinstance(self.ac_space, gym.spaces.Box), "PPO2Agent 動作空間必須為 gym.spaces.Box"
        assert isinstance(self.ob_space, gym.spaces.Dict), "PPO2Agent 觀測空間必須為 gym.spaces.Dict"
        assert "states" in self.ob_space.spaces and "time_states" in self.ob_space.spaces

    def _preprocess_obs(self, obs):
        states = torch.as_tensor(obs["states"], dtype=torch.float32, device=self.device)
        time_states = torch.as_tensor(obs["time_states"], dtype=torch.float32, device=self.device)
        if states.dim() == 2:
            states = states.unsqueeze(0)
            time_states = time_states.unsqueeze(0)
        return states, time_states

    def get_action(self, obs, stochastic: bool = True) -> AgentOutput:
        self.model.eval()
        states, time_states = self._preprocess_obs(obs)
        with torch.no_grad():
            mu, std, value = self.model(states, time_states)
            dist = torch.distributions.Normal(mu, std)
            if stochastic:
                action = dist.sample()
            else:
                action = mu
            
            # 連續動作裁剪至合法邊界 [-1.0, 1.0]
            action = torch.clamp(action, -1.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1)

        return AgentOutput(
            action=action.squeeze(0), # 1D Tensor e.g., shape (1,)
            value=value.squeeze(0),
            log_prob=log_prob.squeeze(0)
        )

    def evaluate_actions(self, states, time_states, actions):
        self.model.train()
        mu, std, value = self.model(states, time_states)
        dist = torch.distributions.Normal(mu, std)
        
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value
```

---

## 6. Experience Layer: RolloutBuffer & Mini-batch / 經驗回放層 (`Brain/PPO2/lib/experience.py`)

在 `Brain/PPO2/lib/experience.py` 中擴充 `get_batches()` 支援打散的 mini-batch 迭代：

```python
class RolloutBuffer:
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        self.buffer = []

    def store(self, *args):
        self.buffer.append(Transition(*args))

    def compute_gae(self, last_value):
        rewards, values, dones = [], [], []
        for t in self.buffer:
            rewards.append(t.reward)
            values.append(t.value.item() if torch.is_tensor(t.value) else float(t.value))
            dones.append(t.done)
        
        values = values + [last_value]
        gae, returns = 0, []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        
        advantages = np.array(returns) - np.array(values[:-1])
        for idx, tr in enumerate(self.buffer):
            self.buffer[idx] = tr._replace(reward=returns[idx], value=advantages[idx])
        return self.buffer

    def get_batches(self, batch_size=64, shuffle=True):
        states = torch.stack([torch.as_tensor(t.state["states"], dtype=torch.float32) for t in self.buffer])
        time_states = torch.stack([torch.as_tensor(t.state["time_states"], dtype=torch.float32) for t in self.buffer])
        actions = torch.stack([torch.as_tensor(t.action, dtype=torch.float32) for t in self.buffer])
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        old_log_probs = torch.tensor([t.logp.item() if torch.is_tensor(t.logp) else float(t.logp) for t in self.buffer], dtype=torch.float32)
        returns = torch.tensor([t.reward for t in self.buffer], dtype=torch.float32)
        advantages = torch.tensor([t.value for t in self.buffer], dtype=torch.float32)
        
        # Advantage 標準化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = len(self.buffer)
        indices = np.arange(dataset_size)
        if shuffle:
            np.random.shuffle(indices)
            
        for start_idx in range(0, dataset_size, batch_size):
            b_idx = indices[start_idx : start_idx + batch_size]
            yield (
                states[b_idx],
                time_states[b_idx],
                actions[b_idx],
                old_log_probs[b_idx],
                returns[b_idx],
                advantages[b_idx],
            )

    def clear(self):
        self.buffer = []
```

---

## 7. Training Script & Optimization / 訓練排程與優化器 (`PPO2_rl_train.py`)

在 `PPO2_rl_train.py` 中實作核心訓練管理器 `PPO2`：

### 7.1 自適應優化器 (保留 DAIN 隔離與權重衰減過濾)
```python
def build_optimizer(model, lr=3e-4, weight_decay=1e-4, base_lr=1e-4):
    dean_params_ids = set()
    if hasattr(model, 'dean'):
        dean_params_ids.update(id(p) for p in model.dean.parameters())

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in dean_params_ids:
            continue
        if "norm" in name or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
    ]

    if hasattr(model, "dean"):
        param_groups.extend([
            {"params": list(model.dean.mean_layer.parameters()), "lr": base_lr * model.dean.mean_lr, "weight_decay": 0.0},
            {"params": list(model.dean.scaling_layer.parameters()), "lr": base_lr * model.dean.scale_lr, "weight_decay": 0.0},
            {"params": list(model.dean.gating_layer.parameters()), "lr": base_lr * model.dean.gate_lr, "weight_decay": 0.0},
        ])

    return optim.AdamW(param_groups)
```

### 7.2 主採樣與更新邏輯 (Fixed Horizon Rollout)
```python
class PPO2:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._prepare_device()
        self._prepare_symbols()
        self._prepare_env()
        self._prepare_agent()
        self._prepare_optimizer()
        self.buffer = RolloutBuffer(gamma=0.99, lam=0.95)
        self.model_tool = ModelTool()
        
    def train(self):
        obs = self.train_env.reset()
        episode_reward = 0.0
        total_steps = 0
        iteration = 0
        n_steps = getattr(self.cfg.training, "N_STEPS", 1000)
        
        while True:
            # 1. 採集 n_steps 個步數
            for _ in range(n_steps):
                out = self.agent.get_action(obs, stochastic=True)
                action_scalar = float(out.action.cpu().numpy()[0])
                next_obs, reward, done, info = self.train_env.step(action_scalar)
                
                self.buffer.store(obs, out.action.cpu(), out.log_prob.cpu(), reward, next_obs, done, out.value.cpu())
                episode_reward += reward
                total_steps += 1
                obs = next_obs
                
                if done:
                    print(f"[Episode Done] Return: {episode_reward:.4f} | Equity: {info.get('position', 0.0)}")
                    obs = self.train_env.reset()
                    episode_reward = 0.0
            
            # 2. 計算 Bootstrap 價值並求解 GAE
            with torch.no_grad():
                last_out = self.agent.get_action(obs, stochastic=False)
                last_value = last_out.value.item()
            
            self.buffer.compute_gae(last_value)
            
            # 3. PPO 剪裁損失多 Epoch 更新
            self.update_ppo()
            self.buffer.clear()
            iteration += 1
            
            # 4. 定期存檔
            if iteration % 10 == 0:
                self.save_checkpoint(iteration)
```

---

## 8. Proposed File Modifications / 預計改動檔案清單

當我們完成討論並獲得您的確認後，預計按以下層次進行模組化實作：

| 檔案路徑 | 變更性質 | 職責說明 |
| :--- | :--- | :--- |
| [Brain/PPO2/lib/model.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/model.py) | **[MODIFY]** | 新增 `MambaContinuousActorCritic` 模型類別，實現 DAIN + SineActivation + MixerModel + Tanh Actor + Critic 頭 |
| [Brain/PPO2/lib/Agent.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/Agent.py) | **[MODIFY]** | 重構 `PPO2Agent` 支援 `Box(-1, 1)` 動作空間、Dict 觀測值前處理、連續動作採樣與邊界裁剪 |
| [Brain/PPO2/lib/experience.py](file:///home/b0457812963/Mamba3RL/SynapseX/Brain/PPO2/lib/experience.py) | **[MODIFY]** | 為 `RolloutBuffer` 擴充 `get_batches()`，提供打散且標準化優勢值的 mini-batch 迭代器 |
| [PPO2_rl_train.py](file:///home/b0457812963/Mamba3RL/SynapseX/PPO2_rl_train.py) | **[MODIFY]** | 徹底重構主訓練腳本：整合 Hydra 配置、DAIN 自適應 AdamW 優化器、Fixed Horizon 採樣迴圈、多 Epoch PPO 更新與監控存檔 |

---

## 9. Implementation Checklist / 實作檢核清單

- [x] **Phase 1: 模型層實作 (Model Layer - `Brain/PPO2/lib/model.py`)**
  - [x] 導入 `DAIN_Layer`、`SineActivation`、`MixerModel`
  - [x] 實作雙流特徵編碼（市場數據 DAIN + 時間特徵 SineActivation）
  - [x] 實作自適應門控融合層（Gate Layer + GELU Feature Embedding）
  - [x] 整合 Mamba/SSM 時序骨幹（支援 `ssm_cfg`, `moe_cfg`）
  - [x] 實作 Critic 頭部（`fc_val` 輸出狀態期望價值 $V(s) \in \mathbb{R}$）
  - [x] 實作 Actor 連續策略頭部（`fc_actor` 輸出均值 $\mu$，經 $\tanh$ 映射至 $[-1.0, 1.0]$）
  - [x] 加入狀態無關的連續動作可學習參數 $\log \sigma$（初始化為 -0.5）
  - [x] 實作 `forward(states, time_states)` 前向傳播回傳 `(mu, std, value)`

- [x] **Phase 2: 代理層重構 (Agent Layer - `Brain/PPO2/lib/Agent.py`)**
  - [x] 移除舊有 `TupleSpace` 混合動作斷言，改為校驗 `Box(-1.0, 1.0)` 與 `Dict` 觀測空間
  - [x] 實作 `_preprocess_obs` 支援單步字典與批次字典張量轉換
  - [x] 實作 `get_action(obs, stochastic)`：
    - [x] 建立高斯分佈 $\mathcal{N}(\mu, \sigma)$
    - [x] 動作採樣與邊界裁切 (`torch.clamp(action, -1.0, 1.0)`)
    - [x] 計算總對數機率 `log_prob`
    - [x] 包裝並回傳 `AgentOutput(action, value, log_prob)`
  - [x] 實作 `evaluate_actions(states, time_states, actions)`：
    - [x] 計算批次動作對數機率 `log_prob`
    - [x] 計算連續高斯微分熵 `entropy`
    - [x] 取得狀態價值預測 `value`

- [x] **Phase 3: 經驗回放層擴充 (Experience Layer - `Brain/PPO2/lib/experience.py`)**
  - [x] 強化 `compute_gae(last_value)`：相容 Tensor 與 Float 格式，正確維護折現回報與 GAE 優勢值
  - [x] 實作 `get_batches(batch_size, shuffle)` 產生器：
    - [x] 解構 Transition 中的字典觀測值 `{"states", "time_states"}`
    - [x] 實現優勢值標準化：$\hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + 1e-8}$
    - [x] 支援隨機洗牌 (Shuffle) 並產出 Mini-batch 迭代器

- [x] **Phase 4: 訓練主腳本重構 (Training Script - `PPO2_rl_train.py`)**
  - [x] 清理冗餘與錯誤導入，適配 Hydra 配置與設備自動偵測
  - [x] 實作專屬 `build_optimizer` (AdamW)：
    - [x] `DAIN_Layer` 特殊層（`mean_layer`, `scaling_layer`, `gating_layer`）專屬學習率且 0 weight decay
    - [x] LayerNorm 與 Bias 排除 weight decay
    - [x] 其餘參數正常施加 weight decay
  - [x] 實作 `PPO2` 核心排程類別：
    - [x] 環境、模型、代理、優化器與 Buffer 初始化
    - [x] Fixed-Horizon 採樣迴圈（$N = \text{N\_STEPS}$ 步，例如 1000 步）
    - [x] 連續動作轉浮點數傳入 `env.step(action)`，解包 4 元組 `(obs, reward, done, info)`
    - [x] Episode 結束統計與重置處理
    - [x] 截斷步 Critic Bootstrap 價值評估
  - [x] 實作 PPO 多 Epoch 更新流程 (`update_ppo`)：
    - [x] 重要性採樣比率 $r_t(\theta) = \exp(\log \pi_\theta - \log \pi_{\text{old}})$
    - [x] 策略剪裁損失 $L^{\text{CLIP}}$（$\epsilon = 0.2$）
    - [x] 價值損失 $L^{\text{VF}}$（MSE）
    - [x] 連續高斯微分熵探索獎勵 $L^{\text{ENT}}$
    - [x] 梯度裁剪 (`torch.nn.utils.clip_grad_norm_`)
    - [x] 優化器單步更新與清空 Buffer
  - [x] 實作即時度量監控（FPS、Loss、Entropy、Episode Return、持倉淨值趨勢）
  - [x] 實作模型 Checkpoint 定期存檔

- [x] **Phase 5: 驗證與測試 (Verification & Testing)**
  - [x] 單元層級前向傳播驗證 (`test_mamba_actor_critic`)
  - [x] 代理與環境 100 步連續互動測試
  - [x] 執行 1~2 個 Rollout 迭代（共 2000 步）實機訓練測試，確認指標正常印出、無 NaN、存檔成功

---

## 10. Verification & Test Plan / 驗證與測試計畫

在獲得確認並執行程式碼編寫後，依照 Phase 5 的檢核項目執行完整三階段驗證：

1. **單元前向傳播驗證**：
   - 撰寫測試驗證 `MambaContinuousActorCritic` 接受 `states` `[1, 300, 22]` 與 `time_states` `[1, 300, 8]`，輸出均值 $\mu \in [-1.0, 1.0]$ 與標量價值 $V(s)$。
2. **環境對接與 GAE 驗證**：
   - 測試 `PPO2Agent` 與 `TrainingEnv` 連續互動 100 步，驗證 `RolloutBuffer` 記錄正確且 `compute_gae` 順暢產出優勢值與回報。
3. **短期完整訓練執行 (1~2 個 Rollout 迭代)**：
   - 透過 Hydra 載入真實數據執行 2000 步訓練，觀察 Policy Loss、Value Loss、Entropy 與 FPS 輸出正常，無 NaN，Checkpoint 成功產出。
