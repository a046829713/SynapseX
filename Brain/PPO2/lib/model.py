import math
from dataclasses import dataclass
from typing import Sequence, Tuple, List
import time
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from gymnasium.spaces import Tuple as TupleSpace, Discrete, Box
from Brain.Common.dain import DAIN_Layer
from Brain.Common.model_components import SineActivation
from Brain.Common.ssm_tool import MixerModel

# policy gradient
# \nabla_{\theta} \bar{R}_{\theta} = \nabla_{\theta}(\frac{1}{N}\sum_{n=1}^{N} \sum_{t=1}^{T_{n}} R(\tau)\log
# p_{\theta}(s_{t+1}\mid s_t,a_t))


# --- util: running mean / std ------------------------------------------------
class RunningMeanStd(nn.Module):
    """
    線上估計輸入張量的逐維 mean / std，與 OpenAI Baselines 相同介面。
    調用 `update(x)` 後即可用屬性 `.mean`、`.var`、`.std`。
    自動註冊為 buffer，`model.to(device)` 時會一起搬到 GPU。
    """

    def __init__(self, shape, epsilon: float = 1e-4):
        """
        Args:
            shape (tuple): 單一觀測值的形狀，例如 (4,) 或 (3, 84, 84)
            epsilon (float): 初始計數，避免剛開始方差為 0
        """
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): 形狀 (batch, *shape) 的觀測批量，
                        必須與 mean / var 在同一個 device。
        """
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.size(0), device=x.device, dtype=self.count.dtype)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        # 就地更新（不改動計算圖）
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(tot_count)

    @property
    def std(self):
        return torch.sqrt(self.var + 1e-8)


class MlpBackbone(nn.Module):
    def __init__(self, in_dim: int, hid: int, num_layers: int):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(num_layers):
            layers += [init_(nn.Linear(last, hid)), nn.Tanh()]
            last = hid
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class ActOut:
    action: torch.Tensor  # 已抽樣 action（或 arg-max）
    log_prob: torch.Tensor  # log π(a|s)
    value: torch.Tensor  # V(s)
    # dist: torch.distributions.Categorical


class MlpPolicy(nn.Module):
    """
    PyTorch 版本；同樣支援 Box 與 Discrete 動作空間

    """

    def __init__(
        self,
        ob_space: gym.Space,
        ac_space: gym.Space,
        hid_size: int = 64,
        num_hid_layers: int = 2,
        gaussian_fixed_var: bool = True,
        obs_norm: bool = True,
    ):
        super().__init__()

        assert isinstance(ob_space, gym.spaces.Box), "目前僅支援 Box observation"
        self.ob_shape = ob_space.shape
        self.ac_space = ac_space
        self.hid_size = hid_size

        # ╭─ observation normalisation ─╮
        self.obs_norm = obs_norm
        if obs_norm:
            self.ob_rms = RunningMeanStd(shape=self.ob_shape)

        # ╭─ backbone shared by actor / critic ─╮
        self.backbone = MlpBackbone(
            int(torch.prod(torch.tensor(self.ob_shape))), hid_size, num_hid_layers
        )

        # ╭─ critic head ─╮
        self.v_head = init_(nn.Linear(hid_size, 1))

        # ╭─ actor head ─╮
        if isinstance(ac_space, gym.spaces.Discrete):
            self.is_continuous = False
            self.logits = init_(nn.Linear(hid_size, ac_space.n), 0.01)
        elif isinstance(ac_space, gym.spaces.Box):
            self.is_continuous = True
            act_dim = ac_space.shape[0]
            self.mu = init_(nn.Linear(hid_size, act_dim), 0.01)
            if gaussian_fixed_var:
                self.log_std = nn.Parameter(torch.zeros(act_dim))
            else:
                self.log_std_head = init_(nn.Linear(hid_size, act_dim), 0.01)
        else:
            raise TypeError("Unsupported action space")

    # --------------------------------------------------------------------- #
    # public API，與 baselines MlpPolicy 對齊
    # --------------------------------------------------------------------- #
    def act(
        self, stochastic: bool, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """單一步驟推論（像 Baselines 一樣回傳 (action, value)）"""
        with torch.no_grad():
            out = self.forward(obs, stochastic=stochastic)
            return out.action.squeeze(0), out.value.squeeze(0)

    def get_variables(self) -> List[torch.nn.Parameter]:
        return list(self.parameters())

    def get_trainable_variables(self) -> List[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def get_initial_state(self):
        return []  # 保留接口，與 Baselines 對齊

    # --------------------------------------------------------------------- #
    # core forward
    # --------------------------------------------------------------------- #
    def _preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_norm:
            self.ob_rms.update(obs)  # 線上更新統計量
            obs = torch.clamp((obs - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        return obs

    def forward(
        self, obs: torch.Tensor, stochastic: bool = True, needDist: bool = False
    ) -> ActOut:

        if obs.dim() == len(self.ob_shape):  # (obs_dim,) → (1, obs_dim)
            obs = obs.unsqueeze(0)

        obs = obs.to(next(self.parameters()).device, dtype=torch.float32)

        obs = self._preprocess_obs(obs)

        flat_obs = obs.view(obs.size(0), -1)

        h = self.backbone(flat_obs)

        # critic
        value = self.v_head(h).squeeze(-1)

        # actor
        if self.is_continuous:
            mu = self.mu(h)
            log_std = self.log_std if hasattr(self, "log_std") else self.log_std_head(h)
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
        else:
            logits = self.logits(h)
            dist = torch.distributions.Categorical(logits=logits)

        if stochastic:
            action = dist.sample()
        else:
            action = dist.mode if hasattr(dist, "mode") else dist.probs.argmax(dim=-1)

        log_prob = dist.log_prob(action)

        print(action)
        time.sleep(100)
        # 對 Box 動作空間，把 action 壓回合法範圍
        if self.is_continuous:
            action = torch.tanh(action)  # [-1, 1]
            lo, hi = self.ac_space.low, self.ac_space.high
            action = lo + (0.5 * (action + 1.0) * (hi - lo))  # 映射到動作空間實際範圍

        return ActOut(action=action, log_prob=log_prob, value=value)

    def evaluate_actions(self, obs: torch.Tensor, act: torch.Tensor):
        """
        回傳 (log_prob, entropy, value)，
        讓 PPO 更新一次 forward 就拿到三樣東西。
        """
        if obs.dim() == len(self.ob_shape):
            obs = obs.unsqueeze(0)
        obs = obs.to(next(self.parameters()).device, dtype=torch.float32)
        obs = self._preprocess_obs(obs)
        flat_obs = obs.view(obs.size(0), -1)

        h = self.backbone(flat_obs)
        value = self.v_head(h).squeeze(-1)  #  critic head ---------

        # actor head -------------------------------------------------
        if self.is_continuous:
            mu = self.mu(h)
            std = (
                self.log_std if hasattr(self, "log_std") else self.log_std_head(h)
            ).exp()
            dist = torch.distributions.Normal(mu, std)
            # 連續動作：熵要把各維 sum 起來
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits = self.logits(h)
            dist = torch.distributions.Categorical(logits=logits)
            entropy = dist.entropy()  # shape = (batch,)

        logp = dist.log_prob(act)
        return logp, entropy, value


# --- policy net --------------------------------------------------------------
def init_(layer, std=1.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, 0)
    return layer





class MambaContinuousActorCritic(nn.Module):
    """
    Mamba 連續動作 Actor-Critic 網路架構：
    - 雙流特徵編碼：DAIN_Layer (市場數據) + SineActivation (時間特徵)
    - 自適應門控融合：Sigmoid gate + GELU feature embedding
    - 時序骨幹：MixerModel (Mamba SSM)
    - Critic 頭：輸出狀態期望價值 V(s)
    - Actor 頭：輸出連續動作均值 mu in [-1.0, 1.0] (經由 Tanh)
    - 獨立可學習 log_std：連續動作高斯分佈標準差
    """

    def __init__(
        self,
        d_model: int,
        time_features_in: int,
        action_dim: int = 1,
        seq_dim: int = 300,
        hidden_size: int = 96,
        nlayers: int = 2,
        time_features_out: int = 32,
        dropout: float = 0.1,
        mode: str = "full",
        init_log_std: float = -0.5,
        ssm_cfg: Optional[dict] = None,
        moe_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.time_embedding = SineActivation(
            in_features=time_features_in, out_features=time_features_out
        )
        self.dean = DAIN_Layer(mode=mode, input_dim=d_model)
        self.market_embedding = nn.Linear(d_model, hidden_size)
        self.time_emb_projection = nn.Linear(time_features_out, hidden_size)

        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )
        self.feature_embedding = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
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
            nn.Linear(256, 1),
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
            nn.Tanh(),
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
        value = self.fc_val(flat).squeeze(-1)  # [B]
        mu = self.fc_actor(flat)  # [B, action_dim]
        std = self.log_std.exp().expand_as(mu)  # [B, action_dim]

        return mu, std, value


@dataclass
class AgentOutput:
    """Agent在與環境互動時的輸出。"""

    action: torch.Tensor
    value: torch.Tensor
    log_prob: torch.Tensor

