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


class TemporalAttentionPool(nn.Module):
    """
    Multi-horizon temporal pooling module for sequential financial RL.
    Extracts three complementary representations from Mamba output H of shape [B, L, D]:
      1. h_last: Causal final hidden state H[:, -1, :] (immediate market state at decision step T).
      2. h_short: Mean-pooled short-term window H[:, -K:, :] (recent micro-trend & momentum).
      3. h_attn: Query-guided cross-time attention over all L steps using h_last as query
                 (captures historical support/resistance pivots and regime shifts).
    """

    def __init__(self, hidden_size: int, short_window: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.short_window = short_window
        self.scale = 1.0 / math.sqrt(hidden_size)

        self.q_proj = init_(nn.Linear(hidden_size, hidden_size), std=1.0)
        self.k_proj = init_(nn.Linear(hidden_size, hidden_size), std=1.0)
        self.v_proj = init_(nn.Linear(hidden_size, hidden_size), std=1.0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, L, D]
        h_last = h[:, -1, :]  # [B, D]

        k_win = min(self.short_window, h.size(1))
        h_short = h[:, -k_win:, :].mean(dim=1)  # [B, D]

        # Query-guided temporal attention: h_last queries the full sequence H
        q = self.q_proj(h_last).unsqueeze(1)  # [B, 1, D]
        k = self.k_proj(h)  # [B, L, D]
        v = self.v_proj(h)  # [B, L, D]

        attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B, 1, L]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 1, L]
        h_attn = torch.bmm(attn_weights, v).squeeze(1)  # [B, D]

        return torch.cat([h_last, h_short, h_attn], dim=-1)  # [B, 3 * D]


class MambaContinuousActorCritic(nn.Module):
    """
    Mamba2 Continuous Action Actor-Critic Network for Financial Reinforcement Learning:
      - Decoupled Dual-Stream Input:
          * Market stream (first d_model - portfolio_dim features) normalized via DAIN_Layer.
          * Portfolio stream (last portfolio_dim features: position, unrealized_return, trade_bar/48)
            bypasses DAIN to preserve real account state signals.
      - Gated Temporal-Market Fusion: SineActivation (Time2Vec) + Sigmoid Gate + GELU embedding.
      - Temporal Backbone: MixerModel (Mamba / Mamba2 SSM, deterministic without dropout).
      - Multi-Horizon Temporal Pooling + Late Portfolio Skip Connection:
          * TemporalAttentionPool compresses [B, L, D] -> [B, 3*D] with O(D^2) parameters.
          * Current-step portfolio context is encoded into [B, portfolio_emb_dim] and concatenated
            directly into the Actor-Critic trunk for unattenuated position & PnL awareness.
      - Orthogonal-Initialized Compact Heads:
          * Critic head outputs scalar state value V(s) (output gain=1.0).
          * Actor head outputs continuous action mean mu via pure linear projection (output gain=0.01, no Tanh).
          * Clamped learnable log_std ensures numerical stability.
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
        mode: str = "full",
        init_log_std: float = 0.0,
        portfolio_dim: int = 3,
        ssm_cfg: Optional[dict] = None,
        moe_cfg: Optional[dict] = None,
        d_intermediate: Optional[int] = None,
        short_window: int = 16,
        portfolio_emb_dim: Optional[int] = None,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
    ):
        super().__init__()
        assert d_model > portfolio_dim, (
            f"d_model ({d_model}) must be greater than portfolio_dim ({portfolio_dim})"
        )
        self.seq_dim = seq_dim
        self.hidden_size = hidden_size
        self.portfolio_dim = portfolio_dim
        self.market_dim = d_model - self.portfolio_dim
        self.log_std_min = log_std_min
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        assert self.log_std_min < self.log_std_max, (
            f"log_std_min ({self.log_std_min}) must be strictly less than log_std_max ({self.log_std_max})"
        )
        assert self.log_std_min <= init_log_std <= self.log_std_max, (
            f"init_log_std ({init_log_std}) must lie within [{self.log_std_min}, {self.log_std_max}]"
        )

        self.time_embedding = SineActivation(
            in_features=time_features_in, out_features=time_features_out
        )
        self.dean = DAIN_Layer(mode=mode, input_dim=self.market_dim)
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

        # When d_intermediate is None, GatedMLP automatically scales to 8/3 * hidden_size (multiple of 128)
        self.mixer = MixerModel(
            d_model=hidden_size,
            n_layer=nlayers,
            d_intermediate=d_intermediate,
            dropout=None,
            ssm_cfg=ssm_cfg,
            moe_cfg=moe_cfg,
        )

        # Multi-horizon temporal pooling: [B, L, D] -> [B, 3 * D]
        self.temporal_pool = TemporalAttentionPool(
            hidden_size=hidden_size, short_window=short_window
        )

        # Late-fusion portfolio state encoder for direct position/PnL/duration awareness
        gain_hid = math.sqrt(2)
        p_emb_dim = (
            portfolio_emb_dim
            if portfolio_emb_dim is not None
            else max(32, hidden_size // 8)
        )
        if self.portfolio_dim > 0:
            self.portfolio_encoder = nn.Sequential(
                init_(nn.Linear(self.portfolio_dim, p_emb_dim), std=gain_hid),
                nn.LayerNorm(p_emb_dim),
                nn.SiLU(),
            )
            trunk_dim = hidden_size * 3 + p_emb_dim
        else:
            self.portfolio_encoder = None
            trunk_dim = hidden_size * 3

        self.trunk_norm = nn.LayerNorm(trunk_dim)

        # Proportional Actor & Critic hidden dimensions (256 -> 128 at D=96; 512 -> 256 at D=512)
        head_h1 = max(256, hidden_size)
        head_h2 = max(128, hidden_size // 2)

        # Critic Head (estimates state value V(s))
        self.fc_val = nn.Sequential(
            init_(nn.Linear(trunk_dim, head_h1), std=gain_hid),
            nn.LayerNorm(head_h1),
            nn.SiLU(),
            init_(nn.Linear(head_h1, head_h2), std=gain_hid),
            nn.LayerNorm(head_h2),
            nn.SiLU(),
            init_(nn.Linear(head_h2, 1), std=1.0),
        )

        # Actor Head (outputs continuous action mean mu via unconstrained linear projection, aligned with OpenAI Baselines)
        self.fc_actor = nn.Sequential(
            init_(nn.Linear(trunk_dim, head_h1), std=gain_hid),
            nn.LayerNorm(head_h1),
            nn.SiLU(),
            init_(nn.Linear(head_h1, head_h2), std=gain_hid),
            nn.LayerNorm(head_h2),
            nn.SiLU(),
            init_(nn.Linear(head_h2, action_dim), std=0.01),
        )

        # State-independent learnable log_std parameter
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

    def forward(self, states: torch.Tensor, time_states: torch.Tensor):
        # 1. Extract time features and decoupled market/portfolio features
        time_emb = self.time_embedding(time_states)
        time_proj = self.time_emb_projection(time_emb)

        # Market stream (only first market_dim features enter DAIN normalization)
        market_raw = states[:, :, : self.market_dim].transpose(1, 2)
        market_norm = self.dean(market_raw).transpose(1, 2)

        # Portfolio stream (last portfolio_dim features bypass DAIN)
        portfolio_raw = states[:, :, self.market_dim :]
        combined_states = torch.cat([market_norm, portfolio_raw], dim=-1)
        market_emb = self.market_embedding(combined_states)

        # 2. Gated feature fusion
        gate = self.gate_layer(torch.cat([market_emb, time_proj], dim=-1))
        fused = self.feature_embedding(gate * market_emb + (1 - gate) * time_proj)

        # 3. Mamba2 temporal backbone
        out, _ = self.mixer(fused)

        # 4. Multi-horizon temporal pooling + late-fusion portfolio skip connection
        temporal_feat = self.temporal_pool(out)  # [B, 3 * hidden_size]
        if self.portfolio_encoder is not None:
            current_portfolio = portfolio_raw[:, -1, :]  # [B, portfolio_dim]
            portfolio_feat = self.portfolio_encoder(current_portfolio)  # [B, portfolio_emb_dim]
            trunk = torch.cat([temporal_feat, portfolio_feat], dim=-1)
        else:
            trunk = temporal_feat

        trunk = self.trunk_norm(trunk)

        # 5. Actor & Critic head outputs
        value = self.fc_val(trunk).squeeze(-1)  # [B]
        mu = self.fc_actor(trunk)  # [B, action_dim]
        clamped_log_std = torch.clamp(
            self.log_std, self.log_std_min, self.log_std_max
        )
        std = clamped_log_std.exp().expand_as(mu)  # [B, action_dim]

        return mu, std, value


@dataclass
class AgentOutput:
    """Output container for PPO2Agent forward interaction."""

    action: torch.Tensor  # Pre-squash latent Gaussian sample u in R, stored in RolloutBuffer
    value: torch.Tensor  # State value estimate V(s)
    log_prob: torch.Tensor  # Exact squashed Change-of-Variables log-density log pi(a | s)
    clipped_action: Optional[torch.Tensor] = None  # Bounded squashed action a in (low, high) for env execution


