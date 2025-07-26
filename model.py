import math
from dataclasses import dataclass
from typing import Sequence, Tuple, List
import time
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional




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
        self.register_buffer("var",  torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): 形狀 (batch, *shape) 的觀測批量，
                        必須與 mean / var 在同一個 device。
        """
        batch_mean = x.mean(dim=0)
        batch_var  = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.size(0), device=x.device, dtype=self.count.dtype)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2  = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        # 就地更新（不改動計算圖）
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(tot_count)

    @property
    def std(self):
        return torch.sqrt(self.var + 1e-8)


# --- policy net --------------------------------------------------------------
def init_(layer, std=1.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, 0)
    return layer


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
    action: torch.Tensor      # 已抽樣 action（或 arg-max）
    log_prob: torch.Tensor    # log π(a|s)
    value: torch.Tensor       # V(s)
    dist: torch.distributions.Categorical
    
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
        self.backbone = MlpBackbone(int(torch.prod(torch.tensor(self.ob_shape))), hid_size, num_hid_layers)

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
    def act(self, stochastic: bool, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def forward(self, obs: torch.Tensor, stochastic: bool = True, needDist: bool = False) -> ActOut:

        if obs.dim() == len(self.ob_shape):       # (obs_dim,) → (1, obs_dim)
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
        

        # 對 Box 動作空間，把 action 壓回合法範圍
        if self.is_continuous:
            action = torch.tanh(action)                       # [-1, 1]
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
        value = self.v_head(h).squeeze(-1)     #  critic head ---------

        # actor head -------------------------------------------------
        if self.is_continuous:
            mu  = self.mu(h)
            std = (self.log_std if hasattr(self, "log_std")
                   else self.log_std_head(h)).exp()
            dist = torch.distributions.Normal(mu, std)
            # 連續動作：熵要把各維 sum 起來
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits = self.logits(h)
            dist   = torch.distributions.Categorical(logits=logits)
            entropy = dist.entropy()            # shape = (batch,)

        logp = dist.log_prob(act)
        return logp, entropy, value
    


class PPOAgent():
    def __init__(self):
        """
            This Agent should generate .
        """
        self.model = HybridMlpPolicy

    




# 我們先更新 ActOut dataclass，讓它能容納複合動作
@dataclass
class HybridActOut:
    # 動作現在是一個元組 (離散動作, 連續動作)
    action: Tuple[torch.Tensor, torch.Tensor]
    # 總的 log_prob
    log_prob: torch.Tensor
    # 價值函數輸出
    value: torch.Tensor
    # 為了方便，也把兩個分佈都存起來
    dist: Tuple[torch.distributions.Distribution, torch.distributions.Distribution]

class HybridMlpPolicy(nn.Module):
    """
    修改後的 MlpPolicy，支援混合動作空間。
    假設動作空間是 gym.spaces.Tuple((gym.spaces.Discrete(N), gym.spaces.Box(...)))
    """
    def __init__(
        self,
        ob_space: gym.Space,
        ac_space: gym.Space,  # <--- MODIFIED: 這裡我們期望傳入一個 Tuple Space
        hid_size: int = 64,
        num_hid_layers: int = 2,
        obs_norm: bool = True,
    ):
        super().__init__()

        # --- 驗證輸入的空間是否正確 ---
        assert isinstance(ac_space, gym.spaces.Tuple), "混合動作空間必須使用 gym.spaces.Tuple"
        assert isinstance(ac_space[0], gym.spaces.Discrete), "Tuple的第一個元素必須是 Discrete"
        assert isinstance(ac_space[1], gym.spaces.Box), "Tuple的第二個元素必須是 Box"
        
        self.discrete_space = ac_space[0]
        self.continuous_space = ac_space[1]
        
        self.ob_shape = ob_space.shape
        self.ac_space = ac_space
        self.hid_size = hid_size

        # ╭─ observation normalisation (與原版相同) ─╮
        self.obs_norm = obs_norm
        if obs_norm:
            self.ob_rms = RunningMeanStd(shape=self.ob_shape)

        # ╭─ backbone shared by actor / critic (與原版相同) ─╮
        self.backbone = MlpBackbone(int(torch.prod(torch.tensor(self.ob_shape))), hid_size, num_hid_layers)

        # ╭─ critic head (與原版相同) ─╮
        self.v_head = init_(nn.Linear(hid_size, 1))

        # ╭─ actor heads (核心修改處) ─╮
        # 我們現在有兩個並行的 actor head
        
        # 1. 離散決策頭 (Discrete Head)
        self.discrete_head = init_(nn.Linear(hid_size, self.discrete_space.n), 0.01)
        
        # 2. 連續參數頭 (Continuous Head)
        act_dim = self.continuous_space.shape[0]
        self.continuous_mu_head = init_(nn.Linear(hid_size, act_dim), 0.01)
        # 為了簡單，我們繼續使用固定的標準差，這在實務中很常見
        self.continuous_log_std = nn.Parameter(torch.zeros(act_dim))

    def _preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # (與原版相同)
        if self.obs_norm:
            self.ob_rms.update(obs)
            obs = torch.clamp((obs - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        return obs

    def forward(self, obs: torch.Tensor, stochastic: bool = True) -> HybridActOut:
        # (核心修改處)
        if obs.dim() == len(self.ob_shape):
            obs = obs.unsqueeze(0)
        
        obs = obs.to(next(self.parameters()).device, dtype=torch.float32)
        obs = self._preprocess_obs(obs)
        flat_obs = obs.view(obs.size(0), -1)

        # 共享的 backbone
        h = self.backbone(flat_obs)

        # Critic
        value = self.v_head(h).squeeze(-1)

        # === Actor Heads ===
        # 1. 離散部分
        discrete_logits = self.discrete_head(h)
        discrete_dist = torch.distributions.Categorical(logits=discrete_logits)

        # 2. 連續部分
        continuous_mu = self.continuous_mu_head(h)
        continuous_std = self.continuous_log_std.exp()
        continuous_dist = torch.distributions.Normal(continuous_mu, continuous_std)

        # === 採樣動作 ===
        if stochastic:
            discrete_action = discrete_dist.sample()
            # 注意：這裡的 continuous_action 是從 Normal 分佈中採樣的原始值
            # 稍後需要映射到 [0, 1] 範圍
            raw_continuous_action = continuous_dist.sample()
        else: # 確定性策略 (用於評估)
            discrete_action = discrete_dist.probs.argmax(dim=-1)
            raw_continuous_action = continuous_mu

        # === 計算總的 log_prob ===
        # log(P(a_total)) = log(P(a_discrete)) + log(P(a_continuous))
        log_prob_discrete = discrete_dist.log_prob(discrete_action)
        log_prob_continuous = continuous_dist.log_prob(raw_continuous_action).sum(dim=-1) # sum over action dimensions
        total_log_prob = log_prob_discrete + log_prob_continuous

        # === 格式化輸出動作 ===
        # 將連續動作映射到 [0, 1] 的有效範圍
        # 假設您的 Box 空間是 [0, 1]
        final_continuous_action = torch.tanh(raw_continuous_action) # 壓到 [-1, 1]
        final_continuous_action = (final_continuous_action + 1) / 2 # 映射到 [0, 1]

        composite_action = (discrete_action, final_continuous_action)

        return HybridActOut(
            action=composite_action,
            log_prob=total_log_prob,
            value=value,
            dist=(discrete_dist, continuous_dist)
        )

    def evaluate_actions(self, obs: torch.Tensor, act: Tuple[torch.Tensor, torch.Tensor]):
        # (核心修改處)
        # act 是一個元組 (discrete_actions, continuous_actions)
        discrete_act, continuous_act = act

        if obs.dim() == len(self.ob_shape):
            obs = obs.unsqueeze(0)
        obs = obs.to(next(self.parameters()).device, dtype=torch.float32)
        obs = self._preprocess_obs(obs)
        flat_obs = obs.view(obs.size(0), -1)

        h = self.backbone(flat_obs)
        value = self.v_head(h).squeeze(-1)

        # === 重建分佈並計算 log_prob 和 entropy ===
        # 1. 離散部分
        discrete_logits = self.discrete_head(h)
        discrete_dist = torch.distributions.Categorical(logits=discrete_logits)
        log_prob_discrete = discrete_dist.log_prob(discrete_act)
        entropy_discrete = discrete_dist.entropy()

        # 2. 連續部分
        # 注意：我們需要從你提供的 [0,1] 動作逆向工程回 raw_action
        # 這是 tanh 變換的逆操作
        # final = (tanh(raw) + 1) / 2  => tanh(raw) = 2*final - 1 => raw = atanh(2*final - 1)
        # 為了避免邊界值 inf，我們用 clamp 做一個保護
        clipped_continuous_act = torch.clamp(continuous_act, 1e-6, 1.0 - 1e-6)
        raw_continuous_act = torch.atanh(2 * clipped_continuous_act - 1)

        continuous_mu = self.continuous_mu_head(h)
        continuous_std = self.continuous_log_std.exp()
        continuous_dist = torch.distributions.Normal(continuous_mu, continuous_std)
        log_prob_continuous = continuous_dist.log_prob(raw_continuous_act).sum(dim=-1)
        entropy_continuous = continuous_dist.entropy().sum(dim=-1)

        # === 合併結果 ===
        total_log_prob = log_prob_discrete + log_prob_continuous
        total_entropy = entropy_discrete + entropy_continuous

        return total_log_prob, total_entropy, value
    


