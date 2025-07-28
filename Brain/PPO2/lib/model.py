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
        
        print(action)
        time.sleep(100)
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
    



# --- policy net --------------------------------------------------------------
def init_(layer, std=1.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, 0)
    return layer
    






class HybridMlpPolicy(nn.Module):
    """
    純模型架構：
    接收扁平化的觀測張量，回傳 actor 的原始輸出和 critic 的價值。
    不包含任何與 gym.space、採樣、log_prob 計算或正規化相關的邏輯。
    """
    def __init__(
        self,
        ob_dim: int,
        discrete_ac_dim: int,
        continuous_ac_dim: int,
        hid_size: int = 64,
        num_hid_layers: int = 2,
    ):
        super().__init__()
        
        # 共享的骨幹網路
        self.backbone = MlpBackbone(ob_dim, hid_size, num_hid_layers)

        # Critic head
        self.v_head = init_(nn.Linear(hid_size, 1))

        # Actor heads
        # 1. 離散動作頭
        self.discrete_head = init_(nn.Linear(hid_size, discrete_ac_dim), 0.01)
        
        # 2. 連續動作頭
        self.continuous_mu_head = init_(nn.Linear(hid_size, continuous_ac_dim), 0.01)
        # 連續動作的 log_std 是一個可學習的參數，獨立於觀測值
        self.continuous_log_std = nn.Parameter(torch.zeros(continuous_ac_dim))

    def forward(self, flat_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        執行前向傳播。
        
        Args:
            flat_obs (torch.Tensor): 經過預處理和扁平化的觀測張量。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - discrete_logits (Tensor): 離散動作的 logits。
            - continuous_mu (Tensor): 連續動作的均值。
            - continuous_log_std (Tensor): 連續動作的 log standard deviation。
            - value (Tensor): 狀態價值。
        """
        h = self.backbone(flat_obs)
        
        value = self.v_head(h).squeeze(-1)
        
        discrete_logits = self.discrete_head(h)
        continuous_mu = self.continuous_mu_head(h)
        
        return discrete_logits, continuous_mu, self.continuous_log_std, value
    

@dataclass
class AgentOutput:
    """Agent在與環境互動時的輸出。"""
    action: Tuple[torch.Tensor, torch.Tensor]
    value: torch.Tensor
    log_prob: torch.Tensor


class PPO2Agent:
    def __init__(
        self,
        ob_space: gym.Space,
        ac_space: gym.Space,
        device: torch.device,
        hid_size: int = 64,
        num_hid_layers: int = 2,
        obs_norm: bool = False,
    ):
        self.device = device
        self.ac_space = ac_space
        self.ob_space = ob_space
        self.obs_norm = obs_norm

        # --- 1. 動作空間驗證 ---
        self._validate_spaces()

        # --- 2. 觀測值標準化 (如果啟用) ---
        if self.obs_norm:
            self.ob_rms = RunningMeanStd(shape=self.ob_space.shape)

        # --- 3. 實例化純模型 ---
        ob_dim = int(torch.prod(torch.tensor(self.ob_space.shape)))
        discrete_ac_dim = self.ac_space[0].n
        continuous_ac_dim = self.ac_space[1].shape[0]
        
        self.model = HybridMlpPolicy(
            ob_dim, discrete_ac_dim, continuous_ac_dim, hid_size, num_hid_layers
        ).to(self.device)



    def _validate_spaces(self):
        """將驗證邏輯移至此處。"""
        assert isinstance(self.ac_space, TupleSpace), "混合動作空間必須使用 gym.spaces.Tuple"
        assert len(self.ac_space.spaces) == 2, "動作元組應包含2個元素"
        assert isinstance(self.ac_space[0], Discrete), "Tuple的第一個元素必須是 Discrete"
        assert isinstance(self.ac_space[1], Box), "Tuple的第二個元素必須是 Box"

    def _preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """處理觀測值：標準化、調整維度、移至 device 並扁平化。"""
        if obs.dim() == len(self.ob_space.shape):
            obs = obs.unsqueeze(0)
        
        obs = obs.to(self.device, dtype=torch.float32)

        if self.obs_norm:
            # 更新 RMS 應在訓練數據收集期間進行，評估時凍結
            # self.ob_rms.update(obs) # 這行通常在收集數據時調用
            obs = torch.clamp((obs - self.ob_rms.mean.to(self.device)) / self.ob_rms.std.to(self.device), -5.0, 5.0)
            
        return obs.view(obs.size(0), -1)

    def _postprocess_continuous_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """將模型輸出的原始連續動作轉換為環境可接受的範圍 [0, 1]。"""
        # Tanh 變換將輸出壓到 [-1, 1]
        action = torch.tanh(raw_action)
        # 縮放和平移到 [0, 1]
        return (action + 1) / 2.0

    def _unprocess_continuous_action(self, processed_action: torch.Tensor) -> torch.Tensor:
        """將 [0, 1] 範圍的動作逆轉換為原始的、未經 Tanh 處理的動作。"""
        # 避免邊界值導致 atanh 為 inf
        clipped_action = torch.clamp(processed_action, 1e-6, 1.0 - 1e-6)
        # 從 [0, 1] 逆向映射到 [-1, 1]
        tanh_output = 2 * clipped_action - 1
        # 執行 atanh 逆運算
        return torch.atanh(tanh_output)

    def get_action(self, obs: torch.Tensor, stochastic: bool = True) -> AgentOutput:
        """
            根據觀測值獲取動作，用於與環境互動。
        """
        self.model.eval() # 確保在 inference 模式        
        flat_obs = self._preprocess_obs(obs)
        
        with torch.no_grad():
            # 從模型獲取策略參數和價值
            discrete_logits, continuous_mu, continuous_log_std, value = self.model(flat_obs)
            
            # 建立機率分佈
            std = continuous_log_std.exp()
            discrete_dist = torch.distributions.Categorical(logits=discrete_logits)
            continuous_dist = torch.distributions.Normal(continuous_mu, std)

            # 根據策略採樣或選擇確定性動作
            if stochastic:
                discrete_action = discrete_dist.sample()
                raw_continuous_action = continuous_dist.sample()
            else:
                discrete_action = discrete_dist.probs.argmax(dim=-1)
                raw_continuous_action = continuous_mu

            # 計算採樣動作的總 log_prob
            log_prob_discrete = discrete_dist.log_prob(discrete_action)
            log_prob_continuous = continuous_dist.log_prob(raw_continuous_action).sum(dim=-1)
            total_log_prob = log_prob_discrete + log_prob_continuous

            # 將連續動作後處理到有效範圍
            final_continuous_action = self._postprocess_continuous_action(raw_continuous_action)

        return AgentOutput(
            action=(discrete_action.squeeze(0), final_continuous_action.squeeze(0)), # 返回單個動作，而不是 batch
            value=value.squeeze(0),
            log_prob=total_log_prob.squeeze(0)
        )

    def evaluate_actions(self, obs_batch: torch.Tensor, discrete_act_batch: torch.Tensor, continuous_act_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            在 PPO 更新期間，評估一個 batch 的 (obs, action) 對。
            計算其 log_prob、entropy 和對應的 value。
        """
        self.model.train() # 確保在 training 模式

        flat_obs_batch = self._preprocess_obs(obs_batch)
        
        # 從模型獲取該 batch 觀測值對應的策略參數和價值
        discrete_logits, continuous_mu, continuous_log_std, value = self.model(flat_obs_batch)

        # 逆處理連續動作以匹配分佈的原始空間
        raw_continuous_act_batch = self._unprocess_continuous_action(continuous_act_batch)

        # 重建機率分佈
        std = continuous_log_std.exp()
        discrete_dist = torch.distributions.Categorical(logits=discrete_logits)
        continuous_dist = torch.distributions.Normal(continuous_mu, std)
        
        # 計算 log_prob
        log_prob_discrete = discrete_dist.log_prob(discrete_act_batch)
        log_prob_continuous = continuous_dist.log_prob(raw_continuous_act_batch).sum(dim=-1)
        total_log_prob = log_prob_discrete + log_prob_continuous
        
        # 計算 entropy
        entropy_discrete = discrete_dist.entropy()
        entropy_continuous = continuous_dist.entropy().sum(dim=-1)
        total_entropy = entropy_discrete + entropy_continuous

        return total_log_prob, total_entropy, value

    def update(self, rollout_buffer):
        """
        PPO 更新的核心邏輯（示意）。
        """
        # 從 rollout_buffer 中獲取數據
        # obs, actions, log_probs, advantages, returns = rollout_buffer.get()
        # discrete_actions, continuous_actions = actions
        
        # # 計算 loss
        # new_log_probs, entropy, values = self.evaluate_actions(obs, discrete_actions, continuous_actions)
        
        # # ... PPO 裁剪目標損失 (clipped surrogate objective) 的計算 ...
        # policy_loss = ...
        # value_loss = F.mse_loss(returns, values)
        # entropy_loss = entropy.mean()
        
        # # loss = policy_loss + c1 * value_loss - c2 * entropy_loss
        
        # # self.optimizer.zero_grad()
        # # loss.backward()
        # # self.optimizer.step()
        pass