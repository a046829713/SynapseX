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
from Brain.PPO2.lib.model import HybridMambaPolicy, AgentOutput

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

        self._validate_spaces()

        discrete_ac_dim = self.ac_space[0].n
        continuous_ac_dim = self.ac_space[1].shape[0]
        


        # discrete_ac_dim 3  # continuous_ac_dim 1
        self.model = HybridMambaPolicy(
            d_model=self.ob_space['states'].shape[1],
            time_features_in = self.ob_space['time_states'].shape[1]
        ).to(self.device)

        print(self.model)
        time.sleep(100)
        print(discrete_ac_dim)
        print(continuous_ac_dim)
        time.sleep(100)

    def _validate_spaces(self):
        assert isinstance(self.ac_space, TupleSpace), "混合動作空間必須使用 gym.spaces.Tuple"
        assert len(self.ac_space.spaces) == 2, "動作元組應包含2個元素"
        assert isinstance(self.ac_space[0], Discrete), "Tuple的第一個元素必須是 Discrete"
        assert isinstance(self.ac_space[1], Box), "Tuple的第二個元素必須是 Box"
        print("環境驗證成功")

    def _preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
            處理觀測值：標準化、調整維度、移至 device 並扁平化。
        """
        if obs.dim() == len(self.ob_space.shape):
            obs = obs.unsqueeze(0)
        
        obs = obs.to(self.device, dtype=torch.float32)

            
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