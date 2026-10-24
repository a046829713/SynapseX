from typing import Optional, Tuple
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import torch
import torch.nn as nn
from Brain.PPO2.lib.model import MambaContinuousActorCritic, AgentOutput


class PPO2Agent:
    def __init__(
        self,
        ob_space: gym.Space,
        ac_space: gym.Space,
        device: torch.device,
        seq_dim: int = 300,
        hidden_size: int = 96,
        nlayers: int = 2,
        time_features_out: int = 32,
        mode: str = "full",
        init_log_std: float = -0.5,
        portfolio_dim: int = 3,
        ssm_cfg: Optional[dict] = None,
        moe_cfg: Optional[dict] = None,
    ):
        self.device = device
        self.ob_space = ob_space
        self.ac_space = ac_space

        self._validate_spaces()
        self._validate_ssm_config_eight_heads(ssm_cfg)



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
            time_features_out=time_features_out,
            mode=mode,
            init_log_std=init_log_std,
            portfolio_dim=portfolio_dim,
            ssm_cfg=ssm_cfg,
            moe_cfg=moe_cfg,
        ).to(self.device)

    def _validate_ssm_config_eight_heads(self, ssm_cfg: dict):
        # {'layer': 'Mamba2', 'expand': 2, 'd_ssm': 96, 'headdim': 32, 'd_state': 64, 'ngroups': 1, 'chunk_size': 64, 'd_conv': 4, 'rmsnorm': True, 'bias': False, 'conv_bias': True}
        assert(ssm_cfg['d_ssm'] / ssm_cfg['headdim']) % 8 == 0,"for best train performance please check the d_ssm size and headdim size"


    def _validate_spaces(self):
        assert isinstance(self.ac_space, Box), "PPO2Agent 動作空間必須為 gym.spaces.Box"
        assert isinstance(
            self.ob_space, Dict
        ), "PPO2Agent 觀測空間必須為 gym.spaces.Dict"
        assert (
            "states" in self.ob_space.spaces and "time_states" in self.ob_space.spaces
        ), "PPO2Agent 觀測字典必須包含 'states' 與 'time_states'"

    def _preprocess_obs(self, obs):
        states = torch.as_tensor(obs["states"], dtype=torch.float32, device=self.device)
        time_states = torch.as_tensor(
            obs["time_states"], dtype=torch.float32, device=self.device
        )
        is_single = False
        if states.dim() == 2:
            states = states.unsqueeze(0)
            time_states = time_states.unsqueeze(0)
            is_single = True
        return states, time_states, is_single

    def get_action(self, obs, stochastic: bool = True) -> AgentOutput:
        self.model.eval()
        states, time_states, is_single = self._preprocess_obs(obs)
        with torch.no_grad():
            mu, std, value = self.model(states, time_states)
            dist = torch.distributions.Normal(mu, std)
            if stochastic:
                action = dist.sample()
            else:
                action = mu

            # 根據 OpenAI Baselines / CleanRL 標準實作：
            # 1. log_prob 必須基於未裁剪的原始高斯採樣動作計算
            # 2. clipped_action 供環境 step 執行，確保環境介面在合法範圍內
            log_prob = dist.log_prob(action).sum(dim=-1)
            clipped_action = torch.clamp(action, -1.0, 1.0)

        if is_single:
            return AgentOutput(
                action=action.squeeze(0),  # 原始未裁剪動作
                clipped_action=clipped_action.squeeze(0),  # 裁剪後動作
                value=value.squeeze(0),
                log_prob=log_prob.squeeze(0),
            )

        return AgentOutput(
            action=action,  # 原始未裁剪動作 (N, action_dim)
            clipped_action=clipped_action,  # 裁剪後動作 (N, action_dim)
            value=value,  # (N,)
            log_prob=log_prob,  # (N,)
        )

    def evaluate_actions(self, states, time_states, actions):
        self.model.train()
        states = states.to(self.device)
        time_states = time_states.to(self.device)
        actions = actions.to(self.device)

        mu, std, value = self.model(states, time_states)
        dist = torch.distributions.Normal(mu, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value
