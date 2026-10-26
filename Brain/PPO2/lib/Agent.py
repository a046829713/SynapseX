from typing import Dict as TypingDict, Optional, Tuple, Union
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        init_log_std: float = 0.0,
        portfolio_dim: int = 3,
        ssm_cfg: Optional[dict] = None,
        moe_cfg: Optional[dict] = None,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
    ):
        self.device = device
        self.ob_space = ob_space
        self.ac_space = ac_space

        self._validate_spaces()
        self._validate_ssm_config_eight_heads(ssm_cfg)

        low_np = np.asarray(self.ac_space.low, dtype=np.float32)
        high_np = np.asarray(self.ac_space.high, dtype=np.float32)
        scale_np = high_np - low_np

        self._act_low = torch.as_tensor(low_np, dtype=torch.float32, device=self.device)
        self._act_scale = torch.as_tensor(
            scale_np, dtype=torch.float32, device=self.device
        )
        self._log_act_scale = torch.log(self._act_scale)

        # Precompute 16-point probabilists' Gauss-Hermite quadrature nodes/weights
        # for deterministic, exact bounded action-space differential entropy evaluation.
        gh_nodes_np, gh_weights_np = np.polynomial.hermite_e.hermegauss(16)
        gh_weights_np = gh_weights_np / np.sqrt(2.0 * np.pi)
        self._gh_nodes = torch.as_tensor(
            gh_nodes_np, dtype=torch.float32, device=self.device
        ).view(1, 1, -1)
        self._gh_weights = torch.as_tensor(
            gh_weights_np, dtype=torch.float32, device=self.device
        ).view(1, 1, -1)

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
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        ).to(self.device)

    def _validate_ssm_config_eight_heads(self, ssm_cfg: dict):
        # {'layer': 'Mamba2', 'expand': 2, 'd_ssm': 96, 'headdim': 32, 'd_state': 64, 'ngroups': 1, 'chunk_size': 64, 'd_conv': 4, 'rmsnorm': True, 'bias': False, 'conv_bias': True}
        assert(ssm_cfg['d_ssm'] / ssm_cfg['headdim']) % 8 == 0,"for best train performance please check the d_ssm size and headdim size"


    def _validate_spaces(self):
        assert isinstance(self.ac_space, Box), "PPO2Agent 動作空間必須為 gym.spaces.Box"
        assert np.all(np.isfinite(self.ac_space.low)) and np.all(
            np.isfinite(self.ac_space.high)
        ), "PPO2Agent requires finite Box action bounds for Sigmoid Squashed-Gaussian bijection"
        assert np.all(
            self.ac_space.high > self.ac_space.low
        ), "PPO2Agent requires ac_space.high > ac_space.low"
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

    def _squash_action(self, latent_u: torch.Tensor) -> torch.Tensor:
        """Bijective Sigmoid transformation from R^D onto (ac_space.low, ac_space.high)."""
        return self._act_low + self._act_scale * torch.sigmoid(latent_u)

    def _squashed_log_prob(
        self, dist: torch.distributions.Normal, latent_u: torch.Tensor
    ) -> torch.Tensor:
        """
        Exact Change-of-Variables log-density of squashed action a = low + scale * sigmoid(u):
          log pi(a | s) = log N(u; mu, std^2) - log(scale) + softplus(u) + softplus(-u)
        """
        neg_log_det_jac = (
            F.softplus(latent_u) + F.softplus(-latent_u) - self._log_act_scale
        )
        return (dist.log_prob(latent_u) + neg_log_det_jac).sum(dim=-1)

    def _squashed_entropy(
        self,
        dist: torch.distributions.Normal,
        mu: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Deterministic 16-point Gauss-Hermite evaluation of exact bounded action-space
        differential entropy H(A | s) <= sum(log(scale)).
        """
        u_quad = mu.unsqueeze(-1) + std.unsqueeze(-1) * self._gh_nodes
        expected_neg_log_jac = (
            (F.softplus(u_quad) + F.softplus(-u_quad)) * self._gh_weights
        ).sum(dim=-1) - self._log_act_scale
        return (dist.entropy() - expected_neg_log_jac).sum(dim=-1)

    def get_action(self, obs, stochastic: bool = True) -> AgentOutput:
        self.model.eval()
        states, time_states, is_single = self._preprocess_obs(obs)
        with torch.no_grad():
            mu, std, value = self.model(states, time_states)
            dist = torch.distributions.Normal(mu, std)
            if stochastic:
                latent_action = dist.sample()
            else:
                latent_action = mu

            log_prob = self._squashed_log_prob(dist, latent_action)
            squashed_action = self._squash_action(latent_action)

        if is_single:
            return AgentOutput(
                action=latent_action.squeeze(0),
                clipped_action=squashed_action.squeeze(0),
                value=value.squeeze(0),
                log_prob=log_prob.squeeze(0),
            )

        return AgentOutput(
            action=latent_action,
            clipped_action=squashed_action,
            value=value,
            log_prob=log_prob,
        )

    def evaluate_actions(
        self,
        states: torch.Tensor,
        time_states: torch.Tensor,
        actions: torch.Tensor,
        return_stats: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, TypingDict[str, float]],
    ]:
        self.model.train()
        states = states.to(self.device)
        time_states = time_states.to(self.device)
        actions = actions.to(self.device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        mu, std, value = self.model(states, time_states)
        dist = torch.distributions.Normal(mu, std)

        log_prob = self._squashed_log_prob(dist, actions)
        entropy = self._squashed_entropy(dist, mu, std)

        if return_stats:
            with torch.no_grad():
                squashed_mu = self._squash_action(mu)
                stats = {
                    "mean_action": float(squashed_mu.mean().item()),
                    "mean_mu": float(mu.mean().item()),
                }
            return log_prob, entropy, value, stats

        return log_prob, entropy, value
