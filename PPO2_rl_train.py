import os
import time
from datetime import datetime
from typing import Optional

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

from Brain.Common.PytorchModelTool import ModelTool
from Brain.PPO2.lib.Agent import PPO2Agent
from Brain.PPO2.lib.environment import TrainingEnv
from Brain.PPO2.lib.experience import RolloutBuffer


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    base_lr: float = 1e-4,
) -> optim.AdamW:
    """
    建立專屬 AdamW 優化器：
    1. `dean` (DAIN_Layer) 的特殊層 (`mean_layer`, `scaling_layer`, `gating_layer`)
       使用獨立學習率且 0 weight decay。
    2. `LayerNorm`、`bias` 與 `log_std` 參數排除 weight decay。
    3. 其餘參數施加標準 weight decay。
    """
    dean_params_ids = set()
    if hasattr(model, "dean"):
        dean_params_ids.update(id(p) for p in model.dean.parameters())

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in dean_params_ids:
            continue
        if "norm" in name or name.endswith(".bias") or "log_std" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
    ]

    if hasattr(model, "dean"):
        param_groups.extend([
            {
                "params": list(model.dean.mean_layer.parameters()),
                "lr": base_lr * model.dean.mean_lr,
                "weight_decay": 0.0,
            },
            {
                "params": list(model.dean.scaling_layer.parameters()),
                "lr": base_lr * model.dean.scale_lr,
                "weight_decay": 0.0,
            },
            {
                "params": list(model.dean.gating_layer.parameters()),
                "lr": base_lr * model.dean.gate_lr,
                "weight_decay": 0.0,
            },
        ])

    return optim.AdamW(param_groups)


class PPO2:
    def __init__(self, cfg: DictConfig):
        self.config = cfg
        self._prepare_device()
        self._prepare_symbols()
        self._prepare_env()
        self._prepare_agent()
        self._prepare_optimizer()

        cfg_training = getattr(self.config, "training", self.config)
        self.buffer = RolloutBuffer(
            gamma=getattr(cfg_training, "GAMMA", 0.99),
            lam=getattr(cfg_training, "GAE_LAMBDA", 0.95),
        )
        self.model_tool = ModelTool()

    def _prepare_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--DEVICE--: {self.device}")

    def _prepare_symbols(self):
        try:
            from hydra.utils import get_original_cwd
            base_dir = get_original_cwd()
        except Exception:
            base_dir = os.getcwd()

        train_data_path = os.path.join(base_dir, "Brain", "simulation", "train_data")
        symbol_files = [f for f in os.listdir(train_data_path) if f.endswith(".csv")]
        unique_symbols = sorted(list(set(f.split(".")[0] for f in symbol_files)))

        cfg_training = getattr(self.config, "training", self.config)
        cfg_training.UNIQUE_SYMBOLS = unique_symbols
        print(f"--SYMBOLNAMES--: Total {len(unique_symbols)} unique symbols found.")

        saves_tag = getattr(cfg_training, "SAVES_TAG", "saves")
        bars_count = getattr(cfg_training, "BARS_COUNT", 300)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.saves_path = os.path.join(base_dir, saves_tag, f"{timestamp}-{bars_count}k-")
        os.makedirs(self.saves_path, exist_ok=True)
        print(f"--SAVES_PATH--: {self.saves_path}")

    def _prepare_env(self):
        self.train_env = TrainingEnv(config=self.config)
        print(f"--TrainingEnv--: {self.train_env}")

    def _prepare_agent(self):
        cfg_training = getattr(self.config, "training", self.config)
        self.agent = PPO2Agent(
            ob_space=self.train_env.observation_space,
            ac_space=self.train_env.action_space,
            device=self.device,
            seq_dim=getattr(cfg_training, "BARS_COUNT", 300),
            hidden_size=getattr(cfg_training, "HIDDEN_SIZE", 96),
            nlayers=getattr(cfg_training, "NLAYERS", 2),
            time_features_out=getattr(cfg_training, "TIME_FEATURES_OUT", 32),
            dropout=getattr(cfg_training, "DROPOUT", 0.1),
        )
        print(f"--Agent--: Initialized with model {self.agent.model.__class__.__name__}")

    def _prepare_optimizer(self):
        cfg_training = getattr(self.config, "training", self.config)
        lr = getattr(cfg_training, "LEARNING_RATE", 3e-4)
        weight_decay = getattr(cfg_training, "WEIGHT_DECAY", 1e-4)
        base_lr = getattr(cfg_training, "BASE_LR", 1e-4)
        self.optimizer = build_optimizer(
            self.agent.model, lr=lr, weight_decay=weight_decay, base_lr=base_lr
        )
        print(f"--Optimizer--: AdamW created (lr={lr}, weight_decay={weight_decay})")

    def update_ppo(
        self,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []

        for _ in range(ppo_epochs):
            for (
                states_b,
                time_states_b,
                actions_b,
                old_log_probs_b,
                returns_b,
                advantages_b,
            ) in self.buffer.get_batches(batch_size=batch_size, shuffle=True):
                states_b = states_b.to(self.device)
                time_states_b = time_states_b.to(self.device)
                actions_b = actions_b.to(self.device)
                old_log_probs_b = old_log_probs_b.to(self.device)
                returns_b = returns_b.to(self.device)
                advantages_b = advantages_b.to(self.device)

                log_probs, entropy, values = self.agent.evaluate_actions(
                    states_b, time_states_b, actions_b
                )

                # Ratio
                log_ratio = log_probs - old_log_probs_b
                ratio = torch.exp(log_ratio)

                # Approximate KL divergence for monitoring
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    approx_kls.append(approx_kl)

                # Clipped surrogate objective
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function MSE loss
                value_loss = F.mse_loss(values, returns_b)

                # Entropy loss (bonus)
                entropy_loss = -entropy.mean()

                total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(-np.mean(entropy_losses)) if entropy_losses else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        }

    def train(self, max_iterations: Optional[int] = None):
        cfg_training = getattr(self.config, "training", self.config)
        n_steps = getattr(cfg_training, "N_STEPS", 1000)
        batch_size = getattr(cfg_training, "BATCH_SIZE", 64)
        ppo_epochs = getattr(cfg_training, "PPO_EPOCHS", 4)
        clip_eps = getattr(cfg_training, "CLIP_EPS", 0.2)
        vf_coef = getattr(cfg_training, "VF_COEF", 0.5)
        ent_coef = getattr(cfg_training, "ENT_COEF", 0.01)
        max_grad_norm = getattr(cfg_training, "MAX_GRAD_NORM", 0.5)
        checkpoint_every = getattr(cfg_training, "CHECKPOINT_EVERY_ITER", 10)

        obs = self.train_env.reset()
        episode_reward = 0.0
        episode_count = 0
        total_steps = 0
        iteration = 0

        fps_step_count = 0
        fps_start_time = time.time()
        recent_episode_returns = []

        print(f"\n{'=' * 30} Starting PPO2 Training {'=' * 30}")
        print(
            f"Rollout Horizon (N_STEPS): {n_steps} | Batch Size: {batch_size} | "
            f"PPO Epochs: {ppo_epochs} | Device: {self.device}"
        )

        try:
            while True:
                if max_iterations is not None and iteration >= max_iterations:
                    print(f"Reached max iterations ({max_iterations}). Stopping training.")
                    break

                # 1. Rollout sampling
                for _ in range(n_steps):
                    out = self.agent.get_action(obs, stochastic=True)
                    action_scalar = float(out.action.cpu().numpy()[0])
                    next_obs, reward, done, info = self.train_env.step(action_scalar)

                    self.buffer.store(
                        obs,
                        out.action.cpu(),
                        out.log_prob.cpu(),
                        reward,
                        next_obs,
                        done,
                        out.value.cpu(),
                    )
                    episode_reward += reward
                    total_steps += 1
                    obs = next_obs

                    if done:
                        episode_count += 1
                        recent_episode_returns.append(episode_reward)
                        pos = info.get("position", 0.0)
                        sym = info.get("instrument", "N/A")
                        print(
                            f"[Ep #{episode_count:04d}] Symbol: {sym} | TotalSteps: {total_steps:07d} | "
                            f"Return: {episode_reward:+.4f} | End Pos: {pos:+.2f}"
                        )
                        obs = self.train_env.reset()
                        episode_reward = 0.0

                # 2. Compute Bootstrap value & GAE
                with torch.no_grad():
                    last_out = self.agent.get_action(obs, stochastic=False)
                    last_value = last_out.value.item()

                self.buffer.compute_gae(last_value)

                # 3. PPO Update
                metrics = self.update_ppo(
                    ppo_epochs=ppo_epochs,
                    batch_size=batch_size,
                    clip_eps=clip_eps,
                    vf_coef=vf_coef,
                    ent_coef=ent_coef,
                    max_grad_norm=max_grad_norm,
                )
                self.buffer.clear()
                iteration += 1

                # 4. Monitoring & FPS
                elapsed = time.time() - fps_start_time
                fps = (total_steps - fps_step_count) / elapsed if elapsed > 0 else 0.0
                fps_step_count = total_steps
                fps_start_time = time.time()

                avg_ret_str = (
                    f"{np.mean(recent_episode_returns[-10:]):+.4f}"
                    if recent_episode_returns
                    else "N/A"
                )
                print(
                    f"[Iter #{iteration:04d}] Steps: {total_steps:07d} | FPS: {fps:6.1f} | "
                    f"PiLoss: {metrics['policy_loss']:+.4f} | VfLoss: {metrics['value_loss']:.4f} | "
                    f"Ent: {metrics['entropy']:.4f} | KL: {metrics['approx_kl']:.5f} | "
                    f"AvgRet(10): {avg_ret_str}"
                )

                # 5. Checkpoint Saving
                if iteration % checkpoint_every == 0:
                    self.save_checkpoint(iteration, total_steps)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving final checkpoint...")
            self.save_checkpoint(iteration, total_steps, is_final=True)

    def save_checkpoint(self, iteration: int, total_steps: int, is_final: bool = False):
        prefix = "final_checkpoint" if is_final else f"checkpoint_iter_{iteration}"
        checkpoint_path = os.path.join(self.saves_path, f"{prefix}.pt")
        self.model_tool.save_checkpoint(
            {
                "iteration": iteration,
                "total_steps": total_steps,
                "model_state_dict": self.agent.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print(f"--> Checkpoint saved: {checkpoint_path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    ppo_instance = PPO2(cfg)
    cfg_training = getattr(cfg, "training", cfg)
    max_iters = getattr(cfg_training, "MAX_ITERATIONS", None)
    ppo_instance.train(max_iterations=max_iters)


if __name__ == "__main__":
    main()