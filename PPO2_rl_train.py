import os
import time
from datetime import datetime
from collections import deque
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from Brain.Common.DataFeature import OriginalDataFeature
from Brain.Common.PytorchModelTool import ModelTool
from Brain.Common.optimization import build_optimizer
from Brain.PPO2.lib.Agent import PPO2Agent
from Brain.PPO2.lib.environment import TrainingEnv, SubprocVecEnv
from Brain.PPO2.lib.experience import RolloutBuffer
import torch.nn as nn


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y - ypred] / Var[y], matching OpenAI Baselines.
    """
    var_y = torch.var(y_true, unbiased=False)
    if var_y < 1e-8:
        return 0.0
    return float((1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y).item())


class SymbolQueue:
    """
    全域商品洗牌佇列 (Global Shuffled Round-Robin Queue)。
    維護訓練集內所有唯一標的，依序取出；當佇列抽空時自動重新洗牌補充，保證 100% 均勻覆蓋。
    """

    def __init__(self, symbols: list):
        self.all_symbols = list(symbols)
        self.queue = deque()
        self._refill()

    def _refill(self):
        shuffled = list(self.all_symbols)
        np.random.shuffle(shuffled)
        self.queue.extend(shuffled)

    def get_next(self) -> str:
        if len(self.queue) == 0:
            self._refill()
        return self.queue.popleft()


class PPO2:
    def __init__(self, cfg: DictConfig):
        self.config = cfg
        self.cfg_training = cfg.training
        self._prepare_device()
        self._prepare_symbols()
        self._prepare_env()
        self._prepare_agent()
        self._prepare_optimizer()

        self.buffer = RolloutBuffer(
            gamma=self.cfg_training.GAMMA,
            lam=self.cfg_training.GAE_LAMBDA,
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

        self.cfg_training.UNIQUE_SYMBOLS = unique_symbols
        print(f"--SYMBOLNAMES--: Total {len(unique_symbols)} unique symbols found.")

        # 全域商品預載入快取，供所有 Worker 零拷貝共享讀取，徹底杜絕重複讀取 CSV 與記憶體膨脹
        print(
            f"--PRELOAD--: Preloading {len(unique_symbols)} unique symbols into memory cache..."
        )
        t0 = time.time()
        self.preloaded_data = OriginalDataFeature().get_train_net_work_data_by_path(
            unique_symbols
        )
        print(
            f"--PRELOAD--: Preloaded {len(self.preloaded_data)} symbols in {time.time() - t0:.2f}s."
        )

        self.symbol_queue = SymbolQueue(unique_symbols)

        saves_tag = self.cfg_training.SAVES_TAG
        bars_count = self.cfg_training.BARS_COUNT
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.saves_path = os.path.join(
            base_dir, saves_tag, f"{timestamp}-{bars_count}k-"
        )
        os.makedirs(self.saves_path, exist_ok=True)
        print(f"--SAVES_PATH--: {self.saves_path}")

    def _prepare_env(self):
        self.num_workers = self.cfg_training.NUM_WORKERS

        def make_env():
            return TrainingEnv(config=self.config, preloaded_data=self.preloaded_data)

        self.vec_env = SubprocVecEnv(
            [make_env for _ in range(self.num_workers)], context="fork"
        )
        print(
            f"--SubprocVecEnv--: Initialized with {self.num_workers} parallel workers."
        )

    def count_parameters(self, net: nn.modules):
        return sum(p.numel() for p in net.parameters() if p.requires_grad)

    def _prepare_agent(self):
        ssm_cfg = OmegaConf.to_container(self.cfg_training.SSM_CFG, resolve=True)

        self.agent = PPO2Agent(
            ob_space=self.vec_env.observation_space,
            ac_space=self.vec_env.action_space,
            device=self.device,
            seq_dim=self.cfg_training.BARS_COUNT,
            hidden_size=self.cfg_training.HIDDEN_SIZE,
            nlayers=self.cfg_training.NLAYERS,
            time_features_out=self.cfg_training.TIME_FEATURES_OUT,
            portfolio_dim=self.cfg_training.PORTFOLIO_DIM,
            ssm_cfg=ssm_cfg,
        )

        print(f"net create current parameter size:{self.count_parameters(self.agent.model)}")

    def _prepare_optimizer(self):
        lr = self.cfg_training.LEARNING_RATE
        weight_decay = self.cfg_training.WEIGHT_DECAY
        base_lr = self.cfg_training.BASE_LR
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
        clipfracs = []
        expl_vars = []

        for _ in range(ppo_epochs):
            for (
                states_b,
                time_states_b,
                actions_b,
                old_log_probs_b,
                returns_b,
                advantages_b,
                old_values_b,
            ) in self.buffer.get_batches(batch_size=batch_size, shuffle=True):
                states_b = states_b.to(self.device)
                time_states_b = time_states_b.to(self.device)
                actions_b = actions_b.to(self.device)
                old_log_probs_b = old_log_probs_b.to(self.device)
                returns_b = returns_b.to(self.device)
                advantages_b = advantages_b.to(self.device)
                old_values_b = old_values_b.to(self.device)

                log_probs, entropy, values = self.agent.evaluate_actions(
                    states_b, time_states_b, actions_b
                )

                # Ratio
                log_ratio = log_probs - old_log_probs_b
                ratio = torch.exp(log_ratio)

                # Approximate KL divergence & Clip Fraction for monitoring (OpenAI Baselines)
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    approx_kls.append(approx_kl)
                    clipfrac = torch.mean(
                        (torch.abs(ratio - 1.0) > clip_eps).float()
                    ).item()
                    clipfracs.append(clipfrac)
                    expl_var = explained_variance(values, returns_b)
                    expl_vars.append(expl_var)

                # Mini-batch Advantage Normalization (OpenAI Baselines / CleanRL Detail #7)
                mb_advantages = (advantages_b - advantages_b.mean()) / (
                    advantages_b.std() + 1e-8
                )

                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clipped Value Function loss with 0.5 factor (OpenAI Baselines Detail #9)
                v_clipped = old_values_b + torch.clamp(
                    values - old_values_b, -clip_eps, clip_eps
                )
                vf_loss_unclipped = (values - returns_b).pow(2)
                vf_loss_clipped = (v_clipped - returns_b).pow(2)
                value_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped).mean()

                # Entropy loss (bonus)
                entropy_loss = -entropy.mean()

                total_loss = (
                    policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.model.parameters(), max_grad_norm
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        policy_std = float(self.agent.model.log_std.exp().mean().item())

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(-np.mean(entropy_losses)) if entropy_losses else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "explained_variance": float(np.mean(expl_vars)) if expl_vars else 0.0,
            "policy_std": policy_std,
        }

    def checkgrad(self, model: nn.Module):
        # 打印梯度統計數據
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(
                    f"Layer: {name}, Grad Min: {param.grad.min()}, Grad Max: {param.grad.max()}, Grad Mean: {param.grad.mean()}"
                )

    def train(self):
        num_workers = self.num_workers
        horizon_steps = self.cfg_training.HORIZON_STEPS
        batch_size = self.cfg_training.BATCH_SIZE
        ppo_epochs = self.cfg_training.PPO_EPOCHS
        clip_eps = self.cfg_training.CLIP_EPS
        vf_coef = self.cfg_training.VF_COEF
        ent_coef = self.cfg_training.ENT_COEF
        max_grad_norm = self.cfg_training.MAX_GRAD_NORM
        checkpoint_every = self.cfg_training.CHECKPOINT_EVERY_ITER

        initial_symbols = [self.symbol_queue.get_next() for _ in range(num_workers)]
        obs = self.vec_env.reset(symbols=initial_symbols)
        current_symbols = list(initial_symbols)

        episode_rewards = np.zeros(num_workers, dtype=np.float64)
        episode_steps = np.zeros(num_workers, dtype=np.int64)
        episode_count = 0
        total_steps = 0
        iteration = 0

        fps_step_count = 0
        fps_start_time = time.time()
        recent_episode_returns = []

        total_rollout_steps = num_workers * horizon_steps
        print(f"\n{'=' * 30} Starting Multi-Instrument PPO2 Training {'=' * 30}")
        print(
            f"Workers: {num_workers} | Horizon/Worker: {horizon_steps} | "
            f"Total Rollout Steps: {total_rollout_steps} | Batch Size: {batch_size} | "
            f"PPO Epochs: {ppo_epochs} | Device: {self.device}"
        )

        try:
            while True:
                # 1. Rollout sampling across all N workers
                for _ in range(horizon_steps):
                    out = self.agent.get_action(obs, stochastic=True)
                    # 傳入環境執行已裁剪動作
                    env_actions_np = (
                        out.clipped_action.cpu().numpy()
                        if out.clipped_action is not None
                        else np.clip(out.action.cpu().numpy(), -1.0, 1.0)
                    )
                    next_obs, rewards, terminateds, truncateds, infos = (
                        self.vec_env.step(env_actions_np)
                    )

                    # Option A: Terminal Value Bootstrap Adjustment
                    stored_rewards = rewards.copy()
                    is_trunc = truncateds & (~terminateds)
                    if np.any(is_trunc):
                        trunc_indices = np.where(is_trunc)[0]
                        trunc_obs = {
                            "states": next_obs["states"][trunc_indices],
                            "time_states": next_obs["time_states"][trunc_indices],
                        }
                        with torch.no_grad():
                            trunc_out = self.agent.get_action(
                                trunc_obs, stochastic=False
                            )
                            trunc_values = trunc_out.value.cpu().numpy().reshape(-1)

                        gamma = float(self.cfg_training.GAMMA)
                        stored_rewards[trunc_indices] += gamma * trunc_values

                    dones = terminateds | truncateds

                    self.buffer.store(
                        obs,
                        out.action.cpu(),
                        out.log_prob.cpu(),
                        stored_rewards,
                        next_obs,
                        dones,
                        out.value.cpu(),
                    )
                    episode_rewards += rewards  # Record raw trading returns
                    episode_steps += 1
                    total_steps += num_workers
                    obs = next_obs

                    for i, done in enumerate(dones):
                        if done:
                            episode_count += 1
                            recent_episode_returns.append(float(episode_rewards[i]))
                            sym = infos[i].get("instrument", current_symbols[i])
                            pos = infos[i].get("position", 0.0)
                            print(
                                f"[Ep #{episode_count:04d} | W{i}] Symbol: {sym:<20s} | "
                                f"Steps: {episode_steps[i]:04d} | Return: {episode_rewards[i]:+.4f} | "
                                f"End Pos: {pos:+.2f} | TotalSteps: {total_steps:07d}"
                            )
                            # Reset worker i with next symbol from round-robin queue
                            next_sym = self.symbol_queue.get_next()
                            current_symbols[i] = next_sym
                            new_obs_i = self.vec_env.reset_at(i, symbol=next_sym)
                            obs["states"][i] = new_obs_i["states"]
                            obs["time_states"][i] = new_obs_i["time_states"]
                            episode_rewards[i] = 0.0
                            episode_steps[i] = 0

                # 2. Compute Bootstrap value & GAE
                with torch.no_grad():
                    last_out = self.agent.get_action(obs, stochastic=False)
                    last_values = last_out.value.cpu()

                self.buffer.compute_gae(last_values)

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
                    f"{np.mean(recent_episode_returns[-20:]):+.4f}"
                    if recent_episode_returns
                    else "N/A"
                )
                print(
                    f"[Iter #{iteration:04d}] Steps: {total_steps:07d} | FPS: {fps:6.1f} | "
                    f"PiLoss: {metrics['policy_loss']:+.4f} | VfLoss: {metrics['value_loss']:.4f} | "
                    f"Ent: {metrics['entropy']:.4f} | KL: {metrics['approx_kl']:.5f} | "
                    f"ClipFrac: {metrics['clipfrac']:.3f} | ExplVar: {metrics['explained_variance']:+.3f} | "
                    f"Std: {metrics['policy_std']:.3f} | AvgRet(20): {avg_ret_str}"
                )

                # 5. Checkpoint Saving
                if iteration % checkpoint_every == 0:
                    self.save_checkpoint(iteration, total_steps)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving final checkpoint...")
            self.save_checkpoint(iteration, total_steps, is_final=True)
        finally:
            self.vec_env.close()

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
    ppo_instance = PPO2(cfg)
    ppo_instance.train()


if __name__ == "__main__":
    main()
