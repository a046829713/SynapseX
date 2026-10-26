import os
import time
from datetime import datetime
from collections import deque
from typing import Optional, Dict, Any
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
    Global Shuffled Weighted Round-Robin Queue.
    Maintains active training instruments and allocates 1..max_weight slots per cycle
    proportional to usable history length, preventing short new-listing sampling skew.
    """

    def __init__(
        self,
        symbols: list,
        symbol_weights: Optional[Dict[str, int]] = None,
        max_weight: int = 5,
    ):
        assert len(symbols) > 0, "SymbolQueue requires at least one symbol"
        self.all_symbols = list(symbols)
        self.max_weight = max(1, int(max_weight))
        if symbol_weights is not None:
            self.symbol_weights = {
                s: max(1, min(self.max_weight, int(symbol_weights.get(s, 1))))
                for s in self.all_symbols
            }
        else:
            self.symbol_weights = {s: 1 for s in self.all_symbols}

        self._expanded_pool = []
        for s in self.all_symbols:
            self._expanded_pool.extend([s] * self.symbol_weights[s])

        self.queue = deque()
        self._refill()

    @staticmethod
    def compute_bar_weights(
        preloaded_data: Dict[str, Any],
        bars_count: int = 0,
        episode_steps: int = 0,
        max_weight: int = 5,
    ) -> Dict[str, int]:
        if not preloaded_data:
            return {}
        max_w = max(1, int(max_weight))
        usable_map = {
            sym: max(1, int(len(prices.close)) - int(bars_count) - int(episode_steps))
            for sym, prices in preloaded_data.items()
        }
        max_usable = max(usable_map.values())
        min_usable = min(usable_map.values())
        if max_usable <= min_usable:
            return {sym: 1 for sym in preloaded_data}
        return {
            sym: max(1, min(max_w, int(round(max_w * u / max_usable))))
            for sym, u in usable_map.items()
        }

    def _refill(self):
        shuffled = list(self._expanded_pool)
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
        all_symbols = sorted(list(set(f.split(".")[0] for f in symbol_files)))
        exclude_cfg = getattr(self.cfg_training, "EXCLUDE_SYMBOLS", None) or []
        exclude_set = set(str(s) for s in exclude_cfg)
        excluded_found = [s for s in all_symbols if s in exclude_set]
        candidate_symbols = [s for s in all_symbols if s not in exclude_set]
        assert (
            len(candidate_symbols) > 0
        ), "No symbols remaining after applying EXCLUDE_SYMBOLS"

        if excluded_found:
            print(
                f"--SYMBOLNAMES--: Excluded {len(excluded_found)} symbols: {excluded_found}"
            )

        train_end_date = getattr(self.cfg_training, "TRAIN_END_DATE", None)
        min_bars = (
            int(self.cfg_training.BARS_COUNT) + int(self.cfg_training.EPISODE_STEPS) + 2
        )

        # Shared memory cache across forked workers with strict temporal cutoff & bar-count filter
        print(
            f"--PRELOAD--: Preloading {len(candidate_symbols)} candidate symbols "
            f"(TRAIN_END_DATE={train_end_date}, min_bars={min_bars})..."
        )
        t0 = time.time()
        self.preloaded_data = OriginalDataFeature().get_train_net_work_data_by_path(
            candidate_symbols,
            end_date=train_end_date,
            min_bars=min_bars,
        )
        retained_symbols = sorted(list(self.preloaded_data.keys()))
        assert len(retained_symbols) > 0, (
            f"No symbols remaining after temporal cutoff (TRAIN_END_DATE={train_end_date}, "
            f"min_bars={min_bars})."
        )
        self.cfg_training.UNIQUE_SYMBOLS = retained_symbols
        skipped_count = len(candidate_symbols) - len(retained_symbols)
        print(
            f"--PRELOAD--: Retained {len(retained_symbols)} active symbols "
            f"(filtered {skipped_count} post-cutoff/short symbols) in {time.time() - t0:.2f}s."
        )

        symbol_weights = SymbolQueue.compute_bar_weights(
            self.preloaded_data,
            bars_count=self.cfg_training.BARS_COUNT,
            episode_steps=self.cfg_training.EPISODE_STEPS,
            max_weight=5,
        )
        self.symbol_queue = SymbolQueue(
            retained_symbols, symbol_weights=symbol_weights, max_weight=5
        )

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
        init_log_std = float(getattr(self.cfg_training, "INIT_LOG_STD", 0.0))
        log_std_min = float(getattr(self.cfg_training, "LOG_STD_MIN", -5.0))
        log_std_max = float(getattr(self.cfg_training, "LOG_STD_MAX", 1.0))

        self.agent = PPO2Agent(
            ob_space=self.vec_env.observation_space,
            ac_space=self.vec_env.action_space,
            device=self.device,
            seq_dim=self.cfg_training.BARS_COUNT,
            hidden_size=self.cfg_training.HIDDEN_SIZE,
            nlayers=self.cfg_training.NLAYERS,
            time_features_out=self.cfg_training.TIME_FEATURES_OUT,
            init_log_std=init_log_std,
            portfolio_dim=self.cfg_training.PORTFOLIO_DIM,
            ssm_cfg=ssm_cfg,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
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
        grad_norms = []
        mean_actions = []
        mean_mus = []

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

                log_probs, entropy, values, stats = self.agent.evaluate_actions(
                    states_b, time_states_b, actions_b, return_stats=True
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
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.model.parameters(), max_grad_norm
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                grad_norms.append(float(grad_norm.item()))
                mean_actions.append(stats["mean_action"])
                mean_mus.append(stats["mean_mu"])

        with torch.no_grad():
            clamped_log_std = torch.clamp(
                self.agent.model.log_std,
                self.agent.model.log_std_min,
                self.agent.model.log_std_max,
            )
            policy_std = float(clamped_log_std.exp().mean().item())

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(-np.mean(entropy_losses)) if entropy_losses else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "explained_variance": float(np.mean(expl_vars)) if expl_vars else 0.0,
            "policy_std": policy_std,
            "mean_action": float(np.mean(mean_actions)) if mean_actions else 0.5,
            "mean_mu": float(np.mean(mean_mus)) if mean_mus else 0.0,
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
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
        episode_net_pnl = np.zeros(num_workers, dtype=np.float64)
        episode_cost = np.zeros(num_workers, dtype=np.float64)
        episode_trades = np.zeros(num_workers, dtype=np.int64)
        episode_turnover = np.zeros(num_workers, dtype=np.float64)
        episode_abs_pos_sum = np.zeros(num_workers, dtype=np.float64)
        episode_steps = np.zeros(num_workers, dtype=np.int64)
        episode_count = 0
        total_steps = 0
        iteration = 0

        fps_step_count = 0
        fps_start_time = time.time()
        recent_episode_returns = []
        recent_episode_net_pnl_pct = []
        recent_episode_cost_pct = []
        recent_episode_trades = []
        recent_episode_avg_abs_pos = []

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
                    # Execute clipped Long-Only [0.0, 1.0] action in environment
                    env_actions_np = (
                        out.clipped_action.cpu().numpy()
                        if out.clipped_action is not None
                        else np.clip(out.action.cpu().numpy(), 0.0, 1.0)
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
                    episode_rewards += rewards
                    for w_idx, info_w in enumerate(infos):
                        episode_net_pnl[w_idx] += float(info_w.get("step_return", 0.0))
                        episode_cost[w_idx] += float(info_w.get("cost", 0.0))
                        episode_trades[w_idx] += int(bool(info_w.get("traded", False)))
                        episode_turnover[w_idx] += float(info_w.get("turnover", 0.0))
                        episode_abs_pos_sum[w_idx] += abs(
                            float(info_w.get("position", 0.0))
                        )
                    episode_steps += 1
                    total_steps += num_workers
                    obs = next_obs

                    for i, done in enumerate(dones):
                        if done:
                            episode_count += 1
                            steps_i = int(max(episode_steps[i], 1))
                            ret_i = float(episode_rewards[i])
                            net_pnl_pct_i = float(episode_net_pnl[i] * 100.0)
                            cost_pct_i = float(episode_cost[i] * 100.0)
                            trades_i = int(episode_trades[i])
                            turnover_i = float(episode_turnover[i])
                            avg_abs_pos_i = float(episode_abs_pos_sum[i] / steps_i)

                            recent_episode_returns.append(ret_i)
                            recent_episode_net_pnl_pct.append(net_pnl_pct_i)
                            recent_episode_cost_pct.append(cost_pct_i)
                            recent_episode_trades.append(trades_i)
                            recent_episode_avg_abs_pos.append(avg_abs_pos_i)

                            sym = infos[i].get("instrument", current_symbols[i])
                            pos = infos[i].get("position", 0.0)
                            print(
                                f"[Ep #{episode_count:04d} | W{i}] Symbol: {sym:<20s} | "
                                f"Steps: {episode_steps[i]:04d} | Return: {ret_i:+.4f} | "
                                f"NetPnL: {net_pnl_pct_i:+.2f}% | Cost: {cost_pct_i:.2f}% | "
                                f"Trades: {trades_i:03d} | Turnover: {turnover_i:5.2f} | "
                                f"Avg|Pos|: {avg_abs_pos_i:.2f} | End Pos: {pos:+.2f} | "
                                f"TotalSteps: {total_steps:07d}"
                            )
                            # Reset worker i with next symbol from round-robin queue
                            next_sym = self.symbol_queue.get_next()
                            current_symbols[i] = next_sym
                            new_obs_i = self.vec_env.reset_at(i, symbol=next_sym)
                            obs["states"][i] = new_obs_i["states"]
                            obs["time_states"][i] = new_obs_i["time_states"]
                            episode_rewards[i] = 0.0
                            episode_net_pnl[i] = 0.0
                            episode_cost[i] = 0.0
                            episode_trades[i] = 0
                            episode_turnover[i] = 0.0
                            episode_abs_pos_sum[i] = 0.0
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

                if recent_episode_returns:
                    avg_ret_str = f"{np.mean(recent_episode_returns[-20:]):+.4f}"
                    med_ret_100_str = f"{np.median(recent_episode_returns[-100:]):+.4f}"
                    net_pnl_100 = recent_episode_net_pnl_pct[-100:]
                    net_pnl_100_str = f"{np.mean(net_pnl_100):+.2f}%"
                    cost_100_str = f"{np.mean(recent_episode_cost_pct[-100:]):.2f}%"
                    win_100_str = f"{np.mean([p > 0.0 for p in net_pnl_100]) * 100.0:.1f}%"
                    trades_100_str = f"{np.mean(recent_episode_trades[-100:]):.1f}"
                    avg_pos_100_str = f"{np.mean(recent_episode_avg_abs_pos[-100:]):.2f}"
                else:
                    avg_ret_str = "N/A"
                    med_ret_100_str = "N/A"
                    net_pnl_100_str = "N/A"
                    cost_100_str = "N/A"
                    win_100_str = "N/A"
                    trades_100_str = "N/A"
                    avg_pos_100_str = "N/A"

                print(
                    f"[Iter #{iteration:04d}] Steps: {total_steps:07d} | FPS: {fps:6.1f} | "
                    f"PiLoss: {metrics['policy_loss']:+.4f} | VfLoss: {metrics['value_loss']:.4f} | "
                    f"Ent: {metrics['entropy']:+.4f} | KL: {metrics['approx_kl']:.5f} | "
                    f"ClipFrac: {metrics['clipfrac']:.3f} | ExplVar: {metrics['explained_variance']:+.3f} | "
                    f"Std: {metrics['policy_std']:.3f} | MeanAct: {metrics['mean_action']:.3f} | "
                    f"MeanMu: {metrics['mean_mu']:+.3f} | GradNorm: {metrics['grad_norm']:.3f} | "
                    f"AvgRet(20): {avg_ret_str} | MedRet(100): {med_ret_100_str} | "
                    f"NetPnL(100): {net_pnl_100_str} | Cost(100): {cost_100_str} | "
                    f"Win(100): {win_100_str} | Trades(100): {trades_100_str} | "
                    f"Avg|Pos|(100): {avg_pos_100_str}"
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
