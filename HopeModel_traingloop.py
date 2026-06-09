from __future__ import annotations

import base64
import json
import os
import pickle
import random
import math
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, Tuple, Deque
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf

# 假設這些是你上一段代碼定義的 HOPE DQN 模型
from HopeModel import HOPEDQN, DQNConfig
from nested_learning.levels import LevelSpec
from nested_learning.logging_utils import BaseLogger, NullLogger, init_logger
import time
from Brain.HopeDQN.lib.environment import TrainingEnv
from utils.AppSetting import UpdateConfig
from Brain.Common.optimization import opitmizar
from Brain.Common.agent import DQNAgent
from Brain.Common.actions import EpsilonGreedyActionSelector
from Brain.Common.experience import SequentialExperienceReplayBuffer
from Brain.DQN.ptan.experience import ExperienceSourceFirstLast
from Brain.HopeDQN.experience import SequentialReplayBuffer
import torch.nn as nn

@dataclass
class DistributedContext:
    rank: int
    world_size: int
    device: torch.device

def unpack_batch(batch):
    (
        first_states,
        first_time_states,
        actions,
        rewards,
        dones,
        last_states,
        last_time_states,
        infos,
        last_infos,
        
    ) = ([], [], [], [], [], [], [], [], [])

    for exp in batch:
        first_state, first_time_state = exp.state
        first_states.append(first_state)
        first_time_states.append(first_time_state)

        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        infos.append(exp.info)
        last_infos.append(exp.last_info)

        if exp.last_state is None:
            # the result will be masked anyway
            last_states.append(first_state)
            last_time_states.append(first_time_state)

        else:
            last_state, last_time_state = exp.last_state
            last_states.append(last_state)
            last_time_states.append(last_time_state)

    return (
        np.array(first_states, copy=False),
        np.array(first_time_states, copy=False),
        np.array(actions),
        np.array(rewards, dtype=np.float32),
        np.array(dones, dtype=np.uint8),
        np.array(last_states, copy=False),
        np.array(last_time_states, copy=False),
        np.array(infos, copy=False),
        np.array(last_infos, copy=False),
        
    )

def update_policy(
    batch,
    net,
    tgt_net,
    gamma,
    optimizer_instance,
    imag_loss_weight=0.1,
    moe_loss_coeff=0.01,
    device="cpu",
):
    """
        計算 DQN 的 MSE loss，並同時計算每筆 transition 的 TD‐error，並整合 MoE 的輔助損失 (auxiliary loss)。
    """
    (
        first_states,
        first_time_states,
        actions,
        rewards,
        dones,
        last_states,
        last_time_states,
        _,
        _,
    ) = unpack_batch(batch)

    first_states_v = torch.tensor(first_states, device=device, dtype=torch.float32)
    
    first_time_states_v = torch.tensor(first_time_states, device=device, dtype=torch.float32 )

    last_states_v = torch.tensor(last_states, device=device, dtype=torch.float32)
    last_time_states_v = torch.tensor(
        last_time_states, device=device, dtype=torch.float32
    )

    actions_v = torch.tensor(actions, device=device, dtype=torch.long)
    rewards_v = torch.tensor(rewards, device=device, dtype=torch.float32)
    done_mask = torch.tensor(dones, device=device, dtype=torch.bool)
    


    # --- 線上網路 (net) ---
    q_values = net(first_states_v, first_time_states_v)

    # 2. 獲取實際採取動作的 Q 值: Q(s,a)
    state_action_values = q_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    # ---目標網路 (tgt_net) & Double DQN ---
    with torch.no_grad():
        # 3. 使用線上網路 `net` 選擇下一狀態的最佳動作 a_max
        next_q_values = net(last_states_v, last_time_states_v)
        next_actions = next_q_values.max(1)[1]

        # 4. 使用目標網路 `tgt_net` 評估 a_max 的 Q 值: Q_target(s', a_max)
        next_state_q_values = tgt_net(last_states_v, last_time_states_v)
        
        next_state_values = next_state_q_values.gather(
            1, next_actions.unsqueeze(-1)
        ).squeeze(-1)

        # 5. 對於終止狀態，其未來價值為 0
        next_state_values[done_mask] = 0.0

        # 6. 計算 TD 目標 (y)
        q_targets = rewards_v + gamma * next_state_values


    loss = nn.MSELoss()(state_action_values, q_targets)
    
    # 5. 優化
    optimizer_instance.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optimizer_instance.step()
    
    


    # --- HOPE Special: Teach Signal Injection ---
    # 這裡我們再次執行一次 forward (不計算梯度)，但是帶入 teach_signal
    # 這樣可以讓 HOPE Block 內部的 Slow/Fast weights 根據 "Surprise" 進行更新
    update_metrics = {}
    with torch.no_grad():
        # 計算 RL 版本的 Teach Signal (基於 TD Error 的梯度方向)
        teach_signal = compute_rl_teach_signal(
            net, current_q_values, b_action, expected_q
        )
        teach_signal_norm = teach_signal.norm(dim=-1).mean().item()

        # 將訊號注入模型，觸發 HOPE Block 內部更新
        net(b_state, teach_signal=teach_signal)

        if hasattr(policy_net, "pop_update_metrics"):
            update_metrics = policy_net.pop_update_metrics()



    return 



def build_model_from_cfg(model_cfg: DictConfig) -> torch.nn.Module:
    """

    Args:
        model_cfg (DictConfig): _description_

    Returns:
        torch.nn.Module: _description_
    """
    optimizer_cfg = {}



    if "optimizers" in model_cfg:
        optimizer_cfg = OmegaConf.to_container(model_cfg.optimizers, resolve=True)

    titan_spec = LevelSpec(**model_cfg.titan_level)
    cms_specs = [LevelSpec(**entry) for entry in model_cfg.cms_levels]

    
    dqn_cfg = DQNConfig(
        state_dim=model_cfg.state_dim,  # 需在 config 中定義
        action_dim=model_cfg.action_dim,  # 需在 config 中定義
        time_dim = model_cfg.time_dim,
        time_features_out = model_cfg.time_features_out,
        mode = model_cfg.mode,
        hidden_size = model_cfg.hidden_size,
        seq_dim = model_cfg.seq_dim,
        dropout = model_cfg.dropout,
        dim=model_cfg.dim,
        num_layers=model_cfg.num_layers,
        heads=model_cfg.heads,
        titan_level=titan_spec,
        cms_levels=cms_specs,
        optimizers=optimizer_cfg,
        teach_scale=model_cfg.get("teach_scale", 1.0),
        teach_clip=model_cfg.get("teach_clip", 0.0),
        teach_schedule=(
            OmegaConf.to_container(model_cfg.teach_schedule, resolve=True)
            if "teach_schedule" in model_cfg
            else {}
        ),
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", False),
    )


    return HOPEDQN(dqn_cfg)


def compute_rl_teach_signal(
    model: torch.nn.Module,
    q_values: torch.Tensor,
    actions: torch.Tensor,
    target_q: torch.Tensor,
) -> torch.Tensor:
    """
    計算 RL 版本的教學訊號。
    原理：我們計算 Q-value 的誤差 (TD Error)，並將其反向投影回 hidden state 空間。
    這告訴 HOPE Blocks：「為了修正這個動作的價值評估錯誤，Hidden State 應該往哪個方向移動」。
    """
    # 1. 取得當前動作的 Q 值
    current_q = q_values.gather(1, actions)

    # 2. 計算殘差 (TD Error), 我們不需要對 target_q 進行微分
    # Residual 形狀: [Batch, 1]
    residual = current_q - target_q.detach()

    # 3. 為了計算對 Hidden State 的影響，我們模擬反向傳播的第一步
    # 取得 Q-Head (Linear Layer) 的權重: [Action_Dim, Hidden_Dim]
    head_weight = model.action_head.weight.detach()

    # 4. 我們只關心被選擇的那個 Action 的權重列
    # 這裡我們建立一個稀疏的梯度向量，只有被選中的 action 位置有 residual 值
    grad_at_output = torch.zeros_like(q_values)
    grad_at_output.scatter_(1, actions, residual)

    # 5. 投影回 Hidden Dimension
    # [Batch, Action_Dim] @ [Action_Dim, Hidden_Dim] -> [Batch, Hidden_Dim]
    # 這代表了 "Loss 對 Hidden State 的梯度方向"
    teach_signal = grad_at_output @ head_weight

    return teach_signal


# ... (保留原有的 checkpoint 和 logging 輔助函數，這裡省略以節省空間，與原版相同) ...
# 注意: maybe_save_checkpoint 需保留
# 注意: write_checkpoint_metadata 需保留


def maybe_save_checkpoint(cfg, model, optimizer, *, step, total_steps, **kwargs):
    # 簡化的存檔邏輯，直接調用 torch.save 即可，邏輯參照原版
    ckpt_cfg = cfg.train.get("checkpoint")
    if not ckpt_cfg or not ckpt_cfg.get("enable", False):
        return

    save_interval = ckpt_cfg.get("save_interval", total_steps)
    if (step + 1) % save_interval == 0:
        ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"step_{step+1}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            path,
        )
        print(f"Saved checkpoint to {path}")


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def prepare_agent(Network: HOPEDQN, device: str, epsilon_start=0.9, epsilon_end=0.05):
    # 貪婪的選擇器
    selector = EpsilonGreedyActionSelector(epsilon_start, epsilon_stop=epsilon_end)

    agent = DQNAgent(Network, selector, device=device)

    return agent


def run_dqn_training_loop(
    cfg: DictConfig,
    *,
    device: torch.device,
) -> Dict[str, float]:

    # 1.參數設定
    batch_size = cfg.train.batch_size
    epsilon_start = cfg.train.get("epsilon_start", 0.9)
    epsilon_end = cfg.train.get("epsilon_end", 0.05)
    epsilon_decay = cfg.train.get("epsilon_decay", 1000)
    target_update_freq = cfg.train.get("target_update_freq", 1000)
    log_interval = cfg.train.get("log_interval", 100)
    logger = init_logger(getattr(cfg, "logging", None), cfg)
    gamma = cfg.train["gamma"]
    step = 0



    # 2.建立環境
    train_env = TrainingEnv(cfg.train)
    

    # update env config
    cfg.model.state_dim = train_env.engine_info()["data_input_size"]
    cfg.model.action_dim = int(train_env.engine_info()["action_space_n"])
    cfg.model.time_dim = train_env.engine_info()["time_input_size"]
    cfg.model.seq_dim = cfg.train.bars_count

    

    # 3. 建立 Policy Network 和 Target Network
    policy_net = build_model_from_cfg(cfg.model).to(device)
    target_net = build_model_from_cfg(cfg.model).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target Net 不訓練

    # 4.建立 Agent
    agent = prepare_agent(
        policy_net,
        device=cfg.train.device,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
    )


    # 已針對dain 進行優化
    optimizer_instance = opitmizar(
        net=policy_net, learning_rate=cfg.optim.lr, lambda_l2=cfg.optim.weight_decay
    ).get_optimizer()

    
    
    
    # 3. 初始化 Replay Buffer
    # memory = SequentialReplayBuffer(
    #     buffer_size= cfg.train.buffer_size,
    #     each_buffer_size=cfg.train.each_buffer_size,
    #     stanstandard_size=cfg.train.stanstandard_size,       
    # )
    
    exp_source = ExperienceSourceFirstLast(
        train_env, agent, cfg.train.gamma, steps_count=cfg.train.REWARD_STEPS)


    buffer = SequentialExperienceReplayBuffer(
        exp_source,
        buffer_size = cfg.train.buffer_size,
        each_buffer_size = cfg.train.each_buffer_size,
        stanstandard_size = cfg.train.stanstandard_size
    )







    
    while True:
        print("step :",step)
        step +=1

        # --- Action Selection (Epsilon Greedy) ---
        epsilon = max((epsilon_start - epsilon_end) * math.exp(
            -1.0 * step / epsilon_decay
        ),epsilon_end) 


        agent.action_selector.update_epsilon(epsilon)
        
        buffer.populate(1)

        # --- Training Step ---
        if not buffer.is_ready():
            continue
        
        


        batch_exp = buffer.sample(batch_size = cfg.train.batch_size)


        loss_v, td_errors = update_policy(
            batch_exp, policy_net, target_net, cfg.train.gamma ** cfg.train.REWARD_STEPS, optimizer_instance=optimizer_instance ,device=device)
        

        
        print(loss_v)
        time.sleep(100)
        
        






        
        
        
        
        
        


        # --- Target Net Update ---
        if step % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        
        
        
        # --- Logging ---
        if step % log_interval == 0:
            metrics_payload = {
                "loss": loss.item(),
                "epsilon": epsilon,
                "q_mean": current_q_a.mean().item(),
                "teach_signal_norm": teach_signal_norm,
            }
            metrics_payload.update(update_metrics)
            logger.log(metrics_payload, step=step)
            print(
                f"[Step {step}] Loss: {loss.item():.4f} | Epsilon: {epsilon:.2f} | TeachNorm: {teach_signal_norm:.4f}"
            )

        maybe_save_checkpoint(
            cfg, policy_net, optimizer_instance, step=step, total_steps=steps
        )





@hydra.main(config_path="configs", config_name="dqn_pilot", version_base=None)
def main(cfg: DictConfig) -> None:
    UpdateConfig(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dqn_training_loop(cfg, device=device)


if __name__ == "__main__":
    main()
