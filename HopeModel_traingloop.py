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
from Brain.DQN.lib.environment import TrainingEnv
from utils.AppSetting import UpdateConfig

@dataclass
class DistributedContext:
    rank: int
    world_size: int
    device: torch.device

class ReplayBuffer:
    """標準的 DQN Replay Buffer"""
    def __init__(self, capacity: int, state_dim: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (
            torch.as_tensor(np.array(state), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(action), dtype=torch.long, device=device).unsqueeze(1),
            torch.as_tensor(np.array(reward), dtype=torch.float32, device=device).unsqueeze(1),
            torch.as_tensor(np.array(next_state), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(done), dtype=torch.float32, device=device).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)



def build_model_from_cfg(model_cfg: DictConfig) -> torch.nn.Module:
    # 建立 DQN 模型
    optimizer_cfg = {}
    if "optimizers" in model_cfg:
        optimizer_cfg = OmegaConf.to_container(model_cfg.optimizers, resolve=True)
    
    titan_spec = LevelSpec(**model_cfg.titan_level)
    cms_specs = [LevelSpec(**entry) for entry in model_cfg.cms_levels]
    
    dqn_cfg = DQNConfig(
        state_dim=model_cfg.state_dim,   # 需在 config 中定義
        action_dim=model_cfg.action_dim, # 需在 config 中定義
        dim=model_cfg.dim,
        num_layers=model_cfg.num_layers,
        heads=model_cfg.heads,
        titan_level=titan_spec,
        cms_levels=cms_specs,
        optimizers=optimizer_cfg,
        teach_scale=model_cfg.get("teach_scale", 1.0),
        teach_clip=model_cfg.get("teach_clip", 0.0),
        teach_schedule=OmegaConf.to_container(model_cfg.teach_schedule, resolve=True) if "teach_schedule" in model_cfg else {},
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", False),
    )
    return HOPEDQN(dqn_cfg)

def compute_rl_teach_signal(model: torch.nn.Module, q_values: torch.Tensor, actions: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
    """
    計算 RL 版本的教學訊號。
    原理：我們計算 Q-value 的誤差 (TD Error)，並將其反向投影回 hidden state 空間。
    這告訴 HOPE Blocks：「為了修正這個動作的價值評估錯誤，Hidden State 應該往哪個方向移動」。
    """
    # 1. 取得當前動作的 Q 值
    current_q = q_values.gather(1, actions)
    
    # 2. 計算殘差 (TD Error), 我們不需要對 target_q 進行微分
    # Residual 形狀: [Batch, 1]
    residual = (current_q - target_q.detach()) 
    
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
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True)
        }, path)
        print(f"Saved checkpoint to {path}")


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)



def run_dqn_training_loop(
    cfg: DictConfig,
    *,
    device: torch.device,
) -> Dict[str, float]:
    

        
    env = TrainingEnv(cfg)
    print(env)
    time.sleep(100)

    # 2. 建立 Policy Network 和 Target Network
    policy_net = build_model_from_cfg(cfg.model).to(device)

    target_net = build_model_from_cfg(cfg.model).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() # Target Net 不訓練

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=cfg.optim.lr)
    
    
    # 3. 初始化 Replay Buffer
    buffer_size = cfg.train.get("buffer_size", 100000)
    memory = ReplayBuffer(buffer_size, cfg.model.state_dim)
    
    # 參數設定
    batch_size = cfg.train.batch_size
    gamma = cfg.train["gamma"]
    epsilon_start = cfg.train.get("epsilon_start", 0.9)
    epsilon_end = cfg.train.get("epsilon_end", 0.05)
    epsilon_decay = cfg.train.get("epsilon_decay", 1000)
    target_update_freq = cfg.train.get("target_update_freq", 1000)
    steps = cfg.train.steps
    log_interval = cfg.train.get("log_interval", 100)
    logger = init_logger(getattr(cfg, "logging", None), cfg)
    state = env.reset()



    metrics: Dict[str, float] = {}
    
    print(f"Start training DQN for {steps} steps...")
    for step in range(steps):
        print(step)

        # --- Action Selection (Epsilon Greedy) ---
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                  math.exp(-1. * step / epsilon_decay)
        
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        else:
            action = env.sample_action() # 隨機動作

        # --- Interaction ---
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        memory.push(state, action, reward, next_state, done)
        state = next_state
        
        if done:
            state, _ = env.reset()

        # --- Training Step ---
        if len(memory) < batch_size:
            continue

        # 1. 取樣
        b_state, b_action, b_reward, b_next_state, b_done = memory.sample(batch_size, device)

        # 2. 計算 Current Q
        # [Batch, Action_Dim]
        current_q_values = policy_net(b_state) 
        # [Batch, 1], 取出對應動作的 Q 值
        current_q_a = current_q_values.gather(1, b_action)

        # 3. 計算 Target Q (使用 Double DQN 或標準 DQN)
        with torch.no_grad():
            next_q_values = target_net(b_next_state)
            max_next_q = next_q_values.max(1)[0].unsqueeze(1)
            expected_q = b_reward + (gamma * max_next_q * (1 - b_done))

        # 4. 計算 Loss
        loss = F.mse_loss(current_q_a, expected_q)

        # 5. 優化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()
        
        # --- HOPE Special: Teach Signal Injection ---
        # 這裡我們再次執行一次 forward (不計算梯度)，但是帶入 teach_signal
        # 這樣可以讓 HOPE Block 內部的 Slow/Fast weights 根據 "Surprise" 進行更新
        update_metrics = {}
        with torch.no_grad():
            # 計算 RL 版本的 Teach Signal (基於 TD Error 的梯度方向)
            teach_signal = compute_rl_teach_signal(policy_net, current_q_values, b_action, expected_q)
            teach_signal_norm = teach_signal.norm(dim=-1).mean().item()
            
            # 將訊號注入模型，觸發 HOPE Block 內部更新
            policy_net(b_state, teach_signal=teach_signal)
            
            if hasattr(policy_net, "pop_update_metrics"):
                update_metrics = policy_net.pop_update_metrics()

        # --- Target Net Update ---
        if step % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # --- Logging ---
        if step % log_interval == 0:
            metrics_payload = {
                "loss": loss.item(), 
                "epsilon": epsilon, 
                "q_mean": current_q_a.mean().item(),
                "teach_signal_norm": teach_signal_norm
            }
            metrics_payload.update(update_metrics)
            logger.log(metrics_payload, step=step)
            print(f"[Step {step}] Loss: {loss.item():.4f} | Epsilon: {epsilon:.2f} | TeachNorm: {teach_signal_norm:.4f}")

        maybe_save_checkpoint(
            cfg, policy_net, optimizer, 
            step=step, total_steps=steps
        )

    logger.finish()
    return metrics

@hydra.main(config_path="configs", config_name="dqn_pilot", version_base=None)
def main(cfg: DictConfig) -> None:
    UpdateConfig(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dqn_training_loop(cfg, device=device)

if __name__ == "__main__":
    main()