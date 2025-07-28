import sys
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from utils.Debug_tool import debug
from typing import Optional,Union
class RewardTracker:
    def __init__(self, stop_reward, group_rewards=1):
        """用來追蹤紀錄獎勵資訊

        Args:
            writer (_type_): _description_
            stop_reward (_type_): _description_
            group_rewards (int, optional): _description_. Defaults to 1.
        """
        self.stop_reward = stop_reward
        self.reward_buf = []
        self.steps_buf = []
        self.group_rewards = group_rewards

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        self.total_steps = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward_steps, frame, epsilon=None) -> Union[bool,np.float64]:
        """
            reward_steps: (-5.116108441842309, 1000)
            frame: 3004
            epsilon: 0.9998331111111111
            Result Type: <class 'bool'>
            <class 'numpy.float64'>
        """

        reward, steps = reward_steps
        self.reward_buf.append(reward)
        self.steps_buf.append(steps)
        
        # 每兩個group 顯示一次
        if len(self.reward_buf) < self.group_rewards:
            return False
        
        reward = np.mean(self.reward_buf)
        steps = np.mean(self.steps_buf)

        self.reward_buf.clear()
        self.steps_buf.clear()

        self.total_rewards.append(reward)        
        self.total_steps.append(steps)

        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        
        mean_reward = np.mean(self.total_rewards[-100:])
        mean_steps = np.mean(self.total_steps[-100:])
        
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        
        print("%d: done %d games, mean reward %.3f, mean steps %.2f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards) *
            self.group_rewards, mean_reward, mean_steps, speed, epsilon_str
        ))

        sys.stdout.flush()
        
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        
        return mean_reward


def calc_values_of_states(states, net, device="cpu"):
    """    action_values_v = net(states_v)：這裡，模型net為給定的states_v預測每個可能動作的價值。

    best_action_values_v = action_values_v.max(1)[0]：接著，我們只考慮每個狀態的最佳動作價值，這是透過取每一行（代表每個狀態）的最大值來完成的。

    結果是所有狀態的最佳動作價值的均值。

    回答你的問題：“假設我部位完全沒有任何的變化下，收益為什麼會改變？”：

    即使部位方向不變，模型的權重和偏差是在訓練過程中不斷更新的。所以，當你使用同一組states重新評估模型時，你會得到不同的動作價值，因為模型已經學到了新的知識。

    動作價值的改變並不直接代表收益的改變。它只是模型對給定狀態應該採取何種動作的估計價值。當你在真實環境中執行這些動作時，真正的收益可能會與模型的估計有所不同。

    訓練過程中，模型試圖學習一個策略，使其預測的動作價值越來越接近真實價值。但這不代表模型總是正確的，只是說它試圖接近真實價值。

    所以，雖然部位方向不變，但模型的估計動作價值可能會變，這反映了模型在訓練過程中的學習進展。

    Args:
        states (_type_): _description_
        net (_type_): _description_
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """

    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


def unpack_batch(batch):
    states, actions, rewards, dones, last_states,infos,last_infos = [], [], [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        infos.append(exp.info)
        last_infos.append(exp.last_info)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
        np.array(dones, dtype=np.uint8), np.array(last_states, copy=False), np.array(infos, copy=False), np.array(last_infos, copy=False)

def turn_to_tensor(infos,device):
    # 使用 NumPy 快速處理
    output_array = np.array([[info.get('postion', 0.0), info.get('diff_percent', 0.0)] for info in infos], dtype=np.float32)
    output_tensor = torch.from_numpy(output_array).to(device)
    return output_tensor 

def calc_loss(batch, net, tgt_net, gamma, moe_loss_coeff=0.01, device="cpu"):
    """
    計算 DQN 的 MSE loss，並同時計算每筆 transition 的 TD‐error，並整合 MoE 的輔助損失 (auxiliary loss)。
    """
    states, actions, rewards, dones, next_states, _, _ = unpack_batch(batch)
    states_v = torch.tensor(states, device=device, dtype=torch.float32)
    next_states_v = torch.tensor(next_states, device=device, dtype=torch.float32)
    actions_v = torch.tensor(actions, device=device, dtype=torch.long)
    rewards_v = torch.tensor(rewards, device=device, dtype=torch.float32)
    done_mask = torch.tensor(dones, device=device, dtype=torch.bool)

    # --- 線上網路 (net) ---
    # 1. 從 MoE 模型獲取 Q 值和輔助損失
    q_values, aux_loss = net(states_v)

    # 2. 獲取實際採取動作的 Q 值: Q(s,a)
    state_action_values = q_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    # ---目標網路 (tgt_net) & Double DQN ---
    with torch.no_grad():
        # 3. 使用線上網路 `net` 選擇下一狀態的最佳動作 a_max
        next_q_values, _ = net(next_states_v)
        next_actions = next_q_values.max(1)[1]

        # 4. 使用目標網路 `tgt_net` 評估 a_max 的 Q 值: Q_target(s', a_max)
        next_state_q_values, _ = tgt_net(next_states_v)
        next_state_values = next_state_q_values.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
        
        # 5. 對於終止狀態，其未來價值為 0
        next_state_values[done_mask] = 0.0

        # 6. 計算 TD 目標 (y)
        q_targets = rewards_v + gamma * next_state_values


    # --- 計算總損失 ---
    # 7. 計算 DQN Loss (MSE)
    dqn_loss = nn.MSELoss()(state_action_values, q_targets)
    
    # 8. 加上 MoE 的輔助損失
    if aux_loss is not None:
        total_loss = dqn_loss + moe_loss_coeff * aux_loss
    else:
        total_loss = dqn_loss

    # 9. 計算 TD-error (用於 PER)
    with torch.no_grad():
        td_errors = torch.abs(q_targets - state_action_values)

    return total_loss, td_errors


def update_eval_states(buffer, STATES_TO_EVALUATE):
    """    
        用來定時更新驗證資料
    Args:
        buffer (_type_): _description_
        STATES_TO_EVALUATE (_type_): _description_
    """
    eval_states = buffer.sample(STATES_TO_EVALUATE)
    eval_states = [np.array(transition.state, copy=False)
                   for transition in eval_states]
    return np.array(eval_states, copy=False)


