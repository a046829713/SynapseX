import sys
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pandas as pd
import time
from utils.Debug_tool import debug


class RewardTracker:
    def __init__(self, writer, stop_reward, group_rewards=1):
        """用來追蹤紀錄獎勵資訊

        Args:
            writer (_type_): _description_
            stop_reward (_type_): _description_
            group_rewards (int, optional): _description_. Defaults to 1.
        """
        self.writer = writer
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

    def reward(self, reward_steps, frame, epsilon=None):
        reward, steps = reward_steps
        self.reward_buf.append(reward)
        self.steps_buf.append(steps)
        
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
        
        # 每兩個group 顯示一次
        mean_reward = np.mean(self.total_rewards[-100:])
        mean_steps = np.mean(self.total_steps[-100:])
        
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        
        print("%d: done %d games, mean reward %.3f, mean steps %.2f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards) *
            self.group_rewards, mean_reward, mean_steps, speed, epsilon_str
        ))

        sys.stdout.flush()
        
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        self.writer.add_scalar("steps_100", mean_steps, frame)
        self.writer.add_scalar("steps", steps, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


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

def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    """
    ************************************************************************************************************************   
    state_action_values like this : tensor([ 0.0645,  0.0453,  0.0322,  0.0556, -0.0476, -0.0432,  0.0252,  0.0906,
         0.0539,  0.0750,  0.0675,  0.0596, -0.0412,  0.0456,  0.0526, -0.0150,
         0.0530,  0.0434,  0.0388, -0.0372,  0.0480,  0.0358,  0.0743,  0.0275,
         0.0687, -0.0173,  0.0859,  0.0522, -0.0125, -0.0301,  0.0224,  0.0628],
       device='cuda:0', grad_fn=<SqueezeBackward1>)
    """
    states, actions, rewards, dones, next_states, infos, last_infos = unpack_batch(batch)
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)
    infos = turn_to_tensor(infos,device=device)
    last_infos = turn_to_tensor(last_infos,device=device)

    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    
    next_state_actions = net(next_states_v).max(1)[1]

    next_state_values = tgt_net(next_states_v).gather(
        1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    
    next_state_values[done_mask] = 0.0
    # detach 單純的tensor 沒有grad_fn
    expected_state_action_values = next_state_values.detach() * gamma + rewards_v    
    
    return nn.MSELoss()(state_action_values, expected_state_action_values)


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


