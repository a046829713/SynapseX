import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque, namedtuple
import time
import os
from model import MlpPolicy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


Transition = namedtuple('Transition',
                        ['state', 'action', 'logp', 'reward', 'next_state', 'done', 'value'])

class RolloutBuffer:
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        self.buffer = []

    def store(self, *args):
        self.buffer.append(Transition(*args))

    def compute_gae(self, last_value):
        rewards, values, dones = [], [], []
        for t in self.buffer:
            rewards.append(t.reward)
            values.append(t.value.item())
            dones.append(t.done)
        
        
        values = values + [last_value]
        gae, returns = 0, []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        
        # 把 advantages 與 returns 放回 transitions
        advantages = np.array(returns) - np.array(values[:-1])
        for idx, tr in enumerate(self.buffer):
            self.buffer[idx] = tr._replace(reward=returns[idx], value=advantages[idx])
        
        return self.buffer

    def clear(self):
        self.buffer = []



def ppo_update(policy_value_net, optimizer, transitions, clip_eps=0.2, epochs=10, batch_size=64):
    """
        this Version implement PPO Clip

        Loss function:
            L(\theta) = L^{\mathrm{CLIP}}(\theta) + c_{1}\,L^{\mathrm{VF}}(\theta) - c_{2}\,L^{\mathrm{H}}(\theta)

            C1(float):0.5 
            C2(float):0.01

        L Clip LaTex:
            L^{\mathrm{CLIP}}(\theta)
                = \hat{\mathbb{E}}_t\!\Bigl[
                    \min\Bigl(
                        r_t(\theta)\,\hat{A}_t,\;
                        \operatorname{clip}\!\bigl(r_t(\theta),\,1-\varepsilon,\,1+\varepsilon\bigr)\,\hat{A}_t
                    \Bigr)
                \Bigr]

        to use in importance sampling.
        Ratio:
            r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
    """
    # 轉為 Tensor
    states      = torch.stack([torch.as_tensor(tr.state,  dtype=torch.float32) for tr in transitions]).to(device)
    actions     = torch.as_tensor([tr.action  for tr in transitions], device=device)
    old_logps   = torch.as_tensor([tr.logp    for tr in transitions], device=device)
    returns     = torch.as_tensor([tr.reward  for tr in transitions], device=device)
    advantages  = torch.as_tensor([tr.value   for tr in transitions], device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset = torch.utils.data.TensorDataset(states, actions, old_logps, returns, advantages)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for s, a, old_lp, ret, adv in loader:
            logp_new, ent , value = policy_value_net.evaluate_actions(s, a)
            entropy = ent.mean()

            # policy ratio
            ratio = torch.exp(logp_new - old_lp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # value loss
            value_loss = nn.functional.mse_loss(value.view(-1), ret)


            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def save_model(model: torch.nn.Module,
               filepath: str = 'model.pt'):
    """
    儲存模型和（可選）優化器的狀態。

    參數：
    - model: 要儲存的 torch.nn.Module    
    - filepath: 檔案路徑（含檔名），例如 'ppo_cartpole.pt'

    儲存內容：
    - 'model_state_dict': model.state_dict()
    """
    # 準備要儲存的 dict
    checkpoint = {
        'model_state_dict': model.state_dict()
    }

    # 確保資料夾存在
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    # 儲存
    torch.save(checkpoint, filepath)
    print(f">>> Saved checkpoint to '{filepath}'")

# 範例使用：
# save_model(net, optimizer, 'ppo_cartpole.pth', epoch=100, extra={'reward_mean': 195.0})


def train(env_name='CartPole-v1', total_timesteps=100000):
    env = gym.make(env_name)
    net = MlpPolicy(env.observation_space, env.action_space).to(device)
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    buffer = RolloutBuffer()

    state = env.reset()[0]
    
    episode_reward = 0
    timesteps = 0

    while timesteps < total_timesteps:
        # collect rollout
        for _ in range(2048):
            state_tensor = torch.tensor(state, dtype=torch.float32 ,device=device)

            # In data collection we don't need computational graph
            with torch.no_grad():
               out = net(state_tensor)

            next_state, reward, Terminated, Truncated, info= env.step(out.action.item())
            episode_reward += reward            
            done = Terminated or Truncated            
            buffer.store(state, out.action,  out.log_prob, reward, next_state, done, out.value)

            state = next_state
            timesteps += 1

            if done:
                print(f"Episode return: {episode_reward:.2f}")
                state, episode_reward = env.reset()[0], 0

        # GAE 與 PPO 更新
        with torch.no_grad():
            out = net(torch.as_tensor(state, dtype=torch.float32, device=device))
        
        
        transitions = buffer.compute_gae(out.value.item())
        ppo_update(net, optimizer, transitions)
        buffer.clear()
    

    save_model(net)



train()