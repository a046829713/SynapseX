import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gymnasium as gym
import torch.multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt
import time

class ActorCritic(nn.Module):  # 定義演員—評論家模型
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4, 25)  # CartPole state維度為4
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)

    def forward(self, x):
        x = F.normalize(x, dim=0)  # 正規化輸入資料
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = self.actor_lin1(
            y)        
          # 演員端輸出兩個動作(向左或向右)的對數機率值

        
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic


def run_episode(worker_env, worker_model):
    # 使用新版 gymnasium 的 reset()
    state, info = worker_env.reset()
    state = torch.from_numpy(state).float()
    values, logprobs, rewards = [], [], []
    done = False
    while not done:
        policy, value = worker_model(state)
        values.append(value)
        logits = policy.view(-1)

        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()        
        logprob_ =action_dist.log_prob(action)
        logprobs.append(logprob_)

        # 使用新版 gymnasium 的 step()
        obs, _, terminated, truncated, info = worker_env.step(action.item())
        done = terminated or truncated
        if done:
            reward = -10
            worker_env.reset()  # 遊戲結束後重置環境
        else:
            reward = 1.0
        rewards.append(reward)
        state = torch.from_numpy(obs).float()
    
    return values, logprobs, rewards, len(rewards)


def update_params(worker_opt, values, logprobs, rewards, clc=0.1, gamma=0.95):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    print(torch.stack(values).flip(dims=(0,)))
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    time.sleep(100)
    Returns = []
    ret_ = torch.Tensor([0])
    
    for r in range(rewards.shape[0]):
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)
    
    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns, dim=0)
    
    actor_loss = -1 * logprobs * (Returns - values.detach())
    critic_loss = (values - Returns) ** 2

    loss = actor_loss.sum() + clc * critic_loss.sum()
    loss.backward()
    worker_opt.step()
    
    return actor_loss, critic_loss, len(rewards)


def worker(t, worker_model, counter, params, buffer):
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4, params=worker_model.parameters())
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards, length = run_episode(worker_env, worker_model)
        actor_loss, critic_loss, eplen = update_params(worker_opt, values, logprobs, rewards)
        
        counter.value = counter.value + 1

        if i % 10 == 0:
            print(f"目前次數:{i}", eplen)
            buffer.put(length)


if __name__ == '__main__':
    
    MasterNode = ActorCritic()
    MasterNode.share_memory()
    processes = []
    params = {
        'epochs': 500,
        'n_workers': 1,
    }
    counter = mp.Value('i', 0)
    buffer = mp.Queue()

    for i in range(params['n_workers']):
        p = mp.Process(target=worker, args=(i, MasterNode, counter, params, buffer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



    print("結束測試")
    # 計算並繪製平均遊戲長度
    score = []
    while not buffer.empty():
        score.append(buffer.get())
    
    data_points = len(score)  # 應該是 params['epochs']/10

    running_mean = []
    total = 0.0
    window = 10  # 例如滑動視窗若是5，代表最近5筆記錄(即50個epoch)

    for idx in range(data_points):
        # idx的score對應實際epoch為 idx*10
        # 計算滑動平均(最近5筆)
        start = max(0, idx - window + 1)
        window_scores = score[start: idx+1]
        mean = sum(window_scores) / len(window_scores)
        running_mean.append(mean)

    plt.figure(figsize=(17, 12))
    plt.ylabel("Mean Episode Length", fontsize=17)
    plt.xlabel("Training Epochs", fontsize=17)
    plt.plot(running_mean)
    plt.show()