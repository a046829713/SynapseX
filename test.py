import torch
import gymnasium as gym
import numpy as np
from model import MlpPolicy  # 請將 your_module 換成你定義網路的模組名稱
import time

def test_trained_policy(env_name='CartPole-v1',
                        model_path='ppo_cartpole.pt',
                        episodes=10,
                        render=False):
    # 建立環境與網路
    env = gym.make(env_name, render_mode="human")
    net = MlpPolicy(env.observation_space, env.action_space)
    net.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
    net.eval()

    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            if render:
                env.render()

            # 前向計算 action 分布，取最大機率
            state_tensor = torch.tensor(state, dtype=torch.float32)
            
            action, _ = net.act(stochastic=False, obs = state_tensor)

            # 執行 action
            next_state, reward, Terminated, Truncated, _ = env.step(action.item())
            done = Terminated or Truncated
            ep_reward += reward
            state = next_state

        rewards.append(ep_reward)
        print(f"[Episode {ep+1:2d}] Reward = {ep_reward:.2f}")

    env.close()
    print(f"\nAverage Reward over {episodes} episodes: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    return rewards

if __name__ == '__main__':
    # 範例：測試 20 個 episode，不開啟畫面
    test_trained_policy(env_name='CartPole-v1',
                        model_path='model.pt',
                        episodes=20,
                        render=True)