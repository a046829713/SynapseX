import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from A2C.lib.Prepare import RL_prepare
import time

class Runner(RL_prepare):
    def __init__(self):
        super().__init__()
        self.num_envs = len(self.train_envs)
        self.train()

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            # 重置环境并获取初始状态
            obs = [env.reset() for env in self.train_envs]
            done = [False] * self.num_envs
            episode_rewards = [0] * self.num_envs

            # 存储每个时间步的数据
            log_probs_list = []
            values_list = []
            rewards_list = []
            entropies_list = []
            masks = []
            next_obs = obs  # 初始化next_obs

            while not all(done):
                # 将观测值转换为张量
                obs_tensor = torch.stack([torch.tensor(o, dtype=torch.float32) for o in obs]).to(self.device)
                # 获取动作概率和状态值
                action_probs, state_values = self.model(obs_tensor)

                # 采样动作并计算对数概率
                dist = Categorical(logits=action_probs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy()
                print(action_probs)
                print(entropy)
                time.sleep(100)

                # 与环境交互
                next_obs = []
                rewards = []
                for i, env in enumerate(self.train_envs):
                    if not done[i]:
                        obs_i, reward_i, done_i, _ = env.step(actions[i].item())
                        next_obs.append(obs_i)
                        rewards.append(reward_i)
                        episode_rewards[i] += reward_i
                        done[i] = done_i
                    else:
                        next_obs.append(obs[i])
                        rewards.append(0.0)

                # 存储数据
                log_probs_list.append(log_probs)
                values_list.append(state_values.squeeze())
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.DEVICE)
                rewards_list.append(rewards_tensor)
                entropies_list.append(entropy)
                mask = torch.tensor([0.0 if d else 1.0 for d in done], dtype=torch.float32).to(self.DEVICE)
                masks.append(mask)

                # 准备下一个时间步
                obs = next_obs

            # 计算累计回报和优势
            returns = self._compute_returns(rewards_list, masks, obs)
            log_probs_tensor = torch.stack(log_probs_list)
            values_tensor = torch.stack(values_list)
            advantages = returns - values_tensor

            # 计算损失
            policy_loss = - (log_probs_tensor * advantages.detach()).mean()
            value_loss = F.mse_loss(values_tensor, returns)
            entropy_loss = - torch.stack(entropies_list).mean()
            total_loss = policy_loss + self.VALUE_LOSS_COEF * value_loss + self.ENTROPY_COEF * entropy_loss

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

            # 更新目标网络
            self.soft_update()

            # 记录日志
            avg_reward = sum(episode_rewards) / self.num_envs
            self.writer.add_scalar('Average Reward', avg_reward, episode)
            print(f"Episode {episode} - Average Reward: {avg_reward}")

    def _compute_returns(self, rewards_list, masks, next_obs):
        returns = []
        with torch.no_grad():
            # 获取目标网络对下一个状态的价值估计
            next_obs_tensor = torch.stack([torch.tensor(o, dtype=torch.float32) for o in next_obs]).to(self.DEVICE)
            _, next_state_values = self.target_model(next_obs_tensor)
            R = next_state_values.squeeze()
        for step in reversed(range(len(rewards_list))):
            R = rewards_list[step] + self.GAMMA * R * masks[step]
            returns.insert(0, R)
        returns = torch.stack(returns)
        return returns.detach()

    def soft_update(self, tau=0.005):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


Runner()