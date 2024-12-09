import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from Brain.A2C.lib.Prepare import RL_prepare
import time
import os
from Brain.Common.PytorchModelTool import ModelTool


class Runner(RL_prepare):
    def __init__(self):
        super().__init__()
        self.num_envs = len(self.train_envs)
        self.model_tool = ModelTool()

        # 記錄計時用，用於計算FPS
        self.start_time = time.time()
        self.last_time = self.start_time
        self.frame_count = 0  # 累計經過的步數
        
        self.train()

    def monitoring(self, step, state_values, episode_rewards):
        # 每隔一定步數（例如每100步）執行監視

        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed > 1.0:  # 每1秒更新一次FPS顯示
            fps = (step - self.frame_count) / elapsed
            self.frame_count = step
            self.last_time = current_time

            # 平均 value
            avg_value = state_values.mean().item()

            # 平均 reward
            avg_reward = sum(episode_rewards) / self.num_envs

            # 將這些數值寫入log
            self.writer.add_scalar('FPS', fps, step)
            self.writer.add_scalar('Average Value', avg_value, step)
            # Avg Reward 已經在下面程式碼紀錄了，可以同樣在這裡顯示

            print(
                f"[Monitor] Step: {step} | FPS: {fps:.2f} | Avg Value: {avg_value:.4f} | Avg Reward: {avg_reward:.4f}")

    def train(self):
        # 重置環境並獲取初始狀態
        obs = [env.reset() for env in self.train_envs]
        done = [False] * self.num_envs
        episode_rewards = [0] * self.num_envs
        step = 0  # 計數器，用於軟更新
        while True:
            step += 1
            # 將觀測值轉換為張量
            obs_tensor = torch.stack(
                [torch.tensor(o, dtype=torch.float32) for o in obs]).to(self.device)
            # 獲取動作概率和狀態值
            action_probs, state_values = self.model(obs_tensor)

            # 采樣動作並計算對數概率
            dist = Categorical(logits=action_probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # 與環境交互
            next_obs = []
            rewards = []
            dones = []
            for i, env in enumerate(self.train_envs):
                if done[i]:
                    obs_i = env.reset()
                    episode_rewards[i] = 0  # 重置该环境的累积奖励
                    done[i] = False  # 重置 done 标志
                    next_obs.append(obs_i)
                    rewards.append(0.0)  # 重置时的奖励为0
                    dones.append(False)  # 重置后的环境未完成
                else:
                    obs_i, reward_i, done_i, info = env.step(actions[i].item())
                    next_obs.append(obs_i)
                    rewards.append(reward_i)
                    episode_rewards[i] += reward_i
                    done[i] = done_i
                    dones.append(done_i)

            # 將獎勵和遮罩轉換為張量
            rewards_tensor = torch.tensor(
                rewards, dtype=torch.float32).to(self.device)
            mask = torch.tensor(
                [0.0 if d else 1.0 for d in dones], dtype=torch.float32).to(self.device)

            # 使用目標網絡計算下一狀態的價值
            next_obs_tensor = torch.stack(
                [torch.tensor(o, dtype=torch.float32) for o in next_obs]).to(self.device)

            with torch.no_grad():
                _, next_state_values = self.target_model(next_obs_tensor)
                next_state_values = next_state_values.squeeze()

            # 計算目標和優勢
            targets = rewards_tensor + self.GAMMA * next_state_values * mask

            advantages = targets - state_values.squeeze()

            # 計算損失
            policy_loss = - (log_probs * advantages.detach()).mean()
            value_loss = F.mse_loss(state_values.squeeze(), targets.detach())
            entropy_loss = - entropy.mean()
            total_loss = policy_loss + self.VALUE_LOSS_COEF * \
                value_loss + self.ENTROPY_COEF * entropy_loss

            # 反向傳播和優化
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

            # 每10個步驟進行一次軟更新
            if step % 100 == 0:
                self.soft_update()

            # 準備下一個時間步
            obs = next_obs

            # 記錄日誌
            avg_reward = sum(episode_rewards) / self.num_envs
            self.writer.add_scalar('Average Reward', avg_reward)
            # 呼叫監視函數來顯示FPS與Value
            self.monitoring(step, state_values, episode_rewards)

            if step % self.CHECKPOINT_EVERY_STEP == 0:
                # 保存檢查點
                self.model_tool.save_checkpoint({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(self.SAVES_PATH, f"checkpoint.pt"))

    def soft_update(self, tau=0.005):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data)


Runner()
