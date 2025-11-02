import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import time
import os
from Brain.Common.PytorchModelTool import ModelTool
import os
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
import time
from abc import ABC, abstractmethod
# from Brain.PPO.lib.model import ActorCriticModel,TransformerModel
import copy 
import pandas as pd
from Brain.PPO2.lib.environment import TrainingEnv
from Brain.PPO2.lib.environment import State_time_step
from Brain.Common.DataFeature import OriginalDataFrature
from Brain.PPO2.lib import model
from Brain.PPO2.lib.experience import RolloutBuffer
from utils.AppSetting import PPO2RLConfig
from Brain.PPO2.lib.Agent import PPO2Agent

def show_setting(title: str, content: str):
    print(f"--{title}--:{content}")

class RL_prepare(ABC):
    def __init__(self):
        self.config = PPO2RLConfig()
        self._prepare_hyperparameters()
        self._prepare_device()
        self._prepare_symbols()
        self._prepare_writer()
        self._prepare_env()
        self._prepare_agent()
        # self._prepare_targer_net()
        # self._prepare_agent()
        self._prepare_optimizer()
        

    def _prepare_device(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        show_setting("DEVICE:", self.device)

    def _prepare_symbols(self):
        symbolNames = os.listdir(os.path.join(os.getcwd() , "Brain","simulation","train_data"))
        symbolNames = [_fileName.split('.')[0] for _fileName in symbolNames]
        unique_symbols = list(set(symbolNames))
        self.config.update_steps_by_symbols(len(unique_symbols))
        self.config.create_saves_path()
        self.config.UNIQUE_SYMBOLS = unique_symbols        
        
        show_setting("SYMBOLNAMES", unique_symbols)

    def _prepare_env(self):
        if self.config.KEYWORD == 'Transformer' or 'Mamba' or "Mamba2":

            # 製作環境
            self.train_env = TrainingEnv(config=self.config)

        show_setting("TrainingEnv", self.train_env)


    def _prepare_writer(self):
        pass


    def _prepare_hyperparameters(self):
        pass
        # self.GAMMA = 0.99
        # self.LEARNING_RATE = 0.00001  # optim 的學習率
        
        # self.ENTROPY_COEF = 0.05  # 熵损失系数
        # self.SAVES_PATH = "saves"  # 儲存的路徑

        # self.CHECKPOINT_EVERY_STEP = 100
        # self.VALUE_LOSS_COEF = 0.1  # 價值損失函數 critic損失函數
        # self.N_STEP = 250
        # self.checkgrad_times = 10

    def _prepare_agent(self):        
        self.Agent = PPO2Agent(self.train_env.observation_space,
                                     self.train_env.action_space,
                                     device=self.device)



    # def _prepare_optimizer(self):
        # # 定義特殊層的學習率
        # param_groups = [
        #     {'params': self.model.dean.mean_layer.parameters(), 'lr': 0.0001 * self.model.dean.mean_lr},
        #     {'params': self.model.dean.scaling_layer.parameters(), 'lr': 0.0001 * self.model.dean.scale_lr},
        #     {'params': self.model.dean.gating_layer.parameters(), 'lr': 0.0001 * self.model.dean.gate_lr},
        # ]

        # # 其餘參數使用默認學習率，直接加入 param_groups
        # for name, param in self.model.named_parameters():
        #     if (
        #         "dean.mean_layer" not in name
        #         and "dean.scaling_layer" not in name
        #         and "dean.gating_layer" not in name
        #     ):
        #         param_groups.append({'params': param, 'lr': self.LEARNING_RATE})

        # # 初始化優化器
        # self.optimizer = optim.Adam(param_groups)
        # self.optimizer = optim.Adam(self.Agent.model.parameters(), lr=3e-4)
    
    def _prepare_optimizer(self, base_lr=1e-4):
        """
        建立 Adam 優化器，以下功能：
        1. `dean` 的三個特殊層 (`mean_layer`, `scaling_layer`, `gating_layer`) 使用獨立的學習率且不做 weight decay。
        2. `LayerNorm` 層和所有 `bias` 參數不做 weight decay。
        3. 其餘參數正常做 weight decay。
        """
        # 存放不同參數組
        decay_params = []
        no_decay_params = []
        
        # 獲取 dean 特殊層的參數 ID，以便後續排除
        dean_params_ids = set()
        if hasattr(self.Agent.model, 'dean'):
            dean_params_ids.update(id(p) for p in self.Agent.model.dean.parameters())

        for name, param in self.Agent.model.named_parameters():
            if not param.requires_grad:
                continue

            # 如果是 dean 層的參數，跳過，因為它們會被單獨處理
            if id(param) in dean_params_ids:
                continue

            # LayerNorm 層和 bias 不做 weight decay
            # 透過 name 來判斷，比 isinstance 更可靠
            if "norm" in name or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # 建立參數組
        param_groups = [
            {
                'params': decay_params,
                'lr': self.config.LEARNING_RATE,
                'weight_decay': self.config.LAMBDA_L2
            },
            {
                'params': no_decay_params,
                'lr': self.config.LEARNING_RATE,
                'weight_decay': 0.0
            }
        ]

        # 為 dean 的特殊層添加獨立的參數組
        if hasattr(self.Agent.model, 'dean'):
            param_groups.extend([
                {'params': list(self.Agent.model.dean.mean_layer.parameters()),
                 'lr': base_lr * self.Agent.model.dean.mean_lr, 'weight_decay': 0.0},
                {'params': list(self.Agent.model.dean.scaling_layer.parameters()),
                 'lr': base_lr * self.Agent.model.dean.scale_lr, 'weight_decay': 0.0},
                {'params': list(self.Agent.model.dean.gating_layer.parameters()),
                 'lr': base_lr * self.Agent.model.dean.gate_lr, 'weight_decay': 0.0},
            ])

        # 用 Adam 建立優化器
        self.optimizer = optim.Adam(param_groups)
        print("optimzer create.")

    def _prepare_targer_net(self):
        # 创建目标网络，作为主模型的副本
        self.target_model = copy.deepcopy(self.model)
        # 将目标网络设置为评估模式
        self.target_model.eval()


class PPO2(RL_prepare):
    def __init__(self):
        super().__init__()
        self.train()


    def train(self):                
        buffer = RolloutBuffer()
        state = self.train_env.reset()
        print(state)
        time.sleep(100)
        episode_reward = 0
        timesteps = 0
        while True:
            # collect rollout
            for _ in range(2048):
                state_tensor = torch.tensor(state, dtype=torch.float32 ,device=self.device)

                # In data collection we don't need computational graph
                with torch.no_grad():
                    out = self.Agent.get_action(state_tensor)

                marketpostion, percentage = out.action
                next_state, reward, Terminated, Truncated, info= self.train_env.step(marketpostion.item(),percentage.item())
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


            

# class Runner(RL_prepare):
#     def __init__(self):
#         super().__init__()        
#         self.model_tool = ModelTool()
#         # 記錄計時用，用於計算FPS
#         self.start_time = time.time()
#         self.last_time = self.start_time
#         self.frame_count = 0  # 累計經過的步數
#         self.avg_rewards = []        
#         self.load_pre_train_model_state() 
#         self.train()
    
#     def load_pre_train_model_state(self):
#         # 加載檢查點如果存在的話
#         checkpoint_path = r''
#         if checkpoint_path and os.path.isfile(checkpoint_path):
#             print("資料繼續運算模式")
#             # 標準化路徑並分割
#             self.saves_path = os.path.dirname(checkpoint_path)
#             checkpoint = torch.load(checkpoint_path)
#             self.model.load_state_dict(checkpoint['model_state_dict'])            
#             self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         else:
#             print("建立新的儲存點")
#             # 用來儲存的位置
#             self.saves_path = os.path.join(self.SAVES_PATH, datetime.strftime(
#                 datetime.now(), "%Y%m%d-%H%M%S") + '-' + str(self.BARS_COUNT) + 'k-')

#             os.makedirs(self.saves_path, exist_ok=True)
    
#     def monitoring(self, step, state_values, episode_rewards):
#         # 每隔一定步數（例如每100步）執行監視
#         current_time = time.time()
#         elapsed = current_time - self.last_time


#         if elapsed > 1.0:  # 每1秒更新一次FPS顯示
#             fps = (step - self.frame_count) / elapsed
#             self.frame_count = step
#             self.last_time = current_time

#             # 平均 value
#             avg_value = state_values.mean().item()

#             # 平均 reward 
#             self.avg_rewards.append(sum(episode_rewards.cpu()))
            
#             # 將這些數值寫入log
#             self.writer.add_scalar('FPS', fps, step)
#             self.writer.add_scalar('Average Value', avg_value, step)
#             # Avg Reward 已經在下面程式碼紀錄了，可以同樣在這裡顯示

#             print(
#                 f"[Monitor] Step: {step} | FPS: {fps:.2f} | Avg Value: {avg_value:.4f} | Avg Reward: {np.mean(self.avg_rewards[-100:]):.4f}")

#     def train(self):
#         step = 0
#         while True:
#             step += 1
#             values, logprobs, rewards, entropies, G, check = self.run_episode(self.train_env, self.model)
#             # 在 update_params 中做反向傳播與更新
#             values, rewards = self.update_params(values, logprobs, rewards, entropies, G)
            
#             # 軟更新 target model
#             # self.soft_update()

#             # 記錄與顯示監控資訊
#             self.monitoring(step, values, rewards)

#             # 定期保存模型參數
#             if step % self.CHECKPOINT_EVERY_STEP == 0:
#                 self.model_tool.save_checkpoint({
#                     'model_state_dict': self.model.state_dict(),
#                     'optimizer_state_dict': self.optimizer.state_dict(),
#                 }, os.path.join(self.saves_path, f"checkpoint{int(step/self.CHECKPOINT_EVERY_STEP)}.pt"))
            
#             if step % self.checkgrad_times == 0:
#                 self.checkgrad()

#     def checkgrad(self):
#         # 打印梯度統計數據
#         for name, param in self.model.named_parameters():
#             if param.grad is not None:
#                 print(f"Layer: {name}, Grad Min: {param.grad.min()}, Grad Max: {param.grad.max()}, Grad Mean: {param.grad.mean()}")
#         print('*'*120)
        
#     def soft_update(self, tau=0.005):
#         for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
#             target_param.data.copy_(
#                 tau * param.data + (1.0 - tau) * target_param.data)
    
#     def update_params(self, values, logprobs, rewards, entropies, G):
#         """
#         使用 value, logprobs, rewards, entropies, G, check 來計算 loss 並進行反向傳播更新。
#         check=1 時表示 environment 完整結束，不需bootstrap。
#         check=0 時表示 N-step截斷，使用 G 做bootstrap。
#         """
#         # values, logprobs, entropies, rewards 已在 run_episode() 中 flip 過
#         # 計算 returns
#         Returns = []
#         ret_ = G
#         for r in rewards:
#             ret_ = r + self.GAMMA * ret_
#             Returns.append(ret_)
        

#         Returns = torch.stack(Returns).view(-1)

#         # Returns = F.normalize(Returns, dim=0)
#         Returns = (Returns - Returns.mean()) / (Returns.std() + 1e-8)

        
#         advantages = Returns - values.detach()
#         policy_loss = -(logprobs * advantages).mean()
#         value_loss = F.mse_loss(values, Returns)
#         entropy_loss = entropies.mean()
#         total_loss = policy_loss + self.VALUE_LOSS_COEF * value_loss - self.ENTROPY_COEF * entropy_loss

#         self.optimizer.zero_grad()
#         total_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
#         self.optimizer.step()
#         return values, rewards
    
#     def run_episode(self, worker_env, worker_model):
#         """
#         此函式其實是在 N-step 更新時使用。
#         根據 N_STEP 截斷或環境 done 來決定是否使用 next_value bootstrap。
#         """
#         state = worker_env.reset()
#         state = torch.from_numpy(state).to(self.device).unsqueeze(0)
#         values, logprobs, rewards, entropies = [], [], [], []
#         done = False
#         j = 0

#         while (j < self.N_STEP and not done):
#             policy, value = worker_model(state)
#             values.append(value)
#             logits = policy.view(-1)            
#             action_dist = Categorical(logits=logits)
#             action = action_dist.sample()
#             logprob_ = action_dist.log_prob(action)
#             entropy_ = action_dist.entropy()
#             logprobs.append(logprob_)
#             entropies.append(entropy_)
#             obs, reward, done, info = worker_env.step(action.item())
#             rewards.append(reward)
#             state = torch.from_numpy(obs).to(self.device).unsqueeze(0)
#             j += 1

#         # Episode/N-step結束後決定 G 與 check
#         if done:
#             # 環境真正結束，無需bootstrap，G=0，check=1
#             G = torch.zeros(1, device=self.device)
#             check = 1
#             # 若需要重新開始下一局，可在此重置環境
#             worker_env.reset()
#         else:
#             # N-step截斷，但未真正結束環境，使用最後一個 state's value bootstrap
#             # 上一迭代已計算出 value，即為values最後一個
#             G = values[-1].detach()
#             check = 0

#         # flip 代表反轉的意思
#         values = torch.stack(values).flip(dims=(0,)).view(-1)
#         logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
#         entropies = torch.stack(entropies).flip(dims=(0,)).view(-1)
#         rewards = torch.tensor(rewards, device=self.device).flip(dims=(0,)).view(-1)

#         return values, logprobs, rewards, entropies, G, check


PPO2()