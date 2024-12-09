import torch
import torch.multiprocessing as mp
from Brain.A2C.lib.Prepare import RL_prepare
import time
import os
from Brain.Common.PytorchModelTool import ModelTool
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import time
from torch.utils.tensorboard import SummaryWriter

class WorkerRunner(RL_prepare):
    def __init__(self, 
                 worker_id,
                 global_model,
                 optimizer,
                 device,
                 env_maker,
                 num_steps,
                 gamma,
                 value_loss_coef,
                 entropy_coef,
                 tau,
                 checkpoint_every_step,
                 model_tool,
                 writer_path,
                 symbols):
        super().__init__()
        self.worker_id = worker_id
        self.global_model = global_model
        self.optimizer = optimizer
        self.device = device
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.tau = tau
        self.checkpoint_every_step = checkpoint_every_step
        self.model_tool = model_tool
        self.writer = SummaryWriter(log_dir=f"{writer_path}/worker_{worker_id}")
        
        # 為這個 worker 建立自己的環境(僅一個環境)
        # 若需要多環境並行在同一 worker，可以自行擴充
        self.env = env_maker(symbols)
        
        # 建立本地模型並同步全局模型參數
        self.local_model = type(self.global_model)().to(self.device)
        self.local_model.load_state_dict(self.global_model.state_dict())
        
        # 訓練超參數
        self.num_steps = num_steps
        self.episode_rewards = 0
        self.global_step = 0

    def run(self):
        obs = self.env.reset()
        done = False
        episode_rewards = 0
        start_time = time.time()
        last_time = start_time
        frame_count = 0

        while True:
            # 在本地暫存多個 step 的資料，然後一起更新
            values = []
            log_probs = []
            rewards = []
            entropies = []
            masks = []
            
            for step in range(self.num_steps):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action_probs, state_value = self.local_model(obs_tensor)
                dist = Categorical(logits=action_probs)
                action = dist.sample()
                
                log_prob = dist.log_prob(action)
                entropy = dist.entropy().mean()
                
                obs_next, reward, done, info = self.env.step(action.item())
                episode_rewards += reward

                mask = 0.0 if done else 1.0
                
                values.append(state_value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)
                masks.append(mask)

                obs = obs_next
                
                self.global_step += 1
                frame_count += 1

                if done:
                    obs = self.env.reset()
                    self.writer.add_scalar('Episode Reward', episode_rewards, self.global_step)
                    episode_rewards = 0
            
            # 最後一個 state 的 value，用於計算 targets
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, next_value = self.local_model(obs_tensor)
            
            # 計算優勢與目標
            returns = []
            R = next_value.detach()
            for step in reversed(range(self.num_steps)):
                R = rewards[step] + self.gamma * R * masks[step]
                returns.insert(0, R)
            
            returns = torch.cat(returns).detach()
            values = torch.cat(values)
            log_probs = torch.cat(log_probs)
            entropies = torch.stack(entropies)
            
            advantages = returns - values
            
            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_loss = entropies.mean()
            
            total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
            
            # 將梯度更新應用到全局模型
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=0.5)

            # 將本地模型梯度複製到全局模型並更新
            for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
                if global_param.grad is None:
                    global_param.grad = local_param.grad
                else:
                    global_param.grad.copy_(local_param.grad)
            
            self.optimizer.step()
            
            # 更新完畢後，將全局模型參數複製回本地模型
            self.local_model.load_state_dict(self.global_model.state_dict())
            
            # 可根據需要增加 soft_update 或省略
            # 在 A3C 中通常不需要 target_model，如需 target_model 可以加入相同邏輯

            # 記錄
            self.writer.add_scalar('Total Loss', total_loss.item(), self.global_step)
            self.writer.add_scalar('Value Loss', value_loss.item(), self.global_step)
            self.writer.add_scalar('Policy Loss', policy_loss.item(), self.global_step)
            self.writer.add_scalar('Entropy', entropy_loss.item(), self.global_step)
            
            current_time = time.time()
            elapsed = current_time - last_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                last_time = current_time
                self.writer.add_scalar('FPS', fps, self.global_step)
                print(f"[Worker {self.worker_id}] Step: {self.global_step}, FPS: {fps:.2f}")

            if self.global_step % self.checkpoint_every_step == 0:
                self.model_tool.save_checkpoint({
                    'model_state_dict': self.global_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(self.SAVES_PATH, f"checkpoint.pt"))




# 這是全局共享模型 (Global Model)
def global_model_init(model_cls, device):
    # 初始化全局模型，放在 CPU 或 GPU
    global_model = model_cls().to(device)
    global_model.share_memory()  # multiprocessing 時需要 share_memory()
    return global_model

def worker_process(worker_id, global_model, optimizer, device, env_maker, num_steps, gamma, value_loss_coef, entropy_coef, tau, checkpoint_every_step, model_tool, writer_path, symbols):
    # 每個 worker 都有自己的 Runner 實例
    runner = WorkerRunner(
        worker_id=worker_id,
        global_model=global_model,
        optimizer=optimizer,
        device=device,
        env_maker=env_maker,
        num_steps=num_steps,
        gamma=gamma,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_coef,
        tau=tau,
        checkpoint_every_step=checkpoint_every_step,
        model_tool=model_tool,
        writer_path=writer_path,
        symbols=symbols
    )
    runner.run()

if __name__ == "__main__":
    # 初始化
    device = torch.device("cpu")  # A3C常用CPU，若想用GPU可嘗試但注意同步問題
    num_workers = 4  # 設定 worker 數量
    gamma = 0.99
    value_loss_coef = 0.5
    entropy_coef = 0.01
    tau = 0.005
    checkpoint_every_step = 10000
    num_steps = 5  # 每個 worker 在送回梯度前收集多少步數
    
    # 環境與模型
    # 假設有一個函數 env_maker(symbol) 可建立環境
    # 你可以根據實務狀況將每個 worker 分配不同 symbol
    symbols = ['BTC', 'ETH', 'XRP', 'LTC']  # 舉例
    model_cls = ...  # 你的 ActorCriticModel 類別
    global_model = global_model_init(model_cls, device)

    optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-4)
    
    model_tool = ModelTool()
    writer_path = "runs"

    processes = []
    for i in range(num_workers):
        p = mp.Process(target=worker_process,
                       args=(i, global_model, optimizer, device, env_maker, num_steps, gamma, value_loss_coef, entropy_coef, tau, checkpoint_every_step, model_tool, writer_path, symbols[i]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
