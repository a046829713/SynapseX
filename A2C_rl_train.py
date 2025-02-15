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
from Brain.Common.DataFeature import OriginalDataFrature
from datetime import datetime
from tensorboardX import SummaryWriter
import time
from abc import ABC, abstractmethod
from Brain.A2C.lib.environment import Env, State_time_step
from Brain.A2C.lib.model import ActorCriticModel,TransformerModel
import copy 

def show_setting(title: str, content: str):
    print(f"--{title}--:{content}")

class RL_prepare(ABC):
    def __init__(self):
        self._prepare_hyperparameters()
        self._prepare_keyword()
        self._prepare_device()
        self._prepare_symbols()
        self._prepare_writer()
        self._prepare_envs()
        self._prepare_model()
        self._prepare_targer_net()
        self._prepare_agent()
        self._prepare_optimizer()

    def _prepare_keyword(self):
        self.keyword = 'Transformer'
        show_setting("KEYWORD:", self.keyword)

    def _prepare_device(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        show_setting("DEVICE:", self.device)

    def _prepare_symbols(self):
        # symbols = ['PEOPLEUSDT','BTCUSDT', 'ENSUSDT', 'LPTUSDT', 'GMXUSDT', 'TRBUSDT', 'ARUSDT', 'XMRUSDT', 'ETHUSDT', 'AAVEUSDT', 'ZECUSDT', 'SOLUSDT', 'DEFIUSDT', 'ETCUSDT', 'LTCUSDT', 'BCHUSDT', 'ORDIUSDT', 'BNBUSDT', 'AVAXUSDT', 'MKRUSDT', 'BTCDOMUSDT']
        # symbols = ['TRBUSDT','BTCUSDT']
        symbols = ['BTCUSDT']
        self.symbols = list(set(symbols))
        show_setting("SYMBOLS", self.symbols)

    def _prepare_envs(self):
        if self.keyword == 'Transformer':
            print("目前商品:",[self.symbols[0]])           
            data = OriginalDataFrature().get_train_net_work_data_by_path([self.symbols[0]])

            state = State_time_step(
                                    init_prices=data[list(data.keys())[0]],
                                    bars_count=self.BARS_COUNT,
                                    commission_perc=self.MODEL_DEFAULT_COMMISSION_PERC,                                    
                                    default_slippage=self.DEFAULT_SLIPPAGE,                                                                       
                                    )

        elif self.keyword == 'EfficientNetV2':
            pass

        show_setting("ENVIRONMENT-STATE", state)
            
        data = OriginalDataFrature().get_train_net_work_data_by_path(self.symbols)
        # 製作環境
        self.train_env = Env(prices=data, state=state, random_ofs_on_reset=True)

    def _prepare_writer(self):
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                'C:\\', 'runs', datetime.strftime(
                    datetime.now(), "%Y%m%d-%H%M%S") + '-conv-'))

    def _prepare_hyperparameters(self):
        self.BARS_COUNT = 300  # 用來準備要取樣的特徵長度,例如:開高低收成交量各取10根K棒
        self.GAMMA = 0.99
        self.MODEL_DEFAULT_COMMISSION_PERC = 0.0025
        self.DEFAULT_SLIPPAGE = 0.0025
        self.LEARNING_RATE = 0.0001  # optim 的學習率
        
        self.ENTROPY_COEF = 0.05  # 熵损失系数
        self.SAVES_PATH = "saves"  # 儲存的路徑

        self.CHECKPOINT_EVERY_STEP = 100
        self.VALUE_LOSS_COEF = 0.1  # 價值損失函數 critic損失函數
        self.N_STEP = 200
        self.checkgrad_times = 10

    def _prepare_model(self):
        engine_info = self.train_env.engine_info()

        # 初始化模型和優化器
        self.model = TransformerModel(
            d_model=engine_info['input_size'],
            nhead=4,
            d_hid=2048,
            nlayers=2,
            n_actions=self.train_env.action_space.n,
            hidden_size=64,
            dropout=0.1,
            batch_first=True,

        ).to(self.device)

    def _prepare_optimizer(self):
        # 定義特殊層的學習率
        param_groups = [
            {'params': self.model.dean.mean_layer.parameters(), 'lr': 0.0001 * self.model.dean.mean_lr},
            {'params': self.model.dean.scaling_layer.parameters(), 'lr': 0.0001 * self.model.dean.scale_lr},
            {'params': self.model.dean.gating_layer.parameters(), 'lr': 0.0001 * self.model.dean.gate_lr},
        ]

        # 其餘參數使用默認學習率，直接加入 param_groups
        for name, param in self.model.named_parameters():
            if (
                "dean.mean_layer" not in name
                and "dean.scaling_layer" not in name
                and "dean.gating_layer" not in name
            ):
                param_groups.append({'params': param, 'lr': self.LEARNING_RATE})

        # 初始化優化器
        self.optimizer = optim.Adam(param_groups)
    
    def _prepare_targer_net(self):
        # 创建目标网络，作为主模型的副本
        self.target_model = copy.deepcopy(self.model)
        # 将目标网络设置为评估模式
        self.target_model.eval()

    def _prepare_agent(self):
        pass
    

            

class Runner(RL_prepare):
    def __init__(self):
        super().__init__()        
        self.model_tool = ModelTool()
        # 記錄計時用，用於計算FPS
        self.start_time = time.time()
        self.last_time = self.start_time
        self.frame_count = 0  # 累計經過的步數
        self.avg_rewards = []        
        self.load_pre_train_model_state() 
        self.train()
    
    def load_pre_train_model_state(self):
        # 加載檢查點如果存在的話
        checkpoint_path = r''
        if checkpoint_path and os.path.isfile(checkpoint_path):
            print("資料繼續運算模式")
            # 標準化路徑並分割
            self.saves_path = os.path.dirname(checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])            
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("建立新的儲存點")
            # 用來儲存的位置
            self.saves_path = os.path.join(self.SAVES_PATH, datetime.strftime(
                datetime.now(), "%Y%m%d-%H%M%S") + '-' + str(self.BARS_COUNT) + 'k-')

            os.makedirs(self.saves_path, exist_ok=True)
    
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
            self.avg_rewards.append(sum(episode_rewards.cpu()))
            
            # 將這些數值寫入log
            self.writer.add_scalar('FPS', fps, step)
            self.writer.add_scalar('Average Value', avg_value, step)
            # Avg Reward 已經在下面程式碼紀錄了，可以同樣在這裡顯示

            print(
                f"[Monitor] Step: {step} | FPS: {fps:.2f} | Avg Value: {avg_value:.4f} | Avg Reward: {np.mean(self.avg_rewards[-100:]):.4f}")

    def train(self):
        step = 0
        while True:
            step += 1
            values, logprobs, rewards, entropies, G, check = self.run_episode(self.train_env, self.model)
            # 在 update_params 中做反向傳播與更新
            values, rewards = self.update_params(values, logprobs, rewards, entropies, G)
            
            # 軟更新 target model
            # self.soft_update()

            # 記錄與顯示監控資訊
            self.monitoring(step, values, rewards)

            # 定期保存模型參數
            if step % self.CHECKPOINT_EVERY_STEP == 0:
                self.model_tool.save_checkpoint({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(self.saves_path, f"checkpoint{int(step/self.CHECKPOINT_EVERY_STEP)}.pt"))
            
            if step % self.checkgrad_times == 0:
                self.checkgrad()

    def checkgrad(self):
        # 打印梯度統計數據
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                print(f"Layer: {name}, Grad Min: {param.grad.min()}, Grad Max: {param.grad.max()}, Grad Mean: {param.grad.mean()}")
        print('*'*120)
        
    def soft_update(self, tau=0.005):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data)
    
    def update_params(self, values, logprobs, rewards, entropies, G):
        """
        使用 value, logprobs, rewards, entropies, G, check 來計算 loss 並進行反向傳播更新。
        check=1 時表示 environment 完整結束，不需bootstrap。
        check=0 時表示 N-step截斷，使用 G 做bootstrap。
        """
        # values, logprobs, entropies, rewards 已在 run_episode() 中 flip 過
        # 計算 returns
        Returns = []
        ret_ = G
        for r in rewards:
            ret_ = r + self.GAMMA * ret_
            Returns.append(ret_)
        
        Returns = torch.stack(Returns).view(-1)

        Returns = F.normalize(Returns, dim=0)
        # Returns = (Returns - Returns.mean()) / (Returns.std() + 1e-8)

        
        advantages = Returns - values.detach()
        policy_loss = -(logprobs * advantages).mean()
        value_loss = F.mse_loss(values, Returns)
        entropy_loss = entropies.mean()
        total_loss = policy_loss + self.VALUE_LOSS_COEF * value_loss - self.ENTROPY_COEF * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        return values, rewards
    
    def run_episode(self, worker_env, worker_model):
        """
        此函式其實是在 N-step 更新時使用。
        根據 N_STEP 截斷或環境 done 來決定是否使用 next_value bootstrap。
        """
        state = worker_env.reset()
        state = torch.from_numpy(state).to(self.device).unsqueeze(0)
        values, logprobs, rewards, entropies = [], [], [], []
        done = False
        j = 0

        while (j < self.N_STEP and not done):
            policy, value = worker_model(state)
            values.append(value)
            logits = policy.view(-1)            
            action_dist = Categorical(logits=logits)
            action = action_dist.sample()
            logprob_ = action_dist.log_prob(action)
            entropy_ = action_dist.entropy()
            logprobs.append(logprob_)
            entropies.append(entropy_)
            obs, reward, done, info = worker_env.step(action.item())
            rewards.append(reward)
            state = torch.from_numpy(obs).to(self.device).unsqueeze(0)
            j += 1

        # Episode/N-step結束後決定 G 與 check
        if done:
            # 環境真正結束，無需bootstrap，G=0，check=1
            G = torch.zeros(1, device=self.device)
            check = 1
            # 若需要重新開始下一局，可在此重置環境
            worker_env.reset()
        else:
            # N-step截斷，但未真正結束環境，使用最後一個 state's value bootstrap
            # 上一迭代已計算出 value，即為values最後一個
            G = values[-1].detach()
            check = 0

        # flip 代表反轉的意思
        values = torch.stack(values).flip(dims=(0,)).view(-1)
        logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
        entropies = torch.stack(entropies).flip(dims=(0,)).view(-1)
        rewards = torch.tensor(rewards, device=self.device).flip(dims=(0,)).view(-1)

        return values, logprobs, rewards, entropies, G, check

Runner()
