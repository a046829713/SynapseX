import torch
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from A3C.lib.environment import TradingEnv
from A3C.lib.models import Actor, Critic
from A3C.lib.distributed import setup_distributed,cleanup_distributed
from abc import ABC

class RL_prepare(ABC):
    def __init__(self):
        self._prepare_keyword()
        self._prepare_device()
        self._prepare_symbols()
        self._prepare_hyperparameters()
        self._prepare_env()
        self._prepare_model()
    
    def show_setting(self,title:str,content:str):
        print(f"--{title}--:{content}")
    
    def _prepare_keyword(self):
        self.KEYWORD = 'Transformer'
        self.show_setting(title="KEYWORD",content=self.KEYWORD)
        

    def _prepare_device(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.show_setting(title="DEVICE",content=self.device)

    def _prepare_symbols(self):
        # symbols = ['BTCUSDT', 'ENSUSDT', 'LPTUSDT', 'GMXUSDT', 'TRBUSDT', 'ARUSDT', 'XMRUSDT',
        #            'ETHUSDT', 'AAVEUSDT',  'ZECUSDT', 'SOLUSDT', 'DEFIUSDT',  'ETCUSDT', 'LTCUSDT', 'BCHUSDT']
        symbols = ['TRBUSDT']
        self.symbols = list(set(symbols))
        self.show_setting(title="SYMBOLS",content=self.symbols)
        

    def _prepare_hyperparameters(self):
        self.BARS_COUNT = 300  # 用來準備要取樣的特徵長度,例如:開高低收成交量各取10根K棒
        self.MODEL_DEFAULT_COMMISSION_PERC = 0.0005
        self.DEFAULT_SLIPPAGE = 0.0025
        self.ACTOR_LR = 0.0001
        self.Q_LR = 0.0001
        self.REPLAY_SIZE = 100000
        self.MAX_EPISODE_LENGTH = 1000  
        self.START_STRPS = 10000
        # self.batch_size = 100
        # self.discount_factor = 0.99
        # self.target_update_freq = 100  # 定義多久進行一次軟更新，例如每100次訓練步驟
        # self.update_count = 0  # 計數器追踪更新次數

    def _prepare_env(self):
        self.train_env = Env(
            keyword=self.KEYWORD,
            bars_count=self.BARS_COUNT,
            commission_perc=self.MODEL_DEFAULT_COMMISSION_PERC,
            default_slippage=self.DEFAULT_SLIPPAGE,
            prices=DataFeature(
                device=self.device).get_train_net_work_data_by_path(self.symbols),
            random_ofs_on_reset=True,
            device=self.device)
        
        

    def _prepare_model(self):
        # 获取环境信息
        env_info = self.train_env.env_info()
        input_size = env_info['input_size']
        action_size = env_info['action_size']

        # 初始化 Actor 和 Target Actor
        def create_actor():
            return Actor(
                d_model=input_size,
                nhead=2,
                d_hid=2048,
                nlayers=8,
                num_actions=action_size,
                hidden_size=64,
                seq_dim=self.BARS_COUNT,
                dropout=0.1
            ).to(self.device)

        self.actor = create_actor()
        self.target_actor = create_actor()
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 初始化 Critic 和 Target Critic
        self.critic = Critic(input_size, action_size)
        self.target_critic = Critic(input_size, action_size)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.Q_LR)

        # 打印信息
        self.show_setting(title="STATE_SIZE",content=input_size)
        self.show_setting(title="ACTION_SIZE",content=action_size)
        
            
def train(rank, world_size, data):
    setup_distributed()

    # 創建環境和模型
    env = TradingEnv(data)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    actor = Actor(state_size, action_size).cuda()
    critic = Critic(state_size).cuda()

    # 包裝為分散式模型
    actor = DDP(actor, device_ids=[rank])
    critic = DDP(critic, device_ids=[rank])

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    num_episodes = 1000
    gamma = 0.99
    gae_lambda = 0.95

    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        for _ in range(len(data)-1):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()
            dist_probs = actor(state_tensor)
            value = critic(state_tensor)

            dist = torch.distributions.Categorical(dist_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            next_state, reward, done, _ = env.step(action.item())

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float32).cuda())
            masks.append(torch.tensor([1 - done], dtype=torch.float32).cuda())

            state = next_state

            if done:
                break

        # 計算優勢函數和目標值
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).cuda()
        next_value = critic(next_state_tensor)
        returns = []
        advantages = []
        R = next_value
        A = torch.zeros(1, 1).cuda()

        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            td_error = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            A = td_error + gamma * gae_lambda * A * masks[step]
            returns.insert(0, R)
            advantages.insert(0, A)

        returns = torch.cat(returns).detach()
        advantages = torch.cat(advantages).detach()
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)

        # 更新Actor
        actor_loss = -(log_probs * advantages).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # 更新Critic
        critic_loss = F.mse_loss(values, returns)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        if rank == 0:
            print(f'Episode {episode}, Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}')

    cleanup_distributed()




def main():
    data = ...  # 載入並預處理您的數據
    world_size = torch.cuda.device_count()
    mp.spawn(train,
             args=(world_size, data),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    main()
