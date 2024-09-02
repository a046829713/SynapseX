# from DQN.lib import environment, models, common
# from DQN.lib.environment import State2D, State_time_step
# from DQN import ptan
# from DQN.lib.DataFeature import DataFeature
# from datetime import datetime
# import time
# from DQN.lib import offical_transformer
# from DQN.lib.EfficientnetV2 import EfficientnetV2SmallDuelingModel

from abc import ABC
import torch
from lib.DataFeature import DataFeature
import time
from DDPG.environment import Env

# import os
# import torch.optim as optim
# from DDPG.model import Actor, Critic
# from DDPG.experience import ReplayBuffer


class RL_prepare(ABC):
    def __init__(self):
        self._prepare_keyword()
        self._prepare_device()
        self._prepare_symbols()
        self._prepare_hyperparameters()
        self._prepare_env()
        self._prepare_model()

    def _prepare_keyword(self):
        self.KEYWORD = 'Transformer'
        print("--KEYWORD--:", self.KEYWORD)

    def _prepare_device(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("There is device:", self.device)

    def _prepare_symbols(self):
        # symbols = ['BTCUSDT', 'ENSUSDT', 'LPTUSDT', 'GMXUSDT', 'TRBUSDT', 'ARUSDT', 'XMRUSDT',
        #            'ETHUSDT', 'AAVEUSDT',  'ZECUSDT', 'SOLUSDT', 'DEFIUSDT',  'ETCUSDT', 'LTCUSDT', 'BCHUSDT']
        symbols = ['TRBUSDT']
        self.symbols = list(set(symbols))
        print("There are symobls:", self.symbols)

    def _prepare_hyperparameters(self):
        self.BARS_COUNT = 300  # 用來準備要取樣的特徵長度,例如:開高低收成交量各取10根K棒
        self.MODEL_DEFAULT_COMMISSION_PERC = 0.0005
        self.DEFAULT_SLIPPAGE = 0.0025
        self.ACTOR_LR = 0.0001
        self.Q_LR = 0.0001
        self.REPLAY_SIZE = 100000
        self.MAX_EPISODE_LENGTH = 1000  # 根據gym的描述 1000步就會結束

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
        
        
        self.train_env.reset()
        

    def _prepare_model(self):
        # get size of state space and action space
        self.state_size = self.train_env._count_state.shape[0]

        # 只有一種行為
        self.action_size = 1

        self.actor = Actor(in_channels=1, action_size=self.action_size)
        self.target_actor = Actor(in_channels=1, action_size=self.action_size)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_size, self.action_size)
        self.target_critic = Critic(self.state_size, self.action_size)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.ACTOR_LR)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.Q_LR)

        print("--STATE_SIZE--:", self.state_size)
        print("--ACTION_SIZE--:", self.action_size)


class DDPG(RL_prepare):
    def __init__(self) -> None:
        super().__init__()

        # Experience replay memory
        # self.replay_buffer = ReplayBuffer(obs_dim=self.state_size, act_dim=self.action_size, size=self.REPLAY_SIZE)

        self.train()

    def get_action(self, s, noise_scale):
        # Convert state to PyTorch tensor
        state_tensor = torch.from_numpy(
            s).float().unsqueeze(0)  # Add batch dimension

        # Get action from the actor network
        a = self.actor(state_tensor)

        # Convert action tensor to numpy array and squeeze the batch dimension
        a = a.detach().numpy().squeeze(0)

        # Add noise for exploration
        noise = noise_scale * np.random.randn(self.action_size)
        a += noise

        # Clip the actions to be within the allowed range
        a = np.clip(a, -self.action_max, self.action_max)
        return a

    def update(self, batch_size):
        # 從回放緩衝區抽樣一批體驗
        batch = self.replay_buffer.sample_batch(batch_size)
        s, a, r, s2, d = batch['s'], batch['a'], batch['r'], batch['s2'], batch['d']

        s = torch.from_numpy(s)
        s2 = torch.from_numpy(s2)
        d = torch.from_numpy(d)
        r = torch.from_numpy(r)
        a = torch.from_numpy(a)

        # 使用下一狀態和目標批評者網絡計算目標Q值
        with torch.no_grad():
            next_actions = self.target_actor(s2)
            target_q_values = self.target_critic(s2, next_actions)
            target_q = r.unsqueeze(1) + self.discount_factor * \
                target_q_values * (1 - d).unsqueeze(1)

        # 更新批評者網絡
        critic_q_values = self.critic(s, a)
        critic_loss = torch.nn.functional.mse_loss(critic_q_values, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新演員網絡
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()

    def soft_update(self, local_model, target_model, tau):
        # 軟更新模型參數
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)

    def train(self):
        i_episode = 0

        returns = []
        q_losses = []
        mu_losses = []
        num_steps = 0

        while True:
            # reset env
            state, episode_return, episode_length, d = self.train_env.reset(), 0, 0, False

            while not (d or (episode_length == self.MAX_EPISODE_LENGTH)):
                # For the first `START_STRPS` steps, use randomly sampled actions
                # in order to encourage exploration.
                if num_steps > self.START_STRPS:
                    action = self.get_action(state, self.action_noise)
                else:
                    action = self.train_env.action_space.sample()

                print(action)
                time.sleep(100)
                # Keep track of the number of steps done
                num_steps += 1
                if num_steps == self.START_STRPS:
                    print("USING AGENT ACTIONS NOW")

                # # Step the env
                next_state, reward, terminated, truncated, info = self.train_env.step(
                    action)
                episode_return += reward
                episode_length += 1

                # Ignore the "done" signal if it comes from hitting the time
                # horizon (that is, when it's an artificial terminal signal
                # that isn't based on the agent's state)
                d_store = False if episode_length == self.MAX_EPISODE_LENGTH else terminated

                # Store experience to replay buffer
                self.replay_buffer.store(
                    state, action, reward, next_state, d_store)

                # Assign next state to be the current state on the next round
                state = next_state

            # Perform the updates
            for _ in range(episode_length):
                self.update(batch_size=self.batch_size)
                self.update_count += 1

                if self.update_count % self.target_update_freq == 0:  # 每隔一定次數進行一次軟更新
                    self.soft_update(
                        self.critic, self.target_critic, tau=0.995)
                    self.soft_update(self.actor, self.target_actor, tau=0.995)

            print("Episode:", i_episode + 1, "Return:",
                  episode_return, 'episode_length:', episode_length)
            returns.append(episode_return)

        # Save the models after training
        self.save_model()
        print("Training finished. Models saved successfully.")

    def save_model(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(self.actor.state_dict(), 'models/actor.pth')
        torch.save(self.critic.state_dict(), 'models/critic.pth')
        print("Models saved successfully.")


class RL_Train(RL_prepare):
    def __init__(self) -> None:
        super().__init__()
        self.count_parameters(self.net)

        self.exp_source = ptan.experience.ExperienceSourceFirstLast(
            self.train_env, self.agent, self.GAMMA, steps_count=self.REWARD_STEPS)

        self.buffer = ptan.experience.ExperienceReplayBuffer(
            self.exp_source, self.REPLAY_SIZE)

        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.LEARNING_RATE)

        self.load_pre_train_model_state()
        self.train()

    def load_pre_train_model_state(self):
        # 加載檢查點如果存在的話
        checkpoint_path = r''
        if checkpoint_path and os.path.isfile(checkpoint_path):
            print("資料繼續運算模式")
            saves_path = checkpoint_path.split('\\')
            self.saves_path = os.path.join(saves_path[0], saves_path[1])
            checkpoint = torch.load(checkpoint_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.step_idx = checkpoint['step_idx']
        else:
            print("建立新的儲存點")
            # 用來儲存的位置
            self.saves_path = os.path.join(self.SAVES_PATH, datetime.strftime(
                datetime.now(), "%Y%m%d-%H%M%S") + '-' + str(self.BARS_COUNT) + 'k-')

            os.makedirs(self.saves_path, exist_ok=True)
            self.step_idx = 0

    def train(self):
        with common.RewardTracker(self.writer, np.inf, group_rewards=2) as reward_tracker:
            while True:
                self.step_idx += 1
                self.buffer.populate(1)
                # 探索率
                self.selector.epsilon = max(
                    self.EPSILON_STOP, self.EPSILON_START - self.step_idx / self.EPSILON_STEPS)

                # [(-2.5305491551459296, 10)]
                # 跑了一輪之後,清空原本的數據,並且取得獎勵
                new_rewards = self.exp_source.pop_rewards_steps()

                if new_rewards:
                    reward_tracker.reward(
                        new_rewards[0], self.step_idx, self.selector.epsilon)

                if len(self.buffer) < self.REPLAY_INITIAL:
                    continue

                self.optimizer.zero_grad()
                batch = self.buffer.sample(self.BATCH_SIZE)

                loss_v = common.calc_loss(
                    batch, self.net, self.tgt_net.target_model, self.GAMMA ** self.REWARD_STEPS, device=self.device)
                if self.step_idx % self.WRITER_EVERY_STEP == 0:
                    self.writer.add_scalar(
                        "Loss_Value", loss_v.item(), self.step_idx)
                loss_v.backward()

                if self.step_idx % self.checkgrad_times == 0:
                    self.checkgrad()

                self.optimizer.step()
                if self.step_idx % self.TARGET_NET_SYNC == 0:
                    self.tgt_net.sync()

                # 在主訓練循環中的合適位置插入保存檢查點的代碼
                if self.step_idx % self.CHECKPOINT_EVERY_STEP == 0:
                    idx = self.step_idx // self.CHECKPOINT_EVERY_STEP
                    checkpoint = {
                        'step_idx': self.step_idx,
                        'model_state_dict': self.net.state_dict(),
                        'selector_state': self.selector.epsilon,

                    }
                    self.save_checkpoint(checkpoint, os.path.join(
                        self.saves_path, f"checkpoint-{idx}.pt"))

    def checkgrad(self):
        # 打印梯度統計數據
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                print(f"Layer: {name}, Grad Min: {param.grad.min()}, Grad Max: {
                      param.grad.max()}, Grad Mean: {param.grad.mean()}")
        print('*'*120)

    def save_checkpoint(self, state, filename):
        # 保存檢查點的函數
        torch.save(state, filename)

    # 計算參數數量
    def count_parameters(self, model):
        data = [p.numel() for p in model.parameters() if p.requires_grad]
        sum_numel = sum(data)
        print("總參數數量:", sum_numel)
        return sum_numel

    def change_torch_script(self, model):
        # 將模型轉換為 TorchScript
        scripted_model = torch.jit.script(model)

        # 保存腳本化後的模型 DQN\Meta\Meta-300B-30K.pt
        scripted_model.save("transformer_dueling_model_scripted.pt")


if __name__ == "__main__":
    # 我認為可以訓練出通用的模型了
    # 多數據供應
    RL_prepare()
