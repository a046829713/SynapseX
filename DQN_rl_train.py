from Brain.DQN.lib import environment, common
from Brain.DQN.lib.environment import State_time_step
import os
import numpy as np
import torch
import torch.optim as optim
from Brain.DQN import ptan
from Brain.Common.DataFeature import OriginalDataFrature
from datetime import datetime
from tensorboardX import SummaryWriter
import time
from Brain.DQN.lib import model
from abc import ABC
from Brain.DQN.lib.EfficientnetV2 import EfficientnetV2SmallDuelingModel
from abc import ABC
from Brain.Common.experience import SequentialExperienceReplayBuffer


class RL_prepare(ABC):
    def __init__(self):
        self._prepare_keyword()
        self._prepare_device()
        self._prepare_symbols()
        self._prepare_data()
        self._prepare_writer()
        self._prepare_hyperparameters()
        self._prepare_env()
        self._prepare_model()
        self._prepare_targer_net()
        self._prepare_agent()
        self._prepare_optimizer()

    def _prepare_keyword(self):
        self.keyword = 'Transformer'
        self.show_setting("KEYWORD:", self.keyword)

    def show_setting(self, title: str, content: str):
        print(f"--{title}--:{content}")

    def _prepare_device(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("There is device:", self.device)

    def _prepare_symbols(self):
        # symbols = ['PEOPLEUSDT','BTCUSDT', 'ENSUSDT', 'LPTUSDT', 'GMXUSDT', 'TRBUSDT', 'ARUSDT', 'XMRUSDT', 'ETHUSDT', 'AAVEUSDT', 'ZECUSDT', 'SOLUSDT', 'DEFIUSDT', 'ETCUSDT', 'LTCUSDT', 'BCHUSDT', 'ORDIUSDT', 'BNBUSDT', 'AVAXUSDT', 'MKRUSDT', 'BTCDOMUSDT']
        # symbols = ['YFIUSDT',
        #      'QNTUSDT',
        #      'XMRUSDT',
        #      'MKRUSDT',
        #      'LTCUSDT',
        #      'INJUSDT',
        #      'DASHUSDT',
        #      'ENSUSDT',
        #      'GMXUSDT',
        #      'ILVUSDT',
        #      'EGLDUSDT',
        #      'SSVUSDT',
        #      "ARUSDT",
        #      "BTCUSDT",
        #      "ETHUSDT",
        #      "SOLUSDT",
        #      "SUIUSDT",
        #      "BNBUSDT"]
        symbols = ['BTCUSDT']
        self.symbols = list(set(symbols))
        print("There are symobls:", self.symbols)

    def _prepare_data(self):
        self.data = OriginalDataFrature().get_train_net_work_data_by_path(self.symbols)

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
        self.REWARD_STEPS = 2
        self.REPLAY_SIZE = 100000
        self.EACH_REPLAY_SIZE = 50000
        self.REPLAY_INITIAL = 1000
        self.LEARNING_RATE = 0.0001  # optim 的學習率
        self.Lambda = 1e-2  # optim L2正則化 Ridge regularization
        self.EPSILON_START = 0.9  # 起始機率(一開始都隨機運行)
        self.SAVES_PATH = "saves"  # 儲存的路徑
        self.EPSILON_STOP = 0.1
        self.TARGET_NET_SYNC = 1000
        self.CHECKPOINT_EVERY_STEP = 20000
        self.VALIDATION_EVERY_STEP = 100000
        self.WRITER_EVERY_STEP = 100
        self.EVAL_EVERY_STEP = 10000  # 每一萬步驗證一次
        self.NUM_EVAL_EPISODES = 10  # 每次评估的样本数
        self.BATCH_SIZE = 32  # 每次要從buffer提取的資料筆數,用來給神經網絡更新權重
        self.STATES_TO_EVALUATE = 10000  # 每次驗證一萬筆資料
        self.checkgrad_times = 1000

    def _prepare_env(self):
        if self.keyword == 'Transformer':
            state = State_time_step(
                init_prices=self.data[np.random.choice(
                    list(self.data.keys()))],
                bars_count=self.BARS_COUNT,
                commission_perc=self.MODEL_DEFAULT_COMMISSION_PERC,
                model_train=True,
                default_slippage=self.DEFAULT_SLIPPAGE
            )

        elif self.keyword == 'EfficientNetV2':
            state = ''

        print("There is state:", state)

        # 製作環境
        self.train_env = environment.Env(
            prices=self.data, state=state, random_ofs_on_reset=True)

    def _prepare_model(self):
        engine_info = self.train_env.engine_info()

        if self.keyword == 'Transformer':
            self.net = model.COT_TransformerDuelingModel(
                d_model=engine_info['input_size'],
                nhead=8,
                d_hid=2048,
                nlayers=4,
                num_actions=self.train_env.action_space.n,  # 假设有5种可能的动作
                hidden_size=64,  # 使用隐藏层
                seq_dim=self.BARS_COUNT,
                dropout=0.1,  # 适度的dropout以防过拟合
                num_iterations=3
            ).to(self.device)

        elif self.keyword == 'EfficientNetV2':
            self.net = EfficientnetV2SmallDuelingModel(
                in_channels=1, num_actions=self.train_env.action_space.n).to(self.device)

        print("There is netWork model:", self.net.__class__)

    def _prepare_targer_net(self):
        self.tgt_net = ptan.agent.TargetNet(self.net)

    def _prepare_agent(self):
        # 貪婪的選擇器
        self.selector = ptan.actions.EpsilonGreedyActionSelector(
            self.EPSILON_START, epsilon_stop=self.EPSILON_STOP)

        self.agent = ptan.agent.DQNAgent(
            self.net, self.selector, device=self.device)

    def _prepare_optimizer(self):
        # 定義特殊層的學習率
        param_groups = [
            {'params': self.net.dean.mean_layer.parameters(), 'lr': 0.0001 *
             self.net.dean.mean_lr},
            {'params': self.net.dean.scaling_layer.parameters(), 'lr': 0.0001 *
             self.net.dean.scale_lr},
            {'params': self.net.dean.gating_layer.parameters(), 'lr': 0.0001 *
             self.net.dean.gate_lr},
        ]

        # 其餘參數使用默認學習率，直接加入 param_groups
        for name, param in self.net.named_parameters():
            if (
                "dean.mean_layer" not in name
                and "dean.scaling_layer" not in name
                and "dean.gating_layer" not in name
            ):
                param_groups.append(
                    {'params': param, 'lr': self.LEARNING_RATE})

        # 初始化優化器
        self.optimizer = optim.Adam(param_groups, weight_decay=self.Lambda)


class RL_Train(RL_prepare):
    def __init__(self) -> None:
        super().__init__()
        self.count_parameters(self.net)

        self.exp_source = ptan.experience.ExperienceSourceFirstLast(
            self.train_env, self.agent, self.GAMMA, steps_count=self.REWARD_STEPS)

        self.buffer = SequentialExperienceReplayBuffer(
            self.exp_source, self.EACH_REPLAY_SIZE, len(self.symbols))

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
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.step_idx = checkpoint['step_idx']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.selector.epsilon = checkpoint['selector_state']
            print("目前epsilon:", self.selector.epsilon)
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

                # [(-2.5305491551459296, 10)]
                # 跑了一輪之後,清空原本的數據,並且取得獎勵
                new_rewards = self.exp_source.pop_rewards_steps()

                if new_rewards:
                    mean_reward = reward_tracker.reward(
                        new_rewards[0], self.step_idx, self.selector.epsilon)

                    if isinstance(mean_reward, np.float64):
                        # 探索率
                        self.selector.update_epsilon(mean_reward)
                        print("目前最新探索率:", self.selector.epsilon)
                    else:
                        print("mean_reward:", mean_reward)

                if not self.buffer.each_num_len_enough(init_size=self.REPLAY_INITIAL):
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
                    # self.checkgrad()
                    self.checkwhight()

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
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }
                    self.save_checkpoint(checkpoint, os.path.join(
                        self.saves_path, f"checkpoint-{idx}.pt"))

    def checkgrad(self):
        # 打印梯度統計數據
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                print(
                    f"Layer: {name}, Grad Min: {param.grad.min()}, Grad Max: {param.grad.max()}, Grad Mean: {param.grad.mean()}")
        print('*'*120)

    def checkwhight(self):
        # 打印梯度統計數據
        for name, param in self.net.named_parameters():
            print(f"該layer 名稱:{name}")
            print(f"該參數為:{param}")
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


# 我認為可以訓練出通用的模型了
# 多數據供應
RL_Train()
