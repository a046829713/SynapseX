from Brain.DQN.lib import environment, common
from Brain.DQN.lib.environment import State_time_step
import os
import numpy as np
import torch
import torch.optim as optim
from Brain.DQN import ptan
from Brain.Common.DataFeature import OriginalDataFrature
from datetime import datetime
import time
from Brain.DQN.lib import model
from abc import ABC
from Brain.DQN.lib.EfficientnetV2 import EfficientnetV2SmallDuelingModel
from Brain.Common.experience import SequentialExperienceReplayBuffer
import torch.nn as nn
import traceback

# import builtins
# import inspect

# original_print = builtins.print

# def custom_print(*args, **kwargs):
#     frame = inspect.currentframe().f_back
#     file_name = frame.f_code.co_filename
#     line_number = frame.f_lineno
#     original_print(f"[TRACE] {file_name}:{line_number} ->", *args, **kwargs)

# # 覆蓋內建 print
# builtins.print = custom_print


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
        self.keyword = 'Mamba'
        self.show_setting("KEYWORD:", self.keyword)

    def show_setting(self, title: str, content: str):
        print(f"--{title}--:{content}")

    def _prepare_device(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("There is device:", self.device)
        

    def _prepare_symbols(self):
        
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'ADAUSDT', 'ENAUSDT', 'LINKUSDT', 'HBARUSDT', 'LTCUSDT', 'XLMUSDT', 'WIFUSDT', 'BNBUSDT', 'ONDOUSDT', 'AAVEUSDT', 'WLDUSDT', 'AVAXUSDT', 'JUPUSDT', 'DOTUSDT', 'TRXUSDT', 'FILUSDT', 'ALGOUSDT', 'ZENUSDT', 'TIAUSDT', 'CRVUSDT', 'AGLDUSDT', 'POPCATUSDT', 'GALAUSDT', 'NEARUSDT']
        
        # symbols = ['BTCUSDT']
        
        self.symbols = list(set(symbols))
        print(len(self.symbols))
        print("There are symobls:", self.symbols)

    def _prepare_data(self):
        self.data = OriginalDataFrature().get_train_net_work_data_by_path(self.symbols)

    def _prepare_writer(self):
        # 取得目前工作目錄，並在此目錄下建立 run 資料夾，並以時間戳記命名子資料夾
        log_dir = os.path.join(os.getcwd(), 'run', datetime.now().strftime("%Y%m%d-%H%M%S") + '-conv-')


    def _prepare_hyperparameters(self):
        self.BARS_COUNT = 300  # 用來準備要取樣的特徵長度,例如:開高低收成交量各取10根K棒
        self.GAMMA = 0.99
        self.MODEL_DEFAULT_COMMISSION_PERC = 0.0045
        self.DEFAULT_SLIPPAGE = 0.0025
        self.REWARD_STEPS = 2
        self.REPLAY_SIZE = 100000
        
        self.EACH_REPLAY_SIZE = 50000
        self.REPLAY_INITIAL = 1000
        self.LEARNING_RATE = 0.00005  # optim 的學習率
        self.Lambda = 0  # optim L2正則化 Ridge regularization
        self.EPSILON_START = 0.9  # 起始機率(一開始都隨機運行)
        self.SAVES_PATH = "saves"  # 儲存的路徑
        self.EPSILON_STOP = 0.1
        self.TARGET_NET_SYNC = 1000
        self.CHECKPOINT_EVERY_STEP = 20000  
        self.VALIDATION_EVERY_STEP = 100000
        self.EPSILON_STEPS = 1000000 * len(self.symbols)
        self.EVAL_EVERY_STEP = 10000  # 每一萬步驗證一次
        self.NUM_EVAL_EPISODES = 10  # 每次评估的样本数
        self.BATCH_SIZE = 32  # 每次要從buffer提取的資料筆數,用來給神經網絡更新權重
        self.STATES_TO_EVALUATE = 10000  # 每次驗證一萬筆資料
        self.checkgrad_times = 10000

    def _prepare_env(self):
        if self.keyword == 'Transformer' or 'Mamba' or "Mamba2":
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
            self.net = model.TransformerDuelingModel(
                d_model=engine_info['input_size'],
                nhead=8,
                d_hid=2048,
                nlayers=4,
                num_actions=self.train_env.action_space.n,  # 假设有5种可能的动作
                hidden_size=64,  # 使用隐藏层
                seq_dim=self.BARS_COUNT,
                dropout=0.1,  # 适度的dropout以防过拟合
                num_iterations = 2
            ).to(self.device)

        elif self.keyword == 'EfficientNetV2':
            self.net = EfficientnetV2SmallDuelingModel(
                in_channels=1, num_actions=self.train_env.action_space.n).to(self.device)
        
        elif self.keyword == "Mamba":
            self.net = model.mambaDuelingModel(
                d_model=engine_info['input_size'],
                nlayers=4,
                num_actions=self.train_env.action_space.n,  # 假设有5种可能的动作
                seq_dim=self.BARS_COUNT,
                dropout=0.3,  # 适度的dropout以防过拟合
            ).to(self.device)

        elif self.keyword == "Mamba2":
            self.net = model.mamba2DuelingModel(
                d_model=engine_info['input_size'],
                nlayers=2,
                num_actions=self.train_env.action_space.n,  # 假设有5种可能的动作
                seq_dim=self.BARS_COUNT,
                dropout=0.2,  # 适度的dropout以防过拟合
            ).to(self.device)

        print("There is netWork model:", self.net.__class__)

    def _prepare_targer_net(self):
        self.tgt_net = ptan.agent.TargetNet(self.net)

    def _prepare_agent(self):
        # 貪婪的選擇器
        self.selector = ptan.actions.EpsilonGreedyActionSelector(
            self.EPSILON_START, epsilon_stop=self.EPSILON_STOP)

        self.agent = ptan.agent.DQNAgent(
            self.net, self.selector, device=self.device)

    def _prepare_optimizer(self, base_lr=1e-4):
        """
        建立 Adam 優化器，以下功能：
        1. BN、LN、Embedding 層的 bias/scale 不做 weight decay
        2. dean.mean_layer、dean.scaling_layer、dean.gating_layer 不做 weight decay
        3. 其餘參數正常做 weight decay
        """

        # 要排除 weight decay 的層 (或參數名稱) 關鍵字
        excluded_keywords = [
            "bias",                  # 一般 bias 都不做或少做 weight decay
            "dean.mean_layer",       # 你的三個特殊層
            "dean.scaling_layer",
            "dean.gating_layer"
        ]

        # 另外，如果想同時排除 BN、LN、Embedding 的權重或 scale/bias，
        # 可以在檢查 layer type 時判斷屬於以下類型：
        excluded_types = (
            nn.LayerNorm,
        )

        # 用來存放不同分組
        decay_params = []
        no_decay_params = []
        for module_name, module in self.net.named_modules():
            # 檢查模組類型是否屬於 BN、LN、Embedding
            if isinstance(module, excluded_types):
                # 這個模組底下所有參數都排除 weight decay
                for param_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        full_name = f"{module_name}.{param_name}"
                        no_decay_params.append(param)
            else:
                # 不是 BN/LN/Embedding 層
                for param_name, param in module.named_parameters(recurse=False):
                    if not param.requires_grad:
                        continue
                    full_name = f"{module_name}.{param_name}"

                    # 檢查是否在 excluded_keywords 裡面
                    if any(kw in full_name for kw in excluded_keywords):
                        # 如果是 dean.* 或 bias 都排除 weight decay                        
                        if not any(layer in full_name for layer in ["dean.mean_layer", "dean.scaling_layer", "dean.gating_layer"]):
                            no_decay_params.append(param)
                    else:
                        # 其他普通參數
                        decay_params.append(param)

        # 建立真正的 param groups
        param_groups = [
            # 1) 正常參數：有 weight decay
            {
                'params': decay_params,
                'lr': self.LEARNING_RATE,
                'weight_decay': self.Lambda
            },
            # 2) 不做 weight decay
            {
                'params': no_decay_params,
                'lr': self.LEARNING_RATE,
                'weight_decay': 0.0
            }
        ]

        # 如果需要對 dean 幾個層有特別的學習率或同樣不做 weight decay：        
        param_groups.extend([
            {'params': list(self.net.dean.mean_layer.parameters()),
                'lr': base_lr *
                self.net.dean.mean_lr, 'weight_decay': 0.0},
            {'params': list(self.net.dean.scaling_layer.parameters()),
                'lr': base_lr *
                self.net.dean.scale_lr, 'weight_decay': 0.0},
            {'params': list(self.net.dean.gating_layer.parameters()),
                'lr': base_lr *
                self.net.dean.gate_lr, 'weight_decay': 0.0},
        ])

        # 用 Adam 建立優化器
        self.optimizer = optim.Adam(param_groups)
        



class RL_Train(RL_prepare):
    def __init__(self) -> None:
        super().__init__()
        self.count_parameters(self.net)

        self.exp_source = ptan.experience.ExperienceSourceFirstLast(
            self.train_env, self.agent, self.GAMMA, steps_count=self.REWARD_STEPS)

        self.buffer = SequentialExperienceReplayBuffer(
            self.exp_source, self.EACH_REPLAY_SIZE, len(self.symbols),replay_initial_size=self.REPLAY_INITIAL)

        self.load_pre_train_model_state()
        self.train()

    def load_pre_train_model_state(self):
        # 加載檢查點如果存在的話
        checkpoint_path = r''
        if checkpoint_path and os.path.isfile(checkpoint_path):
            print("資料繼續運算模式")
            # 標準化路徑並分割
            self.saves_path = os.path.dirname(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, weights_only=False )
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.tgt_net.target_model.load_state_dict(checkpoint['tgt_net_state_dict'])
            self.step_idx = checkpoint['step_idx']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.selector.epsilon = max(self.EPSILON_STOP, self.EPSILON_START - self.step_idx / self.EPSILON_STEPS)
            # 緩衝區資料過大 很難持續保存
            # self.buffer.load_state(checkpoint['buffer_state'])
            print("目前epsilon:", self.selector.epsilon)
        else:
            print("建立新的儲存點")
            # 用來儲存的位置
            self.saves_path = os.path.join(self.SAVES_PATH, datetime.strftime(
                datetime.now(), "%Y%m%d-%H%M%S") + '-' + str(self.BARS_COUNT) + 'k-')

            os.makedirs(self.saves_path, exist_ok=True)
            self.step_idx = 0

    def train(self):
        with common.RewardTracker(np.inf, group_rewards=2) as reward_tracker:
            while True:
                try:

                    self.step_idx += 1
                    self.buffer.populate(1)

                    # [(-2.5305491551459296, 10)]
                    # 跑了一輪之後,清空原本的數據,並且取得獎勵
                    new_rewards = self.exp_source.pop_rewards_steps()

                    if new_rewards:
                        # mean_reward = reward_tracker.reward(
                        #     new_rewards[0], self.step_idx, self.selector.epsilon)

                        # if isinstance(mean_reward, np.float64):
                        #     # 探索率
                        #     self.selector.update_epsilon(mean_reward)
                        #     print("目前最新探索率:", self.selector.epsilon)
                        # else:
                        #     print("mean_reward:", mean_reward)

                        reward_tracker.reward(
                            new_rewards[0], self.step_idx, self.selector.epsilon)
                        
                        self.selector.epsilon = max(
                            self.EPSILON_STOP, self.EPSILON_START - self.step_idx / self.EPSILON_STEPS)

                    if not self.buffer.each_num_len_enough():
                        continue
                    

                    
                    self.optimizer.zero_grad()
                    batch = self.buffer.sample(self.BATCH_SIZE)

                    loss_v = common.calc_loss(
                        batch, self.net, self.tgt_net.target_model, self.GAMMA ** self.REWARD_STEPS, device=self.device)

                    loss_v.backward()

                    if self.step_idx % self.checkgrad_times == 0:
                        
                        self.checkgrad()
                        # self.checkwhight()

                    self.optimizer.step()
                    if self.step_idx % self.TARGET_NET_SYNC == 0:
                        self.tgt_net.sync()

                    # 在主訓練循環中的合適位置插入保存檢查點的代碼
                    if self.step_idx % self.CHECKPOINT_EVERY_STEP == 0:
                        idx = self.step_idx // self.CHECKPOINT_EVERY_STEP
                        checkpoint = {
                            'step_idx': self.step_idx,
                            'model_state_dict': self.net.state_dict(),                            
                            'tgt_net_state_dict': self.tgt_net.target_model.state_dict(),                            
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            # 'buffer_state': self.buffer.get_state(),  # 保存緩衝區
                        }
                        self.save_checkpoint(checkpoint, os.path.join(
                            self.saves_path, f"checkpoint-{idx}.pt"))

                        

                except Exception as e:
                    print("目前錯誤層級：訓練中")
                    print("目前時間：",datetime.now())
                    print("目前錯誤：",e)
                    traceback.print_exc()
                    print('*'*120)
                    time.sleep(10)

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
try:
    RL_Train()
except Exception as e:
    print("目前時間：",datetime.now())
    print("目前錯誤：",e)