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
from Brain.A2C.lib.model import ActorCriticModel
import copy 

def show_setting(title: str, content: str):
    print(f"--{title}--:{content}")


class RL_prepare(ABC):
    def __init__(self):
        self._prepare_keyword()
        self._prepare_device()
        self._prepare_symbols()
        self._prepare_writer()
        self._prepare_hyperparameters()
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
        symbols = ['TRBUSDT','BTCUSDT']
        self.symbols = list(set(symbols))
        show_setting("SYMBOLS", self.symbols)

    def _prepare_envs(self):
        self.train_envs = []
        if self.keyword == 'Transformer':
            print("目前商品:",[self.symbols[0]])
            # self.data[np.random.choice(list(self.data.keys()))]
            data = OriginalDataFrature().get_train_net_work_data_by_path([self.symbols[0]])

            state = State_time_step(
                                    init_prices=data[list(data.keys())[0]],
                                    bars_count=self.BARS_COUNT,
                                    commission_perc=self.MODEL_DEFAULT_COMMISSION_PERC,
                                    model_train=True,
                                    default_slippage=self.DEFAULT_SLIPPAGE
                                    )

        elif self.keyword == 'EfficientNetV2':
            pass

        show_setting("ENVIRONMENT-STATE", state)


        for symbol in self.symbols:            
            data = OriginalDataFrature().get_train_net_work_data_by_path([symbol])
            # 製作環境
            self.train_envs.append(Env(prices=data, state=state, random_ofs_on_reset=True))

    def _prepare_writer(self):
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                'C:\\', 'runs', datetime.strftime(
                    datetime.now(), "%Y%m%d-%H%M%S") + '-conv-'))

    def _prepare_hyperparameters(self):
        self.BARS_COUNT = 300  # 用來準備要取樣的特徵長度,例如:開高低收成交量各取10根K棒
        self.GAMMA = 0.99
        self.MODEL_DEFAULT_COMMISSION_PERC = 0.0005
        self.DEFAULT_SLIPPAGE = 0.0025
        self.LEARNING_RATE = 0.0001  # optim 的學習率
        self.VALUE_LOSS_COEF = 0.5  # 价值损失系数
        self.ENTROPY_COEF = 0.01  # 熵损失系数
        self.SAVES_PATH = "saves"  # 儲存的路徑

        self.CHECKPOINT_EVERY_STEP = 10000
        # self.REWARD_STEPS = 2
        # self.REPLAY_SIZE = 100000
        # self.REPLAY_INITIAL = 10000
        # self.EPSILON_START = 1.0  # 起始機率(一開始都隨機運行)
        # self.EPSILON_STOP = 0.1
        # self.TARGET_NET_SYNC = 1000
        # self.VALIDATION_EVERY_STEP = 100000
        # self.WRITER_EVERY_STEP = 100
        # self.EPSILON_STEPS = 1000000 * len(self.symbols)
        # self.EVAL_EVERY_STEP = 10000  # 每一萬步驗證一次
        # self.NUM_EVAL_EPISODES = 10  # 每次评估的样本数
        # self.BATCH_SIZE = 32  # 每次要從buffer提取的資料筆數,用來給神經網絡更新權重
        # self.STATES_TO_EVALUATE = 10000  # 每次驗證一萬筆資料
        # self.terminate_times = 8000000
        # self.checkgrad_times = 1000

    def _prepare_model(self):
        engine_info = self.train_envs[0].engine_info()
        
        # 初始化模型和優化器
        self.model = ActorCriticModel(
            d_model=engine_info['input_size'],
            nhead=4,
            d_hid=2048,
            nlayers=2,
            n_actions=self.train_envs[0].action_space.n,
            hidden_size=64,
            dropout=0.1,
            batch_first=True,
            num_iterations=3

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
