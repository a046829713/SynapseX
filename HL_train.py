import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC

# ======================
#  Environment 設計
# ======================
class StockSelectionEnv:
    """
    高階環境 (階段 A):
    - 狀態: 全市場日/週等低頻資料 (N 檔股票 x 若干因子), 以及當前持有情況
    - 動作: 選股/資產配置 (維度可大)
    - 獎勵: 後續(在交易階段)的總收益 or 預估收益
    """
    def __init__(self, all_data_daily):
        self.all_data_daily = all_data_daily
        self.current_step = 0
        # ... 其他初始化

    def reset(self):
        self.current_step = 0
        # 重置並返回初始 state
        state = self._get_state()
        return state
    
    def step(self, action):
        """
        action: 在當前 step 選擇哪些股票 (例如 [0,1,0,1,...])
        returns: next_state, reward, done, info
        """
        # 將 action 存起來, 讓子環境(交易階段)知道本週要交易哪些標的
        selected_indices = np.where(action > 0.5)[0]  # 舉例: action 為 [Binarized]
        
        # 進入子階段(交易階段) - 這裡可以呼叫另一個低階環境
        trading_env = TradingEnv(self.all_data_daily, selected_indices, start_step=self.current_step)
        # 在 trading_env 執行多天(或多分鐘)後得到最終收益
        total_reward = trading_env.run_one_period()  
        
        # 計算獎勵
        reward = total_reward  # 可自行設計

        # 更新 time step
        self.current_step += 1
        done = (self.current_step >= len(self.all_data_daily) - 1)
        
        next_state = self._get_state()
        
        return next_state, reward, done, {}
    
    def _get_state(self):
        # 取得 (self.current_step) 位置的日級特徵, 拼裝成可供高階策略使用的 state
        # e.g. shape = [N_stocks, feature_dim]
        # 這裡僅簡化表示
        state = self.all_data_daily[self.current_step]
        return state


class TradingEnv:
    """
    低階環境 (階段 B):
    - 狀態: 已選中的股票的分鐘級資料
    - 動作: 買/賣/持有, 數量 (離散 or 連續)
    - 獎勵: 每日(或每分鐘)的損益, 最後匯總傳回給高階
    """
    def __init__(self, all_data_daily, selected_indices, start_step=0):
        self.all_data_daily = all_data_daily
        self.selected_indices = selected_indices
        self.start_step = start_step
        # minute_data etc...
        # 這裡假設我們能取得對應 minute_data, 省略細節
        self.done = False

    def reset(self):
        # 重置, 返回初始 state (分鐘級別)
        return self._get_minute_state()

    def step(self, action):
        """
        action: 對已選股票的交易行為, shape=[len(selected_indices), ...]
        returns: next_state, reward, done, info
        """
        # 根據 action 計算單步獎勵 e.g. 實際的交易盈虧
        reward = self._calculate_reward(action)
        
        # 移動到下一分鐘
        self._goto_next_minute()
        next_state = self._get_minute_state()
        done = self._check_done()
        
        return next_state, reward, done, {}

    def run_one_period(self):
        """
        用來在 high-level 進行呼叫,
        選定股票後, run_one_period() 負責把一天(或多天)的交易跑完,
        回傳該週期累計的收益 (作為 high-level reward).
        """
        total_reward = 0.0
        state = self.reset()
        done = False
        
        while not done:
            # 這邊可使用低階 policy \pi_B(state)
            action = self._select_action_low_level(state)
            next_state, reward, done, _ = self.step(action)
            total_reward += reward
            state = next_state
        
        return total_reward
    
    # 低階交易策略 - policy_B (可用 NN / Policy Gradient)
    def _select_action_low_level(self, state):
        """
        這裡只是簡化示意, 真實用法會呼叫你訓練好的 \pi_B
        e.g. action = pi_B.sample_action(state)
        """
        # return random or policy-based action
        action = np.zeros(len(self.selected_indices))
        return action

    def _get_minute_state(self):
        # 取得目前分鐘資料作為 low-level 狀態
        return None  # pseudo code
    
    def _goto_next_minute(self):
        # 移動到下一分鐘
        pass

    def _check_done(self):
        # 根據 minute index 判斷是否結束
        return self.done

    def _calculate_reward(self, action):
        return 0.0  # pseudo code


# ======================
#  Policy / Model 設計
# ======================
class HighLevelPolicy(nn.Module):
    """
    高階策略 \pi_A, 負責選股
    input: state (N_stocks, feature_dim)
    output: action (N_stocks) -> 0/1 or (0~1) 代表選或不選
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, state):
        """
        state: shape [N_stocks, feature_dim] 可能需要 flatten 或 pooling
        """
        # pseudo: flatten
        x = state.view(-1)  # 只是示意, 真實需更細緻設計
        logits = self.fc(x)  # shape [output_dim], output_dim = N_stocks
        return logits
    
    def select_action(self, state):
        """
        以 policy gradient 做二元採樣 (Bernoulli) 或 Continuous mapping (Sigmoid)
        這裡僅示意:
        """
        logits = self.forward(state)  # [N_stocks]
        probs = torch.sigmoid(logits) # range in (0,1)
        # 依機率選取
        # action ~ Bernoulli(probs)
        action = torch.bernoulli(probs).detach().cpu().numpy()
        return action, probs


class LowLevelPolicy(nn.Module):
    """
    低階策略 \pi_B, 負責對已選股做交易
    這裡只是示意, 在 TradingEnv.run_one_period() 會被使用
    """
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        # state shape: [some feature dimension]
        logits = self.fc(state)
        return logits
    
    def select_action(self, state):
        # 例如多維離散動作, 或連續動作, 這裡簡化
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        # action ~ Categorical(probs)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action, action_dist.log_prob(action)



def show_setting(title: str, content: str):
    print(f"--{title}--:{content}")

class RL_prepare(ABC):
    def __init__(self):
        # 1.進行日線資料生成        
        self._prepare_hyperparameters()
        
        # self._prepare_keyword()
        # self._prepare_device()
        # self._prepare_symbols()
        # self._prepare_writer()
        # self._prepare_envs()
        # self._prepare_model()
        # self._prepare_targer_net()
        # self._prepare_agent()
        # self._prepare_optimizer()
    
    def _prepare_hyperparameters(self):
        pass
        # self.BARS_COUNT = 300  # 用來準備要取樣的特徵長度,例如:開高低收成交量各取10根K棒
        # self.GAMMA = 0.99
        # self.MODEL_DEFAULT_COMMISSION_PERC = 0.0025
        # self.DEFAULT_SLIPPAGE = 0.0025
        # self.LEARNING_RATE = 0.0001  # optim 的學習率
        
        # self.ENTROPY_COEF = 0.01  # 熵损失系数
        # self.SAVES_PATH = "saves"  # 儲存的路徑

        # self.CHECKPOINT_EVERY_STEP = 100
        # self.VALUE_LOSS_COEF = 0.1  # 價值損失函數 critic損失函數
        # self.N_STEP = 200
        # self.checkgrad_times = 10

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



    def _prepare_model(self):
        engine_info = self.train_env.engine_info()

        # 初始化模型和優化器
        self.model = ActorCriticModel(
            d_model=engine_info['input_size'],
            nhead=4,
            d_hid=2048,
            nlayers=2,
            n_actions=self.train_env.action_space.n,
            hidden_size=64,
            dropout=0.1,
            batch_first=True,
            num_iterations=1

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


























# ======================
#  多階段訓練流程 (Policy Gradient)
# ======================
def train_multi_stage(env, high_level_policy, low_level_policy, 
                      optimizerA, optimizerB, 
                      num_episodes=1000, gamma=0.99):
    """
    env: StockSelectionEnv (內部會呼叫 TradingEnv)
    high_level_policy: \pi_A
    low_level_policy: \pi_B
    optimizerA, optimizerB: 兩個 policy 分開的優化器
    gamma: 折現因子
    """
    for episode in range(num_episodes):
        state = env.reset()
        
        # 儲存高階動作與 log_prob
        high_level_log_probs = []
        high_level_rewards = []

        done = False
        while not done:
            # 1) 高階選股 (ActionA)
            actionA, probsA = high_level_policy.select_action(torch.tensor(state, dtype=torch.float))
            
            # 2) 執行 env.step(actionA) -> 內部會呼叫 TradingEnv.run_one_period()
            next_state, rewardA, done, _ = env.step(actionA)
            
            # 3) 記錄高階 log_prob (簡化處理, 這裡只示意):
            log_probA = torch.distributions.Bernoulli(probsA).log_prob(torch.tensor(actionA))
            high_level_log_probs.append(log_probA.sum())  # sum or mean
            high_level_rewards.append(rewardA)
            
            state = next_state
        
        # =========================
        #  高階策略更新 (REINFORCE)
        # =========================
        # 1) 計算 returns
        returns = []
        G = 0.0
        for r in reversed(high_level_rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float)
        
        # 2) Policy Gradient loss
        #   J(\theta_A) ~ E[ log pi_A(a|s) * G ]
        policy_loss_A = 0
        for log_prob, R in zip(high_level_log_probs, returns):
            policy_loss_A += -log_prob * R
        
        optimizerA.zero_grad()
        policy_loss_A.backward()
        optimizerA.step()

        # =========================
        #  低階策略更新
        # =========================
        #  低階策略的軌跡在 TradingEnv 裡收集
        #  這裡只給示意結構: 
        #  你可能需要在 TradingEnv.run_one_period() 裡
        #  記錄 (s, a, log_prob, reward) 後再傳回上層做更新
        #  或者單獨訓練 \pi_B
        #
        # pseudo:
        low_level_trajectories = gather_trajectories_from_trading_env()
        policy_loss_B = compute_policy_gradient_loss(low_level_trajectories, gamma)
        
        optimizerB.zero_grad()
        policy_loss_B.backward()
        optimizerB.step()

        print(f"Episode: {episode}, Reward: {sum(high_level_rewards):.2f}")

    print("Training finished!")
