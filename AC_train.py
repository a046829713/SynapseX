import os
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
import time
from dataclasses import dataclass
import torch.multiprocessing as mp
from collections import namedtuple
from Brain.DQN.lib import environment, common, model
from Brain.DQN.lib.environment import State_time_step
from Brain.Common.DataFeature import OriginalDataFrature
from Brain.Common.experience import PrioritizedStratifiedReplayBuffer
from Brain.DQN import ptan
import os
import numpy as np
import torch.optim as optim
from queue import Empty
from collections import deque

REWARD_STEPS = 2  # N-step
GAMMA = 0.99
NUM_ACTORS = 4

# 定義 Actor 發送的經驗元組，使其更清晰
Transition = namedtuple("Transition", ("state", "action", "reward", "done"))

# those entries are emitted from ExperienceSourceFirstLast. Reward is discounted over the trajectory piece
ExperienceFirstLast = namedtuple(
    "ExperienceFirstLast",
    ("state", "action", "reward", "last_state", "info", "last_info"),
)


@dataclass
class RLConfig:
    KEYWORD: str = "Mamba"
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVES_PATH: str = "saves"
    BARS_COUNT: int = 300
    GAMMA: float = 0.99
    REWARD_STEPS: int = 2
    MODEL_DEFAULT_COMMISSION_PERC: float = 0.0045
    DEFAULT_SLIPPAGE: float = 0.0025
    LEARNING_RATE: float = 0.00005
    LAMBDA_L2: float = 0.0
    BATCH_SIZE: int = 32
    REPLAY_SIZE: int = 1_000_000
    EACH_REPLAY_SIZE: int = 50_000
    REPLAY_INITIAL: int = 10000
    EPSILON_START: float = 0.9
    EPSILON_STOP: float = 0.1
    EPSILON_STEPS_FACTOR: int = 1_000_000
    TARGET_NET_SYNC: int = 1000
    CHECKPOINT_EVERY_STEP: int = 20_000
    BETA_START: float = 0.4

    def __post_init__(self):
        self.EPSILON_STEPS = 1000
        self.BETA_ANNEALING_STEPS = 1000

    def update_steps_by_symbols(self, num_symbols: int):
        self.EPSILON_STEPS = (
            self.EPSILON_STEPS_FACTOR * 30
            if num_symbols > 30
            else self.EPSILON_STEPS_FACTOR * num_symbols
        )
        self.BETA_ANNEALING_STEPS = self.EPSILON_STEPS


# --- Learner Process (重大修改) ---
class LearnerProcess(mp.Process):
    def __init__(
        self,
        config: RLConfig,
        engine_info: dict,
        symbol_count: int,
        state_queue: mp.Queue,
        action_queues: list[mp.Queue],
        experience_queue: mp.Queue,
        num_actors: int,
    ):
        super().__init__(daemon=True)
        self.config = config
        self.engine_info = engine_info
        # ... (省略與之前相似的初始化) ...
        self.state_queue = state_queue
        self.action_queues = action_queues
        self.experience_queue = experience_queue
        self.num_actors = num_actors

    def _prepare_learner_components(self):
        # ... (這部分與之前 LearnerProcess 創建 model, tgt_net, optimizer 的程式碼相同)
        # 為了簡潔，這裡直接複製貼上
        action_space_n = self.engine_info["action_space_n"]
        input_size = self.engine_info["input_size"]
        if self.config.KEYWORD == "Mamba":
            moe_config = {"num_experts": 16}
            self.net = model.mambaDuelingModel(
                d_model=input_size,
                nlayers=4,
                num_actions=action_space_n,
                seq_dim=self.config.BARS_COUNT,
                dropout=0.3,
                moe_cfg=moe_config,
            ).to(self.config.DEVICE)
        else:
            raise ValueError(f"Unknown model KEYWORD: {self.config.KEYWORD}")

        self.tgt_net = ptan.agent.TargetNet(self.net)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.LEARNING_RATE)
        print(f"Learner model created on {self.config.DEVICE}")

        # 為了讓 PrioritizedStratifiedReplayBuffer 工作，我們需要一個能 pop_rewards_steps 的物件
        # 我們創建一個簡單的類來模擬這個行為
        class ExperienceBufferWrapper:
            def __init__(self, learner_process):
                self.learner_process = learner_process
                self.rewards_steps = deque(
                    maxlen=self.learner_process.config.REPLAY_SIZE
                )

            def __iter__(self):
                # 這個 wrapper 不直接產生經驗，經驗來自 queue
                return self

            def __next__(self):
                # 從 experience_queue 獲取數據並處理
                try:
                    actor_id, transition_history, reward, last_state = (
                        self.learner_process.experience_queue.get_nowait()
                    )
                    # 將收到的原始數據轉換為 ptan 需要的 ExperienceFirstLast 格式
                    first_exp = transition_history[0]
                    exp = ExperienceFirstLast(
                        state=first_exp.state,
                        action=first_exp.action,
                        reward=reward,
                        last_state=last_state,
                    )
                    self.rewards_steps.append((reward, 0.0))  # (reward, loss)
                    return exp
                except Empty:
                    raise StopIteration

            def pop_rewards_steps(self):
                # ptan.buffer.populate 會呼叫這個
                r = list(self.rewards_steps)
                self.rewards_steps.clear()
                return r if r else None

        self.exp_wrapper = ExperienceBufferWrapper(self)
        self.buffer = PrioritizedStratifiedReplayBuffer(
            experience_source=self.exp_wrapper,
            batch_size=self.config.BATCH_SIZE,
            capacity=self.config.REPLAY_SIZE,
            each_capacity=self.config.EACH_REPLAY_SIZE,
            beta_start=self.config.BETA_START,
            beta_annealing_steps=self.config.BETA_ANNEALING_STEPS,
        )

    def _handle_inference_batch(self):
        """處理批次推理"""
        if self.state_queue.qsize() < self.num_actors:
            return  # 等待所有 actor 都提交請求，以達到最大批次效率

        batch = [self.state_queue.get() for _ in range(self.num_actors)]
        actor_ids, states = zip(*batch)

        states_v = torch.from_numpy(np.array(states, dtype=np.float32))
        states_v = states_v.to(self.config.DEVICE)

        with torch.no_grad():
            q_values = self.net(states_v)

        # 這裡我們需要 epsilon-greedy 策略來選擇動作
        # Learner 統一管理 epsilon
        epsilon = max(
            self.config.EPSILON_STOP,
            self.config.EPSILON_START - self.step_idx / self.config.EPSILON_STEPS,
        )

        actions = []
        if np.random.random() < epsilon:
            # 所有 actor 隨機動作
            actions = [
                np.random.randint(0, self.engine_info["action_space_n"])
                for _ in range(self.num_actors)
            ]
        else:
            # 所有 actor 貪婪動作
            actions = q_values.max(dim=1)[1].cpu().numpy()

        for actor_id, action in zip(actor_ids, actions):
            self.action_queues[actor_id].put(action.item())

    def run(self):
        print("--- Learner Process Started ---")
        self._prepare_learner_components()
        self.step_idx = 0

        saves_path = os.path.join(
            self.config.SAVES_PATH,
            datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")
            + "-"
            + str(self.config.BARS_COUNT)
            + "k-",
        )
        os.makedirs(saves_path, exist_ok=True)

        while True:
            # 優先處理推理請求，因為 Actors 正在等待
            self._handle_inference_batch()

            # 填充經驗緩衝區
            self.buffer.populate(self.num_actors)  # 嘗試填充 N 筆經驗

            if not self.buffer.is_ready():
                time.sleep(0.01)  # 避免空轉
                continue

            # 執行訓練
            self.step_idx += 1
            self.optimizer.zero_grad()
            batch_exp, batch_indices, batch_weights = self.buffer.sample()
            loss_v, td_errors = common.calc_loss(
                batch_exp,
                batch_weights,
                self.net,
                self.tgt_net.target_model,
                self.config.GAMMA**self.config.REWARD_STEPS,
                device=self.config.DEVICE,
            )
            loss_v.backward()
            self.optimizer.step()
            self.buffer.update_priorities(batch_indices, td_errors)

            if self.step_idx % self.config.TARGET_NET_SYNC == 0:
                self.tgt_net.sync()

            if self.step_idx % self.config.CHECKPOINT_EVERY_STEP == 0:
                # ... 儲存 checkpoint ...
                pass


# --- Learner Process (重大修改) ---
class LearnerProcess(mp.Process):
    def __init__(
        self,
        config: RLConfig,
        engine_info: dict,
        symbol_count: int,
        state_queue: mp.Queue,
        action_queues: list[mp.Queue],
        experience_queue: mp.Queue,
        num_actors: int,
    ):
        super().__init__(daemon=True)
        self.config = config
        self.engine_info = engine_info
        # ... (省略與之前相似的初始化) ...
        self.state_queue = state_queue
        self.action_queues = action_queues
        self.experience_queue = experience_queue
        self.num_actors = num_actors

    def _prepare_learner_components(self):
        # ... (這部分與之前 LearnerProcess 創建 model, tgt_net, optimizer 的程式碼相同)
        # 為了簡潔，這裡直接複製貼上
        action_space_n = self.engine_info["action_space_n"]
        input_size = self.engine_info["input_size"]
        if self.config.KEYWORD == "Mamba":
            moe_config = {"num_experts": 16}
            self.net = model.mambaDuelingModel(
                d_model=input_size,
                nlayers=4,
                num_actions=action_space_n,
                seq_dim=self.config.BARS_COUNT,
                dropout=0.3,
                moe_cfg=moe_config,
            ).to(self.config.DEVICE)
        else:
            raise ValueError(f"Unknown model KEYWORD: {self.config.KEYWORD}")

        self.tgt_net = ptan.agent.TargetNet(self.net)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.LEARNING_RATE)
        print(f"Learner model created on {self.config.DEVICE}")

        # 為了讓 PrioritizedStratifiedReplayBuffer 工作，我們需要一個能 pop_rewards_steps 的物件
        # 我們創建一個簡單的類來模擬這個行為
        class ExperienceBufferWrapper:
            def __init__(self, learner_process):
                self.learner_process = learner_process
                self.rewards_steps = deque(
                    maxlen=self.learner_process.config.REPLAY_SIZE
                )

            def __iter__(self):
                # 這個 wrapper 不直接產生經驗，經驗來自 queue
                return self

            def __next__(self):
                # 從 experience_queue 獲取數據並處理
                try:
                    actor_id, transition_history, reward, last_state = (
                        self.learner_process.experience_queue.get_nowait()
                    )
                    # 將收到的原始數據轉換為 ptan 需要的 ExperienceFirstLast 格式
                    first_exp = transition_history[0]
                    exp = ExperienceFirstLast(
                        state=first_exp.state,
                        action=first_exp.action,
                        reward=reward,
                        last_state=last_state,
                    )
                    self.rewards_steps.append((reward, 0.0))  # (reward, loss)
                    return exp
                except Empty:
                    raise StopIteration

            def pop_rewards_steps(self):
                # ptan.buffer.populate 會呼叫這個
                r = list(self.rewards_steps)
                self.rewards_steps.clear()
                return r if r else None

        self.exp_wrapper = ExperienceBufferWrapper(self)
        self.buffer = PrioritizedStratifiedReplayBuffer(
            experience_source=self.exp_wrapper,
            batch_size=self.config.BATCH_SIZE,
            capacity=self.config.REPLAY_SIZE,
            each_capacity=self.config.EACH_REPLAY_SIZE,
            beta_start=self.config.BETA_START,
            beta_annealing_steps=self.config.BETA_ANNEALING_STEPS,
        )

    def _handle_inference_batch(self):
        """處理批次推理"""
        if self.state_queue.qsize() < self.num_actors:
            return  # 等待所有 actor 都提交請求，以達到最大批次效率

        batch = [self.state_queue.get() for _ in range(self.num_actors)]
        actor_ids, states = zip(*batch)

        states_v = torch.from_numpy(np.array(states, dtype=np.float32))
        states_v = states_v.to(self.config.DEVICE)

        with torch.no_grad():
            q_values = self.net(states_v)

        # 這裡我們需要 epsilon-greedy 策略來選擇動作
        # Learner 統一管理 epsilon
        epsilon = max(
            self.config.EPSILON_STOP,
            self.config.EPSILON_START - self.step_idx / self.config.EPSILON_STEPS,
        )

        actions = []
        if np.random.random() < epsilon:
            # 所有 actor 隨機動作
            actions = [
                np.random.randint(0, self.engine_info["action_space_n"])
                for _ in range(self.num_actors)
            ]
        else:
            # 所有 actor 貪婪動作
            actions = q_values.max(dim=1)[1].cpu().numpy()

        for actor_id, action in zip(actor_ids, actions):
            self.action_queues[actor_id].put(action.item())

    def run(self):
        print("--- Learner Process Started ---")
        self._prepare_learner_components()
        self.step_idx = 0

        saves_path = os.path.join(
            self.config.SAVES_PATH,
            datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")
            + "-"
            + str(self.config.BARS_COUNT)
            + "k-",
        )
        os.makedirs(saves_path, exist_ok=True)

        while True:
            # 優先處理推理請求，因為 Actors 正在等待
            self._handle_inference_batch()

            # 填充經驗緩衝區
            self.buffer.populate(self.num_actors)  # 嘗試填充 N 筆經驗

            if not self.buffer.is_ready():
                time.sleep(0.01)  # 避免空轉
                continue

            # 執行訓練
            self.step_idx += 1
            self.optimizer.zero_grad()
            batch_exp, batch_indices, batch_weights = self.buffer.sample()
            loss_v, td_errors = common.calc_loss(
                batch_exp,
                batch_weights,
                self.net,
                self.tgt_net.target_model,
                self.config.GAMMA**self.config.REWARD_STEPS,
                device=self.config.DEVICE,
            )
            loss_v.backward()
            self.optimizer.step()
            self.buffer.update_priorities(batch_indices, td_errors)

            if self.step_idx % self.config.TARGET_NET_SYNC == 0:
                self.tgt_net.sync()

            if self.step_idx % self.config.CHECKPOINT_EVERY_STEP == 0:
                # ... 儲存 checkpoint ...
                pass


class ActorProcess(mp.Process):
    def __init__(
        self,
        actor_id: int,
        config: RLConfig,
        all_data: dict,
        state_queue: mp.Queue,
        action_queue: mp.Queue,
        experience_queue: mp.Queue,
    ):
        super().__init__(daemon=True)
        self.actor_id = actor_id
        self.config = config
        self.all_data = all_data
        self.state_queue = state_queue
        self.action_queue = action_queue
        self.experience_queue = experience_queue

    def _prepare_actor_components(self):
        """
            Actor 現在只創建環境
        """
        state_params = {
            "init_prices": self.all_data[np.random.choice(list(self.all_data.keys()))],
            "bars_count": self.config.BARS_COUNT,
            "commission_perc": self.config.MODEL_DEFAULT_COMMISSION_PERC,
            "model_train": True,
            "default_slippage": self.config.DEFAULT_SLIPPAGE,
        }
        state = State_time_step(**state_params)
        self.env = environment.Env(
            prices=self.all_data, state=state, random_ofs_on_reset=True
        )
        print(f"Actor {self.actor_id} ready.")

    def run(self):
        """
        need caculate the N-Step Reward.
        """
        self._prepare_actor_components()
        state = self.env.reset()

        # 每個 Actor 維護一個小型的 n-step 緩衝區
        n_step_buffer = deque(maxlen=REWARD_STEPS)

        while True:
            # 1. 將狀態發送給 Learner 請求動作
            self.state_queue.put((self.actor_id, state))

            # 2. 阻塞等待，直到收到 Learner 回傳的動作
            # action = self.action_queue.get()
            print("random action")
            action = np.random.randint(0, 2)

            # 3. 在環境中執行動作
            next_state, reward, done, info = self.env.step(action)

            n_step_buffer.append((state, action, reward))

            # 只有當緩衝區滿了，我們才能計算出第一個 transition 的 n-step reward
            if len(n_step_buffer) == REWARD_STEPS or done:
                total_reward = 0.0
                for transition in reversed(n_step_buffer):
                    # transition: (s, a, r)
                    reward_in_step = transition[2]
                    total_reward = reward_in_step + GAMMA * total_reward

                # 獲取這個 n-step 軌跡的 đầu (s_t, a_t) 和 cuối (s_{t+n}, d_{t+n})
                first_state, first_action, _ = n_step_buffer[0]
                last_state = None if done else next_state

                # 濃縮後的經驗元組
                # (初始狀態, 初始動作, N步獎勵, N步後的狀態, N步後是否終止)
                # 將處理好的 n-step 經驗發送到隊列
                self.experience_queue.put(
                    ExperienceFirstLast(
                        first_state, first_action, total_reward, last_state, info, done
                    )
                )

            # 更新 state
            state = next_state

            # 5. 處理 Episode 結束的情況
            if done:
                # 將 buffer 中剩餘的 transition 全部處理掉
                while len(n_step_buffer) > 1: # 處理到只剩最後一個
                    # 移除最舊的 transition
                    n_step_buffer.popleft()
                    
                    # 重新計算這個較短軌跡的 return
                    total_reward = 0.0
                    for transition in reversed(n_step_buffer):
                        total_reward = transition[2] + GAMMA * total_reward
                    
                    first_state, first_action, _ = n_step_buffer[0]
                    # last_state 依然是 None，因為 episode 已經結束
                    last_state = None

                    self.experience_queue.put(
                        ExperienceFirstLast(
                            first_state, first_action, total_reward, last_state, info, done
                        )
                    )
                
                # 清空 buffer 並重置環境
                n_step_buffer.clear()
                state = self.env.reset()

# --- 主執行流程 (重大修改) ---
NUM_ACTORS = 4


def main():
    config = RLConfig()
    symbolNames = [
        "BTCUSDT-F-30-Min",
        "AAVEUSDT-F-30-Min",
        "BNBUSDT-F-30-Min",
        "ENSUSDT-F-30-Min",
        "KSMUSDT-F-30-Min",
    ]

    unique_symbols = list(set(symbolNames))
    config.update_steps_by_symbols(len(unique_symbols))
    all_data = OriginalDataFrature().get_train_net_work_data_by_path(unique_symbols)

    # 創建一個臨時環境以獲取 `engine_info`
    # temp_env_state = State_time_step(
    #     init_prices=all_data[np.random.choice(list(all_data.keys()))],
    #     bars_count=config.BARS_COUNT,
    #     commission_perc=config.MODEL_DEFAULT_COMMISSION_PERC,
    #     model_train=True,
    #     default_slippage=config.DEFAULT_SLIPPAGE
    # )

    # temp_env = environment.Env(prices=all_data, state=temp_env_state)
    # engine_info = temp_env.engine_info()
    # del temp_env, temp_env_state

    # 建立新的 IPC Queues
    state_queue = mp.Queue(maxsize=NUM_ACTORS)
    action_queues = [mp.Queue(maxsize=1) for _ in range(NUM_ACTORS)]
    experience_queue = mp.Queue(maxsize=NUM_ACTORS * 5)

    # # 啟動 Learner Process
    # learner_proc = LearnerProcess(
    #     config,
    #     engine_info,
    #     len(unique_symbols),
    #     state_queue,
    #     action_queues,
    #     experience_queue,
    #     NUM_ACTORS,
    # )
    # learner_proc.start()

    # 啟動 Actor Processes
    actor_procs = []

    for i in range(NUM_ACTORS):
        print("目前序列：", i)
        actor = ActorProcess(
            i, config, all_data, state_queue, action_queues[i], experience_queue
        )

        actor.start()
        actor_procs.append(actor)

    time.sleep(100)
    print(f"--- {NUM_ACTORS} Actors and 1 Learner have been started. ---")

    # # 主行程可以做一些監控工作，或者直接等待
    # try:
    #     learner_proc.join()
    #     for actor in actor_procs:
    #         actor.join()
    # except KeyboardInterrupt:
    #     print("--- Main Process: Shutting down all processes. ---")
    #     learner_proc.terminate()
    #     for actor in actor_procs:
    #         actor.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
