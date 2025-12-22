import os
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
import time
import torch.multiprocessing as mp
from collections import namedtuple
from Brain.DQN.lib import environment, common, model
from Brain.DQN import ptan
import os
import numpy as np
import torch.optim as optim
from queue import Empty, Full
from collections import deque
from Brain.Common.experience import ACSequentialExperienceReplayBuffer
import itertools
from utils.AppSetting import RLConfig
from utils.Debug_tool import debug



# those entries are emitted from ExperienceSourceFirstLast. Reward is discounted over the trajectory piece
ExperienceFirstLast = namedtuple(
    "ExperienceFirstLast",
    ("state", "action", "reward", "last_state", "info", "last_info","modelBase_feature"),
)

class MetricsTracker:
    """
    一個用於在主程序中追蹤和報告指標的類別。
    """
    def __init__(self, report_interval_seconds=1.0):
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.report_interval = report_interval_seconds
        
        # 來自 Actor 的指標
        self.total_episodes = 0
        self.rewards_deque = deque(maxlen=100)  # 儲存最近100個episodes的獎勵
        
        # 來自 Learner 的指標
        self.learner_step_idx = 0
        self.epsilon = 0.0
        self.loss = float('nan')
        self.steps_per_sec = 0.0
        
        self._last_learner_step_for_speed = 0
        self._last_time_for_speed = self.start_time

    def update(self, message):
        """根據從佇列收到的消息更新指標"""
        msg_type, *data = message
        if msg_type == "train_step":
            step_idx, loss, epsilon = data
            self.learner_step_idx = step_idx
            self.loss = loss
            self.epsilon = epsilon
        elif msg_type == "episode_done":
            _, total_reward, _ = data  # actor_id, total_reward, episode_steps
            self.total_episodes += 1
            self.rewards_deque.append(total_reward)

    def should_report(self):
        """判斷是否到了該報告的時間"""
        return time.time() - self.last_report_time > self.report_interval

    def report(self):
        """打印格式化的報告到控制台"""
        self.last_report_time = time.time()
        
        # 計算訓練速度
        current_time = time.time()
        time_delta = current_time - self._last_time_for_speed
        steps_delta = self.learner_step_idx - self._last_learner_step_for_speed
        
        if time_delta > 0:
            self.steps_per_sec = steps_delta / time_delta
        
        self._last_learner_step_for_speed = self.learner_step_idx
        self._last_time_for_speed = current_time

        # 計算平均獎勵
        mean_reward = 0.0
        if self.rewards_deque:
            mean_reward = np.mean(list(self.rewards_deque))

        # 格式化輸出
        elapsed_time = datetime.fromtimestamp(current_time) - datetime.fromtimestamp(self.start_time)
        
        report_str = (
            f"Time: {str(elapsed_time).split('.')[0]} | "
            f"Steps: {self.learner_step_idx} | "
            f"Speed: {self.steps_per_sec:7.2f} steps/s | "
            f"Epsilon: {self.epsilon:.3f} | "
            f"Loss: {self.loss:8.4f} | "
            f"Episodes: {self.total_episodes} | "
            f"Mean Reward (100): {mean_reward:8.3f}"
        )
        
        print(f"{report_str}")





class LearnerProcess(mp.Process):
    def __init__(
        self,
        config: RLConfig,
        engine_info: dict,
        state_queue: mp.Queue,
        action_queues: list[mp.Queue],
        experience_queue: mp.Queue,
        metrics_queue: mp.Queue,
        num_actors: int,
    ):
        super().__init__(daemon=True)
        self.config = config
        self.engine_info = engine_info
        self.state_queue = state_queue
        self.action_queues = action_queues
        self.experience_queue = experience_queue
        self.num_actors = num_actors
        self.metrics_queue = metrics_queue

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
        if hasattr(self.net, 'dean'):
            dean_params_ids.update(id(p) for p in self.net.dean.parameters())

        for name, param in self.net.named_parameters():
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
        if hasattr(self.net, 'dean'):
            param_groups.extend([
                {'params': list(self.net.dean.mean_layer.parameters()),
                 'lr': base_lr * self.net.dean.mean_lr, 'weight_decay': 0.0},
                {'params': list(self.net.dean.scaling_layer.parameters()),
                 'lr': base_lr * self.net.dean.scale_lr, 'weight_decay': 0.0},
                {'params': list(self.net.dean.gating_layer.parameters()),
                 'lr': base_lr * self.net.dean.gate_lr, 'weight_decay': 0.0},
            ])

        # 用 Adam 建立優化器
        self.optimizer = optim.Adam(param_groups)
        print("optimzer create.")

    def _prepare_learner_components(self):
        action_space_n = self.engine_info["action_space_n"]
        data_input_size = self.engine_info["data_input_size"]

        if self.config.KEYWORD == "Mamba":
            ssm_cfg = {
                "expand":4
            }
            self.net = model.mambaDuelingModel(
                d_model=data_input_size,
                nlayers=4,
                num_actions=action_space_n,
                time_features_in=self.engine_info["time_input_size"],
                seq_dim=self.config.BARS_COUNT,
                dropout=0.3,
                ssm_cfg=ssm_cfg                
            ).to(self.config.DEVICE)            
            
        else:
            raise ValueError(f"Unknown model KEYWORD: {self.config.KEYWORD}")

        print(f"net create current parameter size:{self.count_parameters()}")
        self.tgt_net = ptan.agent.TargetNet(self.net)
        self._prepare_optimizer()
        self._prepare_buffer()

        # self.net.compile()

    
    def count_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    
    def _handle_inference_batch(self):
        """處理批次推理"""
        q_size = self.state_queue.qsize()
        if q_size == 0:
            return

        batch = [self.state_queue.get() for _ in range(q_size)]
        actor_ids, _states = zip(*batch)
        states, time_states = zip(*_states)

        states_v = torch.from_numpy(np.array(states, dtype=np.float32))
        states_v = states_v.to(self.config.DEVICE)

        time_states_v = torch.from_numpy(np.array(time_states, dtype=np.float32))
        time_states_v = time_states_v.to(self.config.DEVICE)

        with torch.no_grad():
            q_values,_,imagined_features = self.net(states_v,time_states_v)

        # 這裡我們需要 epsilon-greedy 策略來選擇動作 Learner 統一管理 epsilon
        self.epsilon = max(
            self.config.EPSILON_STOP,
            self.config.EPSILON_START - self.step_idx / self.config.EPSILON_STEPS,
        )

        greedy_actions = q_values.max(dim=1)[1].cpu().numpy()
        
        # 2. 產生隨機動作
        num_actions = len(actor_ids)
        random_actions = np.random.randint(
            0, self.engine_info["action_space_n"], size=num_actions
        )
        
        # 3. 決定哪些狀態要進行探索
        should_explore = np.random.random(size=num_actions) < self.epsilon
        actions = np.where(should_explore, random_actions, greedy_actions)

        for actor_id, action in zip(actor_ids, actions):
            self.action_queues[actor_id].put(action.item())
        

    def _prepare_buffer(self):
        self.buffer = ACSequentialExperienceReplayBuffer(
            self.experience_queue,
            del_critical_len =self.config.REPLAY_SIZE,
            capacity = self.config.EACH_REPLAY_SIZE,
            replay_initial_size = self.config.REPLAY_INITIAL,            
        )

    def save_checkpoint(self, state, filename):
        # 保存檢查點的函數
        torch.save(state, filename)

    def checkgrad(self):
        # 打印梯度統計數據
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                print(
                    f"Layer: {name}, Grad Min: {param.grad.min()}, Grad Max: {param.grad.max()}, Grad Mean: {param.grad.mean()}")
    
    def run(self):
        print("--- Learner Process Started ---")
        self._prepare_learner_components()
        self.step_idx = 0

        while True:
            # 優先處理推理請求，因為 Actors 正在等待
            self._handle_inference_batch()

            # 填充經驗緩衝區
            self.buffer.populate()

            if not self.buffer.is_ready():
                continue

            # 執行訓練
            self.step_idx += 1
            self.optimizer.zero_grad()            
            batch_exp = self.buffer.sample(self.config.BATCH_SIZE)

            loss_v, td_errors = common.calc_loss(
                batch=batch_exp, 
                net=self.net, 
                tgt_net=self.tgt_net.target_model, 
                gamma=self.config.GAMMA ** self.config.REWARD_STEPS, 
                device=self.config.DEVICE,
                imag_loss_weight=0.1 # 這是想像損失的權重，您可以將其加入 RLConfig
            )

            loss_v.backward()
            self.optimizer.step()
            

            if self.step_idx % 100 == 0:
                try:
                    # 使用 non-blocking put 避免佇列滿時卡住 Learner
                    self.metrics_queue.put_nowait(
                        ("train_step", self.step_idx, loss_v.item(), self.epsilon)
                    )
                except Full:
                    pass # 如果佇列滿了，就跳過這次發送，不影響訓練

            if self.step_idx % self.config.TARGET_NET_SYNC == 0:
                self.tgt_net.sync()

            if self.step_idx % self.config.CHECKPOINT_EVERY_STEP == 0:
                idx = self.step_idx // self.config.CHECKPOINT_EVERY_STEP
                checkpoint = {
                    'step_idx': self.step_idx,
                    'model_state_dict': self.net.state_dict(),                            
                    # 'tgt_net_state_dict': self.tgt_net.target_model.state_dict(),                            
                    # 'optimizer_state_dict': self.optimizer.state_dict(),
                }
                self.save_checkpoint(checkpoint, os.path.join(
                    self.config.SAVES_PATH, f"checkpoint-{idx}.pt"))
                
            
            if self.step_idx % self.config.CHECK_GRAD_STEP == 0:
                self.checkgrad()
                print("目前buffer size:",len(self.buffer))



class ActorProcess(mp.Process):
    def __init__(
        self,
        actor_id: int,
        config: RLConfig,
        state_queue: mp.Queue,
        task_queue: mp.Queue,
        action_queue: mp.Queue,
        experience_queue: mp.Queue,
        metrics_queue: mp.Queue,
    ):
        super().__init__(daemon=True)
        self.actor_id = actor_id
        self.config = config
        self.state_queue = state_queue
        self.task_queue = task_queue
        self.action_queue = action_queue
        self.experience_queue = experience_queue
        self.metrics_queue = metrics_queue

    def _prepare_actor_components(self):
        """
            Actor 現在只創建環境
        """
        self.env = environment.TrainingEnv(
            config=self.config
        )

        print(f"Actor {self.actor_id} ready.")

    def run(self):
        """
        need caculate the N-Step Reward.
        """
        print(f"--- Actor {self.actor_id} Process Started ---")
        self._prepare_actor_components()

        while True:
            # 1. 從中央協調者獲取一個任務（商品）
            symbol = self.task_queue.get()
            if symbol is None:  # 收到結束信號
                print(f"Actor {self.actor_id} received shutdown signal.")
                break

            # 2. 使用獲取到的商品重置環境
            state = self.env.reset(symbol=symbol)
            modelBase_feature = self.env.getModelBase_feature()

            n_step_buffer = deque(maxlen=self.config.REWARD_STEPS)
            done = False
            episode_accumulated_reward = 0.0
            episode_steps = 0

            # 3. 執行一個完整的 episode
            while not done:
                # 3.1. 將狀態發送給 Learner 請求動作
                self.state_queue.put((self.actor_id, state))

                # 3.2. 阻塞等待，直到收到 Learner 回傳的動作
                action = self.action_queue.get()

                # 3.3. 在環境中執行動作
                next_state, reward, done, info = self.env.step(action)
                
                episode_accumulated_reward += reward
                episode_steps += 1
                n_step_buffer.append((state, action, reward, modelBase_feature))

                # 3.4. 計算 N-Step Reward 並發送經驗
                if len(n_step_buffer) == self.config.REWARD_STEPS or (done and len(n_step_buffer) > 0):
                    total_reward = 0.0
                    for transition in reversed(n_step_buffer):
                        reward_in_step = transition[2]
                        total_reward = reward_in_step + self.config.GAMMA * total_reward

                    first_state, first_action, _, first_modelBase_feature = n_step_buffer[0]
                    last_state = None if done else next_state

                    self.experience_queue.put(
                        ExperienceFirstLast(
                            first_state, first_action, total_reward, last_state, info, done, first_modelBase_feature
                        )
                    )

                state = next_state
                modelBase_feature = self.env.getModelBase_feature()
                
                # 3.5. 如果 episode 結束，處理剩餘的 n-step transitions
                if done:
                    try:                        
                        self.metrics_queue.put_nowait(
                            ("episode_done", self.actor_id, episode_accumulated_reward, episode_steps)
                        )
                    except Full:
                        pass # 如果佇列滿了，就跳過
                    
                    while len(n_step_buffer) > 1:
                        n_step_buffer.popleft()
                        total_reward = 0.0
                        for transition in reversed(n_step_buffer):
                            total_reward = transition[2] + self.config.GAMMA * total_reward
                        
                        first_state, first_action, _,first_modelBase_feature = n_step_buffer[0]
                        self.experience_queue.put(
                            ExperienceFirstLast(
                                first_state, first_action, total_reward, None, info, done, first_modelBase_feature
                            )
                        )


class SymbolProcess(mp.Process):
    def __init__(self, task_queue: mp.Queue, symbols: list[str], shutdown_event):
        """
            這個 Process 專門負責向任務佇列中循環、無限地提供商品名稱。
        """
        super().__init__(daemon=True)
        self.task_queue = task_queue
        self.symbols = symbols
        self.shutdown_event = shutdown_event

    def run(self):
        print("--- Symbol Process Started ---")
        # 使用 itertools.cycle 實現無限循環
        for symbol in itertools.cycle(self.symbols):
            if self.shutdown_event.is_set():
                print("Symbol Process received shutdown signal.")
                break
            # put 會阻塞，直到佇列中有可用空間，這可以自然地調節任務生成速度
            self.task_queue.put(symbol)
        print("--- Symbol Process Exited ---")


# --- 主執行流程 (重大修改) ---
NUM_ACTORS = 4


def main():
    config = RLConfig()
    symbolNames = os.listdir(os.path.join(os.getcwd() , "Brain","simulation","train_data"))
    symbolNames = [_fileName.split('.')[0] for _fileName in symbolNames]
    symbolNames =  ['BTCUSDT-F-30-Min','ETHUSDT-F-30-Min','BNBUSDT-F-30-Min','SOLUSDT-F-30-Min']
    print(symbolNames)

    unique_symbols = list(set(symbolNames))
    config.update_steps_by_symbols(len(unique_symbols))
    config.create_saves_path()
    config.UNIQUE_SYMBOLS = unique_symbols
    

    # 創建一個臨時環境以獲取 `engine_info`
    temp_env = environment.TrainingEnv(
            config=config
    )
    engine_info = temp_env.engine_info()
    del temp_env

    # 建立新的 IPC Queues
    state_queue = mp.Queue(maxsize=NUM_ACTORS)
    action_queues = [mp.Queue(maxsize=1) for _ in range(NUM_ACTORS)]
    experience_queue = mp.Queue(maxsize=NUM_ACTORS * 5)


    # 為指標佇列設定一個合理的上限，防止其無限增長
    metrics_queue = mp.Queue(maxsize=NUM_ACTORS * 100)

    # 建立一個事件來通知子行程關閉
    shutdown_event = mp.Event()

    # 建立任務佇列，並由 SymbolProcess 負責填充
    # 佇列大小可以限制，防止 SymbolProcess 產生過多任務塞爆記憶體
    task_queue = mp.Queue(maxsize=NUM_ACTORS * 2)

    # 啟動 SymbolProcess
    symbol_proc = SymbolProcess(task_queue, config.UNIQUE_SYMBOLS, shutdown_event)
    symbol_proc.start()

    # 啟動 Learner Process
    learner_proc = LearnerProcess(
        config,
        engine_info,
        state_queue,
        action_queues,
        experience_queue,
        metrics_queue,
        NUM_ACTORS,
    )
    learner_proc.start()

    # 啟動 Actor Processes
    actor_procs = []

    for i in range(NUM_ACTORS):
        actor = ActorProcess(
            i, config, state_queue, task_queue, action_queues[i], experience_queue, metrics_queue
        )

        actor.start()
        actor_procs.append(actor)

    print(
        f"--- {NUM_ACTORS} Actors, 1 Learner, and 1 SymbolProvider have been started. ---"
    )
    print("--- Press Ctrl+C to stop the training. ---")


    tracker = MetricsTracker(report_interval_seconds=30.0)
    try:
        while True:
            # 從佇列中獲取所有可用的指標訊息
            while not metrics_queue.empty():
                try:
                    message = metrics_queue.get_nowait()
                    tracker.update(message)
                except Empty:
                    break
            
            # 定期打印報告
            if tracker.should_report():
                tracker.report()

            time.sleep(1)  # 稍微等待，避免CPU佔用過高

    except KeyboardInterrupt:
        print("\n--- Main Process: Shutting down all processes. ---")

        # 1. 設置關閉事件，通知 SymbolProvider 停止
        shutdown_event.set()

        # 2. 發送停止信號給所有 Actor
        print("Sending shutdown signal to actors...")
        for _ in range(NUM_ACTORS):
            try:
                # 使用非阻塞的 put 避免佇列滿時卡住
                task_queue.put(None, timeout=1)
            except Full:
                pass # Actor 可能已經終止

        # 3. 給予 Actor 一些時間來正常結束
        print("Waiting for actors to terminate gracefully (30s timeout)...")
        for actor in actor_procs:
            actor.join(timeout=30)

        # 4. 強制終止仍然在運行的行程
        print("Forcibly terminating any remaining processes...")
        if learner_proc.is_alive():
            learner_proc.terminate()
        if symbol_proc.is_alive():
            symbol_proc.terminate()
        for actor in actor_procs:
            if actor.is_alive():
                actor.terminate()
        
        print("--- All processes have been shut down. ---")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
