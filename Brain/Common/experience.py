import random
import numpy as np
import itertools
from collections import deque
import time
from Brain.DQN.ptan.experience import ExperienceSource
import torch.multiprocessing as mp
import random
from collections import namedtuple, deque
import numpy as np
from utils.Debug_tool import debug
from queue import Empty

# SumTree 是實現 PER 的核心資料結構
# 它能讓我們在 O(log N) 的時間複雜度內完成抽樣與更新
class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity

        # 樹狀結構，使用一個陣列來表示。父節點索引為 (i-1)//2
        # 樹的大小為 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        # data 儲存的是經驗在 deque 中的索引
        self.data = np.zeros(capacity, dtype=object)

        self.write_idx = 0  # 當前寫入的位置
        self.n_entries = 0  # 當前儲存的條目數量

    @property
    def total_priority(self):
        return self.tree[0]

    def _propagate(self, idx, change):
        """向上傳播優先級的變化"""
        parent_idx = (idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate(parent_idx, change)

    def _retrieve(self, idx, s):
        """根據抽樣值 s 查找對應的葉節點"""
        left_child_idx = 2 * idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= len(self.tree):
            return idx

        if s <= self.tree[left_child_idx]:
            return self._retrieve(left_child_idx, s)
        else:
            return self._retrieve(right_child_idx, s - self.tree[left_child_idx])

    def add(self, priority, data_idx):
        """
        添加新的經驗及其優先級
        """
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data_idx
        self.update(tree_idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        """更新指定索引的優先級"""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get_leaf(self, s):
        """根據抽樣值 s 獲取葉節點索引、優先級和對應的數據"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedStratifiedReplayBuffer:
    # PER 的超參數
    epsilon = 0.01  # 避免優先級為0
    alpha = 0.6  # [0~1] 決定優先級的使用程度，0:均勻抽樣, 1:完全按優先級抽樣

    def __init__(
        self,
        experience_source,
        batch_size: int,
        capacity: int,
        each_capacity: int,
        beta_start: float,
        beta_annealing_steps: int,
    ):
        """
        batch_size (int): batch_size input model.
        capacity (int): 整個緩衝區的總容量上限。
        each_capacity (int): 每個 symbol (層) 的容量上限。

        We also save SumTree with every symbol.

        beta [0~1] 重要性抽樣的校正程度，初始值，會線性增加到1
        """
        assert isinstance(experience_source, (ExperienceSource, type(None)))
        assert isinstance(batch_size, int)
        self.experience_source_iter = (
            None if experience_source is None else iter(experience_source)
        )

        self.batch_size = batch_size
        # 分層結構
        self.symbols = []  # 儲存所有 symbol 的名稱
        self.buffer = {}  # key: symbol, value: deque(maxlen=each_capacity)
        self.capacity = capacity
        self.each_capacity = each_capacity
        self.max_priority = 1.0  # 新經驗的初始優先級
        # 優先級結構
        self.priorities = {}  # key: symbol, value: SumTree(each_capacity)

        self.beta = 0.4
        if beta_annealing_steps > 0:
            self.beta_increment_per_sampling = (1.0 - beta_start) / beta_annealing_steps
        else:
            self.beta_increment_per_sampling = 0  # 如果步數為0，則不增加

    def __len__(self):
        # 長度現在由 SumTree 中的 n_entries 決定，這更準確
        return sum(tree.n_entries for tree in self.priorities.values())

    def is_ready(self):
        """檢查緩衝區中的樣本是否足夠開始訓練"""
        return len(self) >= self.batch_size
        # return False

    def populate(self, samples):
        """
        only inference with agent,
        the agent depands epsilon to decide random action.


        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            symbol = entry.info["instrument"]

            if symbol not in self.buffer:
                if len(self) >= self.capacity:
                    self.drop_symbol()

                self.symbols.append(symbol)
                self.buffer[symbol] = [None] * self.each_capacity
                self.priorities[symbol] = SumTree(self.each_capacity)

            symbol_sum_tree = self.priorities[symbol]
            # 這就是 data 在 buffer 中的索引
            data_idx = symbol_sum_tree.write_idx
            # 在 buffer 和 sum_tree 中添加數據
            self.buffer[symbol][data_idx] = entry
            symbol_sum_tree.add(self.max_priority, data_idx)

    def sample(self):
        """
        從緩衝區中抽樣一個批次的數據。
        返回: (批次經驗, 對應的索引(用於更新優先級), 重要性抽樣權重)
        """
        batch = []
        indices = []
        weights = np.empty(self.batch_size, dtype=np.float32)

        # 1. 分層抽樣: 隨機選擇 batch_size 個 symbols (可重複)
        # 這種策略能讓樣本多的 symbol 有更高機率被選中
        total_len = len(self)
        symbol_probs = [self.priorities[s].n_entries / total_len for s in self.symbols]
        # 確保 symbol_probs 的和約等於 1，避免浮點數誤差
        symbol_probs = np.array(symbol_probs)
        symbol_probs /= symbol_probs.sum()

        chosen_symbols = random.choices(
            self.symbols, weights=symbol_probs, k=self.batch_size
        )

        # 2. 優先級抽樣: 從每個選中的 symbol 中抽取一個樣本
        for i, symbol in enumerate(chosen_symbols):
            p_total = self.priorities[symbol].total_priority
            s = random.uniform(0, p_total)
            tree_idx, priority, data_idx = self.priorities[symbol].get_leaf(s)

            experience = self.buffer[symbol][data_idx]
            batch.append(experience)
            indices.append({"symbol": symbol, "tree_idx": tree_idx})

            # 3. 計算重要性抽樣 (IS) 權重
            # P(i) = priority / total_priority_of_buffer
            prob = priority / p_total
            # 這裡的 N 是該層的樣本數，而不是總樣本數
            weights[i] = np.power(self.priorities[symbol].n_entries * prob, -self.beta)

        # 更新 beta
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        # 標準化權重
        if weights.max() > 0:
            weights /= weights.max()

        return batch, indices, weights

    def drop_symbol(self):
        """
        當總容量超過上限時，移除一個 symbol 的所有相關數據。
        """
        # 策略：移除最久沒有被更新的 symbol (最簡單的策略是移除最先加入的)
        if not self.symbols:
            return

        symbol_to_drop = self.symbols.pop(0)
        print(f"總容量超過 {self.capacity}，移除 Symbol: {symbol_to_drop}")
        del self.buffer[symbol_to_drop]
        del self.priorities[symbol_to_drop]

    def update_priorities(self, indices: list, td_errors: np.ndarray):
        """
        在模型訓練後，使用新的 TD-Error 更新經驗的優先級。
        indices (list): sample() 方法返回的索引列表。
        td_errors (np.ndarray): 對應每個經驗的 TD-Error 絕對值。
        """
        priorities = np.power(np.abs(td_errors) + self.epsilon, self.alpha)

        for i, idx_info in enumerate(indices):
            symbol = idx_info["symbol"]
            tree_idx = idx_info["tree_idx"]
            priority = priorities[i]

            self.priorities[symbol].update(tree_idx, priority)
            # 更新記錄到的最大優先級
            self.max_priority = max(self.max_priority, priority)


class SequentialExperienceReplayBuffer:
    def __init__(
        self,
        experience_source,
        buffer_size,
        replay_initial_size: int,
    ):
        """
        Initialize the buffer with a source, capacity, and symbol count.
        :param batch_size: 總的批次大小。
        _count (int) : count the times.
        """
        assert isinstance(experience_source, (ExperienceSource, type(None)))
        assert isinstance(buffer_size, int)

        self.experience_source_iter = (
            None if experience_source is None else iter(experience_source)
        )

        # 用來儲存所有商品的空間
        self.buffer = {}
        self.sample_count_buffer = {}
        self.capacity = buffer_size
        self.replay_initial_size = replay_initial_size
        assert (
            self.capacity > replay_initial_size
        ), "The capacity is smaller than init_size, please change the capacity"

        self._count = 0
        self._del_critical_len = 1000000

    def is_ready(self):
        if self._count > self.replay_initial_size:
            return True
        else:
            return False

    def __len__(self):
        """
        Total number of experiences in the buffer.
        """
        return sum(len(deq) for deq in self.buffer.values())

    def get_random_symbol(self):
        return random.choice(
            [
                symbolName
                for symbolName, data in self.buffer.items()
                if len(data) > self.replay_initial_size
            ]
        )

    def sample(self, batch_size):
        """
        Sample a batch of sequential experiences from a random symbol.

        """
        symbol = self.get_random_symbol()
        data = self.buffer[symbol]
        start = random.randint(0, len(data) - batch_size)
        batch = list(itertools.islice(data, start, start + batch_size))
        offset = batch[0].info["offset"]
        if all(exp.info["offset"] == offset + i for i, exp in enumerate(batch)):
            if symbol not in self.sample_count_buffer:
                self.sample_count_buffer[symbol] = 1
            else:
                self.sample_count_buffer[symbol] += 1
            return batch

        return self.sample(batch_size)

    def populate(self, samples):
        """
        將樣本填入緩衝區中
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)

            if entry.info["instrument"] not in self.buffer:
                self.buffer[entry.info["instrument"]] = deque(maxlen=self.capacity)

            self.buffer[entry.info["instrument"]].append(entry)

        self._count += 1
        if sum(len(deq) for deq in self.buffer.values()) > self._del_critical_len:
            self.dropsymbol()

    def dropsymbol(self):
        best_name = max(self.sample_count_buffer, key=self.sample_count_buffer.get)
        self.buffer.pop(best_name)
        self.sample_count_buffer.pop(best_name)

    def get_state(self):
        """
        保存緩衝區的當前狀態。
        返回一個字典，包含緩衝區數據和其他屬性。
        """
        return {
            "buffer": self.buffer,  # 直接保存 dict of deque
            "capacity": self.capacity,
        }


class MultipleSequentialExperienceReplayBuffer(SequentialExperienceReplayBuffer):
    def __init__(
        self,
        experience_source,
        buffer_size,
        replay_initial_size,
        batch_size,
        num_symbols_to_sample,
    ):
        super().__init__(experience_source, buffer_size, replay_initial_size)
        """
            :param num_symbols_to_sample: 要從多少個不同的商品中抽樣。
        
        """

        # mixer symbol Area
        # 確保 batch_size 可以被均分
        assert (
            batch_size % num_symbols_to_sample == 0
        ), "batch_size 必須可以被 num_symbols_to_sample 整除。"
        self.samples_per_symbol = batch_size // num_symbols_to_sample
        self.num_symbols_to_sample = num_symbols_to_sample

    def each_num_len_enough(self):
        """
        須確定取樣的時候已經超過所需要的資料
        """
        self.available_symbols_len = [
            symbolName
            for symbolName, data in self.buffer.items()
            if len(data) >= self.replay_initial_size
        ]
        return len(self.available_symbols_len) > self.num_symbols_to_sample

    def _sample(self, symbol: str, batch_size):
        """
        Sample a batch of sequential experiencesl.

        """
        symbol = self.get_random_symbol()
        data = self.buffer[symbol]
        start = random.randint(0, len(data) - batch_size)
        batch = list(itertools.islice(data, start, start + batch_size))
        offset = batch[0].info["offset"]
        if all(exp.info["offset"] == offset + i for i, exp in enumerate(batch)):
            if symbol not in self.sample_count_buffer:
                self.sample_count_buffer[symbol] = 1
            else:
                self.sample_count_buffer[symbol] += 1
            return batch

        return self._sample(symbol, batch_size)

    def mixer_sample(self):
        """
        從多個隨機選擇的商品中抽樣，構建成一個混合批次。
        :return: 一個包含混合經驗的列表。
        """
        final_batch = []

        # 2. 隨機選取指定數量的商品 (不重複)
        selected_symbols = random.sample(
            self.available_symbols_len, self.num_symbols_to_sample
        )

        # 3. 從每個選中的商品中抽取樣本
        for symbol in selected_symbols:
            final_batch.extend(self._sample(symbol, batch_size=self.samples_per_symbol))

        return final_batch


class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size):
        assert isinstance(experience_source, (ExperienceSource, type(None)))
        assert isinstance(buffer_size, int)
        self.experience_source_iter = (
            None if experience_source is None else iter(experience_source)
        )
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample):
        """

            將跌代的資料帶入
            萬一超過就覆寫
        Args:
            sample (_type_): _description_
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
            self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples):
        """
         將樣本填入緩衝區中
         Populates samples into the buffer
         :param samples: how many samples to populate

         <class 'ptan.experience.ExperienceFirstLast'>
         entry: ExperienceFirstLast(state=array([ 0.00773994, -0.01083591,  0.00773994,  0.00456621, -0.01065449,
         0.00456621,  0.00607903, -0.00455927,  0.00455927,  0.        ,
        -0.01783061, -0.00148588,  0.00437956, -0.01021898, -0.00291971,
         0.00442478, -0.02359882, -0.02359882,  0.01226994, -0.00153374,
         0.00306748,  0.01076923, -0.00615385,  0.00153846,  0.00310559,
        -0.01086957, -0.00465839,  0.02503912, -0.00312989,  0.02190923,
         0.        ,  0.        ], dtype=float32), action=1, reward=-2.7099031710120034, last_state=array([ 0.00607903, -0.00455927,  0.00455927,  0.        , -0.01783061,
        -0.00148588,  0.00437956, -0.01021898, -0.00291971,  0.00442478,
        -0.02359882, -0.02359882,  0.01226994, -0.00153374,  0.00306748,
         0.01076923, -0.00615385,  0.00153846,  0.00310559, -0.01086957,
        -0.00465839,  0.02503912, -0.00312989,  0.02190923,  0.00311042,
        -0.00777605, -0.00311042,  0.00944882,  0.        ,  0.0015748 ,
         1.        , -0.02603369], dtype=float32))
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)





class ACSequentialExperienceReplayBuffer:
    def __init__(
        self,
        experience_queue:mp.Queue,
        del_critical_len:int,
        capacity:int,
        replay_initial_size: int,
    ):
        """
            This is split train model. so the data come from Queue.

        """
        self.experience_queue = experience_queue


        # 用來儲存所有商品的空間
        self.buffer = {}
        self.sample_count_buffer = {}
        self.capacity = capacity
        self.replay_initial_size = replay_initial_size
        assert (
            self.capacity > replay_initial_size
        ), "The capacity is smaller than init_size, please change the capacity"

        self._count = 0
        self._del_critical_len = del_critical_len

    def is_ready(self):
        if  [
                symbolName
                for symbolName, data in self.buffer.items()
                if len(data) >= self.replay_initial_size
            ]:
            return True
        else:
            return False

    def __len__(self):
        """
        Total number of experiences in the buffer.
        """
        return sum(len(deq) for deq in self.buffer.values())

    def get_random_symbol(self):
        return random.choice(
            [
                symbolName
                for symbolName, data in self.buffer.items()
                if len(data) >= self.replay_initial_size
            ]
        )

    def sample(self, batch_size):
        """
        Sample a batch of sequential experiences from a random symbol.

        """
        symbol = self.get_random_symbol()
        data = self.buffer[symbol]
        start = random.randint(0, len(data) - batch_size)
        batch = list(itertools.islice(data, start, start + batch_size))
        offset = batch[0].info["offset"]

        if all(exp.info["offset"] == offset + i for i, exp in enumerate(batch)):
            if symbol not in self.sample_count_buffer:
                self.sample_count_buffer[symbol] = 1
            else:
                self.sample_count_buffer[symbol] += 1
            return batch
        
        return self.sample(batch_size)

    def populate(self):
        """
            Populate buffer with all available experiences from the queue without blocking.
            The `num_actors` argument is kept for API consistency but not used in the new logic.
        """
        items_added = 0
        while not self.experience_queue.empty():
            try:
                entry = self.experience_queue.get_nowait()
                if entry.info["instrument"] not in self.buffer:
                    self.buffer[entry.info["instrument"]] = deque(maxlen=self.capacity)

                self.buffer[entry.info["instrument"]].append(entry)
                items_added += 1
            except Empty:
                break
        
        if items_added > 0:
            self._count += items_added

        if self.__len__() > self._del_critical_len:
            self.dropsymbol()

    def dropsymbol(self):
        best_name = max(self.sample_count_buffer, key=self.sample_count_buffer.get)
        self.buffer.pop(best_name)
        self.sample_count_buffer.pop(best_name)

    def get_state(self):
        """
        保存緩衝區的當前狀態。
        返回一個字典，包含緩衝區數據和其他屬性。
        """
        return {
            "buffer": self.buffer,  # 直接保存 dict of deque
            "capacity": self.capacity,
        }