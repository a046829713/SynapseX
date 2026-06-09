import random
import itertools
from collections import deque


class SequentialReplayBuffer:
    def __init__(
        self,
        buffer_size:int,
        each_buffer_size:int,
        stanstandard_size: int,
    ):
        """
            Initialize the buffer with capacity, and symbol count.
            :param (batch_size): 總的批次大小。

        """
        assert isinstance(each_buffer_size, int)



        # 用來儲存所有商品的空間
        self.buffer = {}
        self.sample_count_buffer = {}
        self.each_buffer_size = each_buffer_size
        self.stanstandard_size = stanstandard_size
        self._count = 0
        self.buffer_size = buffer_size
        
        assert (
            self.each_buffer_size > self.stanstandard_size
        ), "The capacity is smaller than init_size, please change the capacity"



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

    def push(self, samples):
        """
        將樣本填入緩衝區中
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)

            if entry.info["instrument"] not in self.buffer:
                self.buffer[entry.info["instrument"]] = deque(maxlen=self.capacity)

            self.buffer[entry.info["instrument"]].append(entry)

        self._count += 1
        if sum(len(deq) for deq in self.buffer.values()) > self.self.buffer_size:
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