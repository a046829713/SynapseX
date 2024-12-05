import random
import numpy as np
import itertools
from collections import deque
import time
from Brain.DQN.ptan.experience import ExperienceSource

class SequentialExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size, symbol_size:int):
        """
            Initialize the buffer with a source, capacity, and symbol count.
        """
        assert isinstance(experience_source, (ExperienceSource, type(None)))
        assert isinstance(buffer_size, int)
        
        self.experience_source_iter = None if experience_source is None else iter(
            experience_source)
        
        # 用來儲存所有商品的空間
        self.buffer = {}            
        self.capacity = buffer_size
        self.symbol_size = symbol_size
        self.enough_sign = False

    def __len__(self):
        """
        Total number of experiences in the buffer.
        """
        return sum(len(deq) for deq in self.buffer.values())
    
    def each_symbol_the_drill(self):
        """
        Count experiences per symbol.
        """
        return { key:len(self.buffer[key]) for key in self.buffer}

    def each_num_len_enough(self, init_size:int):        
        assert self.capacity > init_size ,"The capacity is smaller than init_size, please change the capacity"        
        # 用來判斷神經網絡是否需要跳過，希望各商品都有充足的樣本可以使用
        # 需要多個樣本都有了再開始判斷數量
        if self.enough_sign:return self.enough_sign

        if len(self.buffer) != self.symbol_size:
            return False
        
        self.enough_sign = all(len(deq) >= init_size for deq in self.buffer.values())
        return self.enough_sign
    
    def sample(self, batch_size):
        """
        Sample a batch of sequential experiences from a random symbol.
        
        """
        symbol = random.choice(list(self.buffer.keys()))
        data = self.buffer[symbol]
        start = random.randint(0, len(data) - batch_size)
        batch = list(itertools.islice(data, start, start + batch_size))

        offset = batch[0].info['offset']
        if all(exp.info['offset'] == offset + i for i, exp in enumerate(batch)):
            return batch
        
        return self.sample(batch_size)
    
    def populate(self, samples):
        """
        將樣本填入緩衝區中
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)       
            if entry.info['instrument'] not in self.buffer:
                self.buffer[entry.info['instrument']] = deque(maxlen=self.capacity)
            else:
                self.buffer[entry.info['instrument']].append(entry)


class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size):
        assert isinstance(experience_source, (ExperienceSource, type(None)))
        assert isinstance(buffer_size, int)
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
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