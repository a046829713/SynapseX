import random
import numpy as np
import itertools
from collections import deque
import time
from Brain.DQN.ptan.experience import ExperienceSource
import torch
import random
from collections import namedtuple, deque
import numpy as np



class PrioritizedStratifiedReplayBuffer:
    def __init__(self, experience_source, batch_size:int, each_symbol_size:int, capacity:int,each_capacity:int):
        """
            batch_size (int): batch_size input model.
        """
        assert isinstance(experience_source, (ExperienceSource, type(None)))
        assert batch_size % each_symbol_size == 0
        assert isinstance(batch_size, int)
        self.experience_source_iter = None if experience_source is None else iter(
            experience_source)
        
        self.batch_size = batch_size
        self.symbol_len = self.batch_size / each_symbol_size

        self.symbols = []
        self.buffer ={}
        self.capacity = capacity
        self.each_capacity = each_capacity


    def add(self, samples, td_error):
        """
            將樣本填入緩衝區中
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            print(entry)
            time.sleep(100)
            if entry.info['instrument'] not in self.buffer:
                self.buffer[entry.info['instrument']] = deque(maxlen=self.each_capacity)
            
            self.buffer[entry.info['instrument']].append(entry)

    
    def choose_symbols(self)->list:
        pass


    def sample(self,batch_size:int):
        pass

    
    def dropsymbol(self):
        """
            remove the minum prioritized 
        """
        pass










class SequentialExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size,replay_initial_size:int):
        """
            Initialize the buffer with a source, capacity, and symbol count.

            _count (int) : count the times.
        """
        assert isinstance(experience_source, (ExperienceSource, type(None)))
        assert isinstance(buffer_size, int)
        

        self.experience_source_iter = None if experience_source is None else iter(
            experience_source)
        
        # 用來儲存所有商品的空間
        self.buffer = {}
        self.sample_count_buffer = {}            
        self.capacity = buffer_size
        self.replay_initial_size = replay_initial_size
        assert self.capacity > replay_initial_size ,"The capacity is smaller than init_size, please change the capacity"
        
        self._count = 0
        self._del_critical_len = 1000000


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

    def each_num_len_enough(self): 
        if self._count > self.replay_initial_size:
            return True
        else:
            return False
    
    def get_random_symbol(self):
        return random.choice([ symbolName for symbolName,data in self.buffer.items() if len(data) > self.replay_initial_size])


    def sample(self, batch_size):
        """
            Sample a batch of sequential experiences from a random symbol.
        
        """
        symbol  =self.get_random_symbol()
        data = self.buffer[symbol]
        start = random.randint(0, len(data) - batch_size)
        batch = list(itertools.islice(data, start, start + batch_size))
        offset = batch[0].info['offset']
        if all(exp.info['offset'] == offset + i for i, exp in enumerate(batch)):
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

            if entry.info['instrument'] not in self.buffer:
                self.buffer[entry.info['instrument']] = deque(maxlen=self.capacity)
            
            self.buffer[entry.info['instrument']].append(entry)
            

        self._count +=1
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
            'buffer': self.buffer,  # 直接保存 dict of deque
            'capacity': self.capacity,
        }

    def load_state(self, state):
        """
        從保存的狀態加載緩衝區。
        """
        self.buffer = state['buffer']  # 直接還原 dict of deque
        self.capacity = state['capacity']

        


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