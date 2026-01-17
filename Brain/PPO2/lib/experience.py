import numpy as np
from collections import deque, namedtuple



Transition = namedtuple('Transition',
                        ['state', 'action', 'logp', 'reward', 'next_state', 'done', 'value'])

class RolloutBuffer:
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        self.buffer = []

    def store(self, *args):
        self.buffer.append(Transition(*args))

    def compute_gae(self, last_value):
        rewards, values, dones = [], [], []
        for t in self.buffer:
            rewards.append(t.reward)
            values.append(t.value.item())
            dones.append(t.done)
        
        
        values = values + [last_value]
        gae, returns = 0, []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        
        # 把 advantages 與 returns 放回 transitions
        advantages = np.array(returns) - np.array(values[:-1])
        for idx, tr in enumerate(self.buffer):
            self.buffer[idx] = tr._replace(reward=returns[idx], value=advantages[idx])
        
        return self.buffer

    def clear(self):
        self.buffer = []