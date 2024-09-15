import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3)  # 0: 賣出, 1: 持有, 2: 買入
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0  # 0: 空倉, 1: 多頭
        self.current_step = 0
        self.total_asset = self.balance
        return self._next_observation()

    def _next_observation(self):
        return self.data[self.current_step]

    def step(self, action):
        done = False
        reward = 0
        price = self.data[self.current_step][0]  # 假設價格在第一個特徵位置

        # 執行動作
        if action == 0:  # 賣出
            if self.position == 1:
                self.balance += price
                self.position = 0
        elif action == 2:  # 買入
            if self.position == 0 and self.balance >= price:
                self.balance -= price
                self.position = 1

        # 更新資產
        self.total_asset = self.balance + (self.position * price)

        # 獎勵為資產增減
        if self.current_step > 0:
            reward = self.total_asset - self.prev_total_asset

        self.prev_total_asset = self.total_asset

        # 前進一步
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Total Asset: {self.total_asset}')
