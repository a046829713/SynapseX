import pandas as pd
import numpy as np
import gym
from gym import spaces
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 一、特征清洗
data = pd.read_csv('stock_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values(['Stock_ID', 'Date'], inplace=True)
data['Price_Change'] = data.groupby('Stock_ID')['Close'].diff()
data.dropna(inplace=True)
scaler = StandardScaler()
data['Close_Scaled'] = scaler.fit_transform(data[['Close']])

# 二、环境定义
class StockSelectionEnv(gym.Env):
    def __init__(self, data):
        super(StockSelectionEnv, self).__init__()

        self.data = data
        self.stock_ids = data['Stock_ID'].unique()
        self.n_stocks = len(self.stock_ids)
        self.dates = data['Date'].unique()
        self.n_dates = len(self.dates)
        self.current_date_index = 0

        self.action_space = spaces.MultiBinary(self.n_stocks)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_stocks, 1), dtype=np.float32)
        self.previous_selection = set()

    def reset(self):
        self.current_date_index = 0
        self.previous_selection = set()
        return self._next_observation()

    def _next_observation(self):
        obs = []
        current_date = self.dates[self.current_date_index]
        for stock_id in self.stock_ids:
            stock_data = self.data[(self.data['Stock_ID'] == stock_id) & (self.data['Date'] == current_date)]
            if not stock_data.empty:
                obs.append([stock_data['Close_Scaled'].values[0]])
            else:
                obs.append([0])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        current_date = self.dates[self.current_date_index]
        rewards = []
        selected_stocks = set()
        done = False

        for idx, act in enumerate(action[0]):  # 注意action是二维数组
            stock_id = self.stock_ids[idx]
            if act == 1:
                selected_stocks.add(stock_id)
                stock_data = self.data[(self.data['Stock_ID'] == stock_id) & (self.data['Date'] == current_date)]
                if not stock_data.empty:
                    price_change = stock_data['Price_Change'].values[0]
                    if stock_id in self.previous_selection:
                        rewards.append(price_change)
                    else:
                        rewards.append(0)
                else:
                    rewards.append(0)
            else:
                rewards.append(0)

        reward = np.sum(rewards)
        self.previous_selection = selected_stocks
        self.current_date_index += 1
        if self.current_date_index >= self.n_dates:
            done = True

        obs = self._next_observation() if not done else np.zeros((self.n_stocks, 1), dtype=np.float32)
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

# 三、模型训练
env = DummyVecEnv([lambda: StockSelectionEnv(data)])
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_stock_selection")

# 四、模型测试
obs = env.reset()
for i in range(len(data['Date'].unique())):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(f"Step {i}, Reward: {rewards}")
    if dones:
        break
