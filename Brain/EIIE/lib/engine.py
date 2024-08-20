import pandas as pd
import time
from EIIE.lib.environment import PortfolioOptimizationEnv
from EIIE.lib.common import PVM
import numpy as np
from EIIE.lib.model import GradientPolicy
import torch


class EngineBase():
    def __init__(self, Meta_path: str) -> None:
        self.policy = GradientPolicy()
        checkpoint = torch.load(
            Meta_path, map_location=self.policy.device, weights_only=True)

        self.policy.load_state_dict(checkpoint)
        self.policy = self.policy.to(self.policy.device)

    def work(self, df: pd.DataFrame):
        # environment = PortfolioOptimizationEnv(
        #     df,
        #     initial_amount=50000,
        #     comission_fee_pct=0.0025,
        #     time_window=50,
        #     features=["close", "high", "low"]
        # )
        # self.last_order_info = self._performance(environment=environment)
        # this is to stop EIIE,because i find it costs too much taxs and fees
        # [('Cash_asset', 0.10379037), ('ARUSDT', 0.3012763), ('BNBUSDT', 0.2963091), ('BTCDOMUSDT', 0.2986242)]

        self.last_order_info = [('Cash_asset', 0)] + [(each_symbol, 0)
                                                      for each_symbol in list(set(df['tic'].to_list()))]

        new_out_put = []
        for _each_key_value in self.last_order_info:
            _each_key_value = list(_each_key_value)
            if _each_key_value[0] == 'Cash_asset':
                _each_key_value[1] = 0
            else:
                this_percent = 1 / (len(self.last_order_info) - 1)
                _each_key_value[1] = 0.25 if this_percent > 0.25 else this_percent
            new_out_put.append(tuple(_each_key_value))

        self.last_order_info = new_out_put

    def _performance(self, environment: PortfolioOptimizationEnv):
        """
            用來取得模型的最後權重
        Args:
            environment (PortfolioOptimizationEnv): _description_        
        """

        done = False
        obs = environment.reset()
        pvm = PVM(environment.episode_length, environment._stock_dim)
        while not done:
            last_action = pvm.retrieve()
            obs_batch = np.expand_dims(obs, axis=0)
            last_action_batch = np.expand_dims(last_action, axis=0)
            # return numpy.ndarray
            action = self.policy(obs_batch, last_action_batch)
            pvm.add(action)
            obs, _, done, _ = environment.step(action)

        allsymbols = ['Cash_asset']
        allsymbols.extend(environment._tic_list)

        return list(zip(allsymbols, list(environment._actions_memory[-1])))

    def get_order(self, finally_df: pd.DataFrame, balance_balance_map: dict, leverage: float):
        """
            last_order_info:[('Cash_asset', 7.812179e-05), ('BTCUSDT', 0.5025471), ('ETHUSDT', 0.4973748)]
        Args:
            balance_balance_map (dict): 
                {'09XXXXXXXX': 0.0,'09XXXXXXXX': 0.0, '09XXXXXXXX': 7916.93242276}

        Return:
            {'09XXXXXXXX': {'BTCUSDT-15K-OB-DQN': [1, 0.6278047845752174],
              'ETHUSDT-15K-OB-DQN': [1, 20.967342756868344]}}
        """
        last_df = finally_df.groupby('tic').last()
        out_map = {}
        for key, value in balance_balance_map.items():
            for each_data in self.last_order_info[1:]:
                symbol = each_data[0]
                weight = each_data[1]
                shares = value * weight * leverage / \
                    last_df[last_df.index == symbol]['close'].iloc[0]
                if key in out_map:
                    out_map[key].update({symbol: [1, shares]})
                else:
                    out_map[key] = {symbol: [1, shares]}

        return out_map
