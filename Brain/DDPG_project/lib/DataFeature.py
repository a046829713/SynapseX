import pandas as pd
import collections
import torch

class DataFeature():
    """
        I specialize in using tensors to accelerate computations
        because I believe that converting between GPU and CPU consumes a significant amount of computational resources.
    """

    def __init__(self, device: torch.device, formal: bool = False) -> None:
        self.formal = formal
        self.device = device
        self.PricesObject = collections.namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])
        
    def load_relative(self):
        tensor_data = torch.tensor(self.df.values, dtype=torch.float32, device=self.device)
        volume_change = self.calculate_volume_change(tensor_data[:, 4])
        
        open=tensor_data[:, 0]
        high=tensor_data[:, 1]
        low=tensor_data[:, 2]
        close=tensor_data[:, 3]
        volume=volume_change
        
        rh = (high - open) / open
        rl = (low - open) / open
        rc = (close - open) / open
        
        return self.PricesObject(open=open, high=rh, low=rl, close=rc, volume=volume)
        
    def calculate_volume_change(self, volumes):
        """
            Calculate relative volume change
        """
        shift_data = torch.roll(volumes, shifts=1, dims=0)
        shift_data[0] = 0
        diff_data = volumes - shift_data
        volume_change = torch.where(
            volumes != 0, diff_data / volumes, torch.zeros_like(diff_data))
        return volume_change


    def get_train_net_work_data_by_path(self, symbols: list) -> dict:
        """
            用來取得類神經網絡所需要的資料
        """
        out_dict = {}
        for symbol in symbols:
            df = pd.read_csv(f'DDPG/simulation/data/{symbol}-F-30-Min.csv')
            df.set_index('Datetime', inplace=True)
            self.df = df
            # 使用 PyTorch Tensor 的方法
            out_dict.update({symbol: self.load_relative()})
        return out_dict

    def get_train_net_work_data_by_pd(self, symbol: str, df: pd.DataFrame) -> dict:
        """
            用來取得類神經網絡所需要的資料,正式交易的時候
        """
        out_dict = {}
        self.df = df
        # 使用 PyTorch Tensor 的方法
        out_dict.update({symbol: self.load_relative()})
        return out_dict
