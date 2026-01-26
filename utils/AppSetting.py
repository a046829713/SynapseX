import torch
from dataclasses import dataclass
from datetime import datetime
import os
from omegaconf import DictConfig, OmegaConf
import time

class AppSetting():
    def __init__(self) -> None:
        pass

    @staticmethod    
    def systeam_setting():
        permission_data ={
            'execute_orders': False,
            'emergency_times':50
        }   
        return permission_data
         
    @staticmethod
    def engine_setting():
        data = {
            'FREQ_TIME':30,
            'LEVERAGE':5
        }
        return data
    
           
    @staticmethod
    def Trading_setting():
        data = {
            
            "BACKTEST_DEFAULT_COMMISSION_PERC":0.0025,
            "DEFAULT_SLIPPAGE":0.0025
        }
        return data   

@dataclass
class RLConfig:
    KEYWORD: str = "Mamba"
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVES_TAG: str = "saves"
    BARS_COUNT: int = 300
    GAMMA: float = 0.99
    REWARD_STEPS: int = 2
    MODEL_DEFAULT_COMMISSION_PERC_TRAING: float = 0.0045

    MODEL_DEFAULT_COMMISSION_PERC_TEST: float = 0.0005
    DEFAULT_SLIPPAGE: float = 0.0025
    LEARNING_RATE: float = 0.000025
    LAMBDA_L2: float = 0.0
    BATCH_SIZE: int = 32
    REPLAY_SIZE: int = 800000
    EACH_REPLAY_SIZE: int = 50_000
    REPLAY_INITIAL: int = 1000
    EPSILON_START: float = 0.9
    EPSILON_STOP: float = 0.1
    EPSILON_STEPS_FACTOR: int = 1_000_000
    TARGET_NET_SYNC: int = 1000
    CHECKPOINT_EVERY_STEP: int = 20_000
    BETA_START: float = 0.4
    UNIQUE_SYMBOLS: list[str] = None
    CHECK_GRAD_STEP = 1000
    N_STEPS = 1000
    WIN_PAYOFF_WEIGHT = 1.0

    def update_steps_by_symbols(self, num_symbols: int):
        self.EPSILON_STEPS = (
            self.EPSILON_STEPS_FACTOR * 30
            if num_symbols > 30
            else self.EPSILON_STEPS_FACTOR * num_symbols
        )
        
    def create_saves_path(self):
        saves_path = os.path.join(
            self.SAVES_TAG,
            datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")
            + "-"
            + str(self.BARS_COUNT)
            + "k-",
        )
        os.makedirs(saves_path, exist_ok=True)
        self.SAVES_PATH = saves_path



class UpdateConfig():
    def __init__(self, cfg:DictConfig):
        """
            To Update Config from DictConfig.

            To Calculate all parameters.
        """

        self.cfg = cfg


        self.update_steps_by_symbols()
        self.create_saves_path()
        

    def GetInstance(self):
        return self.cfg
    
    def update_steps_by_symbols(self, num_symbols: int):
        self.EPSILON_STEPS = (
            self.cfg.train.epsilon_steps_factor * 30
            if num_symbols > 30
            else self.cfg.train.epsilon_steps_factor * num_symbols
        )
        
    def create_saves_path(self):
        saves_path = os.path.join(
            self.cfg.train.checkpoint.saves_tag,
            datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")
            + "-"
            + str(self.cfg.train.bars_count)
            + "k-",
        )
        os.makedirs(saves_path, exist_ok=True)
        self.cfg.train.saves_path = saves_path
    

    def GetSymbols(self):
        symbolNames = os.listdir(os.path.join(os.getcwd() , *self.cfg.train.train_data_file_path))
        symbolNames = [_fileName.split('.')[0] for _fileName in symbolNames]
        unique_symbols = list(set(symbolNames))
        return unique_symbols