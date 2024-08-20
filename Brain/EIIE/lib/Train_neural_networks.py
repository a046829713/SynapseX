import torch
from EIIE.lib.environment import PortfolioOptimizationEnv
import pandas as pd
from EIIE.lib.common import PG

import logging
logging.getLogger('matplotlib.font_manager').disabled = True


"""
@inproceedings{
    bwaif,
    author = {Caio Costa and Anna Costa},
    title = {POE: A General Portfolio Optimization Environment for FinRL},
    booktitle = {Anais do II Brazilian Workshop on Artificial Intelligence in Finance},
    location = {João Pessoa/PB},
    year = {2023},
    keywords = {},
    issn = {0000-0000},
    pages = {132--143},
    publisher = {SBC},
    address = {Porto Alegre, RS, Brasil},
    doi = {10.5753/bwaif.2023.231144},
    url = {https://sol.sbc.org.br/index.php/bwaif/article/view/24959}
}
"""


def train(Train_data_path: str, Meta_path: str, Train_path: str, episodes: int, save: bool, pre_train: bool):
    portfolio_raw_df = pd.read_csv(Train_data_path)
    portfolio_raw_df.drop(columns=['Unnamed: 0'], inplace=True)
    df_portfolio = portfolio_raw_df[["date", "tic", "close", "high", "low"]]
    environment = PortfolioOptimizationEnv(
        df_portfolio,
        initial_amount=100000,
        comission_fee_pct=0.0025,
        time_window=50,
        features=["close", "high", "low"]
    )
    algo = PG(environment, lr=0.00003)
    # 載入預訓練得模型
    if pre_train:
        algo.policy.load_state_dict(torch.load(Meta_path))
    
    algo.train(episodes=episodes)
    
    if save:
        torch.save(algo.target_policy.state_dict(), Train_path)
