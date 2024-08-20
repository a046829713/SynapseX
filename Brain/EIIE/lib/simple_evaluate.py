import pandas as pd
import time
import matplotlib.pyplot as plt
from EIIE.lib.environment import PortfolioOptimizationEnv
from EIIE.lib.common import PVM
import numpy as np
from EIIE.lib.model import GradientPolicy
import torch
import matplotlib
# matplotlib.use('Agg')
matplotlib.use('tkagg')


def update_test_model_performance(environment: PortfolioOptimizationEnv, Tag, results: dict, Meta_path:str):
    """_summary_

    Args:
        environment (PortfolioOptimizationEnv): _description_        
        Tag (_type_): _description_
        results (dict): 
            EIIE_results = {
                "train": {},
                "test": {},
            }
    """
    policy = GradientPolicy()
    policy.load_state_dict(torch.load(Meta_path))
    policy = policy.to(policy.device)
    done = False
    obs = environment.reset()
    pvm = PVM(environment.episode_length, environment._stock_dim)
    while not done:
        last_action = pvm.retrieve()
        obs_batch = np.expand_dims(obs, axis=0)
        last_action_batch = np.expand_dims(last_action, axis=0)
        # return numpy.ndarray
        action = policy(obs_batch, last_action_batch)
        pvm.add(action)
        obs, _, done, _ = environment.step(action)
    results[Tag]["value"] = environment._asset_memory["final"]


def BuyAndHold(environment: PortfolioOptimizationEnv, Tag, results: dict):
    """_summary_

    Args:
        environment (_type_): _description_
        Tag (_type_): _description_
        results (dict): 
            UBAH_results = {
                "train": {},
                "2020": {},
                "2021": {},
                "2022": {}
            }

    """
    # train period
    terminated = False
    environment.reset()
    while not terminated:
        action = [0] + [1/environment._stock_dim] * environment._stock_dim
        _, _, terminated, _ = environment.step(action)
    results[Tag]["value"] = environment._asset_memory["final"]


def plot_performance(UBAH_results, EIIE_results, period, title):
    """
        绘制指定时间段的投资组合价值表现。
        :param UBAH_results: 买入并持有策略的结果字典。
        :param EIIE_results: EIIE 策略的结果字典。
        :param period: 要绘制的时间段（例如 "train", "test"）。
        :param title: 图表的标题。
    """
    print(UBAH_results[period]["value"])
    print('*'*120)
    print(EIIE_results[period]["value"])
    plt.plot(UBAH_results[period]["value"], label="Buy and Hold")
    plt.plot(EIIE_results[period]["value"], label="EIIE")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.title(title)
    plt.legend()
    plt.show()


def evaluate_train_test_performance(Train_data_path: str, Test_data_path: str, Meta_path: str):
    portfolio_raw_df = pd.read_csv(Train_data_path)
    portfolio_raw_df.drop(columns=['Unnamed: 0'], inplace=True)

    portfolio_test_raw_df = pd.read_csv(Test_data_path)
    portfolio_test_raw_df.drop(columns=['Unnamed: 0'], inplace=True)

    df_portfolio = portfolio_raw_df[["date", "tic", "close", "high", "low"]]
    df_portfolio_test = portfolio_test_raw_df[["date", "tic", "close", "high", "low"]]

    environment = PortfolioOptimizationEnv(
        df_portfolio,
        initial_amount=100000,
        comission_fee_pct=0.0025,
        time_window=50,
        features=["close", "high", "low"]
    )

    environment_test = PortfolioOptimizationEnv(
        df_portfolio_test,
        initial_amount=100000,
        comission_fee_pct=0.0025,
        time_window=50,
        features=["close", "high", "low"]
    )

    EIIE_results = {
        "train": {},
        "test": {},
    }

    UBAH_results = {
        "train": {},
        "test": {},
    }

    merge_list = [
        ('train', environment),
        ('test', environment_test),
    ]

    for each_value in merge_list:
        update_test_model_performance(
            each_value[1], each_value[0], EIIE_results, Meta_path)

    for each_value in merge_list:
        BuyAndHold(each_value[1], each_value[0], UBAH_results)

    # 使用示例
    plot_performance(UBAH_results, EIIE_results, "train",
                     "Performance in train period")
    
    plot_performance(UBAH_results, EIIE_results, "test",
                     "Performance in test period")
