import pandas as pd
import numpy as np
import torch

from Brain.DQN.lib import environment
from Brain.DQN.lib import common
from Brain.DQN.lib.environment import State_time_step
from Brain.DQN.lib import environment, model
from Brain.Count import nb
import matplotlib.pyplot as plt
import quantstats as qs
from pathlib import Path
import time
from utils.AppSetting import RLConfig
import json
from Brain.DQN.lib.Strategy import Strategy


def load_from_json(filename="data.json"):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def save_to_json(data, filename="data.json"):
    """
    將資料保存為本地端的 JSON 檔案

    :param data: 要保存的資料（字典、列表等可 JSON 化的物件）
    :param filename: 保存的檔案名稱或路徑
    """
    with open(filename, "w", encoding="utf-8") as f:
        # indent=4 可以讓產生的 JSON 檔案自動排版，方便閱讀
        # ensure_ascii=False 確保中文字元不會被轉成 Unicode 碼
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"資料已成功保存至：{filename}")


class RL_evaluate:
    def __init__(self, strategy: Strategy, formal: bool) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyperparameters(strategy)

        self.config = RLConfig()

        if not formal:
            self.config.UNIQUE_SYMBOLS = [strategy.symbol_file_name.split(".")[0]]

        # 準備神經網絡的狀態
        state = State_time_step(
            bars_count=self.config.BARS_COUNT,
            commission_perc=self.config.MODEL_DEFAULT_COMMISSION_PERC_TEST,
            model_train=False,
            default_slippage=self.config.DEFAULT_SLIPPAGE,
            N_steps=self.config.N_STEPS,
        )

        # 製作環境
        self.evaluate_env = environment.ProductionEnv(
            prices_data=strategy.datafeature, state=state
        )

        self.agent = self.load_model(model_path=strategy.model_count_path)
        self.test()

    def load_model(self, model_path: str):
        engine_info = self.evaluate_env.engine_info()
        action_space_n = engine_info["action_space_n"]
        data_input_size = engine_info["data_input_size"]

        ssm_cfg = {"expand": 4}
        net = model.mambaDuelingModel(
            d_model=data_input_size,
            nlayers=4,
            num_actions=action_space_n,
            time_features_in=engine_info["time_input_size"],
            seq_dim=self.config.BARS_COUNT,
            dropout=0.3,
            ssm_cfg=ssm_cfg,
        ).to(self.config.DEVICE)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        net.load_state_dict(checkpoint["model_state_dict"])
        print("評估模式開始啟動")
        net.eval()  # 將模型設置為評估模式
        return net

    def test(self):
        """
            order  length : L - B -1
        """
        done = False
        rewards = []
        record_orders = []
        info = [{}]
        obs = self.evaluate_env.reset()
        state, time_state = obs

        state = torch.from_numpy(state).to(self.device)
        state = state.unsqueeze(0)

        time_state = torch.from_numpy(time_state).to(self.device)
        time_state = time_state.unsqueeze(0)

        info = common.turn_to_tensor(info, self.device)

        with torch.no_grad():
            while not done:
                action, _,_ = self.agent(state, time_state)
                action_idx = action.max(dim=1)[1].item()
                record_orders.append(self._parser_order(action_idx))
                _state, reward, done, info = self.evaluate_env.step(action_idx)
                # info = common.turn_to_tensor([info],self.device)
                state, time_state = _state

                state = torch.from_numpy(state).to(self.device)
                state = state.unsqueeze(0)

                time_state = torch.from_numpy(time_state).to(self.device)
                time_state = time_state.unsqueeze(0)
                rewards.append(reward)

        self.record_orders = record_orders
        

    def hyperparameters(self, strategy):
        self.BARS_COUNT = (
            strategy.model_feature_len
        )  # 用來準備要取樣的特徵長度,例如:開高低收成交量各取10根K棒
        self.MODEL_DEFAULT_COMMISSION_PERC = strategy.fee
        self.DEFAULT_SLIPPAGE = strategy.slippage

    def _parser_order(self, action_value: int):
        if action_value == 2:
            return -1
        return action_value


class Backtest(object):
    def __init__(
        self,
        re_evaluate: RL_evaluate,
        strategy: Strategy,
        model_name: str = None,
    ) -> None:
        """
        order (list):
            類神經網絡所產生的訂單
        """
        self.strategy = strategy
        self.bars_count = re_evaluate.BARS_COUNT
        self.order: list = re_evaluate.record_orders
        self.Symbol_data = self.strategy.df
        self.model_name = model_name or Path(strategy.model_count_path).stem

    def order_becktest(self, ifplot: bool):
        """
            透過order 來產生回測績效表
            datetime_list = datetime_list[:-1] 避免在時盤交易的時候 尚未確認收盤 價格變動導致重複交易
        """

        # 從類神經網絡拿order的一個狀態
        self.shiftorder = np.array(self.order)
        self.shiftorder = np.roll(self.shiftorder, 1)
        self.shiftorder[0] = 0  # 一率將其歸零即可
        datetime_list = self.Symbol_data.index.to_list()

        # # 前面10個當樣本
        datetime_list = datetime_list[self.bars_count :]

        # # 最後一個不計算
        datetime_list = datetime_list[:-1]

        # open 平倉版本
        self.Open = self.Symbol_data["Open"].to_numpy()

        # # 前面10個當樣本
        self.Open = self.Open[self.bars_count :]

        # # 最後一個不計算
        self.Open = self.Open[:-1]

        assert len(self.order) == len(
            self.Open
        ), "order not match the open data,please check."

        params = {
            "shiftorder": self.shiftorder,
            "open_array": self.Open,
            "Length": len(self.Open),
            "init_cash": self.strategy.init_cash,
            "slippage": self.strategy.slippage,
            "size": 1.0,
            "fee": self.strategy.fee,
        }

        (
            orders,
            marketpostion_array,
            entryprice_array,
            buy_Fees_array,
            sell_Fees_array,
            OpenPostionprofit_array,
            ClosedPostionprofit_array,
            profit_array,
            Gross_profit_array,
            Gross_loss_array,
            all_Fees_array,
            netprofit_array,
        ) = nb.logic_order(**params)

        final_equity = float(ClosedPostionprofit_array[-1])
        net_profit = float(netprofit_array[-1])
        return_pct = float(
            (final_equity - self.strategy.init_cash) / self.strategy.init_cash * 100
        )

        index = pd.to_datetime(self.Symbol_data.index[self.bars_count : -1])
        data_series = pd.Series(ClosedPostionprofit_array, index=index)


        try:
            max_dd = float(qs.stats.max_drawdown(data_series))
        except Exception:
            cummax = np.maximum.accumulate(ClosedPostionprofit_array)
            drawdown = (ClosedPostionprofit_array - cummax) / np.where(cummax > 0, cummax, 1)
            max_dd = float(np.min(drawdown))


        pos_diff = np.diff(marketpostion_array)
        entry_count = int(np.sum(pos_diff == 1))
        exit_count = int(np.sum(pos_diff == -1))
        total_trades = max(entry_count, exit_count)

        trade_profits = profit_array[profit_array != 0]
        if len(trade_profits) > 0:
            win_rate = float(np.sum(trade_profits > 0) / len(trade_profits) * 100)
        else:
            win_rate = 0.0

        if ifplot:
            self._cwd = Path("./")
            # results file isolated by model_name
            self._results_file = (
                self._cwd / "results" / self.model_name / f"{self.strategy.symbol_name}"
            )
            self._results_file.mkdir(parents=True, exist_ok=True)
            self.plot_max_drawdown(ClosedPostionprofit_array)
            self.detail_image(ClosedPostionprofit_array, orders)

        return {
            "marketpostion_array": marketpostion_array,
            "final_equity": final_equity,
            "net_profit": net_profit,
            "return_pct": return_pct,
            "max_drawdown": max_dd,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "gross_profit": float(Gross_profit_array[-1]),
            "gross_loss": float(Gross_loss_array[-1]),
            "all_fees": float(all_Fees_array[-1]),
        }

    def detail_image(self, ClosedPostionprofit_array, orders):
        self._plot_and_save(
            ClosedPostionprofit_array,
            save_path=self._results_file,
            ylabel="closed_position_profit",
            title="Closed Position Profit",
            file_name="closed_position_profit.png",
        )

        self._plot_and_save(
            orders,
            save_path=self._results_file,
            ylabel="orders",
            title="orders",
            file_name="orders.png",
        )

    def _plot_and_save(self, data, save_path, ylabel: str, title: str, file_name: str):
        index = pd.to_datetime(
            self.Symbol_data.index[self.bars_count : -1]
        )  # 转换为DatetimeIndex
        data_series = pd.Series(data, index=index)  # 将数据转换为Series，并设置索引
        # 这里添加绘图代码
        plt.plot(data_series)
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.title(title)
        # 保存圖片為PNG格式
        plt.savefig(Path(save_path) / file_name)
        plt.close()  # 關閉圖片，釋放資源

    def plot_max_drawdown(self, data):
        index = pd.to_datetime(
            self.Symbol_data.index[self.bars_count : -1]
        )  # 转换为DatetimeIndex
        data_series = pd.Series(data, index=index)  # 将数据转换为Series，并设置索引

        # 计算最大回撤
        max_dd = qs.stats.max_drawdown(data_series)
        print("Maximum DrawDown: {}".format(max_dd))
        # 绘制最大回撤图
        plt.figure(figsize=(10, 6))
        qs.plots.drawdown(
            data_series, show=False, savefig=self._results_file / "max_drawdown.png"
        )
        plt.close()
