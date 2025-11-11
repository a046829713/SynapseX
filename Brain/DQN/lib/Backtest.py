from ...Common.Error import InvalidModeError
import pandas as pd
import numpy as np
import torch
from Brain.Common.DataFeature import OriginalDataFrature
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


class Strategy(object):
    """
    神經網絡的模型基底
    """

    def __init__(
        self,
        strategytype: str,
        symbol_name: str,
        freq_time: int,
        model_feature_len: int,
        fee: float,
        slippage: float,
        model_count_path: str,
        init_cash: float = 10000.0,
        symobl_type: str = "Futures",
        lookback_date: str = None,
        symbol_first_trade_date=None,
        formal: bool = False,
    ) -> None:
        self.strategytype = strategytype
        self.symbol_name = symbol_name  # 商品名稱
        self.freq_time = freq_time  # 商品週期
        self.model_feature_len = model_feature_len  # 商品週期
        self.fee = fee  # 手續費
        self.slippage = slippage  # 滑價
        self.model_count_path = model_count_path  # 模型路徑
        self.init_cash = init_cash  # 起始資金
        self.symobl_type = symobl_type  # 每個策略會有一個商品別(期貨現貨別)
        self.lookback_date = lookback_date  # 策略回測日期
        self.symbol_first_trade_date = symbol_first_trade_date
        self.formal = formal  # 策略是否正式啟動

    def load_data(self, local_data_path: str):
        """
        如果非正式交易的的時候，可以啟用
        """
        if self.formal:
            raise InvalidModeError()

        self.df = pd.read_csv(local_data_path)
        self.df.set_index("Datetime", inplace=True)

    def load_Real_time_data(self, df: pd.DataFrame):
        self.df = df[
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_av",
                "trades",
                "tb_base_av",
                "tb_quote_av",
            ]
        ].copy()
        self.df.rename(
            columns={
                "date": "Datetime",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )

        self.df.set_index("Datetime", inplace=True)

    def _strategy_name(self):
        return f"{self.strategytype}-{self.symbol_name}-{self.freq_time}"


class RL_evaluate:
    def __init__(self, strategy: Strategy, formal: bool) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyperparameters(strategy)

        self.config = RLConfig()

        if not formal:
            self.config.UNIQUE_SYMBOLS = [strategy.symbol_file_name.split(".")[0]]

        data = OriginalDataFrature().get_train_net_work_data_by_pd(
            symbol=strategy.symbol_name,
            df=strategy.df,
            first_date=strategy.symbol_first_trade_date,
        )
        # 準備神經網絡的狀態
        state = State_time_step(
            bars_count=self.config.BARS_COUNT,
            commission_perc=self.config.MODEL_DEFAULT_COMMISSION_PERC_TEST,
            model_train=False,
            default_slippage=self.config.DEFAULT_SLIPPAGE,
            N_steps = self.config.N_STEPS,
            win_payoff_weight = self.config.WIN_PAYOFF_WEIGHT,
        )

        # 製作環境
        self.evaluate_env = environment.ProductionEnv(prices_data=data, state=state)
        self.agent = self.load_model(model_path=strategy.model_count_path)
        self.test()

    def load_model(self, model_path: str):
        engine_info = self.evaluate_env.engine_info()
        action_space_n = engine_info["action_space_n"]
        data_input_size = engine_info["data_input_size"]

        ssm_cfg = {"expand": 4}
        net = model.I2A_MambaDuelingModel(
            d_model=data_input_size,
            nlayers=6,
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
    def __init__(self, re_evaluate: RL_evaluate, strategy: Strategy) -> None:
        """

        order (list):
            類神經網絡所產生的訂單

        params = {'shiftorder': array([0, 0, 0, ..., 0, 0, 0], dtype=int64),
                'open_array': array([ 146.  ,  146.  ,  146.  , ..., 1631.48, 1627.78, 1628.54]),
                'Length': 135450,
                'init_cash': 10000.0,
                'slippage': 0.0025,
                'size': 1.0,
                'fee': 0.002}
        """
        self.strategy = strategy
        self.bars_count = re_evaluate.BARS_COUNT
        self.order: list = re_evaluate.record_orders
        self.Symbol_data = self.strategy.df

    def order_becktest(self, ifplot: bool):
        """
        透過order 來產生回測績效表
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

        if ifplot:
            self._cwd = Path("./")
            # results file
            self._results_file = self._cwd / "results" / f"{self.strategy.symbol_name}"
            self._results_file.mkdir(parents=True, exist_ok=True)
            self.plot_max_drawdown(ClosedPostionprofit_array)
            self.detail_image(ClosedPostionprofit_array, orders)

        return {"marketpostion_array": marketpostion_array}

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
