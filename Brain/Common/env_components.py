import numpy as np

class BasicState():
    def __init__(
        self,
        bars_count,
        commission_perc,
        model_train,
        default_slippage,
        N_steps,
    ):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0

        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.N_steps = N_steps
        self.model_train = model_train
        self.default_slippage = default_slippage


        self.info_list = [
            "log_open",
            "log_high",
            "log_low",
            "log_close",
            "log_volume",
            "log_quote_av",
            "log_trades",
            "log_tb_base_av",
            "log_tb_quote_av",
            "log_ma_30",
            "log_ma_60",
            "log_ma_120",
            "log_ma_240",
            "log_ma_360",
        ]

        self.timelist = [
            "age_log_minutes",
            "age_years",
            "month_sin",
            "month_cos",
            "day_sin",
            "day_cos",
            "hour_sin",
            "hour_cos",
            "minute_sin",
            "minute_cos",
            "dayofweek_sin",
            "dayofweek_cos",
            "week_sin",
            "week_cos"
        ]

    def getStateShape(self):
        return self.bars_count, len(self.info_list) + 3

    def getTimeShape(self):
        return self.bars_count, len(self.timelist)
    
class State_time_step_template(BasicState):
    """
    專門用於transformer的時序函數。
    """

    def encode(self):
        data_res = np.zeros(shape=self.getStateShape(), dtype=np.float32)
        time_res = np.zeros(shape=self.getTimeShape(), dtype=np.float32)

        ofs = self.bars_count

        for bar_idx in range(self.bars_count):
            for idx, field in enumerate(self.info_list):  # 編碼所有字段
                data_res[bar_idx][idx] = getattr(self._prices, field)[
                    self._offset - ofs + bar_idx
                ]



        if self.have_position:
            data_res[:, len(self.info_list)] = 1.0
            data_res[:, len(self.info_list) + 1] = (
                self._prices.close[self._offset] - self.open_price
            ) / self.open_price
            data_res[:, len(self.info_list) + 2] = self.trade_bar

        for bar_idx in range(self.bars_count):
            for idx, field in enumerate(self.timelist):  # 編碼所有字段
                time_res[bar_idx][idx] = getattr(self._prices, field)[
                    self._offset - ofs + bar_idx
                ]

        return data_res, time_res