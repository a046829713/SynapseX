import numpy as np

class BasicState():
    STATIONARY_TIME_FEATURES = (
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
        "dayofweek_sin",
        "dayofweek_cos",
    )

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

        self.timelist = list(self.STATIONARY_TIME_FEATURES)

    def getStateShape(self):
        return self.bars_count, len(self.info_list) + 3

    def getTimeShape(self):
        return self.bars_count, len(self.timelist)