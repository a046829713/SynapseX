import numpy as np

class BasicState():
    MARKET_FEATURES = (
        # K-bar orthogonal 4-DoF geometry (shared causal sigma_{t-1}, uncentered)
        "ret_gap",
        "candle_body",
        "upper_wick",
        "lower_wick",
        # Multi-horizon SMA trend deviation (horizon-scaled by sigma_{t-1} * sqrt(p / 3))
        "ma_dev_30",
        "ma_dev_60",
        "ma_dev_120",
        "ma_dev_240",
        "ma_dev_360",
        # Explicit volatility regime (preserves absolute scale vs friction cost)
        "vol_regime_abs",
        "vol_expansion",
        # Scale-invariant volume & order flow microstructure
        "rel_quote_vol",
        "rel_trades",
        "taker_imbalance",
    )

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

        self.info_list = list(self.MARKET_FEATURES)
        self.timelist = list(self.STATIONARY_TIME_FEATURES)

    def getStateShape(self):
        return self.bars_count, len(self.info_list) + 3

    def getTimeShape(self):
        return self.bars_count, len(self.timelist)