class AppSetting():
    def __init__(self) -> None:
        pass

    @staticmethod
    def RL_test_setting():
        """
            當回測可以將參數擺放至此
        """
        data = {
            
            "BACKTEST_DEFAULT_COMMISSION_PERC":0.0005,
            "DEFAULT_SLIPPAGE":0.0025
        }
        return data        