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
            'LEVERAGE':4
        }
        return data
    
           
    @staticmethod
    def Trading_setting():
        data = {
            
            "BACKTEST_DEFAULT_COMMISSION_PERC":0.0005,
            "DEFAULT_SLIPPAGE":0.0025
        }
        return data
    
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