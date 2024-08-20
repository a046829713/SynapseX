import traceback
import logging
import datetime
import os
from time import time

LOG_DIR = "LogRecord"
    

today = datetime.date.today().strftime('%Y%m%d')
log_file = os.path.join(LOG_DIR, f"{today}.log")
handler = logging.FileHandler(log_file,encoding='utf-8')

logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M', handlers=[handler, ])

class debug():
    _count_map = {}
    _counts = 0

    @staticmethod
    def print_info(error_msg: str = None):
        
        traceback.print_exc()
        logging.error(f'{traceback.format_exc()}')
        if error_msg:
            logging.error(f'{error_msg}')

    @staticmethod
    def record_msg(error_msg: str, log_level=logging.debug):
        print(error_msg)
        log_level(f'{error_msg}')

    @staticmethod
    def record_args_return_type(func):
        def wrapper(*args, **kwargs):
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            for i, arg in enumerate(args):
                print(f"{arg_names[i]}: {arg}")
            for k, v in kwargs.items():
                print(f"{k}: {v}")
            result = func(*args, **kwargs)
            print(f"Result Type: {type(result)}")
            return result
        return wrapper

    @staticmethod
    def record_timemsg(func):
        def wrapper(*args, **kwargs):
            begin_time = time()
            result = func(*args, **kwargs)
            end_time = time()
            print("函數名稱:", func)
            print("使用時間:", end_time-begin_time)
            return result

        return wrapper

    @staticmethod
    def record_time_add(func):
        """
            用來計算函數所累積的時間判斷誰的影響最大

        Args:
            func (_type_): callable
        """
        def wrapper(*args, **kwargs):
            debug._counts += 1
            begin_time = time()
            result = func(*args, **kwargs)  
            end_time = time()            
            elapsed_time = end_time - begin_time
            countMap = debug._count_map
            if func.__name__ in countMap:
                countMap[func.__name__]['time'] += elapsed_time
                countMap[func.__name__]['use_count'] +=1
            else:
                countMap[func.__name__]= {}
                countMap[func.__name__].update({'time':elapsed_time})
                countMap[func.__name__].update({'use_count':1})
            
            if debug._counts % 1000 == 0:
                print(countMap) 
                print('*'*120)
            return result
        return wrapper


def print_syslog(title:str ,msg:str):    
    print(f"--{title}--:{msg}")
    
class ExcessiveTradeException(Exception):
    """異常類：交易次數過多"""
    def __init__(self, message="交易次數過多，已觸發緊急狀況處理"):
        self.message = message
        super().__init__(self.message)