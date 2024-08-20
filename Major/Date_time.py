from datetime import datetime
from pytz import timezone
import pytz
from typing import Union

T = Union[str, datetime]

class parser_time:
    @staticmethod
    def change_ts_to_str(time_stamp: int):
        """
            將時間戳 轉換成字串
        """
        return str(datetime.utcfromtimestamp(time_stamp))

    @staticmethod
    def changetime(input_time: T):
        """
            將輸入的時間(UTC時區) 改成台灣時間

        Args:
            input_time (str): '2019/9/25 08:01:00'
        """
        utc = pytz.utc
        tw = timezone('Asia/Taipei')

        if isinstance(input_time, datetime):
            utctime = utc.localize(input_time)
            newtime = utctime.astimezone(tw).replace(tzinfo=None)

        elif isinstance(input_time, str):
            utctime = utc.localize(datetime.strptime(
                input_time, "%Y/%m/%d %H:%M:%S"))
            newtime = utctime.astimezone(tw).replace(tzinfo=None)

        else:
            raise ValueError("input type is datetime or str")

        return newtime
