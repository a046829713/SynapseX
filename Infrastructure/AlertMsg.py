
"""
    用來產生監控的exe
    用來單獨監控系統是否有正常運作

"""
import time
from datetime import datetime
from Database.SQL_operate import DB_operate
import requests
from typing import Any
from Major.UserManager import UserManager

class LINE_Alert():
    def req_line_alert(self, each_token: str, str_msg: Any):
        # 記錄Line Notify服務資訊
        Line_Notify_Account = {'Client ID': 'xxxxxxxxxxxxxxxxx',
                               'Client Secret': 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
                               'token': f'{each_token}'}

        # 將token放進headers裡面
        headers = {'Authorization': 'Bearer ' + Line_Notify_Account['token'],
                   "Content-Type": "application/x-www-form-urlencoded"}

        # 回傳測試文字
        params = {"message": f"\n{str(str_msg)}"}

        # 執行傳送測試文字
        # 使用post方法
        try:
            r = requests.post("https://notify-api.line.me/api/notify",
                              headers=headers, params=params)

        except Exception as e:
            return "訊息無法傳送"

    def send_author(self, str_msg: Any):
        """只發送給作者

        Args:
            Self (_type_): _description_
        """
        self.req_line_alert(each_token=UserManager.get_author_line_token(), str_msg=str_msg)


def check_sysLive():
    """
        檢察系統是否有正常運作
    """
    while True:
        data = DB_operate().get_db_data('select *  from `sysstatus`;')
        print("系統最後紀錄時間:", data)
        difftime = datetime.now() - \
            datetime.strptime(data[0][1], "%Y-%m-%d %H:%M:%S.%f")

        print("系統差異秒數:", difftime.seconds)
        if difftime.seconds > 600:
            LINE_Alert().send_author("緊急通知>>程式已經停止運作!!!!!")

        time.sleep(60)
