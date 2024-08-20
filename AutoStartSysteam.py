import subprocess
import time
import os
from utils.Debug_tool import debug
import logging
import platform

def main(exe_file_name: str, python_env_path: str):
    """
    Args:
        exe_file_name (str): 執行檔案名稱
        python_env_path (str): python 虛擬環境放置位置
    """

    # 获取当前文件夹的绝对路径
    current_folder = os.getcwd()
    # 设置要运行的可执行文件路径
    exe_path = os.path.join(current_folder, exe_file_name)
    process = subprocess.Popen([python_env_path, exe_path])

    while True:
        if process.poll() is None:
            pass
        else:
            debug.record_msg("System crashed, restart the program.",
                             log_level=logging.error)            
            process = subprocess.Popen([python_env_path, exe_path])
        time.sleep(30)


if __name__ == "__main__":
    # 檢測操作系統
    if platform.system() == 'Windows':
        # 如果是Windows，使用Windows的路徑
        python_env_path = r'C:\Users\user\Desktop\program\Reinforcement_learninng\Scripts\python.exe'
    else:
        # 否則假設是Linux或其他類Unix系統，使用對應的路徑
        python_env_path = '/home/a046829713/RL_trading/bin/python'
    
    # 執行main函數，使用檢測到的python虛擬環境路徑
    main(exe_file_name='TradingSysteam.py', python_env_path=python_env_path)