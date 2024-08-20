from Database.BackUp import BasePreparator, DatabaseBackupRestore
from Major.DataProvider import DataProvider, AsyncDataProvider
from utils.AppSetting import AppSetting
from datetime import timedelta
import datetime
import time
import threading
from Infrastructure.AlertMsg import LINE_Alert
import pandas as pd
import copy
from utils import Debug_tool, Data_parser
from utils.Debug_tool import ExcessiveTradeException, print_syslog
from binance.exceptions import BinanceAPIException
from EIIE.lib.engine import EngineBase
from DQN.lib.engine import EngineBase as DQN_EngineBase
import asyncio
import os


class Trading_system():
    def __init__(self) -> None:
        BasePreparator()  # 系統初始化等檢查

        self.dataprovider = DataProvider()  # 創建資料庫的連線
        self.strategy_keyword = 'ONE_TO_MANY'

        self.buildEngine()  # 建立引擎
        self.systeam_setting = AppSetting.systeam_setting()

        self.GuiStartDay = str(datetime.date.today())
        self.datatransformer = Data_parser.Datatransformer()

        self.symbol_map = {}

        # 這次新產生的資料
        self.new_symbol_map = {}
        self.engine_setting = AppSetting.engine_setting()

    def buildEngine(self) -> None:
        """ 
        用來創建回測系統，並且將DQN判斷是否送出訂單

        """
        meta_path = os.path.join('EIIE', 'Meta', 'policy_EIIE.pt')
        self.engine = EngineBase(Meta_path=meta_path)
        self.DQN_engin = DQN_EngineBase(self.strategy_keyword)

    def DailyChange(self):
        """
            每天都要重新關閉,怕資料量過大,並且會重新讀取每天的強勢標的
        """
        while not exit_event.is_set():
            if str(datetime.date.today()) != self.GuiStartDay:
                self.click_save_data()
            # 每5分鐘判斷一次就好
            time.sleep(300)

    def click_save_data(self):
        """ 
        保存資料並關閉程序 注意不能使用replace 資料長短問題

        """
        for name, each_df in self.new_symbol_map.items():
            # 不保存頭尾 # 異步模式
            each_df.drop(
                [each_df.index[0], each_df.index[-1]], inplace=True)

            each_df = each_df.astype(float)
            if len(each_df) != 0:
                # 準備寫入資料庫裡面
                self.dataprovider.save_data(
                    symbol_name=name, original_df=each_df, symbol_type='FUTURES', time_type='1m', exists="append")

        print("保存資料完成-退出程序")
        exit_event.set()  # 设置事件，导致所有线程退出

    def checkDailydata(self):
        """
            檢查資料庫中的日資料是否已經回補


            if already update then contiune.            
            Originally, daily data only reloaded the FUTURES type, but now it also reloads the SPOT type.

            symbol_type (str):  ('FUTURES', 'SPOT')
        """
        data = self.dataprovider.SQL.get_db_data(
            """select *  from `btcusdt-f-d` order by Datetime desc limit 1""")

        sql_date = str(data[0][0]).split(' ')[0]
        _todaydate = str((datetime.datetime.today() -
                         timedelta(days=1))).split(' ')[0]

        if sql_date != _todaydate:
            # 更新日資料 並且在回補完成後才繼續進行 即時行情的回補
            self.dataprovider.reload_all_data(
                time_type='1d', symbol_type='FUTURES')

        else:
            # 判斷幣安裡面所有可交易的標的
            allsymobl = self.dataprovider.Binanceapp.get_targetsymobls()
            all_tables = self.dataprovider.SQL.read_Dateframe(
                """show tables""")
            all_tables = [
                i for i in all_tables['Tables_in_crypto_data'].to_list() if '-f-d' in i]

            # 當有新的商品出現之後,會導致有錯誤,錯誤修正
            if list(filter(lambda x: False if x.lower() + "-f-d" in all_tables else True, allsymobl)):
                self.dataprovider.reload_all_data(
                    time_type='1d', symbol_type='FUTURES')


    def get_target_symbol(self):
        """ 
            取得交易標的
        """
        all_symbols = self.dataprovider.get_symbols_history_data(
            symbol_type='FUTURES', time_type='1d')

        return self.dataprovider.filter_useful_symbol(all_symbols, tag="MTM_TYPE")

    def reload_all_futures_data(self):
        """
            用來回補所有日線期貨資料
        """
        self.dataprovider.reload_all_data(
            time_type='1m', symbol_type='FUTURES')

    def export_all_tables(self):
        """
            將資料全部從Mysql寫出
        """
        DatabaseBackupRestore().export_all_tables()

    def export_table_data(self, table_name: str):
        """
            單一資料
        """
        DatabaseBackupRestore().export_table_data(table_name)

    def printfunc(self, *args):
        out_str = ''
        for i in args:
            out_str += str(i)+" "

        print(out_str)

    def get_catch(self, name, eachCatchDf):
        """
            取得當次回補的總資料並且不存在於DB之中

        """
        if name in self.new_symbol_map:
            new_df = pd.concat(
                [self.new_symbol_map[name].reset_index(), eachCatchDf])
            new_df.set_index('Datetime', inplace=True)
            # duplicated >> 重複 True 代表重複了
            new_df = new_df[~new_df.index.duplicated(keep='last')]
            self.new_symbol_map.update({name: new_df})
        else:
            eachCatchDf.set_index('Datetime', inplace=True)
            self.new_symbol_map.update({name: eachCatchDf})

    def timewritersql(self):
        """
            寫入保存時間
        """
        self.dataprovider.SQL.change_db_data(
            f""" UPDATE `sysstatus` SET `systeam_datetime`='{str(datetime.datetime.now())}' WHERE `ID`='1';""")


class AsyncTrading_system(Trading_system):
    def __init__(self) -> None:
        """
            這邊才是正式開始交易的地方，父類別大多擔任準備工作和函數提供



            1. 開始研究如何將更多資料寫入DB

        """
        super().__init__()

        self.checkDailydata()  # 檢查日線資料
        self.process_target_symbol()  # 取得要交易的標的

        # 將標得注入引擎 #這邊可能需要獲取更多資料
        self.asyncDataProvider = AsyncDataProvider()

    def process_target_symbol(self):
        """
           商品改成非動態的，因為會變成指定的
        """

        if self.strategy_keyword == 'ONE_TO_MANY':
            # 取得要交易的標的
            market_symobl = list(map(lambda x: x[0], self.get_target_symbol()))

            # 取得binance實際擁有標的,合併 (因為原本有部位的也要持續追蹤)
            self.targetsymbols = self.datatransformer.target_symobl(
                market_symobl, self.dataprovider.Binanceapp.getfutures_account_name())

        elif self.strategy_keyword == 'ONE_TO_ONE':
            old_symbol = self.dataprovider.Binanceapp.getfutures_account_name()  # 第一次運行會是空的
            # 合併舊的商品 因為這樣更新的商品的時候可以把庫存清掉
            self.targetsymbols = list(set(old_symbol + self.DQN_engin.symbols))
        else:
            raise ValueError("STRATEGY_KEYWORD didn't match,please check")

        # 用來判斷一個模型的模式        # 一對多        # 一對一
        print_syslog(title='STRATEGY_KEYWORD', msg=self.strategy_keyword)
        print_syslog(title='TARGETSYMBOLS', msg=self.targetsymbols)

    def check_money_level(self):
        """
            取得實時運作的資金水位
            採用USDT 
            {'09XXXXXXXX': 86619.85787802}

            # 每天平衡一次所以 要製作一個5%的平衡機制
        """
        # 這邊取得的是幣安的保證金(以平倉損益) # 取得所有戶頭的資金
        return self.dataprovider.Binanceapp.get_alluser_futuresaccountbalance()

    def _filter_if_not_trade(self, last_status, if_order_map):
        """
            last_status 是採用EIIE 的資金分配
            if_order_map 是採用DQN

            將兩者合併再一起綜合判斷
        Args:
            last_status (_type_): {'0975730876': {'BNBUSDT': [1, 22.136232357786525],'BTCUSDT': [1, 0.16601471242314805], 'ETHUSDT': [1, 2.9745855751452743]}}
            if_order_map (_type_): {'BTCUSDT': 1.0, 'ETHUSDT': 0.0, 'BNBUSDT': 1.0}

        """
        new_last_status = copy.deepcopy(last_status)
        for user, order_data in last_status.items():
            for symbol in (order_data.keys()):
                if if_order_map.get(symbol) == 0:
                    new_last_status[user].pop(symbol)

        return new_last_status
    
    def generate_order_map(self) -> dict:        
        # # DQN 準備策略
        self.DQN_engin.strategy_prepare(self.targetsymbols)

        # 準備將資料塞入神經網絡或是策略裡面
        finally_df = self.dataprovider.get_trade_data(
            self.targetsymbols, self.symbol_map, freq=self.engine_setting['FREQ_TIME'])

        # 在底層(oderbacktest會將最後一個拋棄)
        if_order_map = self.DQN_engin.get_if_order_map(finally_df)

        finally_df = self.dataprovider.datatransformer.filter_last_time_series(
            finally_df)

        self.engine.work(finally_df)

        # 取得所有戶頭的平衡資金才有辦法去運算口數
        balance_balance_map = self.check_money_level()

        last_status = self.engine.get_order(
            finally_df, balance_balance_map, leverage=self.engine_setting['LEVERAGE'])

        last_status = self._filter_if_not_trade(
            last_status, if_order_map)

        self.printfunc('目前交易狀態,校正之後', last_status)
        
        return last_status
    
    async def main(self):
        self.printfunc("Crypto_trading 正式交易啟動")
        LINE_Alert().send_author("Crypto_trading 正式交易啟動")

        # 先將資料從DB撈取出來
        for name in self.targetsymbols:
            original_df, eachCatchDf = self.dataprovider.get_symboldata(
                name, QMType='Online')
            self.symbol_map.update({name: original_df})
            self.get_catch(name, eachCatchDf)

        last_min = None
        self.printfunc("資料讀取結束")

        # 透過迴圈回補資料
        while not exit_event.is_set():
            try:
                if datetime.datetime.now().minute != last_min or last_min is None:
                    print("現在時間:", datetime.datetime.now())
                    begin_time = time.time()
                    # 取得原始資料
                    all_data_copy = await self.asyncDataProvider.get_all_data()

                    # 避免在self.symbol_map
                    symbol_map_copy = copy.deepcopy(self.symbol_map)
                    for name, each_df in symbol_map_copy.items():
                        # 這裡的name 就是商品名稱,就是symbol_name EX:COMPUSDT,AAVEUSDT
                        original_df, eachCatchDf = self.datatransformer.mergeData(
                            name, each_df, all_data_copy)
                        self.symbol_map[name] = original_df
                        self.get_catch(name, eachCatchDf)

                    self.printfunc("開始進入回測")
                    
                    last_status = self.generate_order_map()

                    current_size = self.dataprovider.Binanceapp.getfutures_account_positions()
                    self.printfunc("目前binance交易所內的部位狀態:", current_size)
                    self.printfunc("*" * 120)

                    # >>比對目前binance 內的部位狀態 進行交易
                    all_order_finally = self.dataprovider.datatransformer.calculation_size(
                        last_status, current_size, self.symbol_map, exchange_info=self.dataprovider.Binanceapp.getfuturesinfo())

                    print("測試all_order_finally", all_order_finally)
                    # 將order_finally 跟下單最小單位相比
                    all_order_finally = self.dataprovider.datatransformer.change_min_postion(
                        all_order_finally, self.dataprovider.Binanceapp.getMinimumOrderQuantity())

                    self.printfunc("差異單", all_order_finally)
                    self.dataprovider.Binanceapp.execute_orders(
                        all_order_finally, current_size=current_size, symbol_map=self.symbol_map, formal=self.systeam_setting['execute_orders'])
                    self.printfunc("時間差", time.time() - begin_time)
                    last_min = datetime.datetime.now().minute
                    self.timewritersql()
                else:
                    time.sleep(1)

                if self.dataprovider.Binanceapp.trade_count > self.systeam_setting['emergency_times']:
                    self.printfunc("緊急狀況處理-交易次數過多")
                    raise ExcessiveTradeException("當前交易次數超過最大限制")

            except Exception as e:
                if isinstance(e, BinanceAPIException) and e.code == -1001:
                    Debug_tool.debug.print_info()
                else:
                    # re-raise the exception if it's not the expected error code
                    Debug_tool.debug.print_info()
                    raise e


def run_asyncio_loop(func, *args, exit_event=None):  # 使用默认参数，使得 exit_event 是可选的
    if func.__name__ == 'subscriptionData':
        asyncio.run(func(*args, exit_event=exit_event))
    elif func.__name__ == 'main':
        asyncio.run(func(*args))


def safe_run(target, *args, **kwargs):
    try:
        target(*args, **kwargs)
    except Exception as e:
        import os
        print(f"安全錯誤線程 : {target.__name__}: {e}")
        exit_event.set()  # 透過事件設置就可以讓所有的thread關閉


# TradingSysteam.py
if __name__ == '__main__':
    exit_event = threading.Event()
    trading_system = AsyncTrading_system()

    # 使用 safe_run 啟動每個線程
    thread1 = threading.Thread(target=safe_run, args=(
        run_asyncio_loop, trading_system.asyncDataProvider.subscriptionData, trading_system.targetsymbols), kwargs={'exit_event': exit_event})
    thread2 = threading.Thread(target=safe_run, args=(
        run_asyncio_loop, trading_system.main))
    thread3 = threading.Thread(
        target=safe_run, args=(trading_system.DailyChange,))

    thread1.start()
    thread2.start()
    thread3.start()
    thread1.join()
    thread2.join()
    thread3.join()
