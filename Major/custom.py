"""
    refer to 'finlab_crypto' package
    url : 'https://github.com/finlab-python/finlab_crypto'

"""

from binance.client import Client
from binance.enums import HistoricalKlinesType
from binance.enums import SIDE_BUY, ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT, SIDE_SELL
from binance.helpers import interval_to_milliseconds, convert_ts_str
from binance.exceptions import BinanceAPIException
import pandas as pd
from dateutil import parser
import math
from datetime import timedelta, datetime, timezone
import re
import time
import json
from Major.Date_time import parser_time
from utils.Debug_tool import debug
import time
from Database import SQL_operate
import copy
from tqdm.auto import tqdm
from .UserManager import UserManager
from Infrastructure.AlertMsg import LINE_Alert


def SetConnectClose(custom_user):
    """
        如果使用者是所有人通常都會從外部進來,作者就從資料庫拿取就好    
    Returns:
        wrapper function
    """

    assert isinstance(custom_user, str), "custom_user must str"

    def actual_decorator(func):
        def wrapper(self, *args, **kwargs):
            if custom_user == 'author':
                account, passwd = UserManager.GetAccount_Passwd(custom_user)
                client = Client(account, passwd)
                result = func(self, client, *args, **kwargs)
                client.close_connection()
                return result
            elif custom_user == 'all_user':
                user_data = UserManager.GetAllUser()
                # 下訂單要特別處理
                if func.__name__ == 'execute_orders':
                    for row in user_data:
                        # 這邊要特別判斷是要下到甚麼戶頭去
                        ID_phonenumber, account, passwd = row[0], row[1], row[2]
                        client = Client(account, passwd)
                        new_args = (copy.deepcopy(args[0])[ID_phonenumber],)
                        new_kwargs = copy.deepcopy(kwargs)
                        new_kwargs['current_size'] = new_kwargs['current_size'][ID_phonenumber]
                        new_kwargs['current_line_token'] = row[3]
                        result = func(self, client, *new_args, **new_kwargs)
                        client.close_connection()
                else:
                    result_map = {}

                    for row in user_data:
                        account, passwd = row[1], row[2]
                        client = Client(account, passwd)
                        result = func(self, client, *args, **kwargs)
                        result_map.update({row[0]: result})
                        client.close_connection()

                    return result_map
        return wrapper
    return actual_decorator


def change_ts_to_str(time_stamp: int):
    """
        將時間戳 轉換成字串
    """
    return str(datetime.fromtimestamp(time_stamp, tz=timezone.utc).replace(tzinfo=None))


class BinanceData(object):
    """
        'pip install python-binance'
    """
    # CONSTANTS
    binsizes = {"1m": 1, "5m": 5, '15m': 15, '30m': 30,
                "1h": 60, '2h': 120, "4h": 240, "1d": 1440}
    batch_size = 750

    @classmethod
    def historicalklines(cls, symbol, interval, start_str=None, end_str=None, limit=1000,
                         klines_type: HistoricalKlinesType = None, client: Client = None):
        """
            Get Historical Klines from Binance (spot or futures)
            to copy binance.client._historical_klines to add tqdm by self


        """
        # init our list
        output_data = []

        # convert interval to useful value in seconds
        timeframe = interval_to_milliseconds(interval)

        # if a start time was passed convert it
        start_ts = convert_ts_str(start_str)

        # establish first available start timestamp
        if start_ts is not None:
            first_valid_ts = client._get_earliest_valid_timestamp(
                symbol, interval, klines_type)
            start_ts = max(start_ts, first_valid_ts)

        # if an end time was passed convert it
        end_ts = convert_ts_str(end_str)
        if end_ts and start_ts and end_ts <= start_ts:
            return output_data

        idx = 0

        with tqdm() as pbar:
            pbar.set_description(change_ts_to_str(start_ts/1000))
            while True:
                # fetch the klines from start_ts up to max 500 entries or the end_ts if set
                temp_data = client._klines(
                    klines_type=klines_type,
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    startTime=start_ts,
                    endTime=end_ts
                )

                # append this loops data to our output data
                if temp_data:
                    output_data += temp_data

                # handle the case where exactly the limit amount of data was returned last loop
                # check if we received less than the required limit and exit the loop
                if not len(temp_data) or len(temp_data) < limit:
                    # exit the while loop
                    break

                # increment next call by our timeframe
                start_ts = temp_data[-1][0] + timeframe

                # exit loop if we reached end_ts before reaching <limit> klines
                if end_ts and start_ts >= end_ts:
                    break

                pbar.set_description("{} - {}".format(
                    change_ts_to_str(start_ts/1000),
                    change_ts_to_str(end_ts/1000)
                ))

                # sleep after every 3rd call to be kind to the API
                idx += 1
                if idx % 3 == 0:
                    time.sleep(1)

                pbar.update(1)

        return output_data

    @classmethod
    def minutes_of_new_data(cls, symbol, kline_size: str, data: pd.DataFrame, source: str, client: Client, symbol_type:str):
        """Process old and new histrical price data format through binance api.

        The boundary between new data and old data is 2017.1.1.

        Args:
        symbol (str): Trading pair (ex: BTCUSDT).
        kline_size (str): A frequency of the price data (ex: "1m", "5m",'15m', '30m', "1h", '2h', "4h", "1d")
        data (dataframe): The data from get_all_binance() crawlers.
        source (str): data source (ex:'binance','bitmex')
        client (Binance.Client) (optional): Binance Client object.
        symbol_type (str): 'SPOT','FUTURES'

        Returns:
            old: OHLCV DataFrame of old format.
            new: OHLCV DataFrame of new format.
        """
        if len(data) > 0:
            old = parser.parse(data["Datetime"].iloc[-1])
        elif source == "binance":
            old = datetime.strptime('1 Jan 2017', '%d %b %Y')
        elif source == "bitmex":
            old = client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=1, reverse=False).result()[0][0][
                'Datetime']

        if source == "binance":
            """ 有些商品只有期貨的部份 所以還是以期貨的API為主 """
            if symbol_type == 'SPOT':
                new = pd.to_datetime(client.get_klines(symbol=symbol, interval=kline_size)[-1][0],
                                    unit='ms')
            else:
                new = pd.to_datetime(client.futures_klines(symbol=symbol, interval=kline_size)[-1][0],
                                    unit='ms')
        if source == "bitmex":
            new = \
                client.Trade.Trade_getBucketed(
                    symbol=symbol, binSize=kline_size, count=1, reverse=True).result()[0][0]['Datetime']

        return old, new + timedelta(minutes=1)

    @classmethod
    @SetConnectClose("author")
    def download(cls, client: Client, original_df: pd.DataFrame, symbol, kline_size, symbol_type=None):
        """
        Getting histrical price data through binance api.

        Original code from: https://medium.com/swlh/retrieving-full-historical-data-for-every-cryptocurrency-on-binance-bitmex-using-the-python-apis-27b47fd8137f

        Args:
        symbol (str): Trading pair (ex: BTCUSDT).
        kline_size (str): A frequency of the price data (ex: "1m", "5m",'15m', '30m', "1h", '2h', "4h", "1d")
        save (bool): Save the results in ./history/ to improve the retreive waiting time.
        client (Binance.Client) (optional): Binance Client object.

        HistoricalKlinesType.SPOT現貨交易
        HistoricalKlinesType.FUTURES期貨交易
        HistoricalKlinesType.FUTURES_COIN幣本位期貨交易

        klines = [
            [
                1499040000000,      // 开盘时间
                "0.01634790",       // 开盘价
                "0.80000000",       // 最高价
                "0.01575800",       // 最低价
                "67954.9",          // 收盘价(当前K线未结束的即为最新价)
                "29.806",           // 成交量
                1499644799999,      // 收盘时间
                "2025230",          // 成交额
                642,                // 成交笔数
                "20.498",           // 主动买入成交量
                "1392760",          // 主动买入成交额
                "17928899.62484339" // 请忽略该参数
            ]
        ]

        Returns:
            pd.DataFrame: OHLCV data for all



        """
        assert symbol_type is not None, "ERROR symbol type can't be None"

        if symbol_type == 'SPOT':
            klines_type = HistoricalKlinesType.SPOT
        elif symbol_type == 'FUTURES':
            klines_type = HistoricalKlinesType.FUTURES

        oldest_point, newest_point = cls.minutes_of_new_data(
            symbol, kline_size, original_df, source="binance", client=client,symbol_type = symbol_type)

        delta_min = (newest_point - oldest_point).total_seconds() / 60
        available_data = math.ceil(delta_min / cls.binsizes[kline_size])

        if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'):
            print('Downloading all available %s data for %s. Be patient..!' %
                  (kline_size, symbol))
        else:
            print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (
                delta_min, symbol, available_data, kline_size))

        # 取得歷史資料改寫
        klines = cls.historicalklines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"),
                                      newest_point.strftime("%d %b %Y %H:%M:%S"), klines_type=klines_type, client=client)
        data = pd.DataFrame(klines,
                            columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time', 'quote_av',
                                     'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])

        data['Datetime'] = pd.to_datetime(data['Datetime'], unit='ms')

        if len(original_df) > 0:
            original_df['Datetime'] = original_df['Datetime'].apply(
                lambda x: pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S")
                if " " in str(x) else pd.to_datetime(x, format="%Y-%m-%d")
            )

            new_df = pd.concat([original_df, data])
        else:
            new_df = data.copy(deep=True)

        new_df.set_index('Datetime', inplace=True)

        # duplicated >> 重複 True 代表重複了 # 如果在極短的時間重複抓取 會有重複的問題
        new_df = new_df[~new_df.index.duplicated(keep='last')]

        print('商品資料回補完成!')
        new_df = new_df.astype(float)
        return new_df, data


class Binance_server(object):
    """
        用來呼叫幣安的相關應用
        每次連線之前重新建立連線,減少斷線次數

    Args:
        object (_type_):
    """

    def __init__(self) -> None:
        self.trade_count = 0
        self.BinanceData = BinanceData()
    
    @SetConnectClose("author")
    def futures_ticker(self, client: Client) -> dict:
        """ 獲取所有幣安合約的 24 小時價格變動統計數據。

        Returns:
            _type_: _description_
        """
        return client.futures_ticker()
    
    @SetConnectClose("author")
    def getfuturesinfo(self, client: Client) -> dict:
        """ 回傳交易所的合約

        Returns:
            _type_: _description_
        """
        return client.futures_exchange_info()

    @SetConnectClose("author")
    def getExchangeInfo(self, client: Client) -> dict:
        """ 回傳交易所的合約

        Returns:
            _type_: _description_
        """
        return client.get_exchange_info()

    @SetConnectClose("author")
    def getMinimumOrderQuantity(self, client: Client):
        """
            取得最小下單數量限制
        """
        data = client.futures_exchange_info()

        out_dict = {}
        for symbol in data['symbols']:
            out_dict.update({symbol['symbol']: symbol['filters'][2]['minQty']})

        return out_dict

    def get_targetsymobls(self, symbol_type: str) -> list:
        """
            to call api get futures symbol and clean data

            只收集USDT使用的合約,並且已經掛牌上可以交易的標的
        Returns:
            list: _description_
        """
        out_list = []
        if symbol_type == 'FUTURES':
            data = self.getfuturesinfo()
            for each_data in data['symbols']:
                if each_data['marginAsset'] == 'USDT' and each_data['status'] == 'TRADING':
                    # 目前怪怪的合約都拋棄不要（有數字的大概論都有特殊意義）
                    clear_name = re.findall(
                        '[A-Z]+USDT', each_data['symbol'])
                    if clear_name:
                        if each_data['symbol'] == clear_name[0]:
                            out_list.append(each_data['symbol'])
        elif symbol_type == 'SPOT':
            data = self.getExchangeInfo()
            i = 0
            for each_data in data['symbols']:
                if each_data['status'] == 'TRADING' and each_data['quoteAsset'] == "USDT":
                    clear_name = re.search(
                        r'^[A-Z]+USDT$', each_data['symbol'])
                    if clear_name:
                        if each_data['symbol'] == clear_name[0]:
                            out_list.append(each_data['symbol'])

        return out_list

    @SetConnectClose('all_user')
    def getfutures_account_positions(self, client: Client):
        """
            取得合約部位 >> 裡面還可以看到其他資訊
        """
        data = client.futures_account()
        out_put = {}
        for i in data['positions']:
            if i['initialMargin'] == '0':
                continue
            out_put.update({i['symbol']: i['positionAmt']})

        return out_put

    @SetConnectClose('author')
    def getfutures_account_name(self, client: Client):
        """
            取得合約部位 >> 裡面還可以看到其他資訊
        """
        data = client.futures_account()
        out_put = []
        for i in data['positions']:
            if i['initialMargin'] == '0':
                continue
            out_put.append(i['symbol'])

        return out_put

    def _limit_order_times_qty(self, max_qty: str, order_quantity: float) -> tuple:
        """
            主要原因在於市價單的拋出數量有限制如果採用市價拋出的話,有可能會報出錯誤
            返回總共要下幾次,跟單次最大數量限制,餘數不管

        Args:
            max_qty (str): _description_
            order_quantity (float): _description_
        """
        max_qty = float(max_qty)
        order_times, _ = divmod(order_quantity, max_qty)
        return order_times, max_qty

    @SetConnectClose('all_user')
    def execute_orders(self, client: Client, order_finally: dict, model=ORDER_TYPE_MARKET, current_size=dict(), symbol_map=dict(), formal=False, current_line_token=''):
        """
            Execute orders to Binance.
            use for futures
            to develop data = 2023-1-15
            note i can't find create_test_order by futures.


            To place a futures limit order:
            binance_client.futures_create_order(
                symbol='BTCUSDT',
                type='LIMIT',
                timeInForce='GTC',  # Can be changed - see link to API doc below
                price=30000,  # The price at which you wish to buy/sell, float
                side='BUY',  # Direction ('BUY' / 'SELL'), string
                quantity=0.001  # Number of coins you wish to buy / sell, float
            )

            To place a futures market order:
            binance_client.futures_create_order(
                symbol='BTCUSDT',
                type='MARKET',
                timeInForce='GTC',
                side='BUY',
                quantity=0.001
            )

            2. 生效時間
            有效時間表示您的訂單在執行或過期之前將保持有效的時間。這樣可以讓您對時間參數更加具體，您可以在下單時自定義時間。
            在幣安，您可以下 GTC（取消前有效）、IOC（立即取消或取消）或 FOK（執行或終止）訂單：
            GTC (Good-Till-Cancel)：訂單將持續到完成或您取消為止。
            IOC (Immediate-Or-Cancel)：訂單將嘗試以可用的價格和數量立即執行全部或部分訂單，然後取消訂單中任何剩餘的、未完成的部分。如果您下單時所選價格沒有數量可用，將立即取消。請注意，不支持冰山訂單。
            FOK (Fill-Or-Kill)：指示訂單立即全額執行（filled），否則將被取消（kill）。請注意，不支持冰山訂單。


            line_alert: to send msg to LINE

            current_size:
                {'BNBUSDT': '21.18', 'ETCUSDT': '2.35', 'XMRUSDT': '27.470',
                'KSMUSDT': '107.1', 'ZECUSDT': '111.190', 'SOLUSDT': '168', 'YFIUSDT': '1.260', 'ETHUSDT': '3.399',
                'EGLDUSDT': '156.8', 'BCHUSDT': '36.130', 'AAVEUSDT': '71.1', 'MKRUSDT': '3.579', 'DEFIUSDT': '9.939',
                'COMPUSDT': '80.300', 'BTCDOMUSDT': '2.410', 'BTCUSDT': '0.200'}

            Response example:{'symbol': 'XMRUSDT', 'leverage': 10, 'maxNotionalValue': '1000000'}

            待修正問題:
                # 在已經最佳化的槓桿下如果爆出已下錯誤的話,我認為是否是下注過大絕對超過4倍槓桿
                # 所以目前不知道該如何修復
                2.binance.exceptions.BinanceAPIException: APIError(code=-2019): Margin is insufficient.
        """
        if not order_finally:
            return
        self.trade_count += 1
        print('目前交易次數', self.trade_count)
        print(f"進入下單,目前下單模式:{model}")

        # 获取合约账户信息
        account_info = client.futures_account()

        # 下單前檢查leverage
        # 商品槓桿
        # 將已經持倉的部位傳入(讀取所有的槓桿)
        leverage_map = {}
        for position in account_info['positions']:
            symbol = position['symbol']
            leverage = position['leverage']
            position_amt = float(position['positionAmt'])
            if position_amt != 0:
                print(f"交易對: {symbol}, leverage numbers: {leverage}")
                leverage_map.update({symbol: leverage})

        for each_symbol, ready_to_order_size in order_finally.items():
            if leverage_map.get(each_symbol, None) is None:
                def _change_leverage(_symbol):
                    last_leverage = 0
                    leverage = 1
                    Response = client.futures_change_leverage(
                        symbol=_symbol, leverage=leverage)
                    while True:
                        try:
                            # 比下單資金更大才行
                            if float(Response['maxNotionalValue']) > ready_to_order_size * symbol_map[each_symbol]['Close'].iloc[-1]:
                                last_leverage = leverage
                                leverage += 1
                                Response = client.futures_change_leverage(
                                    symbol=_symbol, leverage=leverage)
                            else:
                                Response = client.futures_change_leverage(
                                    symbol=_symbol, leverage=last_leverage)
                                break

                        except BinanceAPIException as e:
                            if e.code == -4028:
                                # 槓桿值無效。請選擇一個有效的槓桿值。
                                print(
                                    "Invalid leverage value. Please choose a valid leverage value.")
                                # 已經沒辦法在大開更大的槓桿時,跳出
                                break
                            else:
                                raise e

                _change_leverage(each_symbol)
            else:
                # 如果商品已經存在 直接呼叫
                # 判斷是否需要更改槓桿 不需要管正負號
                Response = client.futures_change_leverage(
                    symbol=each_symbol, leverage=leverage_map.get(each_symbol))

                print("直接取得原始槓桿:", Response)
                beginleverage = leverage_map.get(each_symbol)
                while True:
                    if (float(current_size[each_symbol]) + ready_to_order_size) * symbol_map[each_symbol]['Close'].iloc[-1] < float(Response['maxNotionalValue']):
                        break
                    time.sleep(0.3)
                    beginleverage = beginleverage - 1
                    Response = client.futures_change_leverage(
                        symbol=each_symbol, leverage=beginleverage)
                    print("調整槓桿:", Response)

        # ===========================================================================================
        # 還要測試下單
        # for symbol, ready_to_order_size in order_finally.items():
        #     Response = client.futures_change_leverage(
        #         symbol=symbol, leverage=10)
        #     print("新手無法使用超過20倍之槓桿")
        #     print(Response)

        self._execute_orders(client,
                             model=model, order_finally=order_finally, formal=formal, current_line_token=current_line_token)

    def _execute_orders(self, client: Client, model=ORDER_TYPE_MARKET, order_finally: dict = None, formal=False, current_line_token=''):
        """
            將下單邏輯分開方便除錯

        Args:
            client (Client): _description_
            line_alert (_type_): _description_
            model (_type_, optional): _description_. Defaults to ORDER_TYPE_MARKET.
            order_finally (dict, optional): _description_. Defaults to None.
            formal (bool, optional): _description_. Defaults to False.
        """

        assert isinstance(order_finally, dict), "danger order_finally is None"

        exchange_info_data = self._get_futures_exchange_info()
        for symbol, ready_to_order_size in order_finally.items():
            # 取得下單模式
            if model == 'MARKET':
                order_type = ORDER_TYPE_MARKET
            else:
                order_type = ORDER_TYPE_LIMIT

            # 取得 side 買賣方向
            if ready_to_order_size > 0:
                order_side = SIDE_BUY
            else:
                order_side = SIDE_SELL

            if model == 'MARKET':
                order_timeInForce = 'IOC'
            else:
                order_timeInForce = 'GTC'  # 這邊要在注意

            # 取得 quantity數量
            order_quantity = abs(ready_to_order_size)

            order_times, max_qty = self._limit_order_times_qty(
                exchange_info_data[symbol], order_quantity)

            print("目前商品:",symbol,"下單次數:",order_times,"最大量:",max_qty)
            for _ in range(max([1, int(order_times)])):
                if int(order_times) == 0:
                    real_order = order_quantity
                else:
                    real_order = max_qty

                LINE_Alert().req_line_alert(
                    each_token=current_line_token, str_msg=f"商品:{symbol}\n買賣別:{order_side}\n委託單:{order_type}\n委託類別:{order_timeInForce}\n委託數量:{real_order}")

                if order_side == "SELL":
                    args = dict(side=order_side,
                                type=order_type,
                                symbol=symbol,
                                quantity=real_order,
                                reduceOnly=True)
                else:
                    args = dict(side=order_side,
                                type=order_type,
                                symbol=symbol,
                                quantity=real_order)

                if formal:
                    # 丟入最後create 單裡面
                    result = client.futures_create_order(**args)
                    self.save_order_result(result)
                else:
                    print("警告:,下單功能被關閉,若目前處於正式交易請重新開啟系統")
                print("循環進入測試")
                time.sleep(0.5)

    def save_order_result(self, order_data: dict):
        """
            下單完成之後伺服器會回傳一串dict
            保存至本地磚資料夾

            2023/10/9 發現格式更改,故決定將其改成字串保存減少報錯的機率

            1. 將原始的table刪除
                CREATE TABLE `crypto_data`.`orderresult`(
                `orderId` BIGINT NOT NULL,
                `symbol` varchar(255) NOT NULL,
                `status` varchar(255) NOT NULL,
                `clientOrderId` varchar(255) NOT NULL,
                `price` varchar(255) NOT NULL,
                `avgPrice` varchar(255) NOT NULL,
                `origQty` varchar(255) NOT NULL,
                `executedQty` varchar(255) NOT NULL,
                `cumQty` varchar(255) NOT NULL,
                `cumQuote` varchar(255) NOT NULL,
                `timeInForce` varchar(255) NOT NULL,
                `type` varchar(255) NOT NULL,
                `reduceOnly` BOOLEAN NOT NULL,
                `closePosition` BOOLEAN NOT NULL,
                `side` varchar(255) NOT NULL,
                `positionSide` varchar(255) NOT NULL,
                `stopPrice` varchar(255) NOT NULL,
                `workingType` varchar(255) NOT NULL,
                `priceProtect` BOOLEAN NOT NULL,
                `origType` varchar(255) NOT NULL,
                `updateTime` BIGINT NOT NULL,
                PRIMARY KEY(`orderId`)
                );

            2. 建立新的table
                CREATE TABLE `crypto_data`.`orderresult`(
                `orderId` BIGINT NOT NULL,
                `order_info` varchar(500) NOT NULL,
                PRIMARY KEY(`orderId`)
                );
        Args:
            order_data (dict): #
            版本1:{'orderId': 22019361762, 'symbol': 'SOLUSDT', 'status': 'NEW', 'clientOrderId': 'yfobvBPbosaT0Zz38XNxHv', 'price': '0', 'avgPrice': '0.0000', 'origQty': '1', 'executedQty': '0', 'cumQty': '0', 'cumQuote': '0', 'timeInForce': 'GTC',
                'type': 'MARKET', 'reduceOnly': False, 'closePosition': False, 'side': 'BUY', 'positionSide': 'BOTH', 'stopPrice': '0', 'workingType': 'CONTRACT_PRICE', 'priceProtect': False, 'origType': 'MARKET', 'updateTime': 1675580975203}

            # 實際拿到的respone(24個欄位)
            (15154865689, 'ZECUSDT', 'NEW', '7Xvs36qJXADB3vbPtz6xCZ', '0.00', '0.00', '0.609', '0.000', '0.000', '0.00000', 'GTC',
             'MARKET', False, False, 'BUY', 'BOTH', '0.00', 'CONTRACT_PRICE', False, 'MARKET', 'NONE', 'NONE', 0, 1696819981601);

            當下官方的文檔為26個欄位:
            {
                "clientOrderId": "testOrder",
                "cumQty": "0",
                "cumQuote": "0",
                "executedQty": "0",
                "orderId": 22542179,
                "avgPrice": "0.00000",
                "origQty": "10",
                "price": "0",
                "reduceOnly": False,
                "side": "BUY",
                "positionSide": "SHORT",
                "status": "NEW",
                "stopPrice": "9300",
                "closePosition": False,
                "symbol": "BTCUSDT",
                "timeInForce": "GTD",
                "type": "TRAILING_STOP_MARKET",
                "origType": "TRAILING_STOP_MARKET",
                "activatePrice": "9020",
                "priceRate": "0.3",
                "updateTime": 1566818724722,
                "workingType": "CONTRACT_PRICE",
                "priceProtect": False,
                "priceMatch": "NONE",
                "selfTradePreventionMode": "NONE",
                "goodTillDate": 1693207680000
            }
        """
        try:
            SQL = SQL_operate.DB_operate()
            orderId = order_data.pop('orderId')
            order_info = json.dumps(order_data)
            SQL.change_db_data(
                f""" INSERT INTO `orderresult` (`orderId`, `order_info`) VALUES ({orderId}, '{order_info}');""")
        except:
            debug.print_info(error_msg="保存系統order單錯誤")

    @SetConnectClose('all_user')
    def get_alluser_futuresaccountbalance(self, client: Client) -> float:
        for i in client.futures_account_balance():
            if i['asset'] == 'USDT':
                return float(i['balance'])

    @SetConnectClose('author')
    def _get_futures_exchange_info(self, client: Client):
        """ 用來查看市價單的數量委託限制

        Args:
            client (Client): _description_

        Returns:
            exchange_inf_data.keys = ['timezone', 'serverTime', 'futuresType',
                'rateLimits', 'exchangeFilters', 'assets', 'symbols']
            futuresType = U_MARGINED
            rateLimits = [{'rateLimitType': 'REQUEST_WEIGHT', 'interval': 'MINUTE', 'intervalNum': 1, 'limit': 2400}, {'rateLimitType': 'ORDERS',
                'interval': 'MINUTE', 'intervalNum': 1, 'limit': 1200}, {'rateLimitType': 'ORDERS', 'interval': 'SECOND', 'intervalNum': 10, 'limit': 300}]
            exchangeFilters = []
            assets = [{'asset': 'USDT', 'marginAvailable': True, 'autoAssetExchange': '-10000'}, {'asset': 'BTC', 'marginAvailable': True, 'autoAssetExchange': '-0.00100000'}, {'asset': 'BNB', 'marginAvailable': True, 'autoAssetExchange': '-10'}, {'asset': 'ETH', 'marginAvailable': True, 'autoAssetExchange': '-5'}, {'asset': 'XRP',
                'marginAvailable': True, 'autoAssetExchange': '0'}, {'asset': 'BUSD', 'marginAvailable': True, 'autoAssetExchange': '-10000'}, {'asset': 'USDC', 'marginAvailable': True, 'autoAssetExchange': '0'}, {'asset': 'TUSD', 'marginAvailable': True, 'autoAssetExchange': '0'}, {'asset': 'USDP', 'marginAvailable': True, 'autoAssetExchange': '0'}]
            symbols :用來放相關資料

            'ZECUSDT': '111.190'
        """
        exchange_inf_data = client.futures_exchange_info()

        out_dict = {}
        for data in exchange_inf_data['symbols']:
            symbol = data['symbol']
            for each_filter in data['filters']:
                if each_filter['filterType'] == 'MARKET_LOT_SIZE':
                    out_dict.update({symbol: each_filter['maxQty']})

        return out_dict
