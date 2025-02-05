import pandas as pd
from Major.Date_time import parser_time
import numpy as np
from decimal import Decimal
import time


class Datatransformer:
    def get_tradedata(self, original_df: pd.DataFrame, freq: int = 30):
        """
            將binance 的UTC 資料做轉換 變成可以交易的資料
            採用biance 官方向前機制
            # 如果是使用Mulitcharts 會變成向後機制

        Args:
            original_df:
                data from sql

            freq (int): 
                "this is resample time like"

        """
        df = original_df.copy()
        
        # 讀取資料(UTC原始資料)
        df.reset_index(inplace=True)

        df["Datetime"] = df['Datetime'].apply(parser_time.changetime)
        df.set_index("Datetime", inplace=True, drop=False)
        df = self.drop_colunms(df)

        # 採用biance 向前機制
        new_df = pd.DataFrame()
        new_df['Open'] = df['Open'].resample(
            rule=f'{freq}min', label="left").first()
        new_df['High'] = df['High'].resample(
            rule=f'{freq}min', label="left").max()
        new_df['Low'] = df['Low'].resample(
            rule=f'{freq}min', label="left").min()
        new_df['Close'] = df['Close'].resample(
            rule=f'{freq}min', label="left").last()
        new_df['Volume'] = df['Volume'].resample(
            rule=f'{freq}min', label="left").sum()
        # quote_av// 成交额
        new_df['quote_av'] = df['quote_av'].resample(
            rule=f'{freq}min', label="left").sum()
        #  trades// 成交笔数
        new_df['trades'] = df['trades'].resample(
            rule=f'{freq}min', label="left").sum()
        
        # tb_base_av// 主动买入成交量
        new_df['tb_base_av'] = df['tb_base_av'].resample(
            rule=f'{freq}min', label="left").sum()
        
        # tb_quote_av// 主动买入成交额
        new_df['tb_quote_av'] = df['tb_quote_av'].resample(
            rule=f'{freq}min', label="left").sum()
        return new_df

    def drop_colunms(self, df: pd.DataFrame):
        """
                
            df(pd.DataFrame):                               Datetime    Open    High     Low   Close  Volume    close_time    quote_av  trades  tb_base_av  tb_quote_av  ignore
                                Datetime
                                2020-09-03 15:02:00 2020-09-03 15:02:00  57.000  58.070  55.900  56.893   611.9  1.599120e+12  34902.3000   125.0       329.8   18842.3000     0.0
                                2020-09-03 15:03:00 2020-09-03 15:03:00  56.741  58.284  56.695  58.284  1195.2  1.599120e+12  68911.6000   448.0       730.1   42054.3000     0.0
                                2020-09-03 15:04:00 2020-09-03 15:04:00  58.300  59.200  57.651  58.882  1568.8  1.599120e+12  91677.7000   385.0       957.6   56024.1000     0.0
                                2020-09-03 15:05:00 2020-09-03 15:05:00  58.739  59.077  58.101  58.101   595.4  1.599120e+12  34876.2000   221.0       257.4   15127.4000     0.0
                                2020-09-03 15:06:00 2020-09-03 15:06:00  58.542  58.675  58.211  58.473    81.3  1.599120e+12   4752.1000    41.0        58.9    3448.1300     0.0
                                ...                                 ...     ...     ...     ...     ...     ...           ...         ...     ...         ...          ...     ...
                                2024-08-27 15:23:00 2024-08-27 15:23:00  70.709  70.781  70.702  70.763   405.7  1.724743e+12  28703.8550   247.0       239.0   16908.3223     0.0
                                2024-08-27 15:24:00 2024-08-27 15:24:00  70.762  70.816  70.760  70.760   318.0  1.724743e+12  22508.1561   227.0       187.6   13278.4140     0.0
                                2024-08-27 15:25:00 2024-08-27 15:25:00  70.755  70.805  70.737  70.793   271.2  1.724744e+12  19191.3057   247.0       163.4   11563.4161     0.0
                                2024-08-27 15:26:00 2024-08-27 15:26:00  70.792  70.793  70.716  70.722   315.4  1.724744e+12  22313.2490   216.0        92.8    6564.4374     0.0
                                2024-08-27 15:27:00 2024-08-27 15:27:00  70.721  70.769  70.697  70.769   329.1  1.724744e+12  23279.6646   177.0       282.7   19998.5423     0.0

        """

        for key in df.columns:
            if key in ['close_time', 'ignore']:
                df = df.drop(columns=[key])

        return df

    def calculation_size(self, last_status, current_size, symbol_map, exchange_info: dict) -> dict:
        """_summary_

        Args:
            last_status (_type_): {'09XXXXXXXX': {'BTCUSDT-15K-OB-DQN': [1, 0.6278047845752174], 'ETHUSDT-15K-OB-DQN': [1, 20.967342756868344]}}
            current_size (_type_): {'09XXXXXXXX': {'BNBUSDT': '115.84', 'XMRUSDT': '67.019',
            'DASHUSDT': '341.320', 'FOOTBALLUSDT': '22.15', 'LTCUSDT': '103.450', 'KSMUSDT': '175.1',
            'TRBUSDT': '544.7', 'ZECUSDT': '227.060', 'QNTUSDT': '541.7', 'SOLUSDT': '365', 'GMXUSDT': '141.84', 'YFIUSDT': '0.520', 'ETHUSDT': '19.039', 'EGLDUSDT': '106.3', 'BCHUSDT': '71.770', 'AAVEUSDT': '96.0', 'MKRUSDT': '11.789', 'DEFIUSDT': '4.080', 'COMPUSDT': '51.079', 'BTCDOMUSDT': '19.410', 'BTCUSDT': '0.570'}}
            symbol_map (_type_): _description_
            exchange_info(dict):
                幣安裡面交易所的資訊
        Returns:
            dict: _description_
        """
        min_notional_map = self.Get_MIN_NOTIONAL(exchange_info=exchange_info)

        out_dict = {}
        for ID_phonenumber, value in last_status.items():
            diff_map = self._calculation_size(
                value, current_size[ID_phonenumber], symbol_map, min_notional_map)
            out_dict[ID_phonenumber] = diff_map

        return out_dict

    def _calculation_size(self, systeam_size: dict, true_size: dict, symbol_map: dict, min_notional_map: dict) -> dict:
        """
            用來比較 系統部位 和 Binance 交易所的實際部位

        Args:
            systeam_size (dict): example :{'BTCUSDT-15K-OB': [1.0, 0.37385995823410634], 'ETHUSDT-15K-OB': [1.0, 5.13707471134965],'BTCUSDT-30K-OB': [1.0, 0.995823410634], 'ETHUSDT-17K-OB': [1.0, 2.13707471134965]}
            true_size (dict): example :{'ETHUSDT': '10.980', 'BTCUSDT': '0.420'} 

        Returns:
            dict: {'BTCUSDT': 0.9496833688681066, 'ETHUSDT': -3.7058505773006996}


            當計算出來的結果 + 就是要買 - 就是要賣

        """
        combin_dict = {}
        for name_key, status in systeam_size.items():
            combin_symobl = name_key.split('-')[0]
            if combin_symobl in combin_dict:
                combin_dict.update(
                    {combin_symobl: combin_dict[combin_symobl] + (status[0] * status[1])})
            else:
                combin_dict.update({combin_symobl: status[0] * status[1]})

        # 下單四捨五入
        for each_symbol, each_value in combin_dict.items():
            combin_dict[each_symbol] = round(combin_dict[each_symbol], 2)

        diff_map = {}
        for symbol_name, postition_size in combin_dict.items():
            if true_size.get(symbol_name, None) is None:
                if postition_size == 0:
                    continue
                diff_map.update({symbol_name: postition_size})
            else:
                # 下單要符合幣安各商品最小下單金額限制
                diff = postition_size - float(true_size[symbol_name])
                if diff > 0 and abs(symbol_map[symbol_name]['Close'].iloc[-1] * (diff)) < min_notional_map[symbol_name]:
                    continue
                diff_map.update(
                    {symbol_name: postition_size - float(true_size[symbol_name])})

        # 如果已經不再系統裡面 就需要去close
        for symbol_name, postition_size in true_size.items():
            if symbol_name not in combin_dict:
                diff_map.update({symbol_name: - float(postition_size)})

        return diff_map

    def trans_int16(self, data: dict):
        """
            用來轉變字串(json dumps 的時候)
            轉換np.int16
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, np.int16):
                    data[key] = int(value)
        return data

    def generate_table_name(self, symbol_name, symbol_type, time_type, iflower=True):
        """
        根據給定的參數產生表名

        参数:
        symbol_name (str): 符号名
        symbol_type (str): 符号类型 ('FUTURES', 'SPOT')
        time_type (str): 时间类型 ('1d', '1m')
        iflower (bool): 如果为True，则将表名转换为小写

        返回:
            str: 生成的表名
        """
        # 根据 symbol_type 修改符号名
        if symbol_type == 'FUTURES':
            tb_symbol_name = symbol_name + '-F'
        elif symbol_type == 'SPOT':
            tb_symbol_name = symbol_name

        # 根据 time_type 修改符号名
        if time_type == '1d':
            tb_symbol_name = tb_symbol_name + '-D'
        elif time_type == '1m':
            tb_symbol_name = tb_symbol_name

        # 如果 iflower 为 True，则将符号名转换为小写
        if iflower:
            tb_symbol_name = tb_symbol_name.lower()

        return tb_symbol_name

    def momentum(self, price: pd.Series, periond: int) -> pd.Series:
        """
            取得動量
        """
        lagPrice = price.shift(periond)
        momen = price / lagPrice - 1
        return momen
    
    def get_newthink_symbol(self,all_symbols):
        for each_data in all_symbols:
            symbolname = each_data[0]
            data = each_data[1]

            print(symbolname)
            print(data)
            print('*'*120)

        return []
    def get_mtm_filter_symbol(self, all_symbols):
        """
            將過濾完的標的(can trade symobl)輸出
        """
        out_list = []
        for each_data in all_symbols:
            symbolname = each_data[0]
            data = each_data[1]
            # 不想要太新的商品
            if len(data) > 365:
                # 價格太低的商品不要
                if data.iloc[-1]['Close'] > 20:
                    mom_num = self.momentum(data['Close'], 30)
                    if mom_num.iloc[-1] > 0:
                        out_list.append(
                            [symbolname.split('-')[0].upper(), mom_num.iloc[-1]])

        sort_example = sorted(out_list, key=lambda x: x[1], reverse=True)
        # 這邊要改成可以直接輸出最後結果的就好
        return sort_example

    def get_volume_top_filter_symobl(self, all_symbols, max_symbols: int, last_sum_day:int = 10):
        """
            取得時間範圍內成交金額最高的前幾名
        Args:
            all_symbols (_type_): _description_            

        Returns:
            _type_: _description_
        """
        compare_dict = {}
        for each_data in all_symbols:
            symbolname = each_data[0]
            data = each_data[1]
            
            # 不想要太新的商品
            if len(data) > last_sum_day and len(data) > 90 :
                # 價格太少的也不要
                if data.tail(1).iloc[0]['Close'] < 0.01:continue

                filter_df = data.tail(last_sum_day)
                compare_dict.update({symbolname: sum(filter_df['Close'] * filter_df['Volume'])})
        
        sorted_compare_dict = sorted(
            compare_dict, key=compare_dict.get, reverse=True)
        
        return [symbolname.split('-')[0].upper() for symbolname in sorted_compare_dict[:max_symbols]]

    def change_min_postion(self, all_order_finally: dict, MinimumQuantity: dict):
        out_dict = {}
        for ID_phonenumber, value in all_order_finally.items():
            fix_value = self._change_min_postion(value, MinimumQuantity)
            out_dict[ID_phonenumber] = fix_value

        return out_dict

    def _change_min_postion(self, order_finally: dict, MinimumQuantity: dict):
        """ 
            args:
            MinimumQuantity(dict):取得最小單位限制
            {"SOLUSDT": 1.35}
            SOLUSDT 最小下單數量 1

            在 Binance 上，每種交易對都有最小下單數量限制，
            無論你是增加還是減少下單數量，都需要符合這個限制。
            這個限制是為了確保市場的流動性和穩定性。這也意味著，
            如果你想減少你的下單數量，那麼你減少後的數量仍然需要達到或超過這個最小限制。
            例如，如果一個交易對的最小下單數量限制是0.001，那麼你在更改下單數量時，
            無論是增加還是減少，你的下單數量都必須達到或超過0.001。
            你可以在 Binance 的「交易規則和費率」頁面上查詢每個交易對的最小下單數量限制。
        """
        out_dict = {}
        for symbol, ready_to_order_size in order_finally.items():
            # 取得 quantity數量
            order_quantity = abs(ready_to_order_size)

            if divmod(order_quantity, float(MinimumQuantity[symbol]))[0] != 0:
                filter_size = float(Decimal(str(int(
                    order_quantity / float(MinimumQuantity[symbol])))) * Decimal(str(float(MinimumQuantity[symbol]))))

                # 依然要還原方向
                out_dict.update({symbol: filter_size if order_quantity ==
                                ready_to_order_size else -1 * filter_size})

        return out_dict

    def Get_MIN_NOTIONAL(self, exchange_info: dict):
        """
            每個商品有自己的最小下單金額限制
            為了避免下單的波動設計了2倍的設計
        """
        out_dict = {}
        for symbol_info in exchange_info['symbols']:
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'MIN_NOTIONAL':
                    out_dict.update(
                        {symbol_info['symbol']: float(filter['notional'])*2})

        return out_dict

    def fix_data_different_len_and_na(self, df: pd.DataFrame):
        # 找出最長的歷史數據長度
        max_length = df.groupby('tic').count().max()

        # 取得所有時間
        all_times = set(df['date'])

        new_df = pd.DataFrame()

        print(df)
        # 對每個 tic 進行處理
        for tic in df['tic'].unique():
            # 找出當前 tic 的數據
            tic_data = df[df['tic'] == tic]

            # 取得當下資料的行數
            diff_times = all_times - set(tic_data['date'])
            dif_len = len(diff_times)

            # 如果需要填充
            if dif_len > 0:
                fill_data = pd.DataFrame({
                    'date': list(diff_times),
                    'tic': tic,
                    'open': np.nan,
                    'high': np.nan,
                    'low': np.nan,
                    'close': np.nan,
                    'volume': np.nan,
                })
                # 將填充用的 Series 添加到原始 DataFrame
                tic_data = pd.concat([tic_data, fill_data])

            # 補上虛擬資料
            tic_data = tic_data.sort_values(by=['tic', 'date'])
            tic_data = tic_data.ffill(axis=0)
            tic_data = tic_data.bfill(axis=0)

            new_df = pd.concat([new_df, tic_data])

        # 重新排序
        new_df = new_df.sort_values(by=['tic', 'date'])
        return new_df

    def filter_last_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            移除每個組最後一條記錄
        """
        def _remove_last(group):
            return group.iloc[:-1]

        return df.groupby('tic').apply(_remove_last).reset_index(drop=True)
