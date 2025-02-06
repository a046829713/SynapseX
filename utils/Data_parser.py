import pandas as pd

class Datatransformer:    
    def mergeData(self, symbol_name: str, lastdata: pd.DataFrame, socketdata: dict):
        """合併資料用來

        Args:
            symbol_name (str) : "BTCUSDT" 
            lastdata (pd.DataFrame): 存在Trading_systeam裡面的更新資料
            socketdata (dict): 存在AsyncDataProvider裡面的即時資料

            [1678000140000, '46.77', '46.77', '46.76', '46.76', '6.597', 1678000199999, '308.51848', 9, '0.000', '0.00000', '0']]
        """
        # 先將catch 裡面的資料做轉換 # 由於當次分鐘量不會很大 所以決定不清空 考慮到異步問題

        if socketdata.get(symbol_name, None) is not None:
            if socketdata[symbol_name]:
                df = pd.DataFrame.from_dict(
                    socketdata[symbol_name], orient='index')
                df.reset_index(drop=True, inplace=True)
                df['Datetime'] = pd.to_datetime(df['Datetime'])
            else:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()

        # lastdata
        lastdata.reset_index(inplace=True)
        new_df = pd.concat([lastdata, df])

        new_df.set_index('Datetime', inplace=True)
        # duplicated >> 重複 True 代表重複了 # 過濾相同資料
        new_df = new_df[~new_df.index.duplicated(keep='last')]
        new_df = new_df.astype(float)

        return new_df, df
    
    def target_symobl(self, market_symobl: list, binance_catch: list):
        """
            用來將所有商品合併在一起,當市場狀況很差的時候,只監控比特幣
        Args:
            market_symobl (list): 取得目前要輪動交易的標的
            binance_catch (list): Binance 目前擁有部位的商品 (有部位的要繼續追蹤)
        """

        market_symobl.extend(binance_catch)

        if market_symobl:
            return list(set(market_symobl))
        else:
            return ['BTCUSDT']