from Database import router
from sqlalchemy import text
from utils import Debug_tool
import pandas as pd


class DB_operate():
    def get_db_data(self, text_msg: str) -> list:
        """
            專門用於select from
        """
        try:
            self.userconn = router.Router().mysql_conn
            with self.userconn as conn:

                result = conn.execute(
                    text(text_msg)
                )
                # 資料範例{'Date': '2022/07/01', 'Time': '09:25:00', 'Open': '470', 'High': '470', 'Low': '470', 'Close': '470', 'Volume': '10'}

                return list(result)
        except:
            Debug_tool.debug.print_info()

    def change_db_data(self, text_msg: str) -> None:
        """ 用於下其他指令
        Args:
            text_msg (str): SQL_Query
        Returns:
            None
        """
        try:
            self.userconn = router.Router().mysql_conn
            with self.userconn as conn:
                conn.execute(text(text_msg))
                conn.commit()  # 在这里调用commit来确保更改被保存
        except:
            Debug_tool.debug.print_info()

    def read_Dateframe(self, text_msg: str) -> pd.DataFrame:
        """
            to get pandas Dateframe
            symbol_name: 'btcusdt-f'
        """
        try:
            self.userconn = router.Router().mysql_conn
            with self.userconn as conn:
                return pd.read_sql(text_msg, con=conn)
        except:
            Debug_tool.debug.print_info()

    def write_Dateframe(self, df: pd.DataFrame, symbol_name: str, exists='replace', if_index=True) -> pd.DataFrame:
        """
            # 資料庫配置：某些資料庫預設配置可能會要求明確提交交易。
            to write pandas Dateframe
            symbol_name or tablename: 'btcusdt-f'
        """
        try:
            self.userconn = router.Router().mysql_conn
            with self.userconn as conn:
                df.to_sql(symbol_name, con=conn,
                          if_exists=exists, index=if_index)
                conn.commit()
        except:
            Debug_tool.debug.print_info()


class SqlSentense():
    @staticmethod
    def createUsers() -> str:
        sql_query = """
            CREATE TABLE users (
                phone_number VARCHAR(255) PRIMARY KEY NOT NULL,
                binance_api_account VARCHAR(255) NOT NULL,
                binance_api_passwd VARCHAR(255) NOT NULL,
                line_token VARCHAR(255) NOT NULL
            );

        """
        return sql_query
    
    @staticmethod
    def insert_Users(result: dict) -> str:
        """
        Args:
            result (dict): 一个包含新用户信息的字典，如：
                        {
                            'phone_number': '1234567890',
                            'binance_api_account': 'new_account',
                            'binance_api_passwd': 'new_passwd',
                            'line_token': 'new_token'
                        }
        
        Returns:
            str: 返回要执行的SQL语句
        """

        sql_query = f"""
                    INSERT INTO `users` 
                    (`phone_number`, `binance_api_account`, `binance_api_passwd`, `line_token`)
                    VALUES 
                    ('{result['phone_number']}', '{result['binance_api_account']}', 
                    '{result['binance_api_passwd']}', '{result['line_token']}');
                    """
        print(sql_query)
        return sql_query


    @staticmethod
    def create_table_name(tb_symbol_name: str) -> str:
        """ to create 1 min"""
        sql_query = f"""CREATE TABLE `crypto_data`.`{tb_symbol_name}`(
                `Datetime` DATETIME NOT NULL,
                `Open` FLOAT NOT NULL,
                `High` FLOAT NOT NULL,
                `Low` FLOAT NOT NULL,
                `Close` FLOAT NOT NULL,
                `Volume` FLOAT NOT NULL,
                `close_time` FLOAT NOT NULL,
                `quote_av` FLOAT NOT NULL,
                `trades` FLOAT NOT NULL,
                `tb_base_av` FLOAT NOT NULL,
                `tb_quote_av` FLOAT NOT NULL,
                `ignore` FLOAT NOT NULL,
                PRIMARY KEY(`Datetime`)
                );"""

        return sql_query

    @staticmethod
    def createlastinitcapital() -> str:
        sql_query = """CREATE TABLE `lastinitcapital` (
        `ID` varchar(255) NOT NULL,
        `capital` int NOT NULL,
        PRIMARY KEY (`ID`)
        );"""

        return sql_query

    @staticmethod
    def createsysstatus() -> str:
        sql_query = """CREATE TABLE `crypto_data`.`sysstatus`(`ID` varchar(255) NOT NULL,`systeam_datetime` varchar(255) NOT NULL,PRIMARY KEY(`ID`));"""
        return sql_query

    @staticmethod
    def createorderresult() -> str:
        sql_query = """
                CREATE TABLE `crypto_data`.`orderresult`(
                    `orderId` BIGINT NOT NULL,
                    `order_info` TEXT NOT NULL,
                    PRIMARY KEY(`orderId`)
                );
            """
        return sql_query

    @staticmethod
    def createinterval_record() -> str:
        sql_query = """
                CREATE TABLE `crypto_data`.`interval_record`(
                    `ID` varchar(255) NOT NULL,
                    `lastportfolioadjustmenttime` varchar(255) NOT NULL,
                    PRIMARY KEY(`ID`));
            """
        return sql_query




