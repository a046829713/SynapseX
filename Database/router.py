import time
import typing
from utils import Debug_tool
from sqlalchemy import text
from sqlalchemy import create_engine, engine


class Router:
    def __init__(self):
        """
            負責用來處理資料庫的連線
        """
        self._mysql_conn = self.get_mysql_conn()

    def get_mysql_conn(self) -> engine.base.Connection:
        """    
        user: root
        password: test
        host: localhost
        port: 3306
        database: crypto_data
        如果有實體 IP，以上設定可以自行更改
        Returns:
            engine.base.Connection: _description_
        """
        self.address = "mysql+pymysql://root:test@localhost:3306"        
        engine = create_engine(self.address + '/crypto_data')
        connect = engine.connect()
        return connect

    def check_alive(self, connect: engine.base.Connection):
        """
        在每次使用之前，先確認 connect 是否活者
        """
        connect.execute(text("SELECT 1 + 1;"))

    def check_connect_alive(self, connect: engine.base.Connection, connect_func: typing.Callable):
        if connect:
            try:
                self.check_alive(connect)
                return connect
            except Exception as e:
                Debug_tool.debug.print_info(
                    f"{connect_func.__name__} reconnect error {e}")
                time.sleep(1)
                connect = self.reconnect(connect_func)
                return self.check_connect_alive(connect, connect_func)
        else:
            connect = self.reconnect(connect_func)
            return self.check_connect_alive(connect, connect_func)

    def reconnect(self, connect_func: typing.Callable) -> engine.base.Connection:
        """如果連線斷掉，重新連線"""
        try:
            connect = connect_func()
        except Exception as e:
            Debug_tool.debug.print_info(
                f"{connect_func.__name__} reconnect error {e}")
        return connect

    def check_mysql_conn_alive(self):
        self._mysql_conn = self.check_connect_alive(
            self._mysql_conn, self.get_mysql_conn)
        return self._mysql_conn

    @property
    def mysql_conn(self):
        """
        使用 property，在每次拿取 connect 時，
        都先經過 check alive 檢查 connect 是否活著

        """
        return self.check_mysql_conn_alive()
    
    @staticmethod
    def create_database_if_not_exists(db_name: str):
        """檢查資料庫是否存在，若不存在則創建"""
        """建立並返回一個不指定資料庫的引擎"""
        engine = create_engine("mysql+pymysql://root:test@localhost:3306")
        
        with engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name}"))
            conn.execute(text("commit"))

