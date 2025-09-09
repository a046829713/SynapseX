import time
import typing
from utils import Debug_tool
from sqlalchemy import text
from sqlalchemy import create_engine, engine


class Router:
    _engine = None

    def __init__(self):
        """
        負責用來處理資料庫的連線。
        使用單例模式 (Singleton) 確保整個應用程式只使用一個 Engine 實例。
        """
        if Router._engine is None:
            Router._engine = create_engine(
                "mysql+pymysql://root:test@localhost:3306/crypto_data",
                pool_pre_ping=True,
                pool_recycle=3600
            )
            print("DB Engine 初始化成功")
        self.engine = Router._engine

    def get_mysql_conn(self) -> engine.base.Connection:
        """
        從 Engine 的連線池中取得一個新的連線。
        `pool_pre_ping=True` 會自動處理連線檢查和重連。
        Returns:
            engine.base.Connection: 一個來自連線池的可用連線。
        """
        return self.engine.connect()

    @property
    def mysql_conn(self):
        """
        使用 property，在每次拿取 connect 時，
        都從連線池取得一個新的連線。
        """
        return self.get_mysql_conn()
    
    @staticmethod
    def create_database_if_not_exists(db_name: str):
        """檢查資料庫是否存在，若不存在則創建"""
        """建立並返回一個不指定資料庫的引擎"""
        engine = create_engine("mysql+pymysql://root:test@localhost:3306")
        
        with engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name}"))
            conn.execute(text("commit"))
