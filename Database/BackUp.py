from Database.SQL_operate import SqlSentense
from Database import SQL_operate
from Database.router import Router
import os
from datetime import datetime
import pandas as pd
from sqlalchemy import text


class BasePreparator:
    def __init__(self, backup_folder="backup", log_folder='LogRecord'):
        self.backup_folder = backup_folder
        self.log_folder = log_folder
        self.SQL = SQL_operate.DB_operate()
        # 檢查資料庫是否存在
        self.checkIfDataBase()
        # 檢查所有需要的文件
        self.check_all_need_file()
        # 檢查所有需要的資料庫
        self.check_all_need_table()

    def checkIfDataBase(self):
        """
            用來檢查資料庫是否存在
        """
        try:
            connection = Router().mysql_conn
            with connection as conn:
                databases = conn.execute(
                    text("show databases like 'crypto_data';"))
                if databases.rowcount > 0:
                    pass
        except:
            Router.create_database_if_not_exists('crypto_data')

    def check_file(self, filename: str):
        """ 檢查檔案是否存在 否則創建 """
        if not os.path.exists(filename):
            os.mkdir(filename)

    def check_all_need_file(self):
        # 檢查備份資料夾是否存在
        self.check_file(self.backup_folder)
        # 檢查log資料夾是否存在
        self.check_file(self.log_folder)

    def check_all_need_table(self):
        getAllTablesName = self.SQL.get_db_data('show tables;')
        getAllTablesName = [y[0] for y in getAllTablesName]

        # 訂單的結果
        if 'orderresult' not in getAllTablesName:
            self.SQL.change_db_data(
                SqlSentense.createorderresult()
            )

        if 'lastinitcapital' not in getAllTablesName:
            self.SQL.change_db_data(SqlSentense.createlastinitcapital())
            self.SQL.change_db_data(
                f"""INSERT INTO `lastinitcapital` VALUES ('1',20000);""")

        if 'sysstatus' not in getAllTablesName:            
            # 將其更改為寫入DB
            self.SQL.change_db_data(SqlSentense.createsysstatus())
            self.SQL.change_db_data(
                f"""INSERT INTO `sysstatus` VALUES ('1','{str(datetime.now())}');""")

        if 'users' not in getAllTablesName:
            self.SQL.change_db_data(SqlSentense.createUsers())

        if 'interval_record' not in getAllTablesName:
            self.SQL.change_db_data(SqlSentense.createinterval_record())            
            self.SQL.change_db_data(                
                f"""INSERT INTO `interval_record` VALUES ('1','{str(datetime.now())}');""")
            
    def import_all_tables(self):
        """
            將資料全部寫入MySQL
        """
        # 將所有的資料讀取出來開始處理
        for file in os.listdir(self.backup_folder):
            if file.endswith(".csv"):
                table_name = file[:-4]
                full_file_path = os.path.join(
                    self.backup_folder, file)  # 獲取完整的文件路徑

                # 資料類的(開高低收的)
                if 'usdt' in table_name:
                    self.SQL.change_db_data(
                        SqlSentense.create_table_name(table_name))
                else:
                    #  系統類的
                    pass

                chunk_size = 50000  # or any other reasonable number
                for chunk in pd.read_csv(f"{self.backup_folder}/{file}", chunksize=chunk_size):
                    self.SQL.write_Dateframe(
                        chunk, table_name, exists='append', if_index=False)

                print(f"開始刪除:資料名稱:{table_name}")
                os.remove(full_file_path)


class DatabaseBackupRestore:
    def __init__(self):
        self.SQL = SQL_operate.DB_operate()

    def export_all_tables(self):
        """
            將資料全部從Mysql寫出
        """
        getAllTablesName = self.SQL.get_db_data('show tables;')
        getAllTablesName = [y[0] for y in getAllTablesName]
        for table in getAllTablesName:
            if 'user' in table:
                print(table)
            # df = self.SQL.read_Dateframe(table)
            # path = os.path.join(f"{self.backup_folder}", f"{table}.csv")
            # df.to_csv(path, index=False)
    
    def export_table_data(self, table_name: str):
        """
            將table資料匯出
        """
        df = self.SQL.read_Dateframe(table_name)
        df.to_csv(f"{table_name}.csv")