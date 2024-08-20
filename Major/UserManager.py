from Database.SQL_operate import DB_operate


class UserManager():
    @staticmethod
    def GetAllUser():
        return DB_operate().get_db_data('select * from `users`;')

    @staticmethod
    def GetAccount_Passwd(custom_user):
        assert custom_user == 'author', 'GetAccount_Passwd only use with author'
        if custom_user == 'author':
            for each_row in UserManager.GetAllUser():
                if each_row[0][-6:] == '730876':
                    account = each_row[1]
                    passwd = each_row[2]

        return account, passwd
    
    @staticmethod
    def get_author_line_token():
        for each_row in UserManager.GetAllUser():
            if each_row[0][-6:] == '730876':
                return each_row[3]                    