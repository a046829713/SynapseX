import os

folder_path = r'C:\Users\Louis\Desktop\workSpace\mambaRL\SynapseX\Brain\simulation\data'
# 取得資料夾所有檔案與子資料夾名稱
all_entries = os.listdir(folder_path)


symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'ADAUSDT', 'ENAUSDT', 'LINKUSDT', 'HBARUSDT', 'LTCUSDT', 'XLMUSDT', 'WIFUSDT', 'BNBUSDT', 'ONDOUSDT', 'AAVEUSDT', 'WLDUSDT', 'AVAXUSDT', 'JUPUSDT', 'DOTUSDT', 'TRXUSDT', 'FILUSDT', 'ALGOUSDT', 'ZENUSDT', 'TIAUSDT', 'CRVUSDT', 'AGLDUSDT', 'POPCATUSDT', 'GALAUSDT', 'NEARUSDT']
# 如果只想要檔案（排除子資料夾），可以再過濾一下
file_names = [
    f for f in all_entries
    if os.path.isfile(os.path.join(folder_path, f)) and f.split('-')[0] in symbols
]


print(file_names)