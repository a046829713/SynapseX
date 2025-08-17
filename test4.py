import pickle
import sys

# 確保 Python 可以找到你的 Brain 模組
# 如果你的專案根目錄不是 MambaRL，請相應地修改路徑
# sys.path.append('/path/to/your/MambaRL/project') 

# 從你的模組中導入必要的類別
# 這裡的路徑必須和你的專案結構一致
from Brain.Common.DataFeature import OriginalDataFrature

print("--- Starting Pickle Test ---")

# 1. 驗證 OriginalDataFrature 的模組路徑
#    我們期望看到 SynapseX.Brain.Common.DataFeature
print(f"OriginalDataFrature is defined in module: {OriginalDataFrature.__module__}")

# 2. 建立你的資料物件
try:
    symbolNames = ["BTCUSDT-F-30-Min"] # 只用一個來測試，加快速度
    data_feature_obj = OriginalDataFrature()
    all_data = data_feature_obj.get_train_net_work_data_by_path(symbolNames)
    print("Data created successfully.")
    
    # 提取其中的一個 Prices 物件來檢查
    first_symbol = symbolNames[0]
    prices_instance = all_data[first_symbol]
    
    # 3. 驗證 Prices 物件的類別路徑
    #    我們期望看到 SynapseX.Brain.Common.DataFeature.PricesObject 
    #    (如果你的 namedtuple 變數名是 PricesObject)
    print(f"The type of the data instance is: {type(prices_instance)}")
    print(f"The module of this type is: {type(prices_instance).__module__}")

except Exception as e:
    print(f"❌ Failed to create data: {e}")
    # 如果在這裡就出錯，表示你的 import 路徑或資料讀取有問題
    exit()


# 4. 嘗試 pickle 整個 all_data 字典
try:
    pickled_data = pickle.dumps(all_data)
    print("✅ SUCCESS: The 'all_data' dictionary was pickled successfully!")
    
    # 嘗試 unpickle
    unpickled_data = pickle.loads(pickled_data)
    print("✅ SUCCESS: The 'all_data' dictionary was unpickled successfully!")

except Exception as e:
    print(f"❌ FAILURE: Failed to pickle/unpickle the 'all_data' dictionary.")
    print(f"Error details: {e}")

print("--- Pickle Test Finished ---")