import time
from Brain.Common.engine import EngineBase
import os
from Major.DataProvider import DataProvider


if __name__ == "__main__":
    first_date_map = DataProvider().get_symbol_first_day(
        symbol_type="FUTURES", time_type="1m"
    )

    # 建立回測引擎 (formal=False 為回測測試模式, formal=True 則鎖定 Meta-300B-30K.pt 正式模型)
    engine = EngineBase("ONE_TO_MANY", first_date_map, formal=False)

    test_data_dir = os.path.join(os.getcwd(), "Brain", "simulation", "test_data")
    test_symbols = os.listdir(test_data_dir)


    # 【選項 1: 預設】一次批次測試 Brain/DQN/Meta/ 下所有的 .pt 模型 (包含 Checkpoints 與 Meta-300B-30K.pt)
    engine.strategy_prepare(test_symbols)

    # 【選項 2: 自訂模型清單】僅測試指定模型 (例如單獨測試正式上線模型)
    # custom_models = [
    #     os.path.join("Brain", "DQN", "Meta", "Meta-300B-30K.pt"),
    #     os.path.join("Brain", "DQN", "Meta", "checkpoint-250.pt"),
    # ]
    # engine.strategy_prepare(test_symbols, model_paths=custom_models)

    # 【選項 3: 指定特定訓練目錄】
    # engine.strategy_prepare(test_symbols, model_dir=os.path.join("Brain", "DQN", "Meta"))

    # 執行回測評估、隔離輸出各模型圖表、並產出跨模型排行榜 (results/multi_model_summary.csv)
    summary_df = engine.analyze_result(ifplot=True)

