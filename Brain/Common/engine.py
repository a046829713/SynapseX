from Brain.DQN.lib.Backtest import Strategy, RL_evaluate, Backtest
from utils.AppSetting import AppSetting, RLConfig
import pandas as pd
import numpy as np
import re
import os
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import time


class EngineBase:
    def __init__(
        self,
        strategy_keyword: str,
        first_date_map: dict,
        formal=True,
    ) -> None:
        """
            負責用來協調Backtest與實盤策略生成

        Args:
            strategy_keyword (str): ONE_TO_MANY
            first_date_map (dict): 交易對首日字典
            formal (bool, optional): 是否為正式交易. Defaults to True.
        """
        self.strategy_keyword = strategy_keyword
        self.first_date_map = first_date_map

        self.setting = AppSetting.Trading_setting()
        self.formal = formal

        if not self.formal:
            self.config = RLConfig()

        self.model_paths: List[str] = []
        self.model_strategy_map: Dict[str, List[Strategy]] = {}
        self.strategys: List[Strategy] = []

    def get_if_order_map(self, df: pd.DataFrame) -> dict:
        """
        用來生成正式交易的訂單
        """
        if_order_map = {}
        for each_strategy in self.strategys:
            # 載入所需要的資料
            each_strategy.load_Real_time_data(
                df[df["tic"] == each_strategy.symbol_name]
            )

            re_evaluate = RL_evaluate(each_strategy, formal=self.formal)
            info = Backtest(re_evaluate, each_strategy).order_becktest(ifplot=False)

            if_order_map[each_strategy.symbol_name] = info["marketpostion_array"][-1]

        return if_order_map

    def _discover_models(self, model_dir: Optional[str] = None) -> List[str]:
        """
        自動搜尋目錄下的所有 .pt 模型檔案。
        預設搜尋 Brain/DQN/Meta/*.pt
        """
        if model_dir is None:
            model_dir = os.path.join("Brain", "DQN", "Meta")

        target_dir = Path(model_dir)
        if not target_dir.exists():
            print(
                f"Warning: Model directory {model_dir} not found. Fallback to default Meta model."
            )
            return [os.path.join("Brain", "DQN", "Meta", "Meta-300B-30K.pt")]

        pt_files = sorted(list(target_dir.glob("*.pt")))
        if not pt_files:
            print(
                f"Warning: No .pt files found in {model_dir}. Fallback to default Meta model."
            )
            return [os.path.join("Brain", "DQN", "Meta", "Meta-300B-30K.pt")]

        return [str(p) for p in pt_files]

    def _parse_model_path(self, model_path: str) -> Tuple[str, int, int, str]:
        """
        從模型路徑解析出所需資訊 (具備正則容錯與預設值備援機制)
        標準命名: Meta-300B-30K.pt -> ('Meta', 300, 30, 'DQN')
        檢查點命名: checkpoint-250.pt -> ('checkpoint-250', 300, 30, 'DQN')
        """
        path = Path(model_path)
        stem = path.stem

        # 解析 strategy_type (DQN, A2C, A3C, PPO2, DDPG, EIIE, Cot 等)
        strategy_type = "DQN"
        known_types = ["DQN", "A2C", "A3C", "PPO2", "DDPG", "EIIE", "Cot"]
        for part in path.parts:
            for kt in known_types:
                if kt.lower() == part.lower():
                    strategy_type = kt
                    break

        # 預設特徵長度與頻率週期
        feature_len = (
            getattr(self, "config", RLConfig()).BARS_COUNT if not self.formal else 300
        )
        data_len = AppSetting.engine_setting().get("FREQ_TIME", 30)
        info = stem

        # 嘗試正則匹配特徵與週期 (例如 300B, 30K, 30-Min)
        b_match = re.search(r"(\d+)B", stem, re.IGNORECASE)
        k_match = re.search(r"(\d+)K", stem, re.IGNORECASE)
        min_match = re.search(r"(\d+)-Min", stem, re.IGNORECASE)

        if b_match:
            feature_len = int(b_match.group(1))
        if k_match:
            data_len = int(k_match.group(1))
        elif min_match:
            data_len = int(min_match.group(1))

        # 若符合標準 3 區段命名 (info-feature-data)
        parts = stem.split("-")
        if len(parts) >= 3:
            info = parts[0]
            digits = [int(d) for d in re.findall(r"\d+", stem)]
            if len(digits) >= 2:
                feature_len = digits[0]
                data_len = digits[1]

        return info, int(feature_len), int(data_len), strategy_type

    def create_strategy(self, model_path: str, symbol: str) -> Strategy:
        info, feature_len, data_len, strategytype = self._parse_model_path(model_path)

        return Strategy(
            strategytype=strategytype,
            symbol_name=symbol,
            freq_time=int(data_len),
            model_feature_len=int(feature_len),
            fee=self.setting["BACKTEST_DEFAULT_COMMISSION_PERC"],
            slippage=self.setting["DEFAULT_SLIPPAGE"],
            model_count_path=model_path,
            symbol_first_trade_date=self.first_date_map.get(symbol),
            formal=True,
        )

    def create_strategy_from_csv(
        self, model_path: str, symbol_file_name: str
    ) -> Optional[Strategy]:
        """
        To use in backtest, not in formal environment
        """
        info, feature_len, data_len, strategytype = self._parse_model_path(model_path)
        symbol = symbol_file_name.split("-")[0]

        if symbol not in self.first_date_map.keys():
            return None

        # 建立 Strategy 實例
        strategy = Strategy(
            strategytype=strategytype,
            symbol_name=symbol,
            freq_time=int(data_len),
            model_feature_len=int(feature_len),
            fee=self.config.MODEL_DEFAULT_COMMISSION_PERC_TEST,
            slippage=self.config.DEFAULT_SLIPPAGE,
            model_count_path=model_path,
            symbol_first_trade_date=self.first_date_map[symbol],
            formal=False,
        )

        strategy.symbol_file_name = symbol_file_name

        # 載入資料
        data_path = os.path.join("Brain", "simulation", "test_data", symbol_file_name)
        if not os.path.exists(data_path):
            return None

        strategy.load_data(local_data_path=data_path)
        return strategy

    def strategy_prepare(
        self,
        targetsymbols: list,
        model_paths: Optional[list] = None,
        model_dir: Optional[str] = None,
    ):
        if self.strategy_keyword != "ONE_TO_MANY":
            raise ValueError("STRATEGY_KEYWORD didn't match, please check")

        self.model_strategy_map = {}
        self.strategys = []

        if self.formal:
            # 正式環境一律鎖定指定生產模型
            Meta_model_path = os.path.join("Brain", "DQN", "Meta", "Meta-300B-30K.pt")
            self.model_paths = [Meta_model_path]
            model_strats = []
            for symbol in targetsymbols:
                strat = self.create_strategy(Meta_model_path, symbol=symbol)
                model_strats.append(strat)
                self.strategys.append(strat)
            self.model_strategy_map[Meta_model_path] = model_strats
            print(
                f"[Formal Mode] Loaded production model: {Meta_model_path} with {len(model_strats)} symbols."
            )

        else:
            # 測試/回測環境：支援多模型或自動探索目錄下所有 .pt 檔案
            if model_paths is not None:
                self.model_paths = model_paths
            else:
                self.model_paths = self._discover_models(model_dir=model_dir)

            print(
                f"[Test Mode] Discovered {len(self.model_paths)} models for batch backtest."
            )

            for m_path in self.model_paths:
                m_strats = []
                for symbol_file_name in targetsymbols:
                    _strategy = self.create_strategy_from_csv(
                        m_path, symbol_file_name=symbol_file_name
                    )
                    if _strategy is not None:
                        m_strats.append(_strategy)
                        self.strategys.append(_strategy)

                self.model_strategy_map[m_path] = m_strats

                print(
                    f"  -> Model [{Path(m_path).stem}]: prepared {len(m_strats)} target symbol strategies."
                )

    def analyze_result(self, ifplot: bool = True) -> pd.DataFrame:
        """
        執行多模型 × 多幣種回測評估，輸出隔離目錄並生成跨模型比較總表
        """
        all_results = []

        for model_path, strategies in self.model_strategy_map.items():
            model_name = Path(model_path).stem
            print("\n" + "=" * 80)
            print(
                f"🚀 [Multi-Model Backtest] Evaluating Model: {model_name} (Total Symbols: {len(strategies)})"
            )
            print("=" * 80)

            for idx, each_strategy in enumerate(strategies, 1):
                try:
                    re_evaluate = RL_evaluate(each_strategy, formal=False)
                    backtester = Backtest(
                        re_evaluate, each_strategy, model_name=model_name
                    )
                    backtest_info = backtester.order_becktest(ifplot=ifplot)

                    res = {
                        "model_name": model_name,
                        "symbol": each_strategy.symbol_name,
                        "net_profit": backtest_info.get("net_profit", 0.0),
                        "return_pct": backtest_info.get("return_pct", 0.0),
                        "max_drawdown": backtest_info.get("max_drawdown", 0.0),
                        "total_trades": backtest_info.get("total_trades", 0),
                        "win_rate": backtest_info.get("win_rate", 0.0),
                        "final_equity": backtest_info.get("final_equity", 10000.0),
                    }
                    all_results.append(res)
                    print(
                        f"[{model_name}] ({idx}/{len(strategies)}) {each_strategy.symbol_name}: "
                        f"Return: {res['return_pct']:+.2f}% | MaxDD: {res['max_drawdown']:.2%} | Trades: {res['total_trades']} | WinRate: {res['win_rate']:.1f}%"
                    )
                except Exception as e:
                    print(
                        f"❌ Error evaluating {model_name} on {each_strategy.symbol_name}: {e}"
                    )

        # 生成跨模型績效匯總與排行榜
        summary_df = self._generate_summary_report(all_results)
        return summary_df

    def _generate_summary_report(self, all_results: list) -> pd.DataFrame:
        if not all_results:
            print("No backtest results to summarize.")
            return pd.DataFrame()

        df_results = pd.DataFrame(all_results)

        # 儲存全明細記錄
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        detail_path = results_dir / "all_symbols_detailed_results.csv"
        df_results.to_csv(detail_path, index=False)

        summary_df = self._summary_performance(df_results)

        # 依照平均報酬率降冪排序
        summary_df.sort_values(by="Mean_Return(%)", ascending=False, inplace=True)
        summary_df.reset_index(drop=True, inplace=True)

        summary_path = results_dir / "multi_model_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 95)
        print("🏆 MULTI-MODEL BATCH BACKTEST LEADERBOARD SUMMARY")
        print("=" * 95)
        print(summary_df.to_string(index=False))
        print("=" * 95)
        print(f"📁 Benchmark summary saved to: {summary_path}\n")

        return summary_df

    def _summary_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            caculate  all modes performance
        Args:
            df (pd.DataFrame): all_symbols_detailed_results

        
        """
        summary_rows = []
        for model_name, group in df.groupby("model_name"):
            summary_rows.append(
                {
                    "Model": model_name,
                    "Symbols_Tested": len(group),
                    "Mean_Return(%)": round(group["return_pct"].mean(), 8),
                    "Median_Return(%)": round(group["return_pct"].median(), 8),
                    "Total_Net_Profit": round(group["net_profit"].sum(), 8),
                    "Mean_Max_Drawdown(%)": round(
                        group["max_drawdown"].mean() * 100, 8
                    ),
                    "Avg_Win_Rate(%)": round(group["win_rate"].mean(), 8),
                    "Total_Trades": int(group["total_trades"].sum()),
                    "Profitable_Symbols_Count": int((group["net_profit"] > 0).sum()),
                    "Win_Symbols_Ratio(%)": round(
                        (group["net_profit"] > 0).mean() * 100, 8
                    ),
                }
            )

        return pd.DataFrame(summary_rows)
