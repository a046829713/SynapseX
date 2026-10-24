import os
import gc
import re
import glob
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf, DictConfig
from gymnasium import spaces

from Brain.Common.DataFeature import OriginalDataFeature, Prices
from Brain.PPO2.lib.Agent import PPO2Agent
from Brain.PPO2.lib.environment import State_time_step
from Brain.PPO2.lib.Evaluator import PPO2Evaluator


def _natural_sort_key(s: str):
    """
    Helper key for natural numerical sorting of checkpoint files (e.g. iter_10 before iter_100).
    """
    parts = re.split(r"(\d+)", s)
    return [int(text) if text.isdigit() else text.lower() for text in parts]


class PPO2Engine:
    """
    PPO2 Evaluation and Order Export Engine.
    Coordinates multi-model and multi-symbol streaming evaluation, memory deallocation,
    manifest aggregation, and generates live order maps for formal execution.
    """

    def __init__(
        self,
        formal: bool = False,
        config: Optional[Union[DictConfig, str]] = None,
        device: Optional[torch.device] = None,
        base_data_dir: str = "Brain/simulation/test_data",
    ):
        self.formal = formal
        self.base_data_dir = base_data_dir
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Resolve configuration strictly without silent fallbacks
        if config is None:
            config_path = "configs/training/ppo_v2.yaml"
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            raw_cfg = OmegaConf.load(config_path)
            self.cfg = raw_cfg.training if hasattr(raw_cfg, "training") else raw_cfg
            self.config_path = config_path
        elif isinstance(config, str):
            if not os.path.exists(config):
                raise FileNotFoundError(f"Configuration file not found: {config}")
            raw_cfg = OmegaConf.load(config)
            self.cfg = raw_cfg.training if hasattr(raw_cfg, "training") else raw_cfg
            self.config_path = config
        elif isinstance(config, DictConfig):
            self.cfg = config.training if hasattr(config, "training") else config
            self.config_path = getattr(config, "config_path", "DictConfig")
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        # Strict fail-fast binding of all model and environment hyperparameters
        self.bars_count = int(self.cfg.BARS_COUNT)
        self.deadzone_threshold = float(self.cfg.DEADZONE_THRESHOLD)
        self.rebalance_threshold = float(self.cfg.REBALANCE_THRESHOLD)
        self.commission_perc = float(self.cfg.MODEL_DEFAULT_COMMISSION_PERC_TRAING)
        self.default_slippage = float(self.cfg.DEFAULT_SLIPPAGE)
        self.hidden_size = int(self.cfg.HIDDEN_SIZE)
        self.nlayers = int(self.cfg.NLAYERS)
        self.time_features_out = int(self.cfg.TIME_FEATURES_OUT)
        self.portfolio_dim = int(self.cfg.PORTFOLIO_DIM)
        self.w1_step_return = float(self.cfg.W1_STEP_RETURN)
        self.vol_window = int(self.cfg.VOL_WINDOW)
        self.risk_lambda = float(self.cfg.RISK_LAMBDA)
        self.reward_clip = float(self.cfg.REWARD_CLIP)
        self.min_sigma = float(getattr(self.cfg, "MIN_ASSET_SIGMA", 1e-3))
        self.episode_steps = int(self.cfg.EPISODE_STEPS)

        self.model_paths: List[str] = []
        self.target_symbols: List[str] = []
        self.symbol_file_map: Dict[str, str] = {}
        self.agent: Optional[PPO2Agent] = None
        self.evaluator: Optional[PPO2Evaluator] = None

    def _discover_models(self, model_dir: Optional[str] = None) -> List[str]:
        """
        Discovers model checkpoints.
        Priority:
        1. Explicit model_dir if passed
        2. Formal mode: Brain/PPO2/Meta/policy_PPO2.pt or latest saves/*.pt
        3. Test mode: All .pt files under saves/ or Brain/PPO2/Meta/
        """
        if model_dir is not None:
            target_path = Path(model_dir)
            if target_path.is_file() and target_path.suffix == ".pt":
                return [str(target_path)]
            pt_files = sorted(list(target_path.glob("**/*.pt")), key=lambda p: _natural_sort_key(p.name))
            return [str(p) for p in pt_files]

        if self.formal:
            meta_path = os.path.join("Brain", "PPO2", "Meta", "policy_PPO2.pt")
            if os.path.exists(meta_path):
                return [meta_path]

        # Scan saves/ directory recursively for all .pt checkpoints
        saves_pt = glob.glob(os.path.join("saves", "**", "*.pt"), recursive=True)
        meta_pt = glob.glob(os.path.join("Brain", "PPO2", "Meta", "*.pt"), recursive=True)
        all_pt = saves_pt + meta_pt
        all_pt = sorted(list(set(all_pt)), key=_natural_sort_key)
        return all_pt

    def _build_symbol_file_map(self) -> Dict[str, str]:
        """
        Builds a map from base symbol (e.g. BTCUSDT) to its CSV filename.
        """
        mapping = {}
        if not os.path.exists(self.base_data_dir):
            return mapping

        for fname in os.listdir(self.base_data_dir):
            if fname.endswith(".csv"):
                base_symbol = fname.split("-")[0].replace(".csv", "")
                mapping[base_symbol] = fname
                mapping[fname] = fname
        return mapping

    def prepare(
        self,
        target_symbols: Optional[List[str]] = None,
        model_paths: Optional[List[str]] = None,
        model_dir: Optional[str] = None,
    ):
        """
        Prepares symbols and model checkpoint paths.
        """
        self.symbol_file_map = self._build_symbol_file_map()

        # 1. Resolve symbols
        if target_symbols is not None and len(target_symbols) > 0:
            resolved_symbols = []
            for s in target_symbols:
                base_s = s.split("-")[0].replace(".csv", "")
                if base_s in self.symbol_file_map:
                    resolved_symbols.append(base_s)
                elif s in self.symbol_file_map:
                    resolved_symbols.append(base_s)
                else:
                    # Keep as-is, attempting direct filename resolution later
                    resolved_symbols.append(base_s)
                    self.symbol_file_map[base_s] = s if s.endswith(".csv") else f"{s}.csv"
            self.target_symbols = sorted(list(set(resolved_symbols)))
        else:
            # Default to all available symbols in test_data
            unique_bases = set()
            for fname in self.symbol_file_map.values():
                base_s = fname.split("-")[0].replace(".csv", "")
                unique_bases.add(base_s)
            self.target_symbols = sorted(list(unique_bases))

        # 2. Resolve model paths
        if model_paths is not None and len(model_paths) > 0:
            self.model_paths = sorted(model_paths, key=_natural_sort_key)
        else:
            self.model_paths = self._discover_models(model_dir=model_dir)

        print(
            f"[PPO2Engine] Prepared: {len(self.target_symbols)} symbols, {len(self.model_paths)} models."
        )

    def build_agent(self) -> PPO2Agent:
        """
        Dynamically constructs a PPO2Agent matching the training environment space.
        """
        dummy_state = State_time_step(
            bars_count=self.bars_count,
            commission_perc=self.commission_perc,
            model_train=False,
            default_slippage=self.default_slippage,
            N_steps=self.episode_steps,
            deadzone_threshold=self.deadzone_threshold,
            rebalance_threshold=self.rebalance_threshold,
            w1_step_return=self.w1_step_return,
            vol_window=self.vol_window,
            risk_lambda=self.risk_lambda,
            reward_clip=self.reward_clip,
            min_sigma=self.min_sigma,
        )
        ob_space = spaces.Dict({
            "states": spaces.Box(
                low=-np.inf, high=np.inf, shape=dummy_state.getStateShape(), dtype=np.float32
            ),
            "time_states": spaces.Box(
                low=-np.inf, high=np.inf, shape=dummy_state.getTimeShape(), dtype=np.float32
            ),
        })
        ac_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        ssm_cfg = None
        if hasattr(self.cfg, "SSM_CFG") and self.cfg.SSM_CFG is not None:
            if isinstance(self.cfg.SSM_CFG, (dict, DictConfig)):
                ssm_cfg = OmegaConf.to_container(self.cfg.SSM_CFG, resolve=True)
            else:
                ssm_cfg = self.cfg.SSM_CFG

        agent = PPO2Agent(
            ob_space=ob_space,
            ac_space=ac_space,
            device=self.device,
            seq_dim=self.bars_count,
            hidden_size=self.hidden_size,
            nlayers=self.nlayers,
            time_features_out=self.time_features_out,
            portfolio_dim=self.portfolio_dim,
            ssm_cfg=ssm_cfg,
        )
        return agent

    def load_checkpoint(self, agent: PPO2Agent, checkpoint_path: str):
        """
        Loads weights from a .pt file into the given PPO2Agent.
        Supports both wrapped checkpoint dictionaries and raw state dictionaries.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            agent.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            agent.model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Unrecognized checkpoint format at {checkpoint_path}")
        agent.model.eval()

    def _extract_latest_obs(
        self, prices: Prices, current_position: float = 0.0
    ) -> Dict[str, np.ndarray]:
        """
        Extracts the latest observation window of length self.bars_count for formal live mode.
        """
        N = len(prices.close)
        if N < self.bars_count:
            raise ValueError(
                f"Data length {N} is shorter than required window {self.bars_count}"
            )

        start = N - self.bars_count
        end = N

        info_list = [
            "log_open", "log_high", "log_low", "log_close", "log_volume",
            "log_quote_av", "log_trades", "log_tb_base_av", "log_tb_quote_av",
            "log_ma_30", "log_ma_60", "log_ma_120", "log_ma_240", "log_ma_360",
        ]
        timelist = [
            "age_log_minutes", "age_years", "month_sin", "month_cos",
            "day_sin", "day_cos", "hour_sin", "hour_cos", "minute_sin",
            "minute_cos", "dayofweek_sin", "dayofweek_cos", "week_sin", "week_cos",
        ]

        data_res = np.zeros(shape=(self.bars_count, len(info_list) + 3), dtype=np.float32)
        time_res = np.zeros(shape=(self.bars_count, len(timelist)), dtype=np.float32)

        for idx, field in enumerate(info_list):
            data_res[:, idx] = getattr(prices, field)[start:end]

        data_res[:, len(info_list)] = current_position
        data_res[:, len(info_list) + 1] = 0.0
        data_res[:, len(info_list) + 2] = 0.0

        for idx, field in enumerate(timelist):
            time_res[:, idx] = getattr(prices, field)[start:end]

        return {"states": data_res, "time_states": time_res}

    def run_test(self, output_dir: str = "results/ppo2_orders") -> pd.DataFrame:
        """
        Executes streaming batch evaluation across models and symbols with O(1) memory footprint.
        Exports per-symbol CSV orders, per-model manifest.json, and consolidated multi_model_test_manifest.csv.
        """
        if not self.model_paths:
            print("[PPO2Engine] Warning: No models to test. Please provide model_paths or model_dir.")
            return pd.DataFrame()

        os.makedirs(output_dir, exist_ok=True)
        manifest_rows: List[Dict[str, Any]] = []
        feature_loader = OriginalDataFeature()

        for model_idx, model_path in enumerate(self.model_paths):
            model_name = Path(model_path).stem
            print(f"\n[{model_idx + 1}/{len(self.model_paths)}] Testing Model: {model_name} ({model_path})")

            agent = self.build_agent()
            self.load_checkpoint(agent, model_path)
            evaluator = PPO2Evaluator(
                agent=agent,
                bars_count=self.bars_count,
                device=self.device,
                deadzone_threshold=self.deadzone_threshold,
                rebalance_threshold=self.rebalance_threshold,
                commission_perc=self.commission_perc,
                default_slippage=self.default_slippage,
                w1_step_return=self.w1_step_return,
                vol_window=self.vol_window,
                risk_lambda=self.risk_lambda,
                reward_clip=self.reward_clip,
                min_sigma=self.min_sigma,
                episode_steps=self.episode_steps,
            )

            model_output_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            model_manifest: Dict[str, Any] = {
                "model_name": model_name,
                "total_symbols_evaluated": 0,
                "symbols": {},
            }

            for sym_idx, symbol in enumerate(self.target_symbols):
                symbol_fname = self.symbol_file_map.get(symbol, f"{symbol}.csv")
                symbol_path = os.path.join(self.base_data_dir, symbol_fname)

                if not os.path.exists(symbol_path):
                    continue

                df = pd.read_csv(symbol_path)
                prices = self._process_df_to_prices(feature_loader, df)
                raw_df = feature_loader.df

                out_csv = os.path.join(model_output_dir, f"{symbol}_orders.csv")
                order_df = evaluator.evaluate_symbol(
                    symbol=symbol,
                    raw_df=raw_df,
                    prices=prices,
                    output_csv_path=out_csv,
                )

                actions = order_df["Action"].values
                positions = order_df["Position"].values
                total_bars = len(actions)

                mean_action = float(np.mean(actions)) if total_bars > 0 else 0.0
                std_action = float(np.std(actions)) if total_bars > 0 else 0.0
                long_ratio = float(np.mean(positions > 0)) if total_bars > 0 else 0.0
                short_ratio = float(np.mean(positions < 0)) if total_bars > 0 else 0.0
                flat_ratio = float(np.mean(positions == 0)) if total_bars > 0 else 0.0

                sym_stats = {
                    "total_bars": total_bars,
                    "mean_action": round(mean_action, 4),
                    "std_action": round(std_action, 4),
                    "long_ratio": round(long_ratio, 4),
                    "short_ratio": round(short_ratio, 4),
                    "flat_ratio": round(flat_ratio, 4),
                    "order_file": out_csv,
                }
                model_manifest["symbols"][symbol] = sym_stats

                manifest_rows.append({
                    "model_name": model_name,
                    "symbol": symbol,
                    "total_bars": total_bars,
                    "mean_action": round(mean_action, 4),
                    "std_action": round(std_action, 4),
                    "long_ratio": round(long_ratio, 4),
                    "short_ratio": round(short_ratio, 4),
                    "flat_ratio": round(flat_ratio, 4),
                    "order_file": out_csv,
                })

                # Explicit streaming memory deallocation
                del prices, raw_df, order_df
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            model_manifest["total_symbols_evaluated"] = len(model_manifest["symbols"])
            manifest_json_path = os.path.join(model_output_dir, "manifest.json")
            with open(manifest_json_path, "w", encoding="utf-8") as f:
                json.dump(model_manifest, f, indent=2)

            print(f"--> Saved model manifest: {manifest_json_path} ({model_manifest['total_symbols_evaluated']} symbols)")

            # Explicit deallocation of model and evaluator
            del agent, evaluator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        manifest_df = pd.DataFrame(manifest_rows)
        consolidated_manifest_path = os.path.join(output_dir, "multi_model_test_manifest.csv")
        manifest_df.to_csv(consolidated_manifest_path, index=False)
        print(f"\n[PPO2Engine] Consolidated manifest saved to: {consolidated_manifest_path}")

        return manifest_df

    def _process_df_to_prices(
        self, feature_loader: OriginalDataFeature, df: pd.DataFrame
    ) -> Prices:
        df_copy = df.copy()
        if "Datetime" in df_copy.columns:
            df_copy.set_index("Datetime", inplace=True)
        cleaned = feature_loader.cleanData(df_copy)

        # Defensive handling for missing volume auxiliary columns
        if "Volume" not in cleaned.columns and "volume" in cleaned.columns:
            cleaned["Volume"] = cleaned["volume"]
        if "quote_av" not in cleaned.columns:
            cleaned["quote_av"] = cleaned.get("Volume", 1.0) * cleaned.get("Close", 1.0)
        if "trades" not in cleaned.columns:
            cleaned["trades"] = cleaned.get("Volume", 1.0) * 0.1
        if "tb_base_av" not in cleaned.columns:
            cleaned["tb_base_av"] = cleaned.get("Volume", 1.0) * 0.5
        if "tb_quote_av" not in cleaned.columns:
            cleaned["tb_quote_av"] = cleaned["quote_av"] * 0.5

        if "log_ma_360" not in cleaned.columns:
            cleaned = feature_loader.add_average_metric(cleaned)
        for p in [30, 60, 120, 240, 360]:
            if f"log_ma_{p}" not in cleaned.columns:
                cleaned[f"log_ma_{p}"] = 0.0

        if "age_log_minutes" not in cleaned.columns:
            cleaned = feature_loader.add_time_feature(cleaned)
        for t_col in [
            "age_log_minutes", "age_years", "month_sin", "month_cos",
            "day_sin", "day_cos", "hour_sin", "hour_cos", "minute_sin",
            "minute_cos", "dayofweek_sin", "dayofweek_cos", "week_sin", "week_cos",
        ]:
            if t_col not in cleaned.columns:
                cleaned[t_col] = 0.0

        feature_loader.df = cleaned
        return feature_loader.load_relative()

    def get_if_order_map(
        self,
        data_source: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
        output_path: str = "results/ppo2_orders/formal_if_order_map.json",
    ) -> Dict[str, float]:
        """
        Formal Live Mode: Generates target order map {symbol: action} for immediate live execution by TradingSysteam.py.
        """
        # Ensure agent is instantiated and weights are loaded
        if self.agent is None:
            self.agent = self.build_agent()
            if self.model_paths:
                self.load_checkpoint(self.agent, self.model_paths[0])
            else:
                formal_model_path = os.path.join("Brain", "PPO2", "Meta", "policy_PPO2.pt")
                if os.path.exists(formal_model_path):
                    self.load_checkpoint(self.agent, formal_model_path)
                else:
                    print("[PPO2Engine] Warning: No model checkpoint found for formal mode. Running with initialized weights.")

        evaluator = PPO2Evaluator(
            agent=self.agent,
            bars_count=self.bars_count,
            device=self.device,
            deadzone_threshold=self.deadzone_threshold,
            rebalance_threshold=self.rebalance_threshold,
            commission_perc=self.commission_perc,
            default_slippage=self.default_slippage,
            w1_step_return=self.w1_step_return,
            vol_window=self.vol_window,
            risk_lambda=self.risk_lambda,
            reward_clip=self.reward_clip,
            min_sigma=self.min_sigma,
            episode_steps=self.episode_steps,
        )

        feature_loader = OriginalDataFeature()
        if_order_map: Dict[str, float] = {}

        if data_source is None:
            # Load the latest bar data from base_data_dir for configured target_symbols
            for symbol in self.target_symbols:
                symbol_fname = self.symbol_file_map.get(symbol, f"{symbol}.csv")
                symbol_path = os.path.join(self.base_data_dir, symbol_fname)
                if not os.path.exists(symbol_path):
                    continue
                df = pd.read_csv(symbol_path)
                prices = self._process_df_to_prices(feature_loader, df)
                obs = self._extract_latest_obs(prices)
                action_val = evaluator.evaluate_single_step(obs, apply_deadzone=True)
                if_order_map[symbol] = round(action_val, 4)

        elif isinstance(data_source, pd.DataFrame):
            # Check symbol column
            sym_col = None
            for col in ["tic", "Symbol", "symbol"]:
                if col in data_source.columns:
                    sym_col = col
                    break

            if sym_col is not None:
                symbols = data_source[sym_col].unique()
                for sym in symbols:
                    sym_df = data_source[data_source[sym_col] == sym]
                    prices = self._process_df_to_prices(feature_loader, sym_df)
                    obs = self._extract_latest_obs(prices)
                    action_val = evaluator.evaluate_single_step(obs, apply_deadzone=True)
                    if_order_map[str(sym)] = round(action_val, 4)
            else:
                # Single symbol dataframe
                sym_name = self.target_symbols[0] if self.target_symbols else "DEFAULT"
                prices = self._process_df_to_prices(feature_loader, data_source)
                obs = self._extract_latest_obs(prices)
                action_val = evaluator.evaluate_single_step(obs, apply_deadzone=True)
                if_order_map[sym_name] = round(action_val, 4)

        elif isinstance(data_source, dict):
            for sym, sym_df in data_source.items():
                prices = self._process_df_to_prices(feature_loader, sym_df)
                obs = self._extract_latest_obs(prices)
                action_val = evaluator.evaluate_single_step(obs, apply_deadzone=True)
                if_order_map[sym] = round(action_val, 4)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(if_order_map, f, indent=4)

        print(f"[PPO2Engine] Saved formal live order map to: {output_path} ({len(if_order_map)} symbols)")
        return if_order_map
