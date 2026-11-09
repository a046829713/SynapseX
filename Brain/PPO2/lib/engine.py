import os
import gc
import glob
import time
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf, DictConfig
from gymnasium import spaces

from Brain.Common.DataFeature import OriginalDataFeature, Prices
from Brain.PPO2.lib.Agent import PPO2Agent
from Brain.PPO2.lib.environment import State_time_step
from Brain.PPO2.lib.Evaluator import PPO2Evaluator

# Modular sub-components
from Brain.PPO2.lib.statistics import (
    natural_sort_key as _natural_sort_key,
    norm_cdf as _norm_cdf,
    norm_ppf as _norm_ppf,
    compute_iqm as _compute_iqm,
    compute_deflated_sharpe as _compute_deflated_sharpe,
    compute_cscv_pbo as _compute_cscv_pbo,
    natural_sort_key,
    norm_cdf,
    norm_ppf,
    compute_iqm,
    compute_deflated_sharpe,
    compute_cscv_pbo,
)
from Brain.PPO2.lib.data_adapter import PPO2DataAdapter, PortfolioState
from Brain.PPO2.lib.leaderboard import LeaderboardManager, GateConfig


class PPO2Engine:
    """
    PPO2 Evaluation and Order Export Engine.
    High-level orchestrator coordinating multi-model and multi-symbol streaming evaluation,
    memory deallocation, manifest aggregation, and live order map generation.
    """

    def __init__(
        self,
        formal: bool = False,
        config: Optional[Union[DictConfig, str]] = None,
        device: Optional[torch.device] = None,
        base_data_dir: str = "Brain/simulation/test_data",
        test_start_date: Optional[str] = None,
        test_end_date: Optional[str] = None,
    ):
        self.formal = formal
        self.base_data_dir = base_data_dir
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # 1. Resolve configuration strictly without silent fallbacks
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

        # 2. Strict fail-fast binding of hyperparameters
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
        self.episode_steps = int(self.cfg.EPISODE_STEPS)
        self.init_log_std = float(getattr(self.cfg, "INIT_LOG_STD", 0.0))
        self.log_std_min = float(getattr(self.cfg, "LOG_STD_MIN", -5.0))
        self.log_std_max = float(getattr(self.cfg, "LOG_STD_MAX", 1.0))

        cfg_test_start = getattr(self.cfg, "TEST_START_DATE", None)
        cfg_test_end = getattr(self.cfg, "TEST_END_DATE", None)
        self.test_start_date = (
            test_start_date if test_start_date is not None else cfg_test_start
        )
        self.test_end_date = (
            test_end_date if test_end_date is not None else cfg_test_end
        )

        # 3. Leaderboard & selection configuration
        self.leaderboard_top_k = int(getattr(self.cfg, "LEADERBOARD_TOP_K", 5))
        self.plateau_window = int(getattr(self.cfg, "PLATEAU_WINDOW", 2))
        self.max_mdd_gate_pct = float(getattr(self.cfg, "MAX_MDD_GATE_PCT", 35.0))
        self.min_beat_bh_ratio = float(getattr(self.cfg, "MIN_BEAT_BH_RATIO", 0.50))

        # 4. Initialize specialized sub-managers
        self.gate_config = GateConfig(
            max_mdd_gate_pct=self.max_mdd_gate_pct,
            min_beat_bh_ratio=self.min_beat_bh_ratio,
            plateau_window=self.plateau_window,
            top_k=self.leaderboard_top_k,
        )
        self.leaderboard_manager = LeaderboardManager(self.gate_config)
        self.data_adapter = PPO2DataAdapter()

        self.model_paths: List[str] = []
        self.target_symbols: List[str] = []
        self.symbol_file_map: Dict[str, str] = {}
        self.agent: Optional[PPO2Agent] = None
        self.evaluator: Optional[PPO2Evaluator] = None
        self.last_leaderboard_df: Optional[pd.DataFrame] = None
        self.last_selection_summary: Optional[Dict[str, Any]] = None

    # -------------------------------------------------------------------------
    # Delegated & backward-compatible helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _atomic_write_json(path: str, payload: Dict[str, Any]):
        LeaderboardManager.atomic_write_json(path, payload)

    @staticmethod
    def _atomic_write_csv(df: pd.DataFrame, path: str):
        LeaderboardManager.atomic_write_csv(df, path)

    def _is_valid_manifest_cache(
        self,
        manifest_path: str,
        effective_start: Optional[str],
        effective_end: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        return self.leaderboard_manager.is_valid_manifest_cache(
            manifest_path, effective_start, effective_end, self.target_symbols
        )

    def _evaluate_hard_gate(
        self,
        evaluated_symbols: int,
        mean_exposure: float,
        mean_daily_turnover: float,
        median_mdd_pct: float,
        beat_bh_ratio: float,
        iqm_ir_2x: float,
    ) -> Tuple[bool, str]:
        return self.leaderboard_manager.evaluate_hard_gate(
            evaluated_symbols, mean_exposure, mean_daily_turnover,
            median_mdd_pct, beat_bh_ratio, iqm_ir_2x
        )

    def _summarize_model_manifest(
        self,
        model_name: str,
        model_path: str,
        sym_dict: Dict[str, Dict[str, Any]],
        portfolio_daily_returns: Dict[str, float],
    ) -> Dict[str, Any]:
        return self.leaderboard_manager.summarize_model_manifest(
            model_name, model_path, sym_dict, portfolio_daily_returns
        )

    def _build_and_save_leaderboard(
        self,
        model_records: List[Dict[str, Any]],
        output_dir: str,
        top_k: int = 5,
    ) -> pd.DataFrame:
        leaderboard_df, summary = self.leaderboard_manager.build_and_save_leaderboard(
            model_records=model_records, output_dir=output_dir, top_k=top_k
        )
        self.last_leaderboard_df = leaderboard_df
        self.last_selection_summary = summary
        return leaderboard_df

    def _process_df_to_prices(
        self, feature_loader: OriginalDataFeature, df: pd.DataFrame
    ) -> Prices:
        return self.data_adapter.process_df_to_prices(feature_loader, df)

    def _extract_latest_obs(
        self,
        prices: Prices,
        position: float = 0.0,
        open_price: float = 0.0,
        portfolio_state: Optional[Dict[str, Any]] = None,
        current_position: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        return self.data_adapter.extract_latest_obs(
            prices=prices,
            bars_count=self.bars_count,
            position=position,
            open_price=open_price,
            portfolio_state=portfolio_state,
            current_position=current_position,
        )

    def _resolve_symbol_portfolio_state(
        self,
        symbol: str,
        portfolio_state: Optional[Dict[str, Any]],
    ) -> Tuple[float, float]:
        state = self.data_adapter.resolve_portfolio_state(symbol, portfolio_state)
        return state.position, state.open_price

    def _apply_deadzone_and_rebalance(
        self, raw_action: float, prev_position: float
    ) -> float:
        clipped_action = float(np.clip(raw_action, 0.0, 1.0))
        candidate_pos = 0.0 if clipped_action <= self.deadzone_threshold else clipped_action
        prev_pos = float(np.clip(prev_position, 0.0, 1.0))
        if prev_pos == 0.0 or candidate_pos == 0.0:
            return candidate_pos
        if abs(candidate_pos - prev_pos) < self.rebalance_threshold:
            return prev_pos
        return candidate_pos

    # -------------------------------------------------------------------------
    # Checkpoint Discovery & Preparation
    # -------------------------------------------------------------------------
    @staticmethod
    def _resolve_watch_model_dir(model_dir: Optional[str] = None) -> Optional[str]:
        if model_dir is not None and str(model_dir).strip() != "":
            return model_dir
        saves_root = Path("saves")
        if saves_root.exists() and saves_root.is_dir():
            subdirs = [d for d in saves_root.iterdir() if d.is_dir()]
            if subdirs:
                latest_dir = max(subdirs, key=lambda d: (d.stat().st_mtime, d.name))
                return str(latest_dir)
        return None

    def _discover_models(self, model_dir: Optional[str] = None) -> List[str]:
        if model_dir is not None:
            target_path = Path(model_dir)
            if target_path.is_file() and target_path.suffix == ".pt":
                return [str(target_path)]
            if not target_path.exists():
                return []
            pt_files = sorted(
                list(target_path.glob("**/*.pt")), key=lambda p: natural_sort_key(p.name)
            )
            return [str(p) for p in pt_files]

        if self.formal:
            meta_path = os.path.join("Brain", "PPO2", "Meta", "policy_PPO2.pt")
            if os.path.exists(meta_path):
                return [meta_path]

        saves_pt = glob.glob(os.path.join("saves", "**", "*.pt"), recursive=True)
        meta_pt = glob.glob(os.path.join("Brain", "PPO2", "Meta", "*.pt"), recursive=True)
        all_pt = sorted(list(set(saves_pt + meta_pt)), key=natural_sort_key)
        return all_pt

    def _build_symbol_file_map(self) -> Dict[str, str]:
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
        self.symbol_file_map = self._build_symbol_file_map()

        # 1. Resolve target symbols
        if target_symbols is not None and len(target_symbols) > 0:
            resolved_symbols = []
            for s in target_symbols:
                base_s = s.split("-")[0].replace(".csv", "")
                if base_s in self.symbol_file_map or s in self.symbol_file_map:
                    resolved_symbols.append(base_s)
                else:
                    resolved_symbols.append(base_s)
                    self.symbol_file_map[base_s] = s if s.endswith(".csv") else f"{s}.csv"
            self.target_symbols = sorted(list(set(resolved_symbols)))
        else:
            unique_bases = {
                fname.split("-")[0].replace(".csv", "")
                for fname in self.symbol_file_map.values()
            }
            self.target_symbols = sorted(list(unique_bases))

        # 2. Resolve model paths
        if model_paths is not None and len(model_paths) > 0:
            discovered = sorted(model_paths, key=natural_sort_key)
        else:
            discovered = self._discover_models(model_dir=model_dir)

        self.model_paths = discovered

        print(
            f"[PPO2Engine] Prepared: {len(self.target_symbols)} symbols, "
            f"{len(self.model_paths)} models."
        )

    def build_agent(self) -> PPO2Agent:
        dummy_state = State_time_step(
            bars_count=self.bars_count,
            commission_perc=self.commission_perc,
            model_train=False,
            default_slippage=self.default_slippage,
            N_steps=self.episode_steps,
            deadzone_threshold=self.deadzone_threshold,
            rebalance_threshold=self.rebalance_threshold,
            w1_step_return=self.w1_step_return,
        )
        ob_space = spaces.Dict({
            "states": spaces.Box(
                low=-np.inf, high=np.inf, shape=dummy_state.getStateShape(), dtype=np.float32
            ),
            "time_states": spaces.Box(
                low=-np.inf, high=np.inf, shape=dummy_state.getTimeShape(), dtype=np.float32
            ),
        })
        ac_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        ssm_cfg = None
        if hasattr(self.cfg, "SSM_CFG") and self.cfg.SSM_CFG is not None:
            if isinstance(self.cfg.SSM_CFG, (dict, DictConfig)):
                ssm_cfg = OmegaConf.to_container(self.cfg.SSM_CFG, resolve=True)
            else:
                ssm_cfg = self.cfg.SSM_CFG

        return PPO2Agent(
            ob_space=ob_space,
            ac_space=ac_space,
            device=self.device,
            seq_dim=self.bars_count,
            hidden_size=self.hidden_size,
            nlayers=self.nlayers,
            time_features_out=self.time_features_out,
            init_log_std=self.init_log_std,
            portfolio_dim=self.portfolio_dim,
            ssm_cfg=ssm_cfg,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
        )

    def load_checkpoint(self, agent: PPO2Agent, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            agent.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            agent.model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Unrecognized checkpoint format at {checkpoint_path}")
        agent.model.eval()

    # -------------------------------------------------------------------------
    # Evaluation Execution Pipeline
    # -------------------------------------------------------------------------
    def _stream_evaluate_symbol(
        self,
        symbol: str,
        evaluator: PPO2Evaluator,
        feature_loader: OriginalDataFeature,
        model_output_dir: str,
        effective_start: Optional[str],
        effective_end: Optional[str],
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        """Evaluates a single symbol and extracts order metrics."""
        symbol_fname = self.symbol_file_map.get(symbol, f"{symbol}.csv")
        symbol_path = os.path.join(self.base_data_dir, symbol_fname)
        if not os.path.exists(symbol_path):
            return None

        df = pd.read_csv(symbol_path)
        if len(df) <= 360:
            return None

        prices = self._process_df_to_prices(feature_loader, df)
        raw_df = feature_loader.df
        out_csv = os.path.join(model_output_dir, f"{symbol}_orders.csv")

        order_df = evaluator.evaluate_symbol(
            symbol=symbol,
            raw_df=raw_df,
            prices=prices,
            output_csv_path=out_csv,
            test_start_date=effective_start,
            test_end_date=effective_end,
        )

        del prices, raw_df
        if order_df.empty:
            del order_df
            gc.collect()
            return None

        metrics = PPO2Evaluator.compute_symbol_metrics(
            order_df=order_df,
            commission_perc=self.commission_perc,
            default_slippage=self.default_slippage,
        )
        del order_df
        gc.collect()

        sym_daily_map = metrics.pop("daily_returns", {})
        metrics["order_file"] = out_csv
        return metrics, sym_daily_map

    def _eval_model_checkpoint(
        self,
        model_idx: int,
        model_path: str,
        output_dir: str,
        effective_start: Optional[str],
        effective_end: Optional[str],
        use_cache: bool,
        min_file_age_sec: float,
        feature_loader: OriginalDataFeature,
        manifest_rows: List[Dict[str, Any]],
        model_records: List[Dict[str, Any]],
    ):
        """Processes and evaluates a single model checkpoint."""
        model_name = Path(model_path).stem
        model_output_dir = os.path.join(output_dir, model_name)
        manifest_json_path = os.path.join(model_output_dir, "manifest.json")

        # 1. Incremental cache check
        if use_cache:
            cached = self._is_valid_manifest_cache(
                manifest_json_path, effective_start, effective_end
            )
            if cached is not None:
                print(
                    f"[{model_idx + 1}/{len(self.model_paths)}] Cached Model: "
                    f"{model_name} (loaded from manifest.json)"
                )
                cached_symbols = cached.get("symbols", {})
                for sym_key, sym_stats in cached_symbols.items():
                    manifest_rows.append({
                        "model_name": model_name, "symbol": sym_key, **sym_stats
                    })
                model_records.append(
                    self._summarize_model_manifest(
                        model_name=model_name,
                        model_path=str(model_path),
                        sym_dict=cached_symbols,
                        portfolio_daily_returns=cached.get("portfolio_daily_returns", {}),
                    )
                )
                return

        # 2. File age guard
        if not os.path.exists(model_path):
            return
        if min_file_age_sec > 0.0:
            try:
                age = time.time() - os.path.getmtime(model_path)
                if age < min_file_age_sec:
                    print(
                        f"[PPO2Engine] Deferring recently modified checkpoint "
                        f"'{model_name}' (age={age:.2f}s < {min_file_age_sec:.2f}s) to next cycle."
                    )
                    return
            except OSError:
                return

        print(
            f"\n[{model_idx + 1}/{len(self.model_paths)}] Testing Model: "
            f"{model_name} ({model_path})"
        )

        # 3. Model construction & loading with fail-safe guards
        agent = self.build_agent()
        try:
            self.load_checkpoint(agent, model_path)
        except (EOFError, zipfile.BadZipFile, OSError, RuntimeError, ValueError, KeyError) as exc:
            msg = str(exc)
            if min_file_age_sec > 0.0 and ("unexpected EOF" in msg or "PytorchStreamReader" in msg):
                print(f"[PPO2Engine] Warning: Checkpoint '{model_path}' still being written; deferring.")
            else:
                print(f"[PPO2Engine] Warning: Skipping incompatible checkpoint '{model_path}': {exc}")
            del agent
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return

        evaluator = PPO2Evaluator(
            agent=agent,
            bars_count=self.bars_count,
            device=self.device,
            deadzone_threshold=self.deadzone_threshold,
            rebalance_threshold=self.rebalance_threshold,
            commission_perc=self.commission_perc,
            default_slippage=self.default_slippage,
            w1_step_return=self.w1_step_return,
            episode_steps=self.episode_steps,
            test_start_date=effective_start,
            test_end_date=effective_end,
        )

        os.makedirs(model_output_dir, exist_ok=True)
        model_symbols_dict: Dict[str, Dict[str, Any]] = {}
        daily_sum_by_date: Dict[str, float] = {}
        daily_cnt_by_date: Dict[str, int] = {}

        for symbol in self.target_symbols:
            eval_res = self._stream_evaluate_symbol(
                symbol=symbol,
                evaluator=evaluator,
                feature_loader=feature_loader,
                model_output_dir=model_output_dir,
                effective_start=effective_start,
                effective_end=effective_end,
            )
            if eval_res is None:
                continue

            metrics, sym_daily_map = eval_res
            for day_str, day_ret in sym_daily_map.items():
                daily_sum_by_date[day_str] = daily_sum_by_date.get(day_str, 0.0) + float(day_ret)
                daily_cnt_by_date[day_str] = daily_cnt_by_date.get(day_str, 0) + 1

            model_symbols_dict[symbol] = metrics
            manifest_rows.append({
                "model_name": model_name, "symbol": symbol, **metrics
            })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        portfolio_daily_returns = {
            day: round(daily_sum_by_date[day] / max(daily_cnt_by_date[day], 1), 6)
            for day in sorted(daily_sum_by_date.keys())
        }

        model_manifest = {
            "cache_version": 2,
            "model_name": model_name,
            "model_path": str(model_path),
            "test_start_date": str(effective_start or ""),
            "test_end_date": str(effective_end or ""),
            "commission_perc": float(self.commission_perc),
            "default_slippage": float(self.default_slippage),
            "requested_symbols": list(self.target_symbols),
            "total_symbols_evaluated": len(model_symbols_dict),
            "portfolio_daily_returns": portfolio_daily_returns,
            "symbols": model_symbols_dict,
        }
        self._atomic_write_json(manifest_json_path, model_manifest)
        print(f"--> Saved model manifest: {manifest_json_path} ({len(model_symbols_dict)} symbols)")

        model_records.append(
            self._summarize_model_manifest(
                model_name=model_name,
                model_path=str(model_path),
                sym_dict=model_symbols_dict,
                portfolio_daily_returns=portfolio_daily_returns,
            )
        )

        del agent, evaluator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_test(
        self,
        output_dir: str = "results/ppo2_orders",
        test_start_date: Optional[str] = None,
        test_end_date: Optional[str] = None,
        top_k: Optional[int] = None,
        use_cache: bool = True,
        min_file_age_sec: float = 0.0,
    ) -> pd.DataFrame:
        """
        Streaming multi-model batch test execution with O(1) memory footprint.
        """
        manifest_columns = [
            "model_name", "symbol", "total_bars", "mean_action", "std_action",
            "long_ratio", "short_ratio", "flat_ratio", "order_file", "sharpe",
            "sortino", "ir_1x", "ir_2x", "max_drawdown_pct", "cum_return_pct",
            "bh_return_pct", "daily_turnover",
        ]
        if not self.model_paths:
            print("[PPO2Engine] Warning: No models to test. Please provide model_paths or model_dir.")
            return pd.DataFrame(columns=manifest_columns)

        effective_start = test_start_date if test_start_date is not None else self.test_start_date
        effective_end = test_end_date if test_end_date is not None else self.test_end_date
        effective_top_k = int(top_k) if top_k is not None else self.leaderboard_top_k

        os.makedirs(output_dir, exist_ok=True)
        manifest_rows: List[Dict[str, Any]] = []
        model_records: List[Dict[str, Any]] = []
        feature_loader = OriginalDataFeature()

        for model_idx, model_path in enumerate(self.model_paths):
            self._eval_model_checkpoint(
                model_idx=model_idx,
                model_path=model_path,
                output_dir=output_dir,
                effective_start=effective_start,
                effective_end=effective_end,
                use_cache=use_cache,
                min_file_age_sec=min_file_age_sec,
                feature_loader=feature_loader,
                manifest_rows=manifest_rows,
                model_records=model_records,
            )

        manifest_df = (
            pd.DataFrame(manifest_rows) if manifest_rows else pd.DataFrame(columns=manifest_columns)
        )
        # Ensure only manifest_columns are exported to consolidated CSV
        exported_cols = [c for c in manifest_columns if c in manifest_df.columns]
        manifest_df = manifest_df[exported_cols]
        consolidated_path = os.path.join(output_dir, "multi_model_test_manifest.csv")
        self._atomic_write_csv(manifest_df, consolidated_path)
        print(f"\n[PPO2Engine] Consolidated manifest saved to: {consolidated_path}")

        self._build_and_save_leaderboard(
            model_records=model_records, output_dir=output_dir, top_k=effective_top_k
        )
        return manifest_df

    # -------------------------------------------------------------------------
    # Sidecar Watcher Mode
    # -------------------------------------------------------------------------
    def watch_and_evaluate(
        self,
        output_dir: str = "results/ppo2_orders",
        model_dir: Optional[str] = None,
        target_symbols: Optional[List[str]] = None,
        poll_interval: float = 30.0,
        top_k: Optional[int] = None,
        min_file_age_sec: float = 2.0,
        max_idle_polls: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Periodically scans for new checkpoints and updates leaderboard incrementally.
        """
        effective_top_k = int(top_k) if top_k is not None else self.leaderboard_top_k
        poll_sec = max(0.5, float(poll_interval))
        idle_polls = 0
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 70)
        print(f"[PPO2Engine] Starting Sidecar Watcher Mode (poll_interval={poll_sec}s)")
        print(f"[PPO2Engine] Output Directory: {output_dir}")
        print("[PPO2Engine] Press Ctrl+C at any time to stop watching cleanly.")
        print("=" * 70)

        try:
            while True:
                active_model_dir = self._resolve_watch_model_dir(model_dir)
                self.prepare(target_symbols=target_symbols, model_dir=active_model_dir)

                pending_models = []
                now_ts = time.time()
                for m_path in self.model_paths:
                    if not os.path.exists(m_path):
                        continue
                    if min_file_age_sec > 0.0:
                        try:
                            if (now_ts - os.path.getmtime(m_path)) < min_file_age_sec:
                                continue
                        except OSError:
                            continue
                    m_name = Path(m_path).stem
                    m_manifest = os.path.join(output_dir, m_name, "manifest.json")
                    if self._is_valid_manifest_cache(m_manifest, self.test_start_date, self.test_end_date) is None:
                        pending_models.append(m_path)

                leaderboard_file = os.path.join(output_dir, "leaderboard.csv")
                if pending_models or (self.model_paths and not os.path.exists(leaderboard_file)):
                    print(
                        f"\n[Watcher] Detected {len(pending_models)} new checkpoint(s) "
                        f"in '{active_model_dir or 'default'}'. Updating leaderboard..."
                    )
                    self.run_test(
                        output_dir=output_dir,
                        top_k=effective_top_k,
                        use_cache=True,
                        min_file_age_sec=min_file_age_sec,
                    )
                    idle_polls = 0
                else:
                    idle_polls += 1
                    print(
                        f"[Watcher] No new checkpoints (total tracked: {len(self.model_paths)}). "
                        f"Idle poll #{idle_polls}..."
                    )

                if max_idle_polls is not None and idle_polls >= int(max_idle_polls):
                    print(f"[Watcher] Reached max_idle_polls={max_idle_polls}. Exiting watcher.")
                    break
                time.sleep(poll_sec)
        except KeyboardInterrupt:
            print("\n[Watcher] Interrupted by user (Ctrl+C). Leaderboard state is safely saved.")

        if self.last_leaderboard_df is not None:
            return self.last_leaderboard_df
        leaderboard_file = os.path.join(output_dir, "leaderboard.csv")
        return pd.read_csv(leaderboard_file) if os.path.exists(leaderboard_file) else pd.DataFrame()

    # -------------------------------------------------------------------------
    # Formal Live Mode Order Map
    # -------------------------------------------------------------------------
    def _collect_symbol_observations(
        self,
        data_source: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]],
        portfolio_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Tuple[Dict[str, np.ndarray], float]]:
        """Preprocesses inputs and extracts observation dicts per symbol."""
        feature_loader = OriginalDataFeature()
        symbol_obs_map: Dict[str, Tuple[Dict[str, np.ndarray], float]] = {}

        if data_source is None:
            for symbol in self.target_symbols:
                symbol_fname = self.symbol_file_map.get(symbol, f"{symbol}.csv")
                symbol_path = os.path.join(self.base_data_dir, symbol_fname)
                if not os.path.exists(symbol_path):
                    continue
                df = pd.read_csv(symbol_path)
                prices = self._process_df_to_prices(feature_loader, df)
                pos, open_p = self._resolve_symbol_portfolio_state(symbol, portfolio_state)
                obs = self._extract_latest_obs(prices, position=pos, open_price=open_p)
                symbol_obs_map[symbol] = (obs, pos)

        elif isinstance(data_source, pd.DataFrame):
            sym_col = next((c for c in ["tic", "Symbol", "symbol"] if c in data_source.columns), None)
            if sym_col is not None:
                for sym in data_source[sym_col].unique():
                    sym_str = str(sym)
                    sym_df = data_source[data_source[sym_col] == sym]
                    prices = self._process_df_to_prices(feature_loader, sym_df)
                    pos, open_p = self._resolve_symbol_portfolio_state(sym_str, portfolio_state)
                    obs = self._extract_latest_obs(prices, position=pos, open_price=open_p)
                    symbol_obs_map[sym_str] = (obs, pos)
            else:
                sym_name = self.target_symbols[0] if self.target_symbols else "DEFAULT"
                prices = self._process_df_to_prices(feature_loader, data_source)
                pos, open_p = self._resolve_symbol_portfolio_state(sym_name, portfolio_state)
                obs = self._extract_latest_obs(prices, position=pos, open_price=open_p)
                symbol_obs_map[sym_name] = (obs, pos)

        elif isinstance(data_source, dict):
            for sym, sym_df in data_source.items():
                sym_str = str(sym)
                prices = self._process_df_to_prices(feature_loader, sym_df)
                pos, open_p = self._resolve_symbol_portfolio_state(sym_str, portfolio_state)
                obs = self._extract_latest_obs(prices, position=pos, open_price=open_p)
                symbol_obs_map[sym_str] = (obs, pos)

        return symbol_obs_map

    def _predict_single_model_orders(
        self, symbol_obs_map: Dict[str, Tuple[Dict[str, np.ndarray], float]]
    ) -> Dict[str, float]:
        """Generates target action orders using a single checkpoint."""
        if self.agent is None:
            self.agent = self.build_agent()
            if self.model_paths:
                self.load_checkpoint(self.agent, self.model_paths[0])
            else:
                formal_model_path = os.path.join("Brain", "PPO2", "Meta", "policy_PPO2.pt")
                if os.path.exists(formal_model_path):
                    self.load_checkpoint(self.agent, formal_model_path)
                else:
                    print(
                        "[PPO2Engine] Warning: No model checkpoint found for formal mode. "
                        "Running with initialized weights."
                    )

        evaluator = PPO2Evaluator(
            agent=self.agent,
            bars_count=self.bars_count,
            device=self.device,
            deadzone_threshold=self.deadzone_threshold,
            rebalance_threshold=self.rebalance_threshold,
            commission_perc=self.commission_perc,
            default_slippage=self.default_slippage,
            w1_step_return=self.w1_step_return,
            episode_steps=self.episode_steps,
        )

        order_map: Dict[str, float] = {}
        for sym, (obs, pos) in symbol_obs_map.items():
            action_val = evaluator.evaluate_single_step(
                obs, apply_deadzone=True, prev_position=pos
            )
            order_map[sym] = round(action_val, 4)
        return order_map

    def _predict_ensemble_orders(
        self, symbol_obs_map: Dict[str, Tuple[Dict[str, np.ndarray], float]]
    ) -> Dict[str, float]:
        """Generates target action orders via continuous Action Ensembling across checkpoints."""
        print(
            f"[PPO2Engine] Running Formal Action Ensemble across {len(self.model_paths)} checkpoints..."
        )
        accum_actions: Dict[str, List[float]] = {sym: [] for sym in symbol_obs_map}
        ensemble_agent = self.build_agent()

        for m_path in self.model_paths:
            try:
                self.load_checkpoint(ensemble_agent, m_path)
            except (RuntimeError, ValueError, KeyError, OSError) as exc:
                print(f"[PPO2Engine] Warning: Skipping checkpoint '{m_path}' in ensemble: {exc}")
                continue

            evaluator = PPO2Evaluator(
                agent=ensemble_agent,
                bars_count=self.bars_count,
                device=self.device,
                deadzone_threshold=self.deadzone_threshold,
                rebalance_threshold=self.rebalance_threshold,
                commission_perc=self.commission_perc,
                default_slippage=self.default_slippage,
                w1_step_return=self.w1_step_return,
                episode_steps=self.episode_steps,
            )
            for sym, (obs, pos) in symbol_obs_map.items():
                raw_act = evaluator.evaluate_single_step(
                    obs, apply_deadzone=False, prev_position=pos
                )
                accum_actions[sym].append(float(raw_act))

        del ensemble_agent
        gc.collect()

        order_map: Dict[str, float] = {}
        for sym, (obs, pos) in symbol_obs_map.items():
            preds = accum_actions.get(sym, [])
            mean_raw = float(np.mean(preds)) if preds else 0.0
            final_act = self._apply_deadzone_and_rebalance(mean_raw, pos)
            order_map[sym] = round(final_act, 4)
        return order_map

    def get_if_order_map(
        self,
        data_source: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
        output_path: str = "results/ppo2_orders/formal_if_order_map.json",
        portfolio_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Formal Live Mode: Generates target order map {symbol: action} in [0.0, 1.0].
        """
        symbol_obs_map = self._collect_symbol_observations(data_source, portfolio_state)

        if len(self.model_paths) > 1:
            if_order_map = self._predict_ensemble_orders(symbol_obs_map)
        else:
            if_order_map = self._predict_single_model_orders(symbol_obs_map)

        self._atomic_write_json(output_path, if_order_map)
        print(f"[PPO2Engine] Saved formal live order map to: {output_path} ({len(if_order_map)} symbols)")
        return if_order_map
