import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from Brain.PPO2.lib.statistics import (
    natural_sort_key,
    compute_iqm,
    compute_deflated_sharpe,
    compute_cscv_pbo,
)


@dataclass
class GateConfig:
    """Configuration for model selection gates and leaderboard smoothing."""
    max_mdd_gate_pct: float = 35.0
    min_beat_bh_ratio: float = 0.50
    plateau_window: int = 2
    top_k: int = 5


@dataclass
class ModelSummaryRecord:
    """Summary record for a single model across evaluated symbols."""
    model_name: str
    model_path: str
    evaluated_symbols: int
    iqm_ir_2x: float
    iqm_ir_1x: float
    iqm_sharpe: float
    median_sharpe: float
    iqm_sortino: float
    median_mdd_pct: float
    p90_mdd_pct: float
    iqm_cum_return_pct: float
    beat_bh_ratio: float
    positive_return_ratio: float
    mean_exposure: float
    mean_daily_turnover: float
    gate_pass: bool
    gate_reason: str
    portfolio_daily_returns: Dict[str, float] = field(default_factory=dict)
    plateau_score: float = 0.0
    dsr_prob: float = 0.5


class LeaderboardManager:
    """
    Manages evaluation manifest caching, gatekeeper checks,
    plateau smoothing, DSR/PBO calculations, and leaderboard generation.
    """

    def __init__(self, gate_config: Optional[GateConfig] = None):
        self.config = gate_config if gate_config is not None else GateConfig()

    @staticmethod
    def atomic_write_json(path: str, payload: Dict[str, Any]):
        """Writes JSON payload atomically using a PID-tagged temporary file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)

    @staticmethod
    def atomic_write_csv(df: pd.DataFrame, path: str):
        """Writes DataFrame to CSV atomically using a PID-tagged temporary file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)

    @staticmethod
    def is_valid_manifest_cache(
        manifest_path: str,
        effective_start: Optional[str],
        effective_end: Optional[str],
        requested_symbols: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Validates whether existing manifest.json matches current evaluation parameters.
        """
        if not os.path.exists(manifest_path):
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or data.get("cache_version") != 2:
                return None
            if data.get("test_start_date", "") != str(effective_start or ""):
                return None
            if data.get("test_end_date", "") != str(effective_end or ""):
                return None
            if data.get("requested_symbols") != list(requested_symbols):
                return None
            if "symbols" not in data or "portfolio_daily_returns" not in data:
                return None
            return data
        except Exception:
            return None

    def evaluate_hard_gate(
        self,
        evaluated_symbols: int,
        mean_exposure: float,
        mean_daily_turnover: float,
        median_mdd_pct: float,
        beat_bh_ratio: float,
        iqm_ir_2x: float,
    ) -> Tuple[bool, str]:
        """
        Evaluates checkpoint against fail-fast hard gates.
        """
        if evaluated_symbols <= 0:
            return False, "FAIL:no_symbols"
        if mean_exposure < 0.01:
            return False, "FAIL:flat_collapse"
        if mean_daily_turnover < 0.005 and mean_exposure > 0.95:
            return False, "FAIL:passive_bh_collapse"
        if median_mdd_pct > self.config.max_mdd_gate_pct:
            return False, "FAIL:excessive_mdd"
        if beat_bh_ratio < self.config.min_beat_bh_ratio:
            return False, "FAIL:low_breadth"
        if iqm_ir_2x <= 0.0:
            return False, "FAIL:negative_ir_2x"
        return True, "PASS"

    def summarize_model_manifest(
        self,
        model_name: str,
        model_path: str,
        sym_dict: Dict[str, Dict[str, Any]],
        portfolio_daily_returns: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Aggregates per-symbol metrics into a consolidated model summary record.
        """
        sym_list = list(sym_dict.values())
        n_syms = len(sym_list)
        if n_syms == 0:
            gate_pass, gate_reason = self.evaluate_hard_gate(0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return {
                "model_name": model_name,
                "model_path": model_path,
                "evaluated_symbols": 0,
                "iqm_ir_2x": 0.0,
                "iqm_ir_1x": 0.0,
                "iqm_sharpe": 0.0,
                "median_sharpe": 0.0,
                "iqm_sortino": 0.0,
                "median_mdd_pct": 0.0,
                "p90_mdd_pct": 0.0,
                "iqm_cum_return_pct": 0.0,
                "beat_bh_ratio": 0.0,
                "positive_return_ratio": 0.0,
                "mean_exposure": 0.0,
                "mean_daily_turnover": 0.0,
                "gate_pass": gate_pass,
                "gate_reason": gate_reason,
                "portfolio_daily_returns": portfolio_daily_returns,
            }

        ir_2x_vals = [float(s.get("ir_2x", 0.0)) for s in sym_list]
        ir_1x_vals = [float(s.get("ir_1x", 0.0)) for s in sym_list]
        sharpe_vals = [float(s.get("sharpe", 0.0)) for s in sym_list]
        sortino_vals = [float(s.get("sortino", 0.0)) for s in sym_list]
        mdd_vals = [float(s.get("max_drawdown_pct", 0.0)) for s in sym_list]
        cum_ret_vals = [float(s.get("cum_return_pct", 0.0)) for s in sym_list]
        exp_vals = [
            float(s.get("mean_exposure", s.get("mean_action", 0.0))) for s in sym_list
        ]
        turn_vals = [float(s.get("daily_turnover", 0.0)) for s in sym_list]

        iqm_ir_2x = round(compute_iqm(ir_2x_vals), 4)
        iqm_ir_1x = round(compute_iqm(ir_1x_vals), 4)
        iqm_sharpe = round(compute_iqm(sharpe_vals), 4)
        median_sharpe = round(float(np.median(sharpe_vals)), 4)
        iqm_sortino = round(compute_iqm(sortino_vals), 4)
        median_mdd_pct = round(float(np.median(mdd_vals)), 4)
        p90_mdd_pct = round(float(np.percentile(mdd_vals, 90)), 4)
        iqm_cum_return_pct = round(compute_iqm(cum_ret_vals), 4)
        beat_bh_ratio = round(
            float(np.mean([1.0 if v > 0.0 else 0.0 for v in ir_1x_vals])), 4
        )
        positive_return_ratio = round(
            float(np.mean([1.0 if v > 0.0 else 0.0 for v in cum_ret_vals])), 4
        )
        mean_exposure = round(float(np.mean(exp_vals)), 4)
        mean_daily_turnover = round(float(np.mean(turn_vals)), 4)

        gate_pass, gate_reason = self.evaluate_hard_gate(
            evaluated_symbols=n_syms,
            mean_exposure=mean_exposure,
            mean_daily_turnover=mean_daily_turnover,
            median_mdd_pct=median_mdd_pct,
            beat_bh_ratio=beat_bh_ratio,
            iqm_ir_2x=iqm_ir_2x,
        )

        return {
            "model_name": model_name,
            "model_path": model_path,
            "evaluated_symbols": n_syms,
            "iqm_ir_2x": iqm_ir_2x,
            "iqm_ir_1x": iqm_ir_1x,
            "iqm_sharpe": iqm_sharpe,
            "median_sharpe": median_sharpe,
            "iqm_sortino": iqm_sortino,
            "median_mdd_pct": median_mdd_pct,
            "p90_mdd_pct": p90_mdd_pct,
            "iqm_cum_return_pct": iqm_cum_return_pct,
            "beat_bh_ratio": beat_bh_ratio,
            "positive_return_ratio": positive_return_ratio,
            "mean_exposure": mean_exposure,
            "mean_daily_turnover": mean_daily_turnover,
            "gate_pass": gate_pass,
            "gate_reason": gate_reason,
            "portfolio_daily_returns": portfolio_daily_returns,
        }

    def build_and_save_leaderboard(
        self,
        model_records: List[Dict[str, Any]],
        output_dir: str,
        top_k: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Builds, ranks, and saves the evaluation leaderboard with plateau smoothing,
        CSCV PBO, and Deflated Sharpe Ratio (DSR).
        """
        effective_top_k = int(top_k) if top_k is not None else self.config.top_k

        leaderboard_columns = [
            "rank",
            "model_name",
            "model_path",
            "gate_pass",
            "gate_reason",
            "ensemble_selected",
            "plateau_score",
            "iqm_ir_2x",
            "iqm_ir_1x",
            "iqm_sharpe",
            "median_sharpe",
            "iqm_sortino",
            "median_mdd_pct",
            "p90_mdd_pct",
            "iqm_cum_return_pct",
            "beat_bh_ratio",
            "positive_return_ratio",
            "mean_exposure",
            "mean_daily_turnover",
            "dsr_prob",
            "evaluated_symbols",
        ]

        if not model_records:
            empty_lb = pd.DataFrame(columns=leaderboard_columns)
            self.atomic_write_csv(empty_lb, os.path.join(output_dir, "leaderboard.csv"))
            summary = {
                "total_models_evaluated": 0,
                "passed_gate_count": 0,
                "effective_trials_n_eff": 1.0,
                "pbo_cscv": None,
                "selected_ensemble_models": [],
                "selected_ensemble_paths": [],
            }
            self.atomic_write_json(
                os.path.join(output_dir, "selection_summary.json"), summary
            )
            return empty_lb, summary

        # 1. Natural chronological sort before neighborhood smoothing
        ordered_records = sorted(
            model_records, key=lambda r: natural_sort_key(str(r["model_name"]))
        )
        n_models = len(ordered_records)
        raw_scores = np.array(
            [float(r["iqm_ir_2x"]) for r in ordered_records], dtype=np.float64
        )
        w = max(0, int(self.config.plateau_window))

        for k in range(n_models):
            lo = max(0, k - w)
            hi = min(n_models, k + w + 1)
            win_scores = raw_scores[lo:hi]
            if len(win_scores) <= 1:
                p_score = float(win_scores[0])
            else:
                med = float(np.median(win_scores))
                mad = float(np.median(np.abs(win_scores - med)))
                p_score = med - 0.5 * mad
            ordered_records[k]["plateau_score"] = round(p_score, 4)

        # 2. Build D x N daily portfolio return matrix
        all_days_set = set()
        for r in ordered_records:
            all_days_set.update(r.get("portfolio_daily_returns", {}).keys())
        sorted_days = sorted(list(all_days_set))

        if len(sorted_days) >= 2:
            daily_mat = np.zeros((len(sorted_days), n_models), dtype=np.float64)
            for k, r in enumerate(ordered_records):
                day_map = r.get("portfolio_daily_returns", {})
                for d_idx, day_str in enumerate(sorted_days):
                    daily_mat[d_idx, k] = float(day_map.get(day_str, 0.0))
            dsr_probs, n_eff = compute_deflated_sharpe(daily_mat)
            pbo_val = compute_cscv_pbo(daily_mat, n_partitions=8)
        else:
            dsr_probs = np.full(n_models, 0.5, dtype=np.float64)
            n_eff = 1.0
            pbo_val = None

        for k in range(n_models):
            ordered_records[k]["dsr_prob"] = float(dsr_probs[k])

        # 3. Multi-objective ranking
        ranked_records = sorted(
            ordered_records,
            key=lambda r: (
                bool(r["gate_pass"]),
                float(r["plateau_score"]),
                float(r["iqm_ir_2x"]),
                float(r["iqm_sharpe"]),
            ),
            reverse=True,
        )

        effective_top_k = max(1, effective_top_k)
        selected_count = 0
        rows_for_df: List[Dict[str, Any]] = []
        selected_names: List[str] = []
        selected_paths: List[str] = []

        for idx, rec in enumerate(ranked_records):
            is_selected = False
            if rec["gate_pass"] and selected_count < effective_top_k:
                is_selected = True
                selected_count += 1
                selected_names.append(str(rec["model_name"]))
                selected_paths.append(str(rec["model_path"]))

            rows_for_df.append({
                "rank": idx + 1,
                "model_name": rec["model_name"],
                "model_path": rec["model_path"],
                "gate_pass": bool(rec["gate_pass"]),
                "gate_reason": rec["gate_reason"],
                "ensemble_selected": is_selected,
                "plateau_score": rec["plateau_score"],
                "iqm_ir_2x": rec["iqm_ir_2x"],
                "iqm_ir_1x": rec["iqm_ir_1x"],
                "iqm_sharpe": rec["iqm_sharpe"],
                "median_sharpe": rec["median_sharpe"],
                "iqm_sortino": rec["iqm_sortino"],
                "median_mdd_pct": rec["median_mdd_pct"],
                "p90_mdd_pct": rec["p90_mdd_pct"],
                "iqm_cum_return_pct": rec["iqm_cum_return_pct"],
                "beat_bh_ratio": rec["beat_bh_ratio"],
                "positive_return_ratio": rec["positive_return_ratio"],
                "mean_exposure": rec["mean_exposure"],
                "mean_daily_turnover": rec["mean_daily_turnover"],
                "dsr_prob": rec["dsr_prob"],
                "evaluated_symbols": rec["evaluated_symbols"],
            })

        fallback_paths: List[str] = []
        if not selected_paths and ranked_records:
            fallback_paths = [
                str(r["model_path"])
                for r in ranked_records[: min(effective_top_k, len(ranked_records))]
            ]

        leaderboard_df = pd.DataFrame(rows_for_df, columns=leaderboard_columns)
        leaderboard_path = os.path.join(output_dir, "leaderboard.csv")
        self.atomic_write_csv(leaderboard_df, leaderboard_path)

        passed_count = int(sum(1 for r in ranked_records if r["gate_pass"]))
        summary_payload = {
            "total_models_evaluated": n_models,
            "passed_gate_count": passed_count,
            "effective_trials_n_eff": n_eff,
            "pbo_cscv": pbo_val,
            "top_k": effective_top_k,
            "selected_ensemble_models": selected_names,
            "selected_ensemble_paths": selected_paths,
            "fallback_top_paths": fallback_paths,
            "leaderboard_csv": leaderboard_path,
        }
        summary_path = os.path.join(output_dir, "selection_summary.json")
        self.atomic_write_json(summary_path, summary_payload)

        print(f"[PPO2Engine] Leaderboard saved to: {leaderboard_path} ({n_models} models ranked)")
        if pbo_val is not None:
            print(
                f"[PPO2Engine] Overfitting Diagnostics: PBO (CSCV) = {pbo_val:.2%}, "
                f"Effective Trials N_eff = {n_eff:.2f}/{n_models}"
            )
        return leaderboard_df, summary_payload
