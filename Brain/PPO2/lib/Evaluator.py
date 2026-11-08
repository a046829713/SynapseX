import os
from typing import Optional, Dict, Any, List, Union
import numpy as np
import pandas as pd
import torch
from Brain.Common.DataFeature import Prices
from Brain.PPO2.lib.Agent import PPO2Agent
from Brain.PPO2.lib.environment import State_time_step, ProductionEnv


class PPO2Evaluator:
    """
    Dedicated lightweight deterministic evaluator for continuous PPO2 policies.
    Completely decoupled from financial PnL accounting, charts, and plotting libraries.
    Generates standardized order CSVs (Option B) for external backtesting frameworks.
    """

    def __init__(
        self,
        agent: PPO2Agent,
        bars_count: int = 300,
        device: Optional[torch.device] = None,
        deadzone_threshold: float = 0.01,
        rebalance_threshold: float = 0.03,
        commission_perc: float = 0.00045,
        default_slippage: float = 0.00025,
        w1_step_return: float = 1.0,
        episode_steps: int = 500,
        test_start_date: Optional[str] = None,
        test_end_date: Optional[str] = None,
    ):
        self.agent = agent
        self.bars_count = bars_count
        self.device = device or agent.device
        self.deadzone_threshold = deadzone_threshold
        self.rebalance_threshold = rebalance_threshold
        self.commission_perc = commission_perc
        self.default_slippage = default_slippage
        self.w1_step_return = w1_step_return
        self.episode_steps = episode_steps
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date

        # Force evaluation mode (deterministic inference)
        self.agent.model.eval()

    def evaluate_single_step(
        self,
        obs: Dict[str, Any],
        apply_deadzone: bool = True,
        prev_position: Optional[float] = None,
    ) -> float:
        """
        Evaluates a single observation window for formal live trading mode.
        Returns continuous deterministic target position in [0.0, 1.0],
        applying deadzone_threshold and rebalance_threshold for parity with ProductionEnv.
        """
        self.agent.model.eval()
        with torch.no_grad():
            out = self.agent.get_action(obs, stochastic=False)
            raw_val = (
                float(out.clipped_action.item())
                if out.clipped_action is not None
                else float(out.action.item())
            )
            clipped_action = float(np.clip(raw_val, 0.0, 1.0))

            if not apply_deadzone:
                return clipped_action

            if clipped_action <= self.deadzone_threshold:
                candidate_pos = 0.0
            else:
                candidate_pos = clipped_action

            if prev_position is not None:
                prev_pos = float(np.clip(float(prev_position), 0.0, 1.0))
            elif "states" in obs and np.asarray(obs["states"]).shape[-1] >= 3:
                states_arr = np.asarray(obs["states"])
                prev_pos = float(np.clip(float(states_arr[..., -1, -3].item()), 0.0, 1.0))
            else:
                prev_pos = 0.0

            if prev_pos == 0.0 or candidate_pos == 0.0:
                target_pos = candidate_pos
            elif abs(candidate_pos - prev_pos) < self.rebalance_threshold:
                target_pos = prev_pos
            else:
                target_pos = candidate_pos

            return float(target_pos)

    def evaluate_symbol(
        self,
        symbol: str,
        raw_df: pd.DataFrame,
        prices: Prices,
        output_csv_path: Optional[str] = None,
        test_start_date: Optional[str] = None,
        test_end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Evaluates chronological steps across the dataset deterministically.
        Generates and optionally exports tabular CSV with columns:
        [Datetime, Symbol, Open, Close, Action, Position, StepReturn, GrossReturn, Cost, Turnover, MarketReturn]
        """
        self.agent.model.eval()
        columns = [
            "Datetime",
            "Symbol",
            "Open",
            "Close",
            "Action",
            "Position",
            "StepReturn",
            "GrossReturn",
            "Cost",
            "Turnover",
            "MarketReturn",
        ]

        total_len = len(prices.close)
        if total_len <= self.bars_count + 1:
            return pd.DataFrame(columns=columns)

        # Defensive index alignment: align raw_df with prices if needed
        aligned_df = raw_df
        if len(raw_df) > total_len:
            aligned_df = raw_df.iloc[-total_len:]

        if "Datetime" in aligned_df.columns:
            datetime_series = aligned_df["Datetime"].astype(str).values
        elif "date" in aligned_df.columns:
            datetime_series = aligned_df["date"].astype(str).values
        else:
            datetime_series = aligned_df.index.astype(str).values

        effective_start = (
            test_start_date if test_start_date is not None else self.test_start_date
        )
        effective_end = (
            test_end_date if test_end_date is not None else self.test_end_date
        )

        datetime_ts = pd.to_datetime(datetime_series)

        if effective_start is not None and str(effective_start).strip() != "":
            start_ts = pd.to_datetime(effective_start)
            start_idx = int(datetime_ts.searchsorted(start_ts, side="left"))
        else:
            start_idx = self.bars_count

        if effective_end is not None and str(effective_end).strip() != "":
            end_ts = pd.to_datetime(effective_end)
            end_idx = int(datetime_ts.searchsorted(end_ts, side="right"))
        else:
            end_idx = total_len - 1

        start_offset = max(self.bars_count, start_idx)
        end_offset = min(total_len - 1, end_idx)

        if start_offset >= end_offset:
            return pd.DataFrame(columns=columns)

        state = State_time_step(
            bars_count=self.bars_count,
            commission_perc=self.commission_perc,
            model_train=False,
            default_slippage=self.default_slippage,
            N_steps=self.episode_steps,
            deadzone_threshold=self.deadzone_threshold,
            rebalance_threshold=self.rebalance_threshold,
            w1_step_return=self.w1_step_return,
        )
        env = ProductionEnv(prices_data={symbol: prices}, state=state)
        obs = env.reset(symbol=symbol, start_offset=start_offset)

        open_series = prices.open
        close_series = prices.close
        records: List[Dict[str, Any]] = []
        done = False

        with torch.no_grad():
            while not done and env._state._offset < end_offset:
                curr_offset = env._state._offset
                dt = datetime_series[curr_offset]
                open_val = float(open_series[curr_offset])
                close_val = float(close_series[curr_offset])
                prev_close_val = max(float(close_series[curr_offset - 1]), 1e-8)
                mkt_log_ret = float(np.log(max(close_val, 1e-8) / prev_close_val))

                out = self.agent.get_action(obs, stochastic=False)
                action_val = float(np.clip(float(out.clipped_action.item()), 0.0, 1.0))

                # Step the environment
                obs, reward, terminated, truncated, info = env.step(action_val)
                done = bool(terminated or truncated)

                # Post-step position in environment
                pos_val = float(env._state.position)

                records.append({
                    "Datetime": dt,
                    "Symbol": symbol,
                    "Open": open_val,
                    "Close": close_val,
                    "Action": action_val,
                    "Position": pos_val,
                    "StepReturn": float(info.get("step_return", 0.0)),
                    "GrossReturn": float(info.get("gross_return", 0.0)),
                    "Cost": float(info.get("cost", 0.0)),
                    "Turnover": float(info.get("turnover", 0.0)),
                    "MarketReturn": mkt_log_ret,
                })

        order_df = pd.DataFrame(records, columns=columns)

        if output_csv_path and len(order_df) > 0:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
            order_df.to_csv(output_csv_path, index=False)

        return order_df

    @staticmethod
    def _safe_annualized_ratio(
        daily_series: np.ndarray, annual_factor: float = float(np.sqrt(365.0))
    ) -> float:
        if len(daily_series) < 2:
            return 0.0
        mean_val = float(np.mean(daily_series))
        std_val = float(np.std(daily_series, ddof=1))
        if not np.isfinite(std_val) or std_val < 1e-8:
            return 0.0
        val = (mean_val / std_val) * annual_factor
        return float(np.clip(val, -50.0, 50.0)) if np.isfinite(val) else 0.0

    @staticmethod
    def _safe_annualized_sortino(
        daily_series: np.ndarray, annual_factor: float = float(np.sqrt(365.0))
    ) -> float:
        if len(daily_series) < 2:
            return 0.0
        mean_val = float(np.mean(daily_series))
        downside = np.minimum(daily_series, 0.0)
        downside_std = float(np.sqrt(np.mean(downside ** 2)))
        if not np.isfinite(downside_std) or downside_std < 1e-8:
            if mean_val > 1e-8:
                return PPO2Evaluator._safe_annualized_ratio(daily_series, annual_factor)
            return 0.0
        val = (mean_val / downside_std) * annual_factor
        return float(np.clip(val, -50.0, 50.0)) if np.isfinite(val) else 0.0

    @staticmethod
    def compute_symbol_metrics(
        order_df: pd.DataFrame,
        commission_perc: float = 0.0001,
        default_slippage: float = 0.0001,
    ) -> Dict[str, Any]:
        """
        Computes daily-aggregated risk-adjusted performance metrics and 2x friction stress
        tests from a symbol's step-by-step evaluation DataFrame.
        """
        empty_metrics: Dict[str, Any] = {
            "total_bars": 0,
            "eval_days": 0,
            "mean_action": 0.0,
            "std_action": 0.0,
            "long_ratio": 0.0,
            "short_ratio": 0.0,
            "flat_ratio": 0.0,
            "mean_exposure": 0.0,
            "cum_return_pct": 0.0,
            "bh_return_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "ir_1x": 0.0,
            "ir_2x": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar": 0.0,
            "daily_turnover": 0.0,
            "trade_count": 0,
            "daily_returns": {},
        }
        if order_df is None or order_df.empty:
            return empty_metrics

        total_bars = len(order_df)
        actions = order_df["Action"].to_numpy(dtype=np.float64)
        positions = order_df["Position"].to_numpy(dtype=np.float64)

        step_ret = (
            order_df["StepReturn"].to_numpy(dtype=np.float64)
            if "StepReturn" in order_df.columns
            else np.zeros(total_bars, dtype=np.float64)
        )
        gross_ret = (
            order_df["GrossReturn"].to_numpy(dtype=np.float64)
            if "GrossReturn" in order_df.columns
            else step_ret.copy()
        )
        cost_arr = (
            order_df["Cost"].to_numpy(dtype=np.float64)
            if "Cost" in order_df.columns
            else np.maximum(0.0, gross_ret - step_ret)
        )
        turnover = (
            order_df["Turnover"].to_numpy(dtype=np.float64)
            if "Turnover" in order_df.columns
            else np.abs(np.diff(np.concatenate([[0.0], positions])))
        )
        if "MarketReturn" in order_df.columns:
            mkt_ret = order_df["MarketReturn"].to_numpy(dtype=np.float64)
        elif "Close" in order_df.columns:
            closes = np.maximum(order_df["Close"].to_numpy(dtype=np.float64), 1e-8)
            mkt_ret = np.concatenate([[0.0], np.log(closes[1:] / closes[:-1])])
        else:
            mkt_ret = np.zeros(total_bars, dtype=np.float64)

        friction_rate = max(0.0, float(commission_perc) + float(default_slippage))
        if friction_rate > 0.0:
            step_ret_2x = gross_ret + np.log(
                np.maximum(1.0 - 2.0 * turnover * friction_rate, 1e-8)
            )
        else:
            step_ret_2x = step_ret - cost_arr

        mean_action = float(np.mean(actions))
        std_action = float(np.std(actions))
        long_ratio = float(np.mean(positions > 0.0))
        short_ratio = float(np.mean(positions < 0.0))
        flat_ratio = float(np.mean(positions == 0.0))
        mean_exposure = float(np.clip(np.mean(positions), 0.0, 1.0))

        cum_log_ret = float(np.sum(step_ret))
        bh_log_ret = float(np.sum(mkt_ret))
        cum_return_pct = float((np.exp(np.clip(cum_log_ret, -50.0, 50.0)) - 1.0) * 100.0)
        bh_return_pct = float((np.exp(np.clip(bh_log_ret, -50.0, 50.0)) - 1.0) * 100.0)

        # Bar-level compounded equity curve for exact Max Drawdown
        cum_curve = np.concatenate([[0.0], np.cumsum(step_ret)])
        equity_curve = np.exp(np.clip(cum_curve, -50.0, 50.0))
        running_peak = np.maximum.accumulate(equity_curve)
        drawdowns = 1.0 - (equity_curve / np.maximum(running_peak, 1e-12))
        max_drawdown_pct = float(np.clip(np.max(drawdowns) * 100.0, 0.0, 100.0))

        # Group by calendar date for daily return aggregation
        dt_parsed = pd.to_datetime(order_df["Datetime"], errors="coerce")
        if dt_parsed.isna().all():
            # Fallback: group every 48 bars (24h of 30m bars) as one synthetic day
            day_keys = pd.Series(
                [f"day_{i // 48:04d}" for i in range(total_bars)], dtype=str
            )
        else:
            day_keys = dt_parsed.dt.strftime("%Y-%m-%d").fillna("unknown")

        daily_df = pd.DataFrame({
            "day": day_keys.values,
            "ret_1x": step_ret,
            "ret_2x": step_ret_2x,
            "mkt_ret": mkt_ret,
            "turnover": turnover,
        }).groupby("day", sort=True).sum()

        eval_days = len(daily_df)
        daily_ret_1x = daily_df["ret_1x"].to_numpy(dtype=np.float64)
        daily_ret_2x = daily_df["ret_2x"].to_numpy(dtype=np.float64)
        daily_mkt_ret = daily_df["mkt_ret"].to_numpy(dtype=np.float64)
        daily_turn_arr = daily_df["turnover"].to_numpy(dtype=np.float64)

        bench_daily = mean_exposure * daily_mkt_ret
        active_1x = daily_ret_1x - bench_daily
        active_2x = daily_ret_2x - bench_daily

        sharpe = PPO2Evaluator._safe_annualized_ratio(daily_ret_1x)
        sortino = PPO2Evaluator._safe_annualized_sortino(daily_ret_1x)
        ir_1x = PPO2Evaluator._safe_annualized_ratio(active_1x)
        ir_2x = PPO2Evaluator._safe_annualized_ratio(active_2x)

        ann_log_ret = (cum_log_ret / max(eval_days, 1)) * 365.0
        mdd_frac = max_drawdown_pct / 100.0
        if mdd_frac > 1e-4:
            calmar = float(np.clip(ann_log_ret / mdd_frac, -50.0, 50.0))
        else:
            calmar = float(np.clip(ann_log_ret * 10.0, -50.0, 50.0)) if ann_log_ret > 0 else 0.0

        daily_turnover = float(np.mean(daily_turn_arr)) if eval_days > 0 else 0.0
        trade_count = int(np.sum(turnover > 1e-8))

        daily_returns_dict = {
            str(k): round(float(v), 6)
            for k, v in zip(daily_df.index, daily_ret_1x)
        }

        return {
            "total_bars": int(total_bars),
            "eval_days": int(eval_days),
            "mean_action": round(mean_action, 4),
            "std_action": round(std_action, 4),
            "long_ratio": round(long_ratio, 4),
            "short_ratio": round(short_ratio, 4),
            "flat_ratio": round(flat_ratio, 4),
            "mean_exposure": round(mean_exposure, 4),
            "cum_return_pct": round(cum_return_pct, 4),
            "bh_return_pct": round(bh_return_pct, 4),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "ir_1x": round(ir_1x, 4),
            "ir_2x": round(ir_2x, 4),
            "max_drawdown_pct": round(max_drawdown_pct, 4),
            "calmar": round(calmar, 4),
            "daily_turnover": round(daily_turnover, 4),
            "trade_count": int(trade_count),
            "daily_returns": daily_returns_dict,
        }
