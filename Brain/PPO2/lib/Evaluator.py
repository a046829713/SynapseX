import os
from collections import deque
from typing import Optional, Dict, Any, List, Tuple, Callable
import numpy as np
import pandas as pd
import torch
from Brain.Common.DataFeature import Prices
from Brain.PPO2.lib.Agent import PPO2Agent
from Brain.PPO2.lib.environment import State_time_step


class _SymbolEvalSlot:
    """In-process state container for a single active symbol slot during batched evaluation."""

    def __init__(
        self,
        symbol: str,
        prices: Prices,
        datetime_series: np.ndarray,
        state: State_time_step,
    ):
        self.symbol = symbol
        self.prices = prices
        self.datetime_series = datetime_series
        self.state = state
        self.start_offset = state.bars_count
        self.end_offset = len(prices.close) - 1
        self.n_steps = max(0, self.end_offset - self.start_offset)
        self.step_idx = 0
        self.state.reset(prices, self.start_offset)

        self.actions = np.zeros(self.n_steps, dtype=np.float64)
        self.positions = np.zeros(self.n_steps, dtype=np.float64)
        self.step_returns = np.zeros(self.n_steps, dtype=np.float64)
        self.gross_returns = np.zeros(self.n_steps, dtype=np.float64)
        self.costs = np.zeros(self.n_steps, dtype=np.float64)
        self.turnovers = np.zeros(self.n_steps, dtype=np.float64)

    def build_order_df(self, columns: List[str]) -> pd.DataFrame:
        s = self.start_offset
        e = self.start_offset + self.step_idx
        if e <= s:
            return pd.DataFrame(columns=columns)

        close_slice = self.prices.close[s:e].astype(np.float64)
        prev_close_slice = np.maximum(
            self.prices.close[s - 1 : e - 1].astype(np.float64), 1e-8
        )
        mkt_log_ret = np.log(np.maximum(close_slice, 1e-8) / prev_close_slice)

        return pd.DataFrame(
            {
                "Datetime": self.datetime_series[s:e],
                "Symbol": self.symbol,
                "Open": self.prices.open[s:e].astype(np.float64),
                "Close": close_slice,
                "Action": self.actions[: self.step_idx],
                "Position": self.positions[: self.step_idx],
                "StepReturn": self.step_returns[: self.step_idx],
                "GrossReturn": self.gross_returns[: self.step_idx],
                "Cost": self.costs[: self.step_idx],
                "Turnover": self.turnovers[: self.step_idx],
                "MarketReturn": mkt_log_ret,
            },
            columns=columns,
        )


class PPO2Evaluator:
    """
    Dedicated lightweight deterministic batched evaluator for continuous PPO2 policies.
    Supports multi-symbol parallel GPU inference via dynamic slot refilling (covering N >= 1 symbols)
    and generates standardized order CSVs for external backtesting frameworks.
    """

    ORDER_COLUMNS = [
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
        eval_batch_size: int = 64,
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
        self.eval_batch_size = max(1, int(eval_batch_size))
        assert self.eval_batch_size >= 1, "eval_batch_size must be >= 1"

        # Force evaluation mode (deterministic inference)
        self.agent.model.eval()

    def _make_state(self) -> State_time_step:
        return State_time_step(
            bars_count=self.bars_count,
            commission_perc=self.commission_perc,
            model_train=False,
            default_slippage=self.default_slippage,
            N_steps=self.episode_steps,
            deadzone_threshold=self.deadzone_threshold,
            rebalance_threshold=self.rebalance_threshold,
            w1_step_return=self.w1_step_return,
        )

    def evaluate_batch_step(
        self,
        batch_obs: Dict[str, np.ndarray],
        apply_deadzone: bool = True,
        prev_positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Evaluates a batch of observation windows (N >= 1) for formal live trading mode.
        Returns 1D numpy array of continuous deterministic target positions in [0.0, 1.0],
        applying deadzone_threshold and rebalance_threshold vectorized across symbols.
        """
        states_all = np.asarray(batch_obs["states"], dtype=np.float32)
        time_states_all = np.asarray(batch_obs["time_states"], dtype=np.float32)
        if states_all.ndim == 2:
            states_all = states_all[np.newaxis, ...]
            time_states_all = time_states_all[np.newaxis, ...]

        n_items = states_all.shape[0]
        if n_items == 0:
            return np.zeros(0, dtype=np.float64)

        raw_actions = np.zeros(n_items, dtype=np.float64)
        self.agent.model.eval()
        with torch.no_grad():
            for start_i in range(0, n_items, self.eval_batch_size):
                end_i = min(n_items, start_i + self.eval_batch_size)
                chunk_obs = {
                    "states": states_all[start_i:end_i],
                    "time_states": time_states_all[start_i:end_i],
                }
                out = self.agent.get_action(chunk_obs, stochastic=False)
                act_tensor = (
                    out.clipped_action if out.clipped_action is not None else out.action
                )
                raw_actions[start_i:end_i] = (
                    act_tensor.detach().cpu().numpy().reshape(-1)
                )

        clipped_actions = np.clip(raw_actions, 0.0, 1.0)
        if not apply_deadzone:
            return clipped_actions

        candidate_pos = np.where(
            clipped_actions <= self.deadzone_threshold, 0.0, clipped_actions
        )

        if prev_positions is not None:
            prev_pos = np.clip(
                np.asarray(prev_positions, dtype=np.float64).reshape(-1), 0.0, 1.0
            )
        elif states_all.shape[-1] >= 2:
            prev_pos = np.clip(states_all[:, -1, -2].astype(np.float64), 0.0, 1.0)
        else:
            prev_pos = np.zeros(n_items, dtype=np.float64)

        hold_mask = (
            (prev_pos > 0.0)
            & (candidate_pos > 0.0)
            & (np.abs(candidate_pos - prev_pos) < self.rebalance_threshold)
        )
        return np.where(hold_mask, prev_pos, candidate_pos)

    def evaluate_symbols(
        self,
        symbol_data_map: Dict[str, Tuple[Prices, np.ndarray]],
        model_output_dir: Optional[str] = None,
        on_symbol_complete: Optional[Callable[[str, pd.DataFrame], None]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Evaluates chronological steps across N >= 1 symbols concurrently on the GPU
        using dynamic slot refilling up to self.eval_batch_size.
        """
        self.agent.model.eval()
        columns = self.ORDER_COLUMNS
        results: Dict[str, pd.DataFrame] = {}

        pending_queue = deque(
            (sym, data_pair)
            for sym, data_pair in symbol_data_map.items()
            if len(data_pair[0].close) > self.bars_count + 1
        )
        if not pending_queue:
            return results

        if model_output_dir:
            os.makedirs(os.path.abspath(model_output_dir), exist_ok=True)

        max_slots = min(self.eval_batch_size, len(pending_queue))
        active_slots: List[_SymbolEvalSlot] = []
        for _ in range(max_slots):
            sym, (prices, dt_arr) = pending_queue.popleft()
            active_slots.append(
                _SymbolEvalSlot(sym, prices, dt_arr, self._make_state())
            )

        state_shape = active_slots[0].state.getStateShape()
        time_shape = active_slots[0].state.getTimeShape()
        batch_states_np = np.zeros((max_slots, *state_shape), dtype=np.float32)
        batch_time_np = np.zeros((max_slots, *time_shape), dtype=np.float32)

        with torch.no_grad():
            while active_slots:
                active_count = len(active_slots)
                for i, slot in enumerate(active_slots):
                    obs_i = slot.state.encode()
                    batch_states_np[i] = obs_i["states"]
                    batch_time_np[i] = obs_i["time_states"]

                batch_obs = {
                    "states": batch_states_np[:active_count],
                    "time_states": batch_time_np[:active_count],
                }
                out = self.agent.get_action(batch_obs, stochastic=False)
                act_tensor = (
                    out.clipped_action if out.clipped_action is not None else out.action
                )
                actions_np = np.clip(
                    act_tensor.detach().cpu().numpy().reshape(-1), 0.0, 1.0
                )

                next_active_slots: List[_SymbolEvalSlot] = []
                for i, slot in enumerate(active_slots):
                    action_val = float(actions_np[i])
                    _, terminated, truncated = slot.state.step(action_val)

                    k = slot.step_idx
                    slot.actions[k] = action_val
                    slot.positions[k] = float(slot.state.position)
                    slot.step_returns[k] = float(slot.state.last_step_return)
                    slot.gross_returns[k] = float(slot.state.last_gross_return)
                    slot.costs[k] = float(slot.state.last_cost)
                    slot.turnovers[k] = float(slot.state.last_turnover)
                    slot.step_idx += 1

                    done = bool(
                        terminated
                        or truncated
                        or slot.state._offset >= slot.end_offset
                    )
                    if not done:
                        next_active_slots.append(slot)
                        continue

                    order_df = slot.build_order_df(columns)
                    if model_output_dir and len(order_df) > 0:
                        out_csv = os.path.join(
                            model_output_dir, f"{slot.symbol}_orders.csv"
                        )
                        order_df.to_csv(out_csv, index=False)

                    if on_symbol_complete is not None:
                        on_symbol_complete(slot.symbol, order_df)
                        del order_df
                    else:
                        results[slot.symbol] = order_df

                    if pending_queue:
                        next_sym, (next_prices, next_dt) = pending_queue.popleft()
                        next_active_slots.append(
                            _SymbolEvalSlot(
                                next_sym, next_prices, next_dt, slot.state
                            )
                        )

                active_slots = next_active_slots

        return results

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
