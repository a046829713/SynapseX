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
        Evaluates chronological steps across the dataset deterministically without PnL or plotting.
        Generates and optionally exports tabular CSV with columns:
        [Datetime, Symbol, Open, Close, Action, Position]
        """
        self.agent.model.eval()
        columns = ["Datetime", "Symbol", "Open", "Close", "Action", "Position"]

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
                })

        order_df = pd.DataFrame(records, columns=columns)

        if output_csv_path and len(order_df) > 0:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
            order_df.to_csv(output_csv_path, index=False)

        return order_df
