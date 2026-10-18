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
        commission_perc: float = 0.0045,
        default_slippage: float = 0.0025,
        downside_window_size: int = 60,
        w1_step_return: float = 1.0,
        w2_downside_ratio: float = 1.0,
        episode_steps: int = 500,
    ):
        self.agent = agent
        self.bars_count = bars_count
        self.device = device or agent.device
        self.deadzone_threshold = deadzone_threshold
        self.commission_perc = commission_perc
        self.default_slippage = default_slippage
        self.downside_window_size = downside_window_size
        self.w1_step_return = w1_step_return
        self.w2_downside_ratio = w2_downside_ratio
        self.episode_steps = episode_steps

        # Force evaluation mode (deterministic inference)
        self.agent.model.eval()

    def evaluate_single_step(
        self, obs: Dict[str, Any], apply_deadzone: bool = True
    ) -> float:
        """
        Evaluates a single observation window for formal live trading mode.
        Returns continuous deterministic action / target position in [-1.0, 1.0].
        """
        self.agent.model.eval()
        with torch.no_grad():
            out = self.agent.get_action(obs, stochastic=False)
            action_val = float(out.clipped_action.item())
            if apply_deadzone and abs(action_val) <= self.deadzone_threshold:
                action_val = 0.0
            return action_val

    def evaluate_symbol(
        self,
        symbol: str,
        raw_df: pd.DataFrame,
        prices: Prices,
        output_csv_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Evaluates chronological steps across the dataset deterministically without PnL or plotting.
        Generates and optionally exports tabular CSV with columns:
        [Datetime, Symbol, Close, Action, Position]
        """
        self.agent.model.eval()

        state = State_time_step(
            bars_count=self.bars_count,
            commission_perc=self.commission_perc,
            model_train=False,
            default_slippage=self.default_slippage,
            N_steps=self.episode_steps,
            deadzone_threshold=self.deadzone_threshold,
            downside_window_size=self.downside_window_size,
            w1_step_return=self.w1_step_return,
            w2_downside_ratio=self.w2_downside_ratio,
        )
        env = ProductionEnv(prices_data={symbol: prices}, state=state)
        obs = env.reset(symbol=symbol)

        # Defensive index alignment: align raw_df with prices if needed
        aligned_df = raw_df
        if len(raw_df) > len(prices.close):
            aligned_df = raw_df.iloc[-len(prices.close):]

        if "Datetime" in aligned_df.columns:
            datetime_series = aligned_df["Datetime"].astype(str).values
        else:
            datetime_series = aligned_df.index.astype(str).values

        close_series = prices.close
        records: List[Dict[str, Any]] = []
        done = False

        with torch.no_grad():
            while not done:
                curr_offset = env._state._offset
                dt = datetime_series[curr_offset]
                close_val = float(close_series[curr_offset])

                out = self.agent.get_action(obs, stochastic=False)
                action_val = float(out.clipped_action.item())

                # Step the environment
                obs, reward, terminated, truncated, info = env.step(action_val)
                done = bool(terminated or truncated)

                # Post-step position in environment
                pos_val = float(env._state.position)

                records.append({
                    "Datetime": dt,
                    "Symbol": symbol,
                    "Close": close_val,
                    "Action": action_val,
                    "Position": pos_val,
                })

        order_df = pd.DataFrame(
            records, columns=["Datetime", "Symbol", "Close", "Action", "Position"]
        )

        if output_csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
            order_df.to_csv(output_csv_path, index=False)

        return order_df
