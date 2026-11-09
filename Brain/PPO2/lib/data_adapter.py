from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from Brain.Common.DataFeature import OriginalDataFeature, Prices
from Brain.PPO2.lib.environment import STATIONARY_TIME_FEATURES


@dataclass
class PortfolioState:
    """Dataclass representing portfolio position and entry price for a symbol."""
    position: float = 0.0
    open_price: float = 0.0


class PPO2DataAdapter:
    """
    Data adapter responsible for OHLCV DataFrame standardization,
    technical indicator and time feature enrichment, and observation window extraction.
    """

    @staticmethod
    def resolve_portfolio_state(
        symbol: str, portfolio_state: Optional[Dict[str, Any]]
    ) -> PortfolioState:
        """
        Resolves (position, open_price) for a given symbol from portfolio_state dictionary.
        """
        if not portfolio_state:
            return PortfolioState(0.0, 0.0)

        entry = None
        if symbol in portfolio_state:
            entry = portfolio_state[symbol]
        elif any(k in portfolio_state for k in ("position", "open_price")):
            entry = portfolio_state

        if entry is None:
            return PortfolioState(0.0, 0.0)
        if isinstance(entry, dict):
            pos = float(np.clip(float(entry.get("position", 0.0)), 0.0, 1.0))
            open_p = float(entry.get("open_price", 0.0))
            return PortfolioState(pos, open_p)
        if isinstance(entry, (tuple, list)):
            pos = float(np.clip(float(entry[0]) if len(entry) > 0 else 0.0, 0.0, 1.0))
            open_p = float(entry[1]) if len(entry) > 1 else 0.0
            return PortfolioState(pos, open_p)
        if isinstance(entry, (int, float, np.floating, np.integer)):
            pos = float(np.clip(float(entry), 0.0, 1.0))
            return PortfolioState(pos, 0.0)
        return PortfolioState(0.0, 0.0)

    @staticmethod
    def process_df_to_prices(
        feature_loader: OriginalDataFeature, df: pd.DataFrame
    ) -> Prices:
        """
        Preprocesses and enriches raw OHLCV DataFrame with required moving averages,
        cyclical time features, and returns Prices data structure.
        """
        df_copy = df.copy()
        if "Datetime" not in df_copy.columns:
            for dt_col in ["date", "Date", "datetime", "timestamp", "Timestamp"]:
                if dt_col in df_copy.columns:
                    df_copy.rename(columns={dt_col: "Datetime"}, inplace=True)
                    break
        if "Datetime" in df_copy.columns:
            df_copy.set_index("Datetime", inplace=True)

        ohlcv_rename = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        for low_col, cap_col in ohlcv_rename.items():
            if cap_col not in df_copy.columns and low_col in df_copy.columns:
                df_copy[cap_col] = df_copy[low_col]

        cleaned = feature_loader.cleanData(df_copy)

        # Defensive handling for missing volume auxiliary columns
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

        if "age_log_minutes" not in cleaned.columns and len(cleaned) > 0:
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

    @classmethod
    def prepare_test_symbol_data(
        cls,
        feature_loader: OriginalDataFeature,
        df: pd.DataFrame,
        bars_count: int,
        test_start_date: Optional[str] = None,
        test_end_date: Optional[str] = None,
    ) -> Optional[Tuple[Prices, np.ndarray]]:
        """
        Enriches raw DataFrame into Prices and slices both Prices and Datetime array
        to [start_offset - bars_count : end_offset + 1] so evaluation always runs
        from offset=bars_count until State_time_step naturally terminates.
        """
        if len(df) <= 360:
            return None

        prices = cls.process_df_to_prices(feature_loader, df)
        raw_df = feature_loader.df
        total_len = len(prices.close)
        if total_len <= bars_count + 1:
            return None

        aligned_df = raw_df.iloc[-total_len:] if len(raw_df) > total_len else raw_df
        if "Datetime" in aligned_df.columns:
            datetime_series = aligned_df["Datetime"].astype(str).to_numpy()
        elif "date" in aligned_df.columns:
            datetime_series = aligned_df["date"].astype(str).to_numpy()
        else:
            datetime_series = aligned_df.index.astype(str).to_numpy()

        datetime_ts = pd.to_datetime(datetime_series)
        if test_start_date is not None and str(test_start_date).strip() != "":
            start_idx = int(
                datetime_ts.searchsorted(pd.to_datetime(test_start_date), side="left")
            )
        else:
            start_idx = bars_count

        if test_end_date is not None and str(test_end_date).strip() != "":
            end_idx = int(
                datetime_ts.searchsorted(pd.to_datetime(test_end_date), side="right")
            )
        else:
            end_idx = total_len - 1

        start_offset = max(bars_count, start_idx)
        end_offset = min(total_len - 1, end_idx)
        if start_offset >= end_offset:
            return None

        slice_start = start_offset - bars_count
        slice_end = end_offset + 1
        sliced_prices = Prices(
            *[
                np.ascontiguousarray(getattr(prices, field)[slice_start:slice_end])
                for field in prices._fields
            ]
        )
        sliced_dt = np.ascontiguousarray(datetime_series[slice_start:slice_end])
        return sliced_prices, sliced_dt

    @staticmethod
    def extract_latest_obs(
        prices: Prices,
        bars_count: int,
        position: float = 0.0,
        open_price: float = 0.0,
        portfolio_state: Optional[Dict[str, Any]] = None,
        current_position: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extracts the latest observation window of length bars_count for formal live mode,
        matching State_time_step.encode() bit-for-bit across market, portfolio, and 6-dim time features.
        """
        if portfolio_state is not None:
            position = float(portfolio_state.get("position", position))
            open_price = float(portfolio_state.get("open_price", open_price))
        elif current_position is not None:
            position = float(current_position)

        pos = float(np.clip(position, 0.0, 1.0))

        n_bars = len(prices.close)
        if n_bars < bars_count:
            raise ValueError(
                f"Data length {n_bars} is shorter than required window {bars_count}"
            )

        start = n_bars - bars_count
        end = n_bars

        info_list = [
            "log_open", "log_high", "log_low", "log_close", "log_volume",
            "log_quote_av", "log_trades", "log_tb_base_av", "log_tb_quote_av",
            "log_ma_30", "log_ma_60", "log_ma_120", "log_ma_240", "log_ma_360",
        ]
        timelist = list(STATIONARY_TIME_FEATURES)

        data_res = np.zeros(shape=(bars_count, len(info_list) + 2), dtype=np.float32)
        time_res = np.zeros(shape=(bars_count, len(timelist)), dtype=np.float32)

        for idx, field in enumerate(info_list):
            data_res[:, idx] = getattr(prices, field)[start:end]

        data_res[:, len(info_list)] = pos

        if pos > 0.0 and open_price > 0.0:
            prev_close = max(float(prices.close[end - 1]), 1e-8)
            safe_open = max(float(open_price), 1e-8)
            raw_equity_ratio = 1.0 + pos * ((prev_close - safe_open) / safe_open)
            data_res[:, len(info_list) + 1] = float(np.log(max(raw_equity_ratio, 1e-8)))
        else:
            data_res[:, len(info_list) + 1] = 0.0

        for idx, field in enumerate(timelist):
            time_res[:, idx] = getattr(prices, field)[start:end]

        return {"states": data_res, "time_states": time_res}
