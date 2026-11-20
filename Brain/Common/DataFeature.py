from typing import Optional, Sequence
import collections
import numpy as np
import pandas as pd
from Brain.Common.env_components import BasicState

OHLC_RAW_FIELDS = ("open", "high", "low", "close")

Prices = collections.namedtuple(
    "Prices",
    field_names=(
        *OHLC_RAW_FIELDS,
        *BasicState.MARKET_FEATURES,
        *BasicState.STATIONARY_TIME_FEATURES,
    ),
)

SOFT_CLIP_LIMIT: float = 5.0
VOL_FLOOR: float = 1e-4
VOL_REF: float = 0.005
VOL_LONG_SPAN: int = 60
VOL_LONG_MIN_PERIODS: int = 20
VOL_SHORT_SPAN: int = 12
VOL_SHORT_MIN_PERIODS: int = 6
VOL_EMA_SPAN: int = 60


def soft_clip(x: np.ndarray, limit: float = SOFT_CLIP_LIMIT) -> np.ndarray:
    """
    Smoothly bounds inputs into (-limit, +limit) via limit * tanh(x / limit),
    preserving near-linearity around zero while damping extreme outlier spikes.
    """
    return limit * np.tanh(x / limit)


class OriginalDataFeature:
    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None

    def cleanData(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Causally cleans market OHLCV data without look-ahead bias:
        - Replaces +-inf with NaN.
        - Treats non-positive OHLC prices (<= 0) as invalid (NaN), forward-fills OHLC only,
          and drops leading rows where OHLC remains NaN (no bfill).
        - Clamps negative volume/microstructure columns to 0.0 and fills missing values with 0.0.
        """
        cleaned = df.copy()
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

        ohlc_cols = [c for c in ("Open", "High", "Low", "Close") if c in cleaned.columns]
        for col in ohlc_cols:
            cleaned.loc[cleaned[col] <= 0.0, col] = np.nan
        if ohlc_cols:
            valid_complete = cleaned[ohlc_cols].notna().all(axis=1)
            if not valid_complete.any():
                cleaned = cleaned.iloc[0:0].copy()
            else:
                first_valid_pos = int(np.argmax(valid_complete.to_numpy()))
                cleaned = cleaned.iloc[first_valid_pos:].copy()
                cleaned[ohlc_cols] = cleaned[ohlc_cols].ffill(axis=0)

        vol_cols = [
            c
            for c in ("Volume", "quote_av", "trades", "tb_base_av", "tb_quote_av")
            if c in cleaned.columns
        ]
        for col in vol_cols:
            cleaned.loc[cleaned[col] < 0.0, col] = 0.0
            cleaned[col] = cleaned[col].fillna(0.0)

        return cleaned.copy()

    def get_train_net_work_data_by_pd(
        self, symbol: str, df: pd.DataFrame, first_date=None
    ) -> dict:
        """
        用來取得類神經網絡所需要的資料（正式交易／DataFrame 入口），
        與 get_train_net_work_data_by_path 執行完全一致的清洗與特徵計算流程。
        """
        out_dict = {}
        cleaned = self.cleanData(df)
        cleaned = self.add_market_features(cleaned)
        cleaned = self.add_time_feature(cleaned, first_date)
        self.df = cleaned
        out_dict[symbol] = self.load_relative(cleaned)
        return out_dict

    def get_train_net_work_data_by_path(
        self,
        symbolNames: list,
        typeName: str = "train_data",
        end_date: Optional[str] = None,
        min_bars: int = 0,
    ) -> dict:
        """
        Loads market CSV data for symbolNames, optionally truncating timestamps at end_date
        and skipping instruments with insufficient usable rows.
        """
        out_dict = {}
        cutoff_ts = (
            pd.to_datetime(end_date)
            if end_date is not None and str(end_date).strip() != ""
            else None
        )
        for symbolName in symbolNames:
            df = pd.read_csv(f"Brain/simulation/{typeName}/{symbolName}.csv")
            df.set_index("Datetime", inplace=True)
            if cutoff_ts is not None:
                df = df[pd.to_datetime(df.index) <= cutoff_ts]
            if len(df) <= 360:
                continue
            cleaned = self.cleanData(df)
            if len(cleaned) <= 360:
                continue
            cleaned = self.add_market_features(cleaned)
            if len(cleaned) < max(1, int(min_bars)):
                continue
            cleaned = self.add_time_feature(cleaned)
            self.df = cleaned
            out_dict[symbolName] = self.load_relative(cleaned)

        return out_dict

    def add_market_features(
        self,
        df: pd.DataFrame,
        periods: Sequence[int] = (30, 60, 120, 240, 360),
    ) -> pd.DataFrame:
        """
        計算 14 維嚴格因果、尺度不變、無中心化波動率歸一化市場特徵：
          1. 正交 4-DoF K 棒幾何（共用前一期波動率 sigma_{t-1}，不去均值）：
             ret_gap, candle_body, upper_wick, lower_wick
          2. 多週期 SMA 趨勢偏離（依隨機漫步尺度 sigma_{t-1} * sqrt(p/3) 縮放）：
             ma_dev_30, ma_dev_60, ma_dev_120, ma_dev_240, ma_dev_360
          3. 顯式波動率體制特徵（保留絕對規模與短期擴張/收斂）：
             vol_regime_abs, vol_expansion
          4. 量能與主動買賣微結構特徵（對數空間 Log-EMA 零中心化）：
             rel_quote_vol, rel_trades, taker_imbalance
        """
        out = df.copy()
        if len(out) == 0:
            for col in BasicState.MARKET_FEATURES:
                out[col] = np.empty(0, dtype=np.float64)
            return out

        o = out["Open"].to_numpy(dtype=np.float64)
        h = out["High"].to_numpy(dtype=np.float64)
        l = out["Low"].to_numpy(dtype=np.float64)
        c = out["Close"].to_numpy(dtype=np.float64)

        # Defensive OHLC envelope so upper_wick >= 0 and lower_wick >= 0 always hold
        oc_max = np.maximum(o, c)
        oc_min = np.minimum(o, c)
        h_safe = np.maximum(h, oc_max)
        l_safe = np.minimum(l, oc_min)

        close_s = pd.Series(c, index=out.index, dtype=np.float64)
        prev_c = close_s.shift(1).to_numpy(dtype=np.float64)
        log_ret = np.log(c / prev_c)
        log_ret_s = pd.Series(log_ret, index=out.index, dtype=np.float64)

        # Causal EWMA RMS volatility (zero-centered, no demeaning)
        sigma_long_s = np.sqrt(
            log_ret_s.pow(2)
            .ewm(span=VOL_LONG_SPAN, min_periods=VOL_LONG_MIN_PERIODS, adjust=False)
            .mean()
        ).clip(lower=VOL_FLOOR)
        sigma_short_s = np.sqrt(
            log_ret_s.pow(2)
            .ewm(span=VOL_SHORT_SPAN, min_periods=VOL_SHORT_MIN_PERIODS, adjust=False)
            .mean()
        ).clip(lower=VOL_FLOOR)

        # Ex-ante volatility sigma_{t-1} scales bar t without self-damping breakouts
        sigma_prev = sigma_long_s.shift(1).to_numpy(dtype=np.float64)
        sigma_long = sigma_long_s.to_numpy(dtype=np.float64)
        sigma_short = sigma_short_s.to_numpy(dtype=np.float64)

        # 1. Orthogonal 4-DoF candlestick features
        out["ret_gap"] = soft_clip(np.log(o / prev_c) / sigma_prev)
        out["candle_body"] = soft_clip(np.log(c / o) / sigma_prev)
        out["upper_wick"] = soft_clip(np.log(h_safe / oc_max) / sigma_prev)
        out["lower_wick"] = soft_clip(np.log(oc_min / l_safe) / sigma_prev)

        # 2. Multi-horizon SMA trend deviations
        for p in periods:
            p_int = int(p)
            sma = close_s.rolling(window=p_int).mean().to_numpy(dtype=np.float64)
            scale_p = sigma_prev * np.sqrt(float(p_int) / 3.0)
            out[f"ma_dev_{p_int}"] = soft_clip(np.log(c / sma) / scale_p)

        # 3. Explicit volatility regime features
        out["vol_regime_abs"] = soft_clip(np.log(sigma_long / VOL_REF))
        out["vol_expansion"] = soft_clip(np.log(sigma_short / sigma_long))

        # 4. Volume & order flow microstructure features (causal log-EMA)
        q = np.maximum(out["quote_av"].to_numpy(dtype=np.float64), 0.0)
        trades = np.maximum(out["trades"].to_numpy(dtype=np.float64), 0.0)
        tbq = np.maximum(out["tb_quote_av"].to_numpy(dtype=np.float64), 0.0)

        log_q_s = pd.Series(np.log1p(q), index=out.index, dtype=np.float64)
        log_t_s = pd.Series(np.log1p(trades), index=out.index, dtype=np.float64)
        log_q_ema = (
            log_q_s.ewm(span=VOL_EMA_SPAN, adjust=False).mean().to_numpy(dtype=np.float64)
        )
        log_t_ema = (
            log_t_s.ewm(span=VOL_EMA_SPAN, adjust=False).mean().to_numpy(dtype=np.float64)
        )

        out["rel_quote_vol"] = soft_clip(log_q_s.to_numpy(dtype=np.float64) - log_q_ema)
        out["rel_trades"] = soft_clip(log_t_s.to_numpy(dtype=np.float64) - log_t_ema)
        imb = np.where(q > 0.0, 2.0 * (tbq / np.maximum(q, 1e-12)) - 1.0, 0.0)
        out["taker_imbalance"] = np.clip(imb, -1.0, 1.0)

        out = out.dropna(subset=list(BasicState.MARKET_FEATURES)).copy()
        return out

    def add_time_feature(self, df: pd.DataFrame, first_date=None) -> pd.DataFrame:
        out = df.copy()
        datetime_index = pd.to_datetime(out.index)

        # --- 平穩週期性時間特徵 (天然標準化在 [-1, 1]) ---
        out["hour_sin"] = np.sin(2 * np.pi * datetime_index.hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * datetime_index.hour / 24)

        out["minute_sin"] = np.sin(2 * np.pi * datetime_index.minute / 60)
        out["minute_cos"] = np.cos(2 * np.pi * datetime_index.minute / 60)

        out["dayofweek_sin"] = np.sin(2 * np.pi * datetime_index.dayofweek / 7)
        out["dayofweek_cos"] = np.cos(2 * np.pi * datetime_index.dayofweek / 7)

        return out

    def load_relative(self, df: Optional[pd.DataFrame] = None) -> Prices:
        """
        從已計算完成特徵的 DataFrame 封裝 24 欄 Prices namedtuple：
          - 4 原始執行價格：open, high, low, close
          - 14 維波動率歸一化市場特徵：BasicState.MARKET_FEATURES
          - 6 維平穩週期時間特徵：BasicState.STATIONARY_TIME_FEATURES
        """
        src = df if df is not None else self.df
        if src is None:
            raise ValueError("DataFrame is not loaded; pass df or run feature enrichment first.")

        return Prices(
            open=src["Open"].to_numpy(dtype=np.float64),
            high=src["High"].to_numpy(dtype=np.float64),
            low=src["Low"].to_numpy(dtype=np.float64),
            close=src["Close"].to_numpy(dtype=np.float64),
            **{
                col: src[col].to_numpy(dtype=np.float64)
                for col in BasicState.MARKET_FEATURES
            },
            **{
                col: src[col].to_numpy(dtype=np.float64)
                for col in BasicState.STATIONARY_TIME_FEATURES
            },
        )
