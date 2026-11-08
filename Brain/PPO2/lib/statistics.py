import math
import re
from itertools import combinations
from typing import Optional, List, Union, Tuple
import numpy as np


def natural_sort_key(s: str) -> List[Union[int, str]]:
    """
    Helper key for natural numerical sorting of checkpoint files (e.g. iter_10 before iter_100).
    """
    parts = re.split(r"(\d+)", s)
    return [int(text) if text.isdigit() else text.lower() for text in parts]


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    if not np.isfinite(x):
        return 0.5
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """
    Rational approximation for the standard normal inverse CDF (Acklam's algorithm).
    """
    p_clipped = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1.0 - plow
    if p_clipped < plow:
        q = math.sqrt(-2.0 * math.log(p_clipped))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if phigh < p_clipped:
        q = math.sqrt(-2.0 * math.log(1.0 - p_clipped))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p_clipped - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def compute_iqm(values: Union[List[float], np.ndarray]) -> float:
    """
    Computes the Interquartile Mean (25% trimmed mean across [Q25, Q75]),
    following the Google Brain / DeepMind `rliable` standard.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return 0.0
    if n < 4:
        return float(np.median(arr))
    sorted_arr = np.sort(arr)
    low_idx = int(n // 4)
    high_idx = int(n - low_idx)
    if low_idx >= high_idx:
        return float(np.median(sorted_arr))
    return float(np.mean(sorted_arr[low_idx:high_idx]))


def compute_deflated_sharpe(
    daily_returns_matrix: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Computes Deflated Sharpe Ratio probabilities (Bailey & López de Prado, 2014)
    for a D x N matrix of daily portfolio log-returns across D days and N checkpoints.
    Returns (dsr_probs_array_of_len_N, effective_trials_N_eff).
    """
    if daily_returns_matrix.ndim != 2:
        return np.zeros(0, dtype=np.float64), 1.0
    d_days, n_models = daily_returns_matrix.shape
    if n_models == 0:
        return np.zeros(0, dtype=np.float64), 1.0
    if d_days < 3:
        return np.full(n_models, 0.5, dtype=np.float64), 1.0

    means = np.mean(daily_returns_matrix, axis=0)
    stds = np.std(daily_returns_matrix, axis=0, ddof=1)
    valid_std_mask = stds > 1e-8

    sr_daily = np.zeros(n_models, dtype=np.float64)
    sr_daily[valid_std_mask] = means[valid_std_mask] / stds[valid_std_mask]

    # Estimate effective number of independent trials N_eff via eigenvalue participation ratio
    n_eff = 1.0
    if n_models > 1 and np.sum(valid_std_mask) > 1:
        active_mat = daily_returns_matrix[:, valid_std_mask]
        centered = active_mat - np.mean(active_mat, axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normalized = centered / norms
        corr = normalized.T @ normalized
        eigvals = np.linalg.eigvalsh(corr)
        eigvals = np.maximum(eigvals, 0.0)
        denom_eig = float(np.sum(eigvals ** 2))
        if denom_eig > 1e-12:
            pr = float((np.sum(eigvals) ** 2) / denom_eig)
            n_eff = float(np.clip(pr, 1.0, float(n_models)))

    # Expected maximum daily Sharpe under the null hypothesis (False Strategy Theorem)
    sr_std_cross = float(np.std(sr_daily, ddof=1)) if n_models > 1 else 0.0
    if n_eff > 1.05 and sr_std_cross > 1e-8:
        euler_gamma = 0.5772156649015329
        z1 = norm_ppf(1.0 - 1.0 / n_eff)
        z2 = norm_ppf(1.0 - 1.0 / (n_eff * math.e))
        sr_star = sr_std_cross * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)
    else:
        sr_star = 0.0

    dsr_probs = np.zeros(n_models, dtype=np.float64)
    for k in range(n_models):
        if not valid_std_mask[k]:
            dsr_probs[k] = 0.0
            continue
        series = daily_returns_matrix[:, k]
        z = (series - means[k]) / stds[k]
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4))
        sr_k = float(sr_daily[k])
        var_term = 1.0 - skew * sr_k + ((kurt - 1.0) / 4.0) * (sr_k ** 2)
        denom = math.sqrt(max(var_term, 1e-8))
        stat = ((sr_k - sr_star) * math.sqrt(max(d_days - 1, 1))) / denom
        dsr_probs[k] = round(norm_cdf(stat), 4)

    return dsr_probs, round(n_eff, 2)


def compute_cscv_pbo(
    daily_returns_matrix: np.ndarray, n_partitions: int = 8
) -> Optional[float]:
    """
    Computes Probability of Backtest Overfitting (PBO) using Combinatorially Symmetric
    Cross-Validation (CSCV, Bailey et al., 2017) on a D x N daily return matrix.
    """
    if daily_returns_matrix.ndim != 2:
        return None
    d_days, n_models = daily_returns_matrix.shape
    if n_models < 2 or d_days < 8:
        return None

    s_parts = min(int(n_partitions), (d_days // 2) * 2)
    if s_parts % 2 != 0:
        s_parts -= 1
    if s_parts < 4:
        return None

    blocks = np.array_split(daily_returns_matrix, s_parts, axis=0)
    half = s_parts // 2
    all_indices = set(range(s_parts))

    logits: List[float] = []
    for is_idx_tuple in combinations(range(s_parts), half):
        is_set = set(is_idx_tuple)
        oos_tuple = tuple(sorted(all_indices - is_set))

        is_mat = np.concatenate([blocks[i] for i in is_idx_tuple], axis=0)
        oos_mat = np.concatenate([blocks[i] for i in oos_tuple], axis=0)

        is_std = np.maximum(np.std(is_mat, axis=0, ddof=1), 1e-8)
        oos_std = np.maximum(np.std(oos_mat, axis=0, ddof=1), 1e-8)

        is_sr = np.mean(is_mat, axis=0) / is_std
        oos_sr = np.mean(oos_mat, axis=0) / oos_std

        best_is_idx = int(np.argmax(is_sr))
        target_oos_sr = oos_sr[best_is_idx]

        # Relative rank in (0, 1) using mid-rank for ties
        less_count = float(np.sum(oos_sr < target_oos_sr))
        equal_count = float(np.sum(np.isclose(oos_sr, target_oos_sr, atol=1e-12)))
        rank = less_count + 0.5 * (equal_count + 1.0)
        omega = float(np.clip(rank / (n_models + 1.0), 1e-6, 1.0 - 1e-6))
        logits.append(math.log(omega / (1.0 - omega)))

    if not logits:
        return None
    pbo = float(np.mean(np.asarray(logits) <= 0.0))
    return round(pbo, 4)


# Backward-compatible aliases with leading underscores
_natural_sort_key = natural_sort_key
_norm_cdf = norm_cdf
_norm_ppf = norm_ppf
_compute_iqm = compute_iqm
_compute_deflated_sharpe = compute_deflated_sharpe
_compute_cscv_pbo = compute_cscv_pbo
