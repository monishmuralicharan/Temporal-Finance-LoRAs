"""Metric helpers for checkpoint 4 outputs."""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def compute_simple_return(start_value: float, end_value: float) -> float:
    if start_value == 0:
        raise ValueError("Cannot compute return from a zero start value.")
    return float((end_value / start_value) - 1.0)


def compute_metrics(predictions: pd.DataFrame) -> Dict[str, Optional[float]]:
    pred = predictions["pred_5d_return"].to_numpy(dtype=float)
    actual = predictions["actual_5d_return"].to_numpy(dtype=float)
    metrics = {
        "sample_count": int(len(pred)),
        "mse": float(np.mean(np.square(pred - actual))),
        "mae": float(np.mean(np.abs(pred - actual))),
        "directional_accuracy": float(np.mean(np.sign(pred) == np.sign(actual))),
        "pearson_ic": _safe_corr(pred, actual, method="pearson"),
        "spearman_rank_ic": _safe_corr(pred, actual, method="spearman"),
    }
    return metrics


def _safe_corr(
    pred: np.ndarray, actual: np.ndarray, method: str
) -> Optional[float]:
    if len(pred) < 2:
        return None
    if np.std(pred) == 0 or np.std(actual) == 0:
        return None
    if method == "pearson":
        value = stats.pearsonr(pred, actual)[0]
    elif method == "spearman":
        value = stats.spearmanr(pred, actual)[0]
    else:
        raise ValueError("Unsupported correlation method: {0}".format(method))
    if np.isnan(value):
        return None
    return float(value)

