"""Window construction for rolling evaluation."""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class EvaluationWindow:
    context_end_date: pd.Timestamp
    target_date: pd.Timestamp
    context: np.ndarray
    last_context_close: float
    actual_close_t5: float


def build_evaluation_windows(
    series: pd.Series,
    context_length: int,
    forecast_horizon: int,
    eval_start_date: str,
) -> List[EvaluationWindow]:
    series = series.dropna().astype(float)
    if len(series) < context_length + forecast_horizon:
        raise RuntimeError(
            "Not enough data for {0} context steps and {1} horizon steps.".format(
                context_length,
                forecast_horizon,
            )
        )

    eval_start = pd.Timestamp(eval_start_date)
    windows = []
    values = series.to_numpy(dtype=float)
    dates = pd.to_datetime(series.index)

    for end_idx in range(context_length - 1, len(series) - forecast_horizon):
        context_end_date = dates[end_idx]
        if context_end_date < eval_start:
            continue
        target_idx = end_idx + forecast_horizon
        context = values[end_idx - context_length + 1 : end_idx + 1]
        windows.append(
            EvaluationWindow(
                context_end_date=context_end_date,
                target_date=dates[target_idx],
                context=context,
                last_context_close=float(values[end_idx]),
                actual_close_t5=float(values[target_idx]),
            )
        )

    if not windows:
        raise RuntimeError(
            "No evaluation windows found on or after {0}.".format(eval_start_date)
        )
    return windows

