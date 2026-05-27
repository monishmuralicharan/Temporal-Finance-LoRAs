import pandas as pd

from temporal_finance.windows import build_evaluation_windows


def test_build_evaluation_windows_respects_context_and_horizon():
    series = pd.Series(
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        index=pd.bdate_range("2020-01-01", periods=7),
    )

    windows = build_evaluation_windows(
        series=series,
        context_length=3,
        forecast_horizon=2,
        eval_start_date="2020-01-06",
    )

    assert len(windows) == 2
    assert list(windows[0].context) == [11.0, 12.0, 13.0]
    assert windows[0].last_context_close == 13.0
    assert windows[0].actual_close_t5 == 15.0
