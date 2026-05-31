import numpy as np
import pandas as pd

from temporal_finance.temporal_lora_config import TemporalLoraSignalConfig
from temporal_finance.temporal_lora_data import (
    TemporalLoraExample,
    TemporalLoraWindowPlan,
    build_normalized_sequence,
    build_training_examples_for_window,
    build_validation_examples_for_window,
    build_window_plans,
)


def test_window_examples_do_not_leak_forward_targets():
    histories = {"AAA": _history("2024-01-01", 12)}
    plan = TemporalLoraWindowPlan(
        window_id="W1",
        slow_train_start="2024-01-04",
        slow_train_end="2024-01-10",
        medium_train_start="2024-01-08",
        medium_train_end="2024-01-10",
        fast_train_start="2024-01-09",
        fast_train_end="2024-01-10",
        validation_start="2024-01-11",
        validation_end="2024-01-12",
    )

    training = build_training_examples_for_window(
        histories,
        plan,
        adapter_name="slow",
        context_length=3,
    )
    validation = build_validation_examples_for_window(
        histories,
        plan,
        context_length=3,
    )

    assert training
    assert validation
    assert all(example.context_end_index < example.target_index for example in training)
    assert all(example.context_end_index < example.target_index for example in validation)
    assert max(example.target_date for example in training) <= "2024-01-10"
    assert min(example.target_date for example in validation) >= "2024-01-11"


def test_build_window_plans_uses_trailing_trading_dates():
    histories = {"AAA": _history("2024-01-01", 10)}
    config = TemporalLoraSignalConfig.from_dict(
        {
            "data_dir": "unused",
            "output_dir": "unused",
            "context_length": 3,
            "windows": [
                {
                    "window_id": "W1",
                    "slow_train_start": "2024-01-01",
                    "slow_train_end": "2024-01-12",
                    "medium_trading_days": 4,
                    "fast_trading_days": 2,
                    "validation_start": "2024-01-16",
                    "validation_end": "2024-01-17",
                }
            ],
        }
    )

    plan = build_window_plans(config, histories)[0]

    assert plan.medium_train_start == "2024-01-09"
    assert plan.fast_train_start == "2024-01-11"
    assert plan.medium_train_end == "2024-01-12"


def test_capped_training_examples_span_the_full_window():
    histories = {
        "AAA": _history("2024-01-01", 30),
        "BBB": _history("2024-01-01", 30),
    }
    plan_w1 = TemporalLoraWindowPlan(
        window_id="W1",
        slow_train_start="2024-01-04",
        slow_train_end="2024-01-26",
        medium_train_start="2024-01-18",
        medium_train_end="2024-01-26",
        fast_train_start="2024-01-24",
        fast_train_end="2024-01-26",
        validation_start="2024-01-29",
        validation_end="2024-02-02",
    )
    plan_w2 = TemporalLoraWindowPlan(
        window_id="W2",
        slow_train_start="2024-01-04",
        slow_train_end="2024-02-09",
        medium_train_start="2024-01-31",
        medium_train_end="2024-02-09",
        fast_train_start="2024-02-07",
        fast_train_end="2024-02-09",
        validation_start="2024-02-12",
        validation_end="2024-02-16",
    )

    capped_w1 = build_training_examples_for_window(
        histories,
        plan_w1,
        adapter_name="slow",
        context_length=3,
        max_examples=6,
    )
    capped_w2 = build_training_examples_for_window(
        histories,
        plan_w2,
        adapter_name="slow",
        context_length=3,
        max_examples=6,
    )

    assert len(capped_w1) == 6
    assert len(capped_w2) == 6
    assert min(example.target_date for example in capped_w1) == "2024-01-04"
    assert max(example.target_date for example in capped_w1) == "2024-01-26"
    assert max(example.target_date for example in capped_w2) == "2024-02-09"
    assert {example.target_date for example in capped_w1} != {
        example.target_date for example in capped_w2
    }


def test_context_only_normalization_excludes_target_row():
    history = _history("2024-01-01", 5)
    history.loc[history.index[-1], "close"] = 1000.0
    example = TemporalLoraExample(
        ticker="AAA",
        window_id="W1",
        split="train",
        adapter_name="slow",
        context_start_index=0,
        context_end_index=3,
        target_index=4,
        context_start_date="2024-01-01",
        context_end_date="2024-01-04",
        target_date="2024-01-05",
    )

    normalized, mean, std, raw = build_normalized_sequence(history, example, clip=10000)

    assert raw.shape == (5, 6)
    assert np.isclose(mean[3], np.mean([10.0, 11.0, 12.0, 13.0]))
    assert np.isclose(std[3], np.std([10.0, 11.0, 12.0, 13.0]))
    assert normalized[-1, 3] > 100.0


def _history(start, periods):
    index = pd.bdate_range(start, periods=periods)
    close = np.arange(10.0, 10.0 + periods)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.arange(100.0, 100.0 + periods),
            "amount": close * np.arange(100.0, 100.0 + periods),
        },
        index=index,
    )
