import json
from pathlib import Path

import pandas as pd
import pytest

from temporal_finance.evaluation import compute_metrics, return_column_names
from temporal_finance.kronos_base_search import (
    build_candidate_grid,
    generate_search_suite,
    select_best_candidate,
    _write_json,
)
from temporal_finance.kronos_benchmark import (
    build_benchmark_config,
    load_kronos_benchmark_manifest,
)
from temporal_finance.kronos_config import load_kronos_checkpoint4_config


def test_compute_metrics_prefers_generic_return_columns():
    predictions = pd.DataFrame(
        {
            "pred_return": [0.01, 0.02, -0.03],
            "actual_return": [0.01, 0.03, -0.01],
            "pred_5d_return": [10.0, 10.0, 10.0],
            "actual_5d_return": [-10.0, -10.0, -10.0],
        }
    )

    assert return_column_names(predictions) == ("pred_return", "actual_return")
    metrics = compute_metrics(predictions)

    assert metrics["directional_accuracy"] == pytest.approx(1.0)
    assert metrics["mae"] < 0.02


def test_generate_search_suite_writes_expected_candidate_manifests(tmp_path):
    paths = generate_search_suite(str(tmp_path / "suite"))

    grid = pd.read_csv(paths["candidate_grid"])
    assert len(grid) == 32
    assert {"small_h1_greedy", "base_h10_t06_p09_s10"}.issubset(
        set(grid["name"])
    )

    manifest_path = (
        tmp_path
        / "suite"
        / "manifests"
        / "stage1"
        / "small_h3_t04_p08_s3.yaml"
    )
    manifest = load_kronos_benchmark_manifest(str(manifest_path))
    base_config = load_kronos_checkpoint4_config(manifest.base_config_path)
    config = build_benchmark_config(
        base_config,
        manifest.experiments[0],
        Path(manifest.output_dir),
    )

    assert config.forecast_horizon == 3
    assert config.eval_stride == 21
    assert config.model_id == "NeoQuasar/Kronos-small"
    assert config.sample_count == 3
    assert manifest.experiments[0].calibration_modes == [
        "none",
        "vol_rescale",
        "vol_rescale_mean_center",
    ]


def test_select_best_candidate_uses_naive_gap_then_stability():
    rows = pd.DataFrame(
        [
            {
                "name": "monthly_like",
                "rank_ic_gap_to_best_naive": 0.01,
                "daily_rank_ic_positive_rate": 0.80,
                "directional_accuracy": 0.60,
                "spearman_rank_ic": 0.20,
                "pred_abs_return_multiple": 1.0,
            },
            {
                "name": "stable_stride",
                "rank_ic_gap_to_best_naive": 0.03,
                "daily_rank_ic_positive_rate": 0.55,
                "directional_accuracy": 0.53,
                "spearman_rank_ic": 0.08,
                "pred_abs_return_multiple": 1.2,
            },
        ]
    )

    selected = select_best_candidate(rows)

    assert selected["name"] == "stable_stride"


def test_candidate_grid_contains_models_horizons_and_decoding_presets():
    grid = build_candidate_grid()
    names = {candidate.name for candidate in grid}

    assert len(grid) == 32
    assert "small_h1_greedy" in names
    assert "small_h10_t06_p09_s10" in names
    assert "base_h1_greedy" in names
    assert "base_h10_t06_p09_s10" in names


def test_write_json_sanitizes_non_finite_values(tmp_path):
    output_path = tmp_path / "selected.json"

    _write_json(
        output_path,
        {
            "score": float("nan"),
            "nested": [float("inf"), float("-inf"), "ok"],
        },
    )

    raw = output_path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert json.loads(raw) == {"nested": [None, None, "ok"], "score": None}
