"""Kronos-focused temporal finance utilities."""

from typing import TYPE_CHECKING

__all__ = [
    "KronosCheckpoint4Config",
    "evaluate_kronos_output",
    "load_kronos_checkpoint4_config",
    "run_kronos_checkpoint4",
    "run_kronos_output_evaluation",
]

if TYPE_CHECKING:
    from temporal_finance.kronos_config import (
        KronosCheckpoint4Config,
        load_kronos_checkpoint4_config,
    )
    from temporal_finance.kronos_output_evaluation import (
        evaluate_kronos_output,
        run_kronos_output_evaluation,
    )
    from temporal_finance.kronos_pipeline import run_kronos_checkpoint4


def __getattr__(name):
    if name in ("KronosCheckpoint4Config", "load_kronos_checkpoint4_config"):
        from temporal_finance.kronos_config import (
            KronosCheckpoint4Config,
            load_kronos_checkpoint4_config,
        )

        exports = {
            "KronosCheckpoint4Config": KronosCheckpoint4Config,
            "load_kronos_checkpoint4_config": load_kronos_checkpoint4_config,
        }
        return exports[name]
    if name == "run_kronos_checkpoint4":
        from temporal_finance.kronos_pipeline import run_kronos_checkpoint4

        return run_kronos_checkpoint4
    if name in ("evaluate_kronos_output", "run_kronos_output_evaluation"):
        from temporal_finance.kronos_output_evaluation import (
            evaluate_kronos_output,
            run_kronos_output_evaluation,
        )

        exports = {
            "evaluate_kronos_output": evaluate_kronos_output,
            "run_kronos_output_evaluation": run_kronos_output_evaluation,
        }
        return exports[name]
    raise AttributeError(name)
