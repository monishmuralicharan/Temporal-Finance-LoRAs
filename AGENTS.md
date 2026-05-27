# Repository Guidelines

## Project Structure & Module Organization
The repo now has three active areas: research notes in `docs/`, runtime configs in `configs/`, and Python code in `temporal_finance/`. [`docs/checkpoints.md`](docs/checkpoints.md) is the roadmap; [`docs/thoughts.md`](docs/thoughts.md) is the decision log for concise research and design calls. Record material decisions there as short dated bullets before they spread across code or PR comments. Keep tests in `tests/`, and treat `external/` as an ignored workspace for upstream model checkouts such as Kronos.

## Build, Test, and Development Commands
Use the repository root as the working directory.

- `python -m temporal_finance.modal_kronos_checkpoint4 download-models --config configs/kronos_checkpoint4_best.yaml` downloads Kronos model files into the shared Modal cache Volume.
- `python -m temporal_finance.modal_kronos_checkpoint4 run --config configs/kronos_checkpoint4_best.yaml --gpu T4` runs the current Kronos smoke config remotely on Modal and writes artifacts back locally.
- `python -m temporal_finance.run_kronos_output_evaluation --predictions outputs/kronos_checkpoint4_best/predictions.csv --output-dir outputs/kronos_checkpoint4_best/evaluation` grades a Kronos output file.
- `python -m pytest` runs the local test suite.
- `git diff --check` catches whitespace issues and merge markers before commit.
- `rg --files` is the preferred fast file index for repo exploration.

## Coding Style & Naming Conventions
Prefer Python for research code unless a different stack is necessary. Use 4-space indentation, `snake_case` for files, functions, and variables, and `PascalCase` for classes. Name configs by purpose and time window, for example `configs/rolling_q1_2020.yaml`. Keep modules narrow in scope, and add short comments only where split logic, tensor shapes, or routing behavior are not obvious.

## Testing Guidelines
Place tests under `tests/` and mirror the behavior being validated rather than the file layout exactly. Focus first on leakage prevention, split integrity, metric computation, config validation, and deterministic preprocessing. Keep unit tests independent of live model weights or network calls; use mocked forecasters for pipeline smoke coverage.

## Commit & Pull Request Guidelines
Git history currently contains a single bootstrap commit (`first  commit`), so there is no strong established convention yet. Use short imperative commit subjects such as `Add rolling-window split generator`. Keep pull requests small, explain the research or engineering intent, list the paths changed, and include sample metrics or output snippets when behavior changes. Link the relevant checkpoint section when possible.

## Configuration & External Models
Do not commit model weights, downloaded market data, or upstream model repos. Keep local artifacts under ignored directories such as `external/` and `outputs/`. When integrating a third-party model, pin the upstream commit in `docs/thoughts.md` and keep the repo wrapper code in this project thin enough to swap or update later.
