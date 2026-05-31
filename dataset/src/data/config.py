from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_data_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or ROOT / "config" / "data_config.yaml"
    with cfg_path.open() as f:
        return yaml.safe_load(f)
