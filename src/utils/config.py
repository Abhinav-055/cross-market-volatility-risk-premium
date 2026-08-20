
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or (REPO_ROOT / "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
