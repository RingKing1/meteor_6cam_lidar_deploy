from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CALIBRATION_DIR = Path(__file__).resolve().parents[1]
DEPLOY_DIR = CALIBRATION_DIR.parent
WORKSPACE_DIR = DEPLOY_DIR.parent


def load_config() -> dict[str, Any]:
    with (CALIBRATION_DIR / "config" / "calibration.json").open(
        "r", encoding="utf-8"
    ) as handle:
        return json.load(handle)


def raw_data_dir(config: dict[str, Any]) -> Path:
    return (CALIBRATION_DIR / config["raw_data_dir"]).resolve()


def xcalib_source_dir() -> Path:
    return WORKSPACE_DIR / "XCalib"
