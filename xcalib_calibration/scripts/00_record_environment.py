#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from common_paths import CALIBRATION_DIR, WORKSPACE_DIR, xcalib_source_dir


def run_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except OSError as exc:
        return {"command": command, "error": str(exc)}


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def optional_import_report() -> dict[str, Any]:
    report: dict[str, Any] = {}
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        report["torch"] = None

    try:
        import xcalib

        report["xcalib"] = {"version": xcalib.__version__}
    except ImportError:
        report["xcalib"] = None

    return report


def git_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    commit = run_command(["git", "-C", str(path), "rev-parse", "HEAD"])
    describe = run_command(["git", "-C", str(path), "describe", "--tags", "--always"])
    status = run_command(["git", "-C", str(path), "status", "--short"])
    return {
        "path": str(path),
        "exists": True,
        "commit": commit.get("stdout"),
        "describe": describe.get("stdout"),
        "status": status.get("stdout", "").splitlines(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="output filename label",
    )
    args = parser.parse_args()

    report = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "in_virtualenv": sys.prefix != sys.base_prefix,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "packages": {
            name: package_version(name)
            for name in ("XCalib", "torch", "numpy", "opencv-python", "pillow")
        },
        "imports": optional_import_report(),
        "gpu": run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,cuda_version",
                "--format=csv,noheader",
            ]
        ),
        "xcalib_repository": git_report(xcalib_source_dir()),
        "meteor_deploy_repository": git_report(CALIBRATION_DIR.parent),
    }

    output = CALIBRATION_DIR / "logs" / f"environment_{args.label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
