#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np

from _calibration_utils import CAMERA_TO_MANIFEST, load_manifest
from common_paths import CALIBRATION_DIR, load_config, raw_data_dir


def main() -> int:
    config = load_config()
    raw_root = raw_data_dir(config)
    scenes_root = (CALIBRATION_DIR / config["scenes_dir"]).resolve()
    scale = float(config["scene_image_scale"])

    output = {
        "image_width": int(config["scene_image_width"]),
        "image_height": int(config["scene_image_height"]),
        "scale_from_raw": scale,
        "note": "T_lidar_camera is direct and unchanged by resize; resize changes K and pixel coordinates.",
        "scenes": {},
    }

    for scene in config["scenes"]:
        manifest = load_manifest(scenes_root / scene)
        cameras_output = {}

        for camera in config["cameras"]:
            raw_path = raw_root / scene / "calib" / "camera" / camera["calibration"]
            raw_data = json.loads(raw_path.read_text(encoding="utf-8"))
            raw_K = np.asarray(raw_data["intrinsic"], dtype=np.float64).reshape(3, 3)
            lidar_to_camera = np.asarray(
                raw_data["extrinsic"], dtype=np.float64
            ).reshape(4, 4)

            manifest_name = CAMERA_TO_MANIFEST[camera["name"]]
            scaled_K = np.asarray(
                manifest["cams"][manifest_name]["K"], dtype=np.float64
            )
            expected_K = raw_K * scale
            expected_K[2, 2] = 1.0

            cameras_output[camera["name"]] = {
                "manifest_name": manifest_name,
                "K_scaled": scaled_K.tolist(),
                "T_lidar_camera": lidar_to_camera.tolist(),
                "checks": {
                    "scaled_K_matches_manifest": bool(
                        np.allclose(scaled_K, expected_K, atol=1e-5)
                    ),
                    "T_lidar_camera_from_raw_camera_calibration": True,
                },
            }

        output["scenes"][scene] = {"cameras": cameras_output}

    output_path = CALIBRATION_DIR / "converted" / "scaled_calibrations.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
