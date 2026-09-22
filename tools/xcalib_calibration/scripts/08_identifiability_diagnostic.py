#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from common_paths import CALIBRATION_DIR, load_config


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


refine = load_module(
    "refine_extrinsics",
    CALIBRATION_DIR / "scripts" / "06_refine_extrinsics.py",
)


def clean_stats(stats: dict) -> dict:
    return {key: value for key, value in stats.items() if key != "sorted"}


def optimize_model(
    mode: str,
    train_points: torch.Tensor,
    train_targets: torch.Tensor,
    original_rotation: torch.Tensor,
    original_translation: torch.Tensor,
    camera_matrix: torch.Tensor,
    max_rotation_rad: float,
    max_translation_m: float,
    huber_delta: float,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_rotation = torch.full(
        (3,), 1e-6, dtype=torch.float64, device=original_rotation.device
    )
    raw_translation = torch.full(
        (3,), 1e-6, dtype=torch.float64, device=original_rotation.device
    )
    parameters = []
    if mode in {"rotation", "both"}:
        raw_rotation.requires_grad_(True)
        parameters.append(raw_rotation)
    if mode in {"translation", "both"}:
        raw_translation.requires_grad_(True)
        parameters.append(raw_translation)

    optimizer = torch.optim.LBFGS(
        parameters,
        max_iter=max_iter,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def pose():
        rotation_delta = None
        rotation_residual = torch.zeros(
            3, dtype=torch.float64, device=original_rotation.device
        )
        translation_residual = torch.zeros(
            3, dtype=torch.float64, device=original_rotation.device
        )
        if mode in {"rotation", "both"}:
            rotation_residual = refine.bounded_vector(
                raw_rotation, max_rotation_rad
            )
            rotation_delta = refine.axis_angle_matrix(rotation_residual)
            rotation = rotation_delta @ original_rotation
            translation = rotation_delta @ original_translation
        else:
            rotation = original_rotation
            translation = original_translation

        if mode in {"translation", "both"}:
            translation_residual = refine.bounded_vector(
                raw_translation, max_translation_m
            )
            translation = translation + translation_residual

        return rotation, translation, rotation_residual, translation_residual

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        rotation, translation, _, _ = pose()
        projected, _ = refine.project_points(
            train_points, rotation, translation, camera_matrix
        )
        loss = torch.nn.functional.huber_loss(
            projected, train_targets, delta=huber_delta
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return pose()


def main() -> int:
    config = load_config()
    refinement_config = config["refinement"]
    matcher_config = config["matcher"]
    scene_name = config["baseline_scene"]
    camera_names = [camera["name"] for camera in config["cameras"]]
    scene_dir = (CALIBRATION_DIR / config["scenes_dir"] / scene_name).resolve()
    calibration_data = refine.read_scaled_calibrations()
    device = torch.device(str(matcher_config["device"]))

    pairs = refine.load_selected_pairs(
        scene_name,
        float(refinement_config["score_threshold"]),
        float(refinement_config["max_center_distance_px"]),
    )
    slot_by_name = refine.camera_slot_by_name(camera_names)
    for row in pairs:
        row["camera_slot"] = slot_by_name[row["camera"]]

    max_rotation_rad = float(
        np.deg2rad(float(refinement_config["max_rotation_deg"]))
    )
    max_translation_m = float(refinement_config["max_translation_m"])
    huber_delta = float(refinement_config["huber_delta_px"])
    max_iter = int(refinement_config["max_iter"])

    results = []
    for camera_name in camera_names:
        camera_pairs = [row for row in pairs if row["camera"] == camera_name]
        frames = sorted({int(row["frame"]) for row in camera_pairs})
        observations = refine.observation_data(camera_pairs, scene_dir)
        train_frames, validation_frames = refine.split_frames(
            frames, int(refinement_config["validation_every"])
        )
        train_points, train_targets = refine.stack_observations(
            observations, train_frames
        )
        validation_points, validation_targets = refine.stack_observations(
            observations, validation_frames
        )

        camera_calib = calibration_data["scenes"][scene_name]["cameras"][camera_name]
        camera_matrix = torch.as_tensor(
            camera_calib["K_scaled"], dtype=torch.float64, device=device
        )
        original_rotation = torch.as_tensor(
            camera_calib["T_lidar_camera"], dtype=torch.float64, device=device
        )[:3, :3]
        original_translation = torch.as_tensor(
            camera_calib["T_lidar_camera"], dtype=torch.float64, device=device
        )[:3, 3]
        train_points = train_points.to(device)
        train_targets = train_targets.to(device)
        validation_points = validation_points.to(device)
        validation_targets = validation_targets.to(device)

        model_results = {}
        for mode in ("rotation", "translation", "both"):
            rotation, translation, rotation_vector, translation_residual = (
                optimize_model(
                    mode,
                    train_points,
                    train_targets,
                    original_rotation,
                    original_translation,
                    camera_matrix,
                    max_rotation_rad,
                    max_translation_m,
                    huber_delta,
                    max_iter,
                )
            )
            model_results[mode] = {
                "train": clean_stats(
                    refine.error_stats(
                        train_points,
                        train_targets,
                        rotation,
                        translation,
                        camera_matrix,
                    )
                ),
                "validation": clean_stats(
                    refine.error_stats(
                        validation_points,
                        validation_targets,
                        rotation,
                        translation,
                        camera_matrix,
                    )
                ) if len(validation_points) else None,
                "rotation_residual_deg": float(
                    torch.rad2deg(torch.linalg.norm(rotation_vector))
                ),
                "translation_residual_m": float(
                    torch.linalg.norm(translation_residual)
                ),
            }

        results.append(
            {
                "camera": camera_name,
                "train_pairs": int(len(train_points)),
                "validation_pairs": int(len(validation_points)),
                "train_frames": sorted(train_frames),
                "validation_frames": sorted(validation_frames),
                "models": model_results,
            }
        )

    output = {
        "scene": scene_name,
        "image_size": [
            int(config["scene_image_width"]),
            int(config["scene_image_height"]),
        ],
        "bounds": {
            "max_rotation_deg": float(refinement_config["max_rotation_deg"]),
            "max_translation_m": float(refinement_config["max_translation_m"]),
        },
        "cameras": results,
    }
    output_path = CALIBRATION_DIR / "validation" / "identifiability.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
