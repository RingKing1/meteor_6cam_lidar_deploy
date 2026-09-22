#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _calibration_utils import macro_class
from common_paths import CALIBRATION_DIR, load_config


def read_scaled_calibrations() -> dict:
    return json.loads(
        (CALIBRATION_DIR / "converted" / "scaled_calibrations.json").read_text(
            encoding="utf-8"
        )
    )


def load_selected_pairs(
    scene_name: str,
    score_threshold: float,
    max_center_distance_px: float,
) -> list[dict]:
    path = CALIBRATION_DIR / "matches" / "xcalib_match_pairs.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        selected = [
            row
            for row in rows
            if row["scene"] == scene_name
            and float(row["score"]) >= score_threshold
            and float(row["projected_center_distance_px"]) <= max_center_distance_px
            and float(row["projected_camera_depth_m"]) > 0.5
        ]
    deduped = []
    used_pairs: dict[tuple[str, int], tuple[set[int], set[int]]] = {}
    for row in sorted(
        selected,
        key=lambda item: (
            -float(item["score"]),
            float(item["projected_center_distance_px"]),
        ),
    ):
        key = (row["camera"], int(row["frame"]))
        image_index = int(row["image_box_index"])
        lidar_index = int(row["lidar_box_index"])
        used = used_pairs.setdefault(key, (set(), set()))
        if image_index in used[0] or lidar_index in used[1]:
            continue
        deduped.append(row)
        used[0].add(image_index)
        used[1].add(lidar_index)
    return deduped


def axis_angle_matrix(rotation_vector: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(rotation_vector)
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    zero = torch.zeros((), dtype=rotation_vector.dtype, device=rotation_vector.device)
    cross_matrix = torch.stack(
        [
            torch.stack([zero, -rotation_vector[2], rotation_vector[1]]),
            torch.stack([rotation_vector[2], zero, -rotation_vector[0]]),
            torch.stack([-rotation_vector[1], rotation_vector[0], zero]),
        ]
    )
    first_order = torch.sinc(angle / torch.pi)
    second_order = 0.5 * torch.sinc(angle / (2.0 * torch.pi)) ** 2
    return (
        identity
        + first_order * cross_matrix
        + second_order * cross_matrix @ cross_matrix
    )


def bounded_vector(raw: torch.Tensor, maximum: float) -> torch.Tensor:
    norm = torch.linalg.norm(raw)
    scale = maximum * torch.tanh(norm) / norm.clamp_min(1e-12)
    return raw * scale


def pose_matrices(
    original_rotation: torch.Tensor,
    original_translation: torch.Tensor,
    raw_rotation: torch.Tensor,
    raw_translation: torch.Tensor,
    max_rotation_rad: float,
    max_translation_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rotation_residual_vector = bounded_vector(raw_rotation, max_rotation_rad)
    translation_residual = bounded_vector(raw_translation, max_translation_m)
    rotation_delta = axis_angle_matrix(rotation_residual_vector)
    rotation = rotation_delta @ original_rotation
    translation = rotation_delta @ original_translation + translation_residual
    return rotation, translation, rotation_residual_vector, translation_residual


def project_points(
    points: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    camera_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera_points = points @ rotation.T + translation
    depth = camera_points[:, 2]
    safe_depth = torch.maximum(depth, torch.full_like(depth, 1e-6))
    u = camera_matrix[0, 0] * camera_points[:, 0] / safe_depth + camera_matrix[0, 2]
    v = camera_matrix[1, 1] * camera_points[:, 1] / safe_depth + camera_matrix[1, 2]
    return torch.stack([u, v], dim=1), depth


def error_stats(
    points: torch.Tensor,
    targets: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    camera_matrix: torch.Tensor,
) -> dict:
    projected, depth = project_points(points, rotation, translation, camera_matrix)
    errors = torch.linalg.norm(projected - targets, dim=1)
    errors = torch.where(depth > 0.5, errors, torch.full_like(errors, 10000.0))
    sorted_errors, _ = torch.sort(errors)
    return {
        "mean": float(errors.mean()),
        "median": float(errors.median()),
        "p90": float(torch.quantile(errors, 0.90)),
        "p95": float(torch.quantile(errors, 0.95)),
        "behind_camera": int((depth <= 0.5).sum()),
        "pairs": int(errors.numel()),
        "sorted": sorted_errors.detach().cpu(),
    }


def observation_data(
    pairs: list[dict],
    scene_dir: Path,
) -> dict[int, dict]:
    by_frame: dict[int, dict] = {}
    for frame in sorted({int(row["frame"]) for row in pairs}):
        frame_pairs = [row for row in pairs if int(row["frame"]) == frame]
        boxes_3d = np.load(scene_dir / "bev_box" / f"{frame:04d}.npz")["boxes_3d"]
        bbox_data = np.load(
            scene_dir / "bbox2d" / f"{frame:04d}.npz", allow_pickle=True
        )
        bbox_array = bbox_data["boxes"]
        observations = []
        for row in frame_pairs:
            camera_slot = int(row["camera_slot"])
            image_index = int(row["image_box_index"])
            lidar_index = int(row["lidar_box_index"])
            observations.append(
                {
                    "frame": frame,
                    "camera_slot": camera_slot,
                    "image_index": image_index,
                    "lidar_index": lidar_index,
                    "point": boxes_3d[lidar_index, 1:4].astype(np.float64),
                    "target": bbox_array[camera_slot, image_index, 1:3].astype(np.float64),
                    "class_2d_macro": macro_class(
                        float(bbox_array[camera_slot, image_index, 0])
                    ),
                    "class_3d_macro": macro_class(
                        float(boxes_3d[lidar_index, 0])
                    ),
                }
            )
        by_frame[frame] = {"observations": observations}
    return by_frame


def camera_slot_by_name(camera_names: list[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(camera_names)}


def split_frames(frames: list[int], every: int) -> tuple[set[int], set[int]]:
    train, validation = set(), set()
    for position, frame in enumerate(frames):
        if every > 1 and position % every == every - 1:
            validation.add(frame)
        else:
            train.add(frame)
    return train, validation


def stack_observations(
    observations_by_frame: dict[int, dict], frames: set[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = [
        observation
        for frame in frames
        for observation in observations_by_frame[frame]["observations"]
    ]
    points = torch.as_tensor(
        np.stack([item["point"] for item in selected]), dtype=torch.float64
    )
    targets = torch.as_tensor(
        np.stack([item["target"] for item in selected]), dtype=torch.float64
    )
    return points, targets


def refine_camera(
    camera_name: str,
    pairs: list[dict],
    scene_dir: Path,
    camera_calibration: dict,
    config: dict,
    device: torch.device,
) -> dict:
    refinement_config = config["refinement"]
    frames = sorted({int(row["frame"]) for row in pairs})
    train_frames, validation_frames = split_frames(
        frames, int(refinement_config["validation_every"])
    )
    observations_by_frame = observation_data(pairs, scene_dir)
    train_points, train_targets = stack_observations(
        observations_by_frame, train_frames
    )
    validation_points, validation_targets = stack_observations(
        observations_by_frame, validation_frames
    )

    camera_matrix = torch.as_tensor(
        camera_calibration["K_scaled"], dtype=torch.float64, device=device
    )
    original_transform = torch.as_tensor(
        camera_calibration["T_lidar_camera"], dtype=torch.float64, device=device
    )
    original_rotation = original_transform[:3, :3]
    original_translation = original_transform[:3, 3]
    train_points = train_points.to(device)
    train_targets = train_targets.to(device)
    validation_points = validation_points.to(device)
    validation_targets = validation_targets.to(device)

    enough = (
        len(train_points) >= int(refinement_config["min_train_pairs"])
        and len(validation_points) >= int(refinement_config["min_validation_pairs"])
    )
    baseline_train = error_stats(
        train_points, train_targets,
        original_rotation, original_translation, camera_matrix,
    )
    baseline_validation = error_stats(
        validation_points, validation_targets,
        original_rotation, original_translation, camera_matrix,
    )

    result = {
        "camera": camera_name,
        "enough_data": enough,
        "train_frames": sorted(train_frames),
        "validation_frames": sorted(validation_frames),
        "train_pairs": int(len(train_points)),
        "validation_pairs": int(len(validation_points)),
        "baseline_train": {key: value for key, value in baseline_train.items() if key != "sorted"},
        "baseline_validation": {
            key: value for key, value in baseline_validation.items() if key != "sorted"
        },
    }
    if not enough:
        result["accepted"] = False
        result["message"] = "insufficient train or validation pairs"
        return result

    raw_rotation = torch.full(
        (3,), 1e-6, dtype=torch.float64, device=device, requires_grad=True
    )
    raw_translation = torch.full(
        (3,), 1e-6, dtype=torch.float64, device=device, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [raw_rotation, raw_translation],
        max_iter=int(refinement_config["max_iter"]),
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    huber_delta = float(refinement_config["huber_delta_px"])
    max_rotation_rad = float(np.deg2rad(float(refinement_config["max_rotation_deg"])))
    max_translation_m = float(refinement_config["max_translation_m"])

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        rotation, translation, _, _ = pose_matrices(
            original_rotation,
            original_translation,
            raw_rotation,
            raw_translation,
            max_rotation_rad,
            max_translation_m,
        )
        projected, _ = project_points(
            train_points, rotation, translation, camera_matrix
        )
        loss = F.huber_loss(projected, train_targets, delta=huber_delta)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        refined_rotation, refined_translation, rotation_vector, translation_residual = (
            pose_matrices(
                original_rotation,
                original_translation,
                raw_rotation,
                raw_translation,
                max_rotation_rad,
                max_translation_m,
            )
        )

    refined_train = error_stats(
        train_points, train_targets,
        refined_rotation, refined_translation, camera_matrix,
    )
    refined_validation = error_stats(
        validation_points, validation_targets,
        refined_rotation, refined_translation, camera_matrix,
    )
    accepted = (
        refined_validation["median"] <= baseline_validation["median"]
        and refined_validation["mean"] <= baseline_validation["mean"]
    )
    transform = torch.eye(4, dtype=torch.float64, device=device)
    transform[:3, :3] = refined_rotation
    transform[:3, 3] = refined_translation

    result.update(
        {
            "accepted": bool(accepted),
            "refined_train": {key: value for key, value in refined_train.items() if key != "sorted"},
            "refined_validation": {
                key: value for key, value in refined_validation.items() if key != "sorted"
            },
            "rotation_residual_deg": float(torch.rad2deg(torch.linalg.norm(rotation_vector))),
            "translation_residual_m": float(torch.linalg.norm(translation_residual)),
            "T_lidar_camera_refined": transform.detach().cpu().tolist(),
        }
    )
    return result


def main() -> int:
    config = load_config()
    matcher_config = config["matcher"]
    refinement_config = config["refinement"]
    scene_name = config["baseline_scene"]
    camera_names = [camera["name"] for camera in config["cameras"]]
    scene_dir = (CALIBRATION_DIR / config["scenes_dir"] / scene_name).resolve()
    calibration_data = read_scaled_calibrations()
    device = torch.device(str(matcher_config["device"]))

    pairs = load_selected_pairs(
        scene_name,
        float(refinement_config["score_threshold"]),
        float(refinement_config["max_center_distance_px"]),
    )
    slot_by_name = camera_slot_by_name(camera_names)
    pairs_by_camera: dict[str, list[dict]] = defaultdict(list)
    for row in pairs:
        row = dict(row)
        row["camera_slot"] = slot_by_name[row["camera"]]
        pairs_by_camera[row["camera"]].append(row)

    results = []
    for camera_name in camera_names:
        camera_calibration = calibration_data["scenes"][scene_name]["cameras"][camera_name]
        results.append(
            refine_camera(
                camera_name,
                pairs_by_camera[camera_name],
                scene_dir,
                camera_calibration,
                config,
                device,
            )
        )

    output = {
        "scene": scene_name,
        "image_size": [
            int(config["scene_image_width"]),
            int(config["scene_image_height"]),
        ],
        "direct_transform": "T_lidar_camera",
        "selection": {
            "score_threshold": float(refinement_config["score_threshold"]),
            "max_center_distance_px": float(refinement_config["max_center_distance_px"]),
        },
        "bounds": {
            "max_rotation_deg": float(refinement_config["max_rotation_deg"]),
            "max_translation_m": float(refinement_config["max_translation_m"]),
        },
        "cameras": results,
    }
    refined_dir = CALIBRATION_DIR / "refined"
    refined_dir.mkdir(parents=True, exist_ok=True)
    output_path = refined_dir / "refined_extrinsics.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    validation_dir = CALIBRATION_DIR / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    validation_path = validation_dir / "refinement_validation.csv"
    fields = [
        "scene",
        "camera",
        "accepted",
        "train_pairs",
        "validation_pairs",
        "baseline_train_median_px",
        "refined_train_median_px",
        "baseline_validation_median_px",
        "refined_validation_median_px",
        "baseline_validation_mean_px",
        "refined_validation_mean_px",
        "rotation_residual_deg",
        "translation_residual_m",
    ]
    with validation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "scene": scene_name,
                    "camera": result["camera"],
                    "accepted": result["accepted"],
                    "train_pairs": result["train_pairs"],
                    "validation_pairs": result["validation_pairs"],
                    "baseline_train_median_px": result["baseline_train"]["median"],
                    "refined_train_median_px": result.get("refined_train", {}).get("median"),
                    "baseline_validation_median_px": result["baseline_validation"]["median"],
                    "refined_validation_median_px": result.get("refined_validation", {}).get("median"),
                    "baseline_validation_mean_px": result["baseline_validation"]["mean"],
                    "refined_validation_mean_px": result.get("refined_validation", {}).get("mean"),
                    "rotation_residual_deg": result.get("rotation_residual_deg"),
                    "translation_residual_m": result.get("translation_residual_m"),
                }
            )

    print(output_path)
    print(validation_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
