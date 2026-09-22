#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from _calibration_utils import box3d_corners, project_points, transform_points
from common_paths import CALIBRATION_DIR, load_config


BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def draw_detections(image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    for bbox in bboxes:
        cv2.rectangle(
            canvas,
            (int(bbox[1]), int(bbox[2])),
            (int(bbox[3]), int(bbox[4])),
            (230, 160, 20),
            1,
        )
    return canvas


def draw_boxes(
    image: np.ndarray,
    boxes_3d: np.ndarray,
    transform: np.ndarray,
    camera_matrix: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    canvas = image.copy()
    for box in boxes_3d:
        corners_camera = transform_points(box3d_corners(box), transform)
        corners_uv, depths = project_points(corners_camera, camera_matrix)
        for first, second in BOX_EDGES:
            if depths[first] > 0.5 and depths[second] > 0.5:
                cv2.line(
                    canvas,
                    tuple(corners_uv[first].astype(int)),
                    tuple(corners_uv[second].astype(int)),
                    color,
                    2,
                )
    return canvas


def make_grid(images: dict[str, np.ndarray], camera_names: list[str]) -> np.ndarray:
    rows = []
    for row_names in (camera_names[:3], camera_names[3:]):
        row_images = []
        for camera_name in row_names:
            image = images[camera_name]
            cv2.putText(
                image,
                camera_name,
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            row_images.append(image)
        rows.append(np.hstack(row_images))
    return np.vstack(rows)


def main() -> int:
    config = load_config()
    scene_name = config["baseline_scene"]
    camera_names = [camera["name"] for camera in config["cameras"]]
    scene_dir = (CALIBRATION_DIR / config["scenes_dir"] / scene_name).resolve()

    scaled_calib = read_json(
        CALIBRATION_DIR / "converted" / "scaled_calibrations.json"
    )
    refined_data = read_json(
        CALIBRATION_DIR / "refined" / "refined_extrinsics.json"
    )
    refined_by_camera = {
        item["camera"]: item for item in refined_data["cameras"]
    }

    with (CALIBRATION_DIR / "baseline" / "baseline_frames.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        baseline_frames = [int(row["frame"]) for row in csv.DictReader(handle)]

    overlay_indices = [
        baseline_frames[0],
        baseline_frames[len(baseline_frames) // 2],
        baseline_frames[-1],
    ]
    overlay_dir = CALIBRATION_DIR / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for frame in dict.fromkeys(overlay_indices):
        boxes_3d = np.load(
            scene_dir / "bev_box" / f"{frame:04d}.npz"
        )["boxes_3d"]
        bbox_data = np.load(
            scene_dir / "bbox2d" / f"{frame:04d}.npz", allow_pickle=True
        )
        bbox_array = bbox_data["boxes"]
        original_images = {}
        refined_images = {}

        for camera_slot, camera_name in enumerate(camera_names):
            camera_calib = scaled_calib["scenes"][scene_name]["cameras"][camera_name]
            manifest_name = camera_calib["manifest_name"]
            camera_matrix = np.asarray(camera_calib["K_scaled"], dtype=np.float64)
            original_transform = np.asarray(
                camera_calib["T_lidar_camera"], dtype=np.float64
            )
            refined_transform_values = refined_by_camera[camera_name].get(
                "T_lidar_camera_refined"
            )
            refined_transform = np.asarray(
                refined_transform_values
                if refined_transform_values is not None
                else camera_calib["T_lidar_camera"],
                dtype=np.float64,
            )

            image = cv2.imread(
                str(scene_dir / "img" / manifest_name / f"{frame:04d}.jpg")
            )
            if image is None:
                raise FileNotFoundError(
                    scene_dir / "img" / manifest_name / f"{frame:04d}.jpg"
                )

            count = int(bbox_data["counts"][camera_slot])
            image_with_detections = draw_detections(
                image, bbox_array[camera_slot, :count]
            )
            original_images[camera_name] = draw_boxes(
                image_with_detections,
                boxes_3d,
                original_transform,
                camera_matrix,
                (40, 230, 40),
            )
            refined_images[camera_name] = draw_boxes(
                image_with_detections,
                boxes_3d,
                refined_transform,
                camera_matrix,
                (40, 120, 255),
            )

        combined = np.vstack(
            [
                make_grid(original_images, camera_names),
                make_grid(refined_images, camera_names),
            ]
        )
        output_path = overlay_dir / f"compare_refined_{frame:04d}.jpg"
        cv2.imwrite(str(output_path), combined)
        print(output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
