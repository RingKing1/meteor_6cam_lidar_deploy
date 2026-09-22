#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from _calibration_utils import (
    box3d_corners,
    box_iou,
    macro_class,
    percentile,
    project_points,
    transform_points,
    uniform_sample,
)
from common_paths import CALIBRATION_DIR, load_config


BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def read_scaled_calibrations() -> dict:
    path = CALIBRATION_DIR / "converted" / "scaled_calibrations.json"
    return json.loads(path.read_text(encoding="utf-8"))


def xyxy_box(box: np.ndarray) -> np.ndarray:
    _, cx, cy, width, height = box
    return np.array(
        [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
        dtype=np.float64,
    )


def distance_bin(lidar_distance: float) -> str:
    if lidar_distance < 15.0:
        return "near_0_15m"
    if lidar_distance < 30.0:
        return "mid_15_30m"
    return "far_30m_plus"


def load_camera_data(
    scene_dir: Path,
    frame_index: int,
    camera_names: list[str],
) -> tuple[np.ndarray, dict[str, list[dict]]]:
    boxes_3d = np.load(
        scene_dir / "bev_box" / f"{frame_index:04d}.npz"
    )["boxes_3d"]
    bbox_data = np.load(
        scene_dir / "bbox2d" / f"{frame_index:04d}.npz",
        allow_pickle=True,
    )
    bbox_array = bbox_data["boxes"]
    bbox_counts = bbox_data["counts"]

    detections_by_camera: dict[str, list[dict]] = {}
    for camera_index, camera_name in enumerate(camera_names):
        count = int(bbox_counts[camera_index])
        detections = []
        for detection_index in range(count):
            box = bbox_array[camera_index, detection_index]
            detections.append(
                {
                    "index": detection_index,
                    "cls": int(round(float(box[0]))),
                    "macro": macro_class(float(box[0])),
                    "xyxy": xyxy_box(box),
                }
            )
        detections_by_camera[camera_name] = detections

    return boxes_3d, detections_by_camera


def project_3d_box(
    box: np.ndarray,
    lidar_to_camera: np.ndarray,
    camera_matrix: np.ndarray,
    min_depth: float,
) -> dict:
    center_lidar = box[1:4].astype(np.float64)
    center_camera = transform_points(center_lidar[None], lidar_to_camera)[0]
    center_uv, center_depth = project_points(center_camera[None], camera_matrix)

    corners_lidar = box3d_corners(box)
    corners_camera = transform_points(corners_lidar, lidar_to_camera)
    corners_uv, corners_depth = project_points(corners_camera, camera_matrix)
    visible = corners_depth > min_depth

    projected_box = None
    if visible.sum() >= 4:
        projected_box = np.array(
            [
                corners_uv[visible, 0].min(),
                corners_uv[visible, 1].min(),
                corners_uv[visible, 0].max(),
                corners_uv[visible, 1].max(),
            ],
            dtype=np.float64,
        )

    return {
        "center_lidar": center_lidar,
        "center_camera": center_camera,
        "center_uv": center_uv[0],
        "center_depth": float(center_depth[0]),
        "corners_uv": corners_uv,
        "corners_depth": corners_depth,
        "visible_corner_count": int(visible.sum()),
        "projected_box": projected_box,
    }


def assign_detections(
    projections: list[dict],
    detections: list[dict],
    max_assignment_distance: float,
) -> dict[int, dict]:
    candidates = []
    for projection in projections:
        if projection["center_depth"] <= 0.5:
            continue
        for detection in detections:
            if detection["macro"] != macro_class(float(projection["cls"])):
                continue
            distance = float(
                np.linalg.norm(projection["center_uv"] - detection["xyxy"][:2].mean())
            )
            detection_center = np.array(
                [
                    (detection["xyxy"][0] + detection["xyxy"][2]) / 2,
                    (detection["xyxy"][1] + detection["xyxy"][3]) / 2,
                ]
            )
            distance = float(np.linalg.norm(projection["center_uv"] - detection_center))
            candidates.append((distance, projection["box_index"], detection["index"]))

    assignments: dict[int, dict] = {}
    used_detections: set[int] = set()
    for distance, projection_index, detection_index in sorted(candidates):
        if projection_index in assignments or detection_index in used_detections:
            continue
        assignments[projection_index] = {
            "detection_index": detection_index,
            "distance": distance,
            "within_threshold": distance <= max_assignment_distance,
        }
        used_detections.add(detection_index)
    return assignments


def summarize_errors(records: list[dict]) -> dict:
    center_errors = [
        record["center_distance_px"]
        for record in records
        if record["assigned"] and record["center_distance_px"] is not None
    ]
    ious = [
        record["iou"]
        for record in records
        if record["assigned"] and record["iou"] is not None
    ]
    return {
        "center_px_mean": float(np.mean(center_errors)) if center_errors else None,
        "center_px_median": percentile(center_errors, 50),
        "center_px_p90": percentile(center_errors, 90),
        "center_px_p95": percentile(center_errors, 95),
        "iou_mean": float(np.mean(ious)) if ious else None,
        "iou_median": percentile(ious, 50),
        "iou_p90": percentile(ious, 90),
    }


def draw_overlay(
    image: np.ndarray,
    projections: list[dict],
    detections: list[dict],
    assignments: dict[int, dict],
    min_depth: float,
) -> np.ndarray:
    canvas = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["xyxy"].astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (230, 160, 20), 1)

    for projection in projections:
        if projection["center_depth"] <= min_depth:
            continue
        corners_uv = projection["corners_uv"]
        corners_depth = projection["corners_depth"]
        for first, second in BOX_EDGES:
            if corners_depth[first] > min_depth and corners_depth[second] > min_depth:
                point_first = tuple(corners_uv[first].astype(int))
                point_second = tuple(corners_uv[second].astype(int))
                cv2.line(canvas, point_first, point_second, (40, 230, 40), 2)

        center = tuple(projection["center_uv"].astype(int))
        cv2.drawMarker(
            canvas, center, (40, 40, 230), cv2.MARKER_CROSS, 14, 2
        )

        assignment = assignments.get(projection["box_index"])
        if assignment is not None:
            label = f"{assignment['distance']:.0f}px"
            cv2.putText(
                canvas, label,
                (int(center[0]) + 8, int(center[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 230), 1,
                cv2.LINE_AA,
            )

    return canvas


def make_grid(images: dict[str, np.ndarray], camera_names: list[str]) -> np.ndarray:
    rows = []
    for row_names in (camera_names[:3], camera_names[3:]):
        row_images = []
        for camera_name in row_names:
            image = images[camera_name]
            cv2.putText(
                image, camera_name, (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                cv2.LINE_AA,
            )
            row_images.append(image)
        rows.append(np.hstack(row_images))
    return np.vstack(rows)


def main() -> int:
    config = load_config()
    calibration_data = read_scaled_calibrations()
    scene_name = config["baseline_scene"]
    scene_dir = (CALIBRATION_DIR / config["scenes_dir"] / scene_name).resolve()
    camera_names = [camera["name"] for camera in config["cameras"]]
    baseline_config = config["baseline"]
    frame_count = int(baseline_config["frames"])
    max_assignment_distance = float(baseline_config["max_assignment_distance_px"])
    min_depth = float(baseline_config["minimum_camera_depth_m"])

    total_frames = len(list((scene_dir / "bev_box").glob("*.npz")))
    selected_frames = uniform_sample(frame_count, total_frames)
    records = []
    projection_cache = {}

    for frame_index in selected_frames:
        boxes_3d, detections_by_camera = load_camera_data(
            scene_dir, frame_index, camera_names
        )
        frame_cache = {}

        for camera_name in camera_names:
            camera_calib = calibration_data["scenes"][scene_name]["cameras"][camera_name]
            K = np.asarray(camera_calib["K_scaled"], dtype=np.float64)
            T = np.asarray(camera_calib["T_lidar_camera"], dtype=np.float64)
            projections = []

            for box_index, box in enumerate(boxes_3d):
                projected = project_3d_box(box, T, K, min_depth)
                projected["box_index"] = box_index
                projected["cls"] = int(round(float(box[0])))
                projected["macro"] = macro_class(float(box[0]))
                projected["lidar_distance"] = float(
                    np.linalg.norm(box[1:3])
                )
                projections.append(projected)

            detections = detections_by_camera[camera_name]
            assignments = assign_detections(
                projections, detections, max_assignment_distance
            )
            frame_cache[camera_name] = (projections, detections, assignments)

            for projection in projections:
                box_index = projection["box_index"]
                assignment = assignments.get(box_index)
                iou = None
                center_distance = None
                assigned = False
                matched_detection = None

                if assignment is not None:
                    assigned = True
                    center_distance = assignment["distance"]
                    matched_detection = next(
                        detection
                        for detection in detections
                        if detection["index"] == assignment["detection_index"]
                    )
                    if (
                        projection["projected_box"] is not None
                        and matched_detection is not None
                    ):
                        iou = box_iou(
                            projection["projected_box"],
                            matched_detection["xyxy"],
                        )

                records.append(
                    {
                        "scene": scene_name,
                        "frame": frame_index,
                        "camera": camera_name,
                        "box_index": box_index,
                        "class": projection["cls"],
                        "lidar_distance_m": projection["lidar_distance"],
                        "distance_bin": distance_bin(projection["lidar_distance"]),
                        "camera_depth_m": projection["center_depth"],
                        "visible_corner_count": projection["visible_corner_count"],
                        "assigned": assigned,
                        "center_distance_px": center_distance,
                        "iou": iou,
                    }
                )

        projection_cache[frame_index] = frame_cache

    baseline_dir = CALIBRATION_DIR / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    frame_list_path = baseline_dir / "baseline_frames.csv"
    with frame_list_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene", "frame"])
        writer.writerows([[scene_name, frame] for frame in selected_frames])

    records_path = baseline_dir / "baseline_projection_pairs.csv"
    fields = list(records[0].keys()) if records else []
    with records_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    by_camera = {}
    for camera_name in camera_names:
        camera_records = [record for record in records if record["camera"] == camera_name]
        assigned_records = [record for record in camera_records if record["assigned"]]
        by_camera[camera_name] = {
            **summarize_errors(camera_records),
            "projected_boxes": len(camera_records),
            "assigned_boxes": len(assigned_records),
            "matched_within_threshold": sum(
                1
                for record in assigned_records
                if record["center_distance_px"] <= max_assignment_distance
            ),
        }

    by_distance_bin = {}
    for bin_name in ("near_0_15m", "mid_15_30m", "far_30m_plus"):
        bin_records = [record for record in records if record["distance_bin"] == bin_name]
        by_distance_bin[bin_name] = {
            **summarize_errors(bin_records),
            "projected_boxes": len(bin_records),
            "assigned_boxes": sum(1 for record in bin_records if record["assigned"]),
        }

    summary = {
        "scene": scene_name,
        "frames_evaluated": len(selected_frames),
        "image_size": [
            int(config["scene_image_width"]),
            int(config["scene_image_height"]),
        ],
        "direct_transform": "T_lidar_camera",
        "max_assignment_distance_px": max_assignment_distance,
        "overall": summarize_errors(records),
        "by_camera": by_camera,
        "by_distance_bin": by_distance_bin,
    }
    summary_path = baseline_dir / "baseline_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    overlay_indices = [
        selected_frames[0],
        selected_frames[len(selected_frames) // 2],
        selected_frames[-1],
    ]
    for frame_index in dict.fromkeys(overlay_indices):
        images = {}
        for camera_name in camera_names:
            manifest_name = (
                calibration_data["scenes"][scene_name]["cameras"][camera_name]["manifest_name"]
            )
            image_path = (
                scene_dir / "img" / manifest_name / f"{frame_index:04d}.jpg"
            )
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            projections, detections, assignments = projection_cache[frame_index][camera_name]
            images[camera_name] = draw_overlay(
                image, projections, detections, assignments, min_depth
            )
        grid = make_grid(images, camera_names)
        overlay_dir = CALIBRATION_DIR / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            overlay_dir / f"baseline_grid_{frame_index:04d}.jpg"
        )
        cv2.imwrite(str(output_path), grid)

    print(frame_list_path)
    print(records_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
