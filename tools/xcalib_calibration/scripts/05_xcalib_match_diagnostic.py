#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from _calibration_utils import (
    box3d_corners,
    macro_class,
    percentile,
    project_points,
    read_pcd_binary,
    rotated_box_to_aabb,
    transform_points,
    uniform_sample,
)
from common_paths import CALIBRATION_DIR, load_config, raw_data_dir


BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def read_scaled_calibrations() -> dict:
    return json.loads(
        (CALIBRATION_DIR / "converted" / "scaled_calibrations.json").read_text(
            encoding="utf-8"
        )
    )


def read_sample_frames(scene_name: str, frame_count: int) -> list[dict]:
    path = CALIBRATION_DIR / "index" / "frame_index.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["scene"] == scene_name
        ]

    selected = uniform_sample(frame_count, len(rows))
    return [
        {
            "frame": int(rows[index]["frame_index"]),
            "timestamp": str(rows[index]["timestamp"]),
        }
        for index in selected
    ]


def cxcywh_to_class_xyxy(box: np.ndarray) -> np.ndarray:
    cls, cx, cy, width, height = box
    return np.array(
        [cls, cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
        dtype=np.float32,
    )


def load_frame_inputs(
    scene_dir: Path,
    frame: int,
    camera_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    boxes_3d = np.load(scene_dir / "bev_box" / f"{frame:04d}.npz")["boxes_3d"]
    bbox_data = np.load(
        scene_dir / "bbox2d" / f"{frame:04d}.npz", allow_pickle=True
    )
    bbox_array = bbox_data["boxes"]
    bbox_counts = bbox_data["counts"]
    bboxes_by_camera: dict[str, np.ndarray] = {}

    for camera_index, camera_name in enumerate(camera_names):
        count = int(bbox_counts[camera_index])
        bboxes_by_camera[camera_name] = np.stack(
            [
                cxcywh_to_class_xyxy(bbox_array[camera_index, index])
                for index in range(count)
            ]
        ) if count else np.zeros((0, 5), dtype=np.float32)

    boxes_aabb = np.stack(
        [rotated_box_to_aabb(box) for box in boxes_3d]
    ).astype(np.float32) if len(boxes_3d) else np.zeros((0, 6), dtype=np.float32)
    return boxes_3d, boxes_aabb, bboxes_by_camera


def box_center_xyxy(box: np.ndarray) -> np.ndarray:
    return np.array(
        [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0],
        dtype=np.float64,
    )


def projected_centers(
    boxes_3d: np.ndarray,
    transform: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centers_lidar = boxes_3d[:, 1:4].astype(np.float64)
    centers_camera = transform_points(centers_lidar, transform)
    centers_uv, depths = project_points(centers_camera, camera_matrix)
    return centers_uv, depths


def visible_box_indices(
    boxes_3d: np.ndarray,
    transform: np.ndarray,
    camera_matrix: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    margin_px: float = 160.0,
) -> np.ndarray:
    visible = []
    for box_index, box in enumerate(boxes_3d):
        corners_camera = transform_points(box3d_corners(box), transform)
        corners_uv, depths = project_points(corners_camera, camera_matrix)
        in_front = depths > 0.5
        in_image = (
            (corners_uv[:, 0] > -margin_px)
            & (corners_uv[:, 0] < image_width + margin_px)
            & (corners_uv[:, 1] > -margin_px)
            & (corners_uv[:, 1] < image_height + margin_px)
        )
        if int((in_front & in_image).sum()) >= 2:
            visible.append(box_index)
    return np.asarray(visible, dtype=np.int64)


def remap_match_lidar_indices(
    matches: list[tuple[int, int, float]],
    visible_indices: np.ndarray,
) -> list[tuple[int, int, float]]:
    return [
        (
            image_index,
            int(visible_indices[lidar_index]),
            score,
        )
        for image_index, lidar_index, score in matches
    ]


def nearest_same_class_detection(
    lidar_index: int,
    boxes_3d: np.ndarray,
    bboxes_2d: np.ndarray,
    centers_uv: np.ndarray,
    max_distance: float,
) -> dict | None:
    lidar_macro = macro_class(float(boxes_3d[lidar_index, 0]))
    best = None
    for detection_index, bbox_2d in enumerate(bboxes_2d):
        detection_macro = macro_class(float(bbox_2d[0]))
        if detection_macro != lidar_macro:
            continue
        distance = float(
            np.linalg.norm(centers_uv[lidar_index] - box_center_xyxy(bbox_2d[1:5]))
        )
        if distance <= max_distance and (best is None or distance < best["distance_px"]):
            best = {
                "detection_index": detection_index,
                "distance_px": distance,
            }
    return best


def one_to_one_top_matches(
    matches: list[tuple[int, int, float]],
) -> dict[int, tuple[int, float]]:
    selected: dict[int, tuple[int, float]] = {}
    used_lidar: set[int] = set()
    for image_index, lidar_index, score in sorted(
        matches, key=lambda item: item[2], reverse=True
    ):
        if image_index in selected or lidar_index in used_lidar:
            continue
        selected[image_index] = (lidar_index, float(score))
        used_lidar.add(lidar_index)
    return selected


def summarize(values: list[float]) -> dict:
    return {
        "mean": float(np.mean(values)) if values else None,
        "median": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
    }


def draw_match_overlay(
    image_bgr: np.ndarray,
    bboxes_2d: np.ndarray,
    boxes_3d: np.ndarray,
    box_3d_index: int,
    transform: np.ndarray,
    camera_matrix: np.ndarray,
    score: float,
) -> np.ndarray:
    canvas = image_bgr.copy()
    for bbox in bboxes_2d:
        cv2.rectangle(
            canvas,
            (int(bbox[1]), int(bbox[2])),
            (int(bbox[3]), int(bbox[4])),
            (230, 160, 20),
            1,
        )

    corners_uv = []
    corners_camera = transform_points(
        box3d_corners(boxes_3d[box_3d_index]), transform
    )
    corners_uv, depths = project_points(corners_camera, camera_matrix)
    for first, second in BOX_EDGES:
        if depths[first] > 0.5 and depths[second] > 0.5:
            cv2.line(
                canvas,
                tuple(corners_uv[first].astype(int)),
                tuple(corners_uv[second].astype(int)),
                (40, 230, 40),
                2,
            )

    cv2.putText(
        canvas,
        f"score {score:.3f}",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (40, 40, 230),
        2,
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
                image,
                camera_name,
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            row_images.append(image)
        rows.append(np.hstack(row_images))
    return np.vstack(rows)


def main() -> int:
    from xcalib import Matcher

    config = load_config()
    matcher_config = config["matcher"]
    scene_name = config["baseline_scene"]
    camera_names = [camera["name"] for camera in config["cameras"]]
    scene_dir = (CALIBRATION_DIR / config["scenes_dir"] / scene_name).resolve()
    raw_root = raw_data_dir(config)
    calibration_data = read_scaled_calibrations()
    sample_frames = read_sample_frames(
        scene_name, int(matcher_config["frames"])
    )
    max_geometry_distance = float(
        config["baseline"]["max_assignment_distance_px"]
    )

    matcher = Matcher.from_pretrained(
        str(matcher_config["model"]),
        site=str(matcher_config["site"]),
        device=str(matcher_config["device"]),
    )
    top_k = int(matcher_config["top_k"])
    pair_records = []
    frame_records = []
    overlay_cache = {}

    for sample in sample_frames:
        frame = sample["frame"]
        point_cloud = read_pcd_binary(
            raw_root / scene_name / "lidar" / f"{sample['timestamp']}.pcd"
        )
        boxes_3d, boxes_aabb, bboxes_by_camera = load_frame_inputs(
            scene_dir, frame, camera_names
        )
        overlay_frame = {}

        for camera_name in camera_names:
            camera_calib = calibration_data["scenes"][scene_name]["cameras"][camera_name]
            manifest_name = camera_calib["manifest_name"]
            camera_matrix = np.asarray(camera_calib["K_scaled"], dtype=np.float64)
            transform = np.asarray(
                camera_calib["T_lidar_camera"], dtype=np.float64
            )
            bboxes_2d = bboxes_by_camera[camera_name]
            image_bgr = cv2.imread(
                str(scene_dir / "img" / manifest_name / f"{frame:04d}.jpg")
            )
            if image_bgr is None:
                raise FileNotFoundError(
                    scene_dir / "img" / manifest_name / f"{frame:04d}.jpg"
                )
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            centers_uv, depths = projected_centers(
                boxes_3d, transform, camera_matrix
            )
            visible_indices = visible_box_indices(
                boxes_3d,
                transform,
                camera_matrix,
                int(config["scene_image_width"]),
                int(config["scene_image_height"]),
            )
            visible_aabb = (
                boxes_aabb[visible_indices]
                if len(visible_indices)
                else np.zeros((0, 6), dtype=np.float32)
            )

            result = matcher.match(
                image_rgb,
                point_cloud,
                bboxes_2d[:, 1:5],
                visible_aabb,
                top_k=top_k,
                validate="warn",
            )
            global_matches = remap_match_lidar_indices(
                result.matches, visible_indices
            )
            top_matches = one_to_one_top_matches(global_matches)
            match_rank_by_image = defaultdict(list)
            for image_index, lidar_index, score in global_matches:
                match_rank_by_image[image_index].append((lidar_index, score))

            agreements = 0
            geometry_distances = []
            for image_index, (lidar_index, score) in top_matches.items():
                geometry = nearest_same_class_detection(
                    lidar_index,
                    boxes_3d,
                    bboxes_2d,
                    centers_uv,
                    max_geometry_distance,
                )
                projected_distance = float(
                    np.linalg.norm(
                        centers_uv[lidar_index]
                        - box_center_xyxy(bboxes_2d[image_index, 1:5])
                    )
                )
                geometry_distances.append(projected_distance)
                agrees = (
                    geometry is not None
                    and geometry["detection_index"] == image_index
                )
                agreements += int(agrees)
                pair_records.append(
                    {
                        "scene": scene_name,
                        "frame": frame,
                        "camera": camera_name,
                        "image_box_index": image_index,
                        "lidar_box_index": lidar_index,
                        "score": score,
                        "projected_center_distance_px": projected_distance,
                        "projected_camera_depth_m": float(depths[lidar_index]),
                        "class_2d_macro": macro_class(
                            float(bboxes_2d[image_index, 0])
                        ),
                        "class_3d_macro": macro_class(
                            float(boxes_3d[lidar_index, 0])
                        ),
                        "geometry_agrees_with_original_pose": agrees,
                    }
                )

            frame_records.append(
                {
                    "scene": scene_name,
                    "frame": frame,
                    "camera": camera_name,
                    "boxes_2d": len(bboxes_2d),
                    "boxes_3d": len(boxes_3d),
                    "visible_boxes_3d": len(visible_indices),
                    "valid_2d_after_crop": len(result.kept_2d_indices),
                    "valid_3d_after_crop": len(result.kept_3d_indices),
                    "top1_matches": len(top_matches),
                    "geometry_agreements": agreements,
                    "mean_score": float(
                        np.mean([score for _, score in top_matches.values()])
                    ) if top_matches else None,
                    "mean_projected_distance_px": float(
                        np.mean(geometry_distances)
                    ) if geometry_distances else None,
                    "latency_ms": result.latency_ms,
                }
            )

            if top_matches:
                image_index, (lidar_index, score) = max(
                    top_matches.items(), key=lambda item: item[1][1]
                )
                overlay_frame[camera_name] = draw_match_overlay(
                    image_bgr,
                    bboxes_2d,
                    boxes_3d,
                    lidar_index,
                    transform,
                    camera_matrix,
                    score,
                )

        if len(overlay_frame) == len(camera_names):
            overlay_cache[frame] = make_grid(overlay_frame, camera_names)

    matches_dir = CALIBRATION_DIR / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = matches_dir / "xcalib_match_pairs.csv"
    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene",
                "frame",
                "camera",
                "image_box_index",
                "lidar_box_index",
                "score",
                "projected_center_distance_px",
                "projected_camera_depth_m",
                "class_2d_macro",
                "class_3d_macro",
                "geometry_agrees_with_original_pose",
            ],
        )
        writer.writeheader()
        writer.writerows(pair_records)

    frames_path = matches_dir / "xcalib_frame_summary.csv"
    with frames_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_records[0].keys()))
        writer.writeheader()
        writer.writerows(frame_records)

    by_camera = {}
    for camera_name in camera_names:
        camera_pairs = [
            record for record in pair_records if record["camera"] == camera_name
        ]
        by_camera[camera_name] = {
            "top1_pairs": len(camera_pairs),
            "score": summarize([record["score"] for record in camera_pairs]),
            "projected_center_px": summarize(
                [record["projected_center_distance_px"] for record in camera_pairs]
            ),
            "geometry_agreement_rate": float(
                np.mean(
                    [
                        record["geometry_agrees_with_original_pose"]
                        for record in camera_pairs
                    ]
                )
            ) if camera_pairs else None,
        }

    summary = {
        "scene": scene_name,
        "frames_evaluated": len(sample_frames),
        "model": str(matcher_config["model"]),
        "site": str(matcher_config["site"]),
        "top_k": top_k,
        "image_size": [
            int(config["scene_image_width"]),
            int(config["scene_image_height"]),
        ],
        "total_top1_pairs": len(pair_records),
        "score": summarize([record["score"] for record in pair_records]),
        "projected_center_px": summarize(
            [record["projected_center_distance_px"] for record in pair_records]
        ),
        "geometry_agreement_rate": float(
            np.mean(
                [
                    record["geometry_agrees_with_original_pose"]
                    for record in pair_records
                ]
            )
        ) if pair_records else None,
        "by_camera": by_camera,
    }
    summary_path = matches_dir / "xcalib_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    overlay_dir = CALIBRATION_DIR / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_frames = [sample_frames[0], sample_frames[len(sample_frames) // 2], sample_frames[-1]]
    for frame in dict.fromkeys(sample["frame"] for sample in overlay_frames):
        if frame in overlay_cache:
            cv2.imwrite(
                str(overlay_dir / f"xcalib_top1_grid_{frame:04d}.jpg"),
                overlay_cache[frame],
            )

    print(pairs_path)
    print(frames_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
