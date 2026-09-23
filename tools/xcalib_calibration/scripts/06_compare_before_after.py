#!/usr/bin/env python3
"""Before/after extrinsic comparison overlays.

Per scene, randomly sample 20 frames. For each frame build a 6-camera grid:
3D box corners projected with the ORIGINAL extrinsics in red and with the
REFINED extrinsics in green on the same image. Output:
  overlays/compare_before_after/<scene>/frame_<XXXX>.jpg
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from _calibration_utils import box3d_corners, project_points, transform_points
from common_paths import CALIBRATION_DIR, DEPLOY_DIR, load_config


FRAMES_PER_SCENE = 20
RANDOM_SEED = 20260923
MIN_DEPTH_M = 0.5

BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

MANIFEST_NAMES = [
    "CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


def lidar_to_ego(scene: str) -> np.ndarray:
    rows = []
    inb = False
    for line in (DEPLOY_DIR / "raw_data" / scene / "calib/lidar/lidar2imu_calib.txt").read_text().splitlines():
        if "4x4" in line:
            inb = True
            continue
        if inb:
            values = line.split()
            if len(values) == 4:
                rows.append([float(v) for v in values])
            if len(rows) == 4:
                break
    return np.array(rows)


def draw_boxes(image, boxes_3d, T, K, ego_to_lidar, color):
    canvas = image.copy()
    for box in boxes_3d:
        corners_lidar = transform_points(box3d_corners(box), ego_to_lidar)
        corners_camera = transform_points(corners_lidar, T)
        corners_uv, depth = project_points(corners_camera, K)
        if (depth > MIN_DEPTH_M).sum() < 2:
            continue
        for a, b in BOX_EDGES:
            if depth[a] > MIN_DEPTH_M and depth[b] > MIN_DEPTH_M:
                cv2.line(canvas, tuple(corners_uv[a].astype(int)),
                         tuple(corners_uv[b].astype(int)), color, 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    config = load_config()
    scenes = config["scenes"]
    camera_names = [camera["name"] for camera in config["cameras"]]
    calibrations = json.loads(
        (CALIBRATION_DIR / "converted/scaled_calibrations.json").read_text()
    )["scenes"]
    deploy = json.loads(
        (CALIBRATION_DIR / "refined_single_dof/deploy_extrinsics_final.json").read_text()
    )["cameras"]
    refined_by_camera = {
        camera: np.asarray(deploy[camera]["T_lidar_camera"])
        for camera in deploy
    }

    rng = np.random.default_rng(RANDOM_SEED)
    output_root = CALIBRATION_DIR / "overlays" / "compare_before_after"
    output_root.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for scene in scenes:
        scene_dir = DEPLOY_DIR / "scenes" / scene
        all_frames = [
            int(path.stem)
            for path in (scene_dir / "bev_box").glob("*.npz")
        ]
        chosen = sorted(
            rng.choice(all_frames, size=min(FRAMES_PER_SCENE, len(all_frames)),
                       replace=False).tolist()
        )
        ego_to_lidar = np.linalg.inv(lidar_to_ego(scene))
        cameras = calibrations[scene]["cameras"]
        scene_out = output_root / scene
        scene_out.mkdir(exist_ok=True)

        for frame in chosen:
            boxes_3d = np.load(
                scene_dir / "bev_box" / f"{frame:04d}.npz"
            )["boxes_3d"]
            tiles = []
            for slot, camera in enumerate(camera_names):
                image_path = (
                    scene_dir / "img" / MANIFEST_NAMES[slot] / f"{frame:04d}.jpg"
                )
                image = cv2.imread(str(image_path))
                K = np.asarray(cameras[camera]["K_scaled"])
                T_original = np.asarray(cameras[camera]["T_lidar_camera"])
                T_refined = refined_by_camera.get(camera, T_original)

                with_original = draw_boxes(
                    image, boxes_3d, T_original, K, ego_to_lidar, (0, 0, 255)
                )
                combined = draw_boxes(
                    with_original, boxes_3d, T_refined, K, ego_to_lidar, (0, 200, 0)
                )
                cv2.putText(combined, camera, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                            cv2.LINE_AA)
                tiles.append(combined)

            grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
            header = np.full((30, grid.shape[1], 3), 24, np.uint8)
            cv2.putText(
                header,
                f"{scene}  frame {frame:04d}   RED=before  GREEN=after",
                (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA,
            )
            grid = np.vstack([header, grid])
            cv2.imwrite(str(scene_out / f"frame_{frame:04d}.jpg"), grid)
            total_written += 1

    print(f"wrote {total_written} comparison grids under {output_root}")


if __name__ == "__main__":
    main()
