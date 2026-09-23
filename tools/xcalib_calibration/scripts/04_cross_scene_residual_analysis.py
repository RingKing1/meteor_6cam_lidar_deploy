#!/usr/bin/env python3
"""Cross-scene residual analysis on the correct lidar<->camera chain.

For every scene, uniformly sample a fixed number of frames; transform each
3D box center ego->lidar (inverse lidar2imu), project with the direct
T_lidar_camera, and associate to 2D detections by nearest center subject
to macro-class agreement. Then, per camera:
  * pooled signed residual medians (x/y) with bootstrap intervals;
  * physical decomposition dv = a + b/depth (a~pitch, b~camera-y offset);
  * per-scene medians to check whether the bias is stable across scenes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _calibration_utils import macro_class, transform_points
from common_paths import CALIBRATION_DIR, DEPLOY_DIR, load_config


FRAMES_PER_SCENE = 30
MAX_MATCH_DISTANCE_PX = 100.0
MIN_DEPTH_M = 0.5


def load_lidar_to_ego(scene: str) -> np.ndarray:
    path = DEPLOY_DIR / "raw_data" / scene / "calib" / "lidar" / "lidar2imu_calib.txt"
    rows: list[list[float]] = []
    in_matrix = False
    for line in path.read_text().splitlines():
        if "4x4" in line:
            in_matrix = True
            continue
        if in_matrix:
            values = line.split()
            if len(values) == 4:
                rows.append([float(value) for value in values])
            if len(rows) == 4:
                break
    assert len(rows) == 4
    return np.asarray(rows, dtype=np.float64)


def uniform_sample(count: int, total: int) -> list[int]:
    if total <= count:
        return list(range(total))
    return sorted(
        {int(round(i * (total - 1) / (count - 1))) for i in range(count)}
    )


def associate(boxes_3d: np.ndarray, projected_uv: np.ndarray, depth: np.ndarray,
              bbox_array: np.ndarray, slot: int) -> list[dict]:
    count = 0
    valid_detections = []
    for j in range(bbox_array.shape[1]):
        box = bbox_array[slot, j]
        if box[3] <= 0 or box[4] <= 0:
            break
        count += 1
        valid_detections.append(box)
    candidates: list[tuple[float, int, int]] = []
    for bi in range(len(boxes_3d)):
        if depth[bi] <= MIN_DEPTH_M:
            continue
        for di, det in enumerate(valid_detections):
            if macro_class(float(boxes_3d[bi, 0])) != macro_class(float(det[0])):
                continue
            det_center = det[1:3]
            distance = float(np.linalg.norm(projected_uv[bi] - det_center))
            candidates.append((distance, bi, di))
    assignments: dict[int, int] = {}
    used: set[int] = set()
    for distance, bi, di in sorted(candidates):
        if distance > MAX_MATCH_DISTANCE_PX or bi in assignments or di in used:
            continue
        assignments[bi] = di
        used.add(di)
    return [
        {"box_index": bi, "det_index": di}
        for bi, di in assignments.items()
    ]


def main() -> int:
    rng = np.random.default_rng(2)
    config = load_config()
    scenes = config["scenes"]
    camera_names = [camera["name"] for camera in config["cameras"]]
    calibrations = json.loads(
        (CALIBRATION_DIR / "converted" / "scaled_calibrations.json").read_text()
    )["scenes"]

    records: list[dict] = []
    for scene in scenes:
        scene_dir = DEPLOY_DIR / "scenes" / scene
        total = len(list((scene_dir / "bev_box").glob("*.npz")))
        frames = uniform_sample(FRAMES_PER_SCENE, total)
        ego_to_lidar = np.linalg.inv(load_lidar_to_ego(scene))
        cameras = calibrations[scene]["cameras"]
        for frame in frames:
            boxes_3d = np.load(
                scene_dir / "bev_box" / f"{frame:04d}.npz"
            )["boxes_3d"]
            bbox_array = np.load(
                scene_dir / "bbox2d" / f"{frame:04d}.npz", allow_pickle=True
            )["boxes"]
            if len(boxes_3d) == 0:
                continue
            for slot, camera in enumerate(camera_names):
                K = np.asarray(cameras[camera]["K_scaled"], dtype=np.float64)
                T_lc = np.asarray(cameras[camera]["T_lidar_camera"], dtype=np.float64)
                centers_lidar = transform_points(
                    boxes_3d[:, 1:4].astype(np.float64), ego_to_lidar
                )
                centers_camera = transform_points(centers_lidar, T_lc)
                depth = centers_camera[:, 2]
                safe = np.maximum(depth, 1e-6)
                u = K[0, 0] * centers_camera[:, 0] / safe + K[0, 2]
                v = K[1, 1] * centers_camera[:, 1] / safe + K[1, 2]
                projected_uv = np.column_stack([u, v])
                for match in associate(
                    boxes_3d, projected_uv, depth, bbox_array, slot
                ):
                    bi = match["box_index"]
                    det = bbox_array[slot, match["det_index"]]
                    dx = projected_uv[bi, 0] - det[1]
                    dy = projected_uv[bi, 1] - det[2]
                    records.append(
                        {
                            "scene": scene,
                            "frame": frame,
                            "camera": camera,
                            "depth_m": float(depth[bi]),
                            "signed_dx_px": float(dx),
                            "signed_dy_px": float(dy),
                        }
                    )

    out_dir = CALIBRATION_DIR / "validation"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "cross_scene_residuals.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    def lstsq_fit(values: np.ndarray) -> tuple[np.ndarray, float]:
        z = values[:, 0]
        dv = values[:, 1]
        X = np.column_stack([np.ones_like(z), 1.0 / z])
        coef, _, _, _ = np.linalg.lstsq(X, dv, rcond=None)
        pred = X @ coef
        r2 = 1.0 - ((dv - pred) ** 2).sum() / ((dv - dv.mean()) ** 2).sum()
        return coef, float(r2)

    summary: dict = {"frames_per_scene": FRAMES_PER_SCENE, "cameras": {}}
    for camera in camera_names:
        rows = [r for r in records if r["camera"] == camera]
        values = np.array([[r["depth_m"], r["signed_dy_px"]] for r in rows])
        dx_values = np.array([r["signed_dx_px"] for r in rows])
        fy = None
        first_scene = scenes[0]
        fy = calibrations[first_scene]["cameras"][camera]["K_scaled"][1][1]

        def boot_median(sequence: np.ndarray) -> tuple[float, np.ndarray]:
            medians = [
                np.median(rng.choice(sequence, len(sequence), replace=True))
                for _ in range(2000)
            ]
            return float(np.median(sequence)), np.percentile(medians, [2.5, 97.5])

        med_y, ci_y = boot_median(values[:, 1])
        med_x, ci_x = boot_median(dx_values)
        coef, r2 = lstsq_fit(values)
        boot_coef = np.array([
            lstsq_fit(rng.choice(values, len(values), replace=True))[0]
            for _ in range(1500)
        ])
        ci_a = np.percentile(boot_coef[:, 0], [2.5, 97.5])
        ci_b = np.percentile(boot_coef[:, 1], [2.5, 97.5])

        per_scene: dict[str, dict] = {}
        for scene in scenes:
            subset = [r for r in rows if r["scene"] == scene]
            if not subset:
                per_scene[scene] = None
                continue
            ys = np.array([r["signed_dy_px"] for r in subset])
            per_scene[scene] = {
                "n": len(subset),
                "median_dy_px": float(np.median(ys)),
            }

        summary["cameras"][camera] = {
            "n": len(rows),
            "median_dy_px": med_y,
            "median_dy_ci": ci_y.tolist(),
            "median_dx_px": med_x,
            "median_dx_ci": ci_x.tolist(),
            "decomp_const_a_px": float(coef[0]),
            "decomp_const_a_ci": ci_a.tolist(),
            "decomp_pitch_deg": float(np.degrees(coef[0] / fy)),
            "decomp_b_px_m": float(coef[1]),
            "decomp_b_ci": ci_b.tolist(),
            "decomp_cam_y_cm": float(coef[1] / fy * 100.0),
            "decomp_r2": r2,
            "per_scene": per_scene,
        }

    json_path = out_dir / "cross_scene_residual_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    scene_colors = plt.cm.tab10(np.linspace(0, 1, len(scenes)))
    for axis, camera in zip(axes.flat, camera_names):
        for color, scene in zip(scene_colors, scenes):
            subset = [
                r for r in records
                if r["camera"] == camera and r["scene"] == scene
            ]
            if subset:
                z = [r["depth_m"] for r in subset]
                dy = [r["signed_dy_px"] for r in subset]
                axis.scatter(z, dy, s=15, alpha=0.7, color=color, label=scene[-6:])
        axis.axhline(0, color="k", lw=0.5)
        info = summary["cameras"][camera]
        axis.set_title(
            f"{camera}: med dy={info['median_dy_px']:.1f}, "
            f"pitch={info['decomp_pitch_deg']:.2f}deg"
        )
        axis.set_xlabel("depth (m)")
        axis.set_ylabel("signed dy (px)")
        axis.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    plot_path = CALIBRATION_DIR / "overlays" / "cross_scene_residuals.png"
    fig.savefig(plot_path, dpi=130)

    print(csv_path)
    print(json_path)
    print(plot_path)
    for camera in camera_names:
        info = summary["cameras"][camera]
        print(
            f"{camera:12s} n={info['n']:4d} med_dy={info['median_dy_px']:6.1f} "
            f"[{info['median_dy_ci'][0]:6.1f},{info['median_dy_ci'][1]:6.1f}] "
            f"pitch={info['decomp_pitch_deg']:6.2f} r2={info['decomp_r2']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
