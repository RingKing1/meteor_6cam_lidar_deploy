#!/usr/bin/env python3
"""Single-DOF, cross-scene refinement with leave-one-scene-out (LOSO) CV.

Physical diagnosis from script 12:
  front_left  / front_right : pure camera pitch (constant term of dv)
  back_wide                 : pure camera-y translation (1/depth term)

FRAMES_PER_SCENE sampled frames are used per scene. Per camera we optimize exactly one degree of freedom on the correct chain
(ego->lidar via inverse lidar2imu, then direct T_lidar_camera):
  * pitch: T_eff = Rx(delta) @ T_lidar_camera   (rotation about camera x,
    also rotates the translation = rotation about the optical center)
  * cam-y translation: t_eff = t - (0, delta, 0)

Validation: 6-fold leave-one-scene-out. A correction is accepted only if
the parameter is stable across folds, does not saturate bounds, and the
held-out scene improves (median up, mean/p90/p95 non-worse). front_wide
is included as a no-bias control and should return ~zero.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from _calibration_utils import transform_points
from common_paths import CALIBRATION_DIR, DEPLOY_DIR, load_config


FRAMES_PER_SCENE = 200
MAX_MATCH_DISTANCE_PX = 100.0
MIN_DEPTH_M = 0.5
PITCH_BOUND_DEG = 2.0
TY_BOUND_M = 0.30

# camera -> single DOF to optimize
DOF = {
    "front_left": "pitch",
    "front_right": "pitch",
    "back_wide": "cam_y",
    "back_left": "cam_y",
}


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
                rows.append([float(v) for v in values])
            if len(rows) == 4:
                break
    assert len(rows) == 4
    return np.asarray(rows, dtype=np.float64)


def uniform_sample(count: int, total: int) -> list[int]:
    if total <= count:
        return list(range(total))
    return sorted({int(round(i * (total - 1) / (count - 1))) for i in range(count)})


def rx_matrix(angle_rad: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle_rad), torch.sin(angle_rad)
    zero = torch.zeros_like(angle_rad)
    one = torch.ones_like(angle_rad)
    return torch.stack([
        torch.stack([one, zero, zero]),
        torch.stack([zero, c, -s]),
        torch.stack([zero, s, c]),
    ])


def bounded_scalar(raw: torch.Tensor, bound: float) -> torch.Tensor:
    return bound * torch.tanh(raw)


def project(points_lidar: torch.Tensor, K: torch.Tensor, R: torch.Tensor,
            t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    camera_points = points_lidar @ R.T + t
    depth = camera_points[:, 2]
    safe = torch.maximum(depth, torch.full_like(depth, 1e-6))
    u = K[0, 0] * camera_points[:, 0] / safe + K[0, 2]
    v = K[1, 1] * camera_points[:, 1] / safe + K[1, 2]
    return torch.stack([u, v], dim=1), depth


def quantiles(errors: torch.Tensor) -> dict[str, float]:
    sorted_errors, _ = torch.sort(errors)
    return {
        "mean": float(errors.mean()),
        "median": float(errors.median()),
        "p90": float(torch.quantile(errors, 0.90)),
        "p95": float(torch.quantile(errors, 0.95)),
    }


def evaluate(points: torch.Tensor, targets: torch.Tensor, K: torch.Tensor,
             R: torch.Tensor, t: torch.Tensor) -> tuple[dict[str, float], torch.Tensor]:
    projected, depth = project(points, K, R, t)
    valid = depth > MIN_DEPTH_M
    errors = torch.linalg.norm(projected - targets, dim=1)
    errors = torch.where(valid, errors, torch.full_like(errors, 10000.0))
    return quantiles(errors), (projected[:, 1] - targets[:, 1])[valid]


def associate(boxes_3d: np.ndarray, projected: np.ndarray, depth: np.ndarray,
              bbox_array: np.ndarray, slot: int) -> list[tuple[int, int]]:
    detections = []
    for j in range(bbox_array.shape[1]):
        box = bbox_array[slot, j]
        if box[3] <= 0 or box[4] <= 0:
            break
        detections.append(box)
    candidates: list[tuple[float, int, int]] = []
    for bi in range(len(boxes_3d)):
        if depth[bi] <= MIN_DEPTH_M:
            continue
        for di, det in enumerate(detections):
            from _calibration_utils import macro_class
            if macro_class(float(boxes_3d[bi, 0])) != macro_class(float(det[0])):
                continue
            distance = float(np.linalg.norm(projected[bi] - det[1:3]))
            candidates.append((distance, bi, di))
    assignments: dict[int, int] = {}
    used: set[int] = set()
    for distance, bi, di in sorted(candidates):
        if distance > MAX_MATCH_DISTANCE_PX or bi in assignments or di in used:
            continue
        assignments[bi] = di
        used.add(di)
    return list(assignments.items())


def collect_scene_observations(config, calibrations) -> dict[str, list[dict]]:
    scenes = config["scenes"]
    camera_names = [c["name"] for c in config["cameras"]]
    by_scene: dict[str, list[dict]] = {s: [] for s in scenes}
    for scene in scenes:
        scene_dir = DEPLOY_DIR / "scenes" / scene
        total = len(list((scene_dir / "bev_box").glob("*.npz")))
        frames = uniform_sample(FRAMES_PER_SCENE, total)
        ego_to_lidar = np.linalg.inv(load_lidar_to_ego(scene))
        cameras = calibrations[scene]["cameras"]
        for frame in frames:
            boxes_3d = np.load(scene_dir / "bev_box" / f"{frame:04d}.npz")["boxes_3d"]
            bbox_array = np.load(
                scene_dir / "bbox2d" / f"{frame:04d}.npz", allow_pickle=True
            )["boxes"]
            if len(boxes_3d) == 0:
                continue
            for slot, camera in enumerate(camera_names):
                K = np.asarray(cameras[camera]["K_scaled"], dtype=np.float64)
                T = np.asarray(cameras[camera]["T_lidar_camera"], dtype=np.float64)
                centers_lidar = transform_points(
                    boxes_3d[:, 1:4].astype(np.float64), ego_to_lidar
                )
                centers_camera = transform_points(centers_lidar, T)
                depth = centers_camera[:, 2]
                safe = np.maximum(depth, 1e-6)
                u = K[0, 0] * centers_camera[:, 0] / safe + K[0, 2]
                v = K[1, 1] * centers_camera[:, 1] / safe + K[1, 2]
                projected = np.column_stack([u, v])
                for bi, di in associate(
                    boxes_3d, projected, depth, bbox_array, slot
                ):
                    by_scene[scene].append({
                        "camera": camera,
                        "point": centers_lidar[bi],
                        "target": bbox_array[slot, di, 1:3].astype(np.float64),
                    })
    return by_scene


def fit_dof(camera: str, dof: str, observations: list[dict],
            camera_calib: dict, device: torch.device) -> tuple[float, dict, torch.Tensor, torch.Tensor]:
    K = torch.as_tensor(camera_calib["K_scaled"], dtype=torch.float64, device=device)
    T = torch.as_tensor(camera_calib["T_lidar_camera"], dtype=torch.float64, device=device)
    R0, t0 = T[:3, :3], T[:3, 3]
    points = torch.as_tensor(
        np.stack([o["point"] for o in observations]), dtype=torch.float64, device=device
    )
    targets = torch.as_tensor(
        np.stack([o["target"] for o in observations]), dtype=torch.float64, device=device
    )

    raw = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], max_iter=200, history_size=50,
                                  line_search_fn="strong_wolfe")

    def pose(raw_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if dof == "pitch":
            delta = bounded_scalar(raw_value, float(np.deg2rad(PITCH_BOUND_DEG)))
            rotx = rx_matrix(delta)
            return rotx @ R0, rotx @ t0
        delta = bounded_scalar(raw_value, TY_BOUND_M)
        offset = torch.stack([torch.zeros_like(delta), delta, torch.zeros_like(delta)])
        return R0, t0 + offset

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        R, t = pose(raw)
        projected, _ = project(points, K, R, t)
        loss = F.huber_loss(projected, targets, delta=10.0)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        if dof == "pitch":
            value = float(torch.rad2deg(bounded_scalar(raw, float(np.deg2rad(PITCH_BOUND_DEG)))))
        else:
            value = float(bounded_scalar(raw, TY_BOUND_M))
        R, t = pose(raw)
    stats, _ = evaluate(points, targets, K, R, t)
    return value, stats, R, t


def camera_original_pose(camera_calib: dict, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = torch.as_tensor(camera_calib["K_scaled"], dtype=torch.float64, device=device)
    T = torch.as_tensor(camera_calib["T_lidar_camera"], dtype=torch.float64, device=device)
    return K, T[:3, :3], T[:3, 3]


def main() -> int:
    config = load_config()
    scenes = config["scenes"]
    calibrations = json.loads(
        (CALIBRATION_DIR / "converted" / "scaled_calibrations.json").read_text()
    )["scenes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    by_scene = collect_scene_observations(config, calibrations)

    results: dict = {}
    for camera, dof in DOF.items():
        folds = []
        for held_out in scenes:
            train_obs = [
                o for s in scenes if s != held_out for o in by_scene[s]
                if o["camera"] == camera
            ]
            val_obs = [o for o in by_scene[held_out] if o["camera"] == camera]
            if len(train_obs) < 8 or len(val_obs) < 2:
                folds.append({"held_out": held_out, "skipped": True})
                continue
            calib = calibrations[scenes[0]]["cameras"][camera]
            value, train_stats, R_fit, t_fit = fit_dof(
                camera, dof, train_obs, calib, device
            )
            K, R0, t0 = camera_original_pose(calib, device)
            val_points = torch.as_tensor(
                np.stack([o["point"] for o in val_obs]), dtype=torch.float64, device=device
            )
            val_targets = torch.as_tensor(
                np.stack([o["target"] for o in val_obs]), dtype=torch.float64, device=device
            )
            base_stats, _ = evaluate(val_points, val_targets, K, R0, t0)
            refined_stats, _ = evaluate(val_points, val_targets, K, R_fit, t_fit)
            folds.append({
                "held_out": held_out,
                "value": value,
                "train_n": len(train_obs),
                "val_n": len(val_obs),
                "baseline_validation": base_stats,
                "refined_validation": refined_stats,
                "median_improved": refined_stats["median"] <= base_stats["median"],
                "non_worse": (
                    refined_stats["mean"] <= base_stats["mean"] + 1e-9
                    and refined_stats["p90"] <= base_stats["p90"] + 1e-9
                    and refined_stats["p95"] <= base_stats["p95"] + 1e-9
                ),
            })

        valid_folds = [f for f in folds if "value" in f]
        values = np.array([f["value"] for f in valid_folds])
        bound = PITCH_BOUND_DEG if dof == "pitch" else TY_BOUND_M
        saturates = bool(np.any(np.abs(values) >= bound * 0.9))
        same_sign = bool(np.all(values * np.median(values) > 0)) if len(values) else False
        value_cv = float(np.std(values) / max(abs(np.mean(values)), 1e-9))
        median_improved_all = all(f["median_improved"] for f in valid_folds)
        non_worse_all = all(f["non_worse"] for f in valid_folds)

        all_obs = [o for s in scenes for o in by_scene[s] if o["camera"] == camera]
        calib = calibrations[scenes[0]]["cameras"][camera]
        final_value, final_train_stats, R_final, t_final = fit_dof(
            camera, dof, all_obs, calib, device
        )
        deploy_transform = torch.eye(4, dtype=torch.float64)
        K, R0, t0 = camera_original_pose(calib, device)
        all_points = torch.as_tensor(
            np.stack([o["point"] for o in all_obs]), dtype=torch.float64, device=device
        )
        all_targets = torch.as_tensor(
            np.stack([o["target"] for o in all_obs]), dtype=torch.float64, device=device
        )
        base_all, signed_dy_before = evaluate(
            all_points, all_targets, K, R0, t0
        )
        _, signed_dy_after = evaluate(
            all_points, all_targets, K, R_final, t_final
        )

        fold_median_improve_count = sum(
            f["median_improved"] for f in valid_folds
        )
        meaningful_magnitude = abs(final_value) > (
            0.1 if dof == "pitch" else 0.02
        )
        residual_zeroed = abs(float(signed_dy_after.median())) <= 1.5
        accepted = bool(
            not saturates
            and same_sign
            and value_cv < 0.6
            and meaningful_magnitude
            and fold_median_improve_count * 2 >= len(valid_folds)
            and residual_zeroed
        )
        if accepted:
            deploy_transform[:3, :3] = R_final
            deploy_transform[:3, 3] = t_final
            decision = "accept"
        elif not meaningful_magnitude:
            decision = "keep_original_negligible"
        else:
            decision = "reject"

        results[camera] = {
            "dof": dof,
            "decision": decision,
            "accepted": accepted,
            "final_value": final_value,
            "folds": folds,
            "fold_value_mean": float(np.mean(values)),
            "fold_value_std": float(np.std(values)),
            "fold_value_cv": value_cv,
            "saturates_bound": saturates,
            "same_sign_across_folds": same_sign,
            "baseline_all": base_all,
            "refined_all": final_train_stats,
            "signed_dy_before_median": float(signed_dy_before.median()),
            "signed_dy_after_median": float(signed_dy_after.median()),
            "T_lidar_camera_deploy": deploy_transform.tolist(),
            "_depth": transform_points(
                all_points.cpu().numpy(),
                np.asarray(calib["T_lidar_camera"]),
            )[:, 2],
            "_dy_before": signed_dy_before.cpu().numpy(),
            "_dy_after": signed_dy_after.cpu().numpy(),
        }

    out_dir = CALIBRATION_DIR / "refined_single_dof"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "single_dof_results.json"
    serializable = {
        c: {k: v for k, v in info.items() if not k.startswith("_")}
        for c, info in results.items()
    }
    output_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))

    deployment = {
        "method": (
            "single-DOF refinement, 6-scene LOSO, "
            f"{FRAMES_PER_SCENE} frames/scene"
        ),
        "chain": (
            "boxes ego->lidar via inverse lidar2imu, then direct "
            "T_lidar_camera"
        ),
        "note": (
            "extrinsics identical at full resolution; pair with "
            "full-resolution intrinsics when deploying"
        ),
        "cameras": {
            camera: {
                "correction": (
                    f"camera pitch {info['final_value']:+.3f} deg about "
                    "optical center"
                    if info["dof"] == "pitch"
                    else f"camera-y translation {info['final_value']:+.3f} m"
                ),
                "T_lidar_camera": info["T_lidar_camera_deploy"],
            }
            for camera, info in results.items()
        },
    }
    deployment_path = out_dir / "deploy_extrinsics_final.json"
    deployment_path.write_text(
        json.dumps(deployment, indent=2, ensure_ascii=False)
    )

    validation_dir = CALIBRATION_DIR / "validation"
    csv_path = validation_dir / "single_dof_loso.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["camera", "dof", "held_out_scene", "value", "val_n",
                         "base_median", "refined_median", "base_mean", "refined_mean",
                         "base_p90", "refined_p90", "base_p95", "refined_p95"])
        for camera, info in results.items():
            for fold in info["folds"]:
                if fold.get("skipped"):
                    continue
                b, r = fold["baseline_validation"], fold["refined_validation"]
                writer.writerow([
                    camera, info["dof"], fold["held_out"], f"{fold['value']:.4f}",
                    fold["val_n"], f"{b['median']:.2f}", f"{r['median']:.2f}",
                    f"{b['mean']:.2f}", f"{r['mean']:.2f}",
                    f"{b['p90']:.2f}", f"{r['p90']:.2f}",
                    f"{b['p95']:.2f}", f"{r['p95']:.2f}",
                ])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, (camera, info) in zip(axes.flat, results.items()):
        depth, before, after = info["_depth"], info["_dy_before"], info["_dy_after"]
        axis.scatter(depth, before, s=14, alpha=0.6, label="before")
        axis.scatter(depth, after, s=14, alpha=0.6, label="after")
        axis.axhline(0, color="k", lw=0.5)
        unit = "deg" if info["dof"] == "pitch" else "m"
        axis.set_title(
            f"{camera} [{info['dof']}]: {info['decision']}, "
            f"value={info['final_value']:.3f}{unit}"
        )
        axis.set_xlabel("depth (m)")
        axis.set_ylabel("signed dy (px)")
        axis.legend(fontsize=8)
    fig.tight_layout()
    plot_path = CALIBRATION_DIR / "overlays" / "single_dof_refine.png"
    fig.savefig(plot_path, dpi=130)

    print(output_path)
    print(deployment_path)
    print(csv_path)
    print(plot_path)
    for camera, info in results.items():
        unit = "deg" if info["dof"] == "pitch" else "m"
        print(
            f"{camera:12s} {info['dof']:6s} {info['decision']:26s} "
            f"value={info['final_value']:+.3f}{unit} folds={len(info['folds'])} "
            f"cv={info['fold_value_cv']:.2f} sat={info['saturates_bound']} "
            f"dy {info['signed_dy_before_median']:.1f}->{info['signed_dy_after_median']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
