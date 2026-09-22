#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from common_paths import CALIBRATION_DIR, load_config, raw_data_dir


def parse_pcd(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    with path.open("rb") as handle:
        header_lines: list[bytes] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"missing DATA marker: {path}")
            header_lines.append(line)
            if line.startswith(b"DATA"):
                data_type = line.decode("ascii").split()[1].strip()
                break

        header: dict[str, Any] = {"data": data_type}
        for raw_line in header_lines:
            line = raw_line.decode("ascii").strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] in {"FIELDS", "SIZE", "TYPE", "COUNT"}:
                header[parts[0].lower()] = parts[1:]
            elif parts[0] in {"WIDTH", "HEIGHT", "POINTS"}:
                header[parts[0].lower()] = int(parts[1])

        fields = header.get("fields", [])
        if data_type != "binary":
            raise ValueError(f"unsupported PCD DATA type: {data_type}")
        if tuple(fields[:4]) != ("x", "y", "z", "intensity"):
            raise ValueError(f"unexpected PCD fields: {fields}")

        point_count = int(header["points"])
        values = np.fromfile(
            handle,
            dtype=np.float32,
            count=point_count * len(fields),
        )

    points = values.reshape(point_count, len(fields))
    return header, points


def check_image(path: Path, expected_size: tuple[int, int]) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        return {
            "ok": image.size == expected_size and image.mode == "RGB",
            "width": image.size[0],
            "height": image.size[1],
            "mode": image.mode,
        }


def check_calibration(
    path: Path,
    expected_size: tuple[int, int],
) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    extrinsic = np.asarray(data["extrinsic"], dtype=np.float64).reshape(4, 4)
    intrinsic = np.asarray(data["intrinsic"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(data.get("distortion", []), dtype=np.float64).reshape(-1)
    rotation = extrinsic[:3, :3]
    principal_x = float(intrinsic[0, 2])
    principal_y = float(intrinsic[1, 2])

    warnings: list[str] = []
    if not np.isfinite(extrinsic).all():
        warnings.append("extrinsic contains non-finite values")
    if not np.isfinite(intrinsic).all():
        warnings.append("intrinsic contains non-finite values")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-4):
        warnings.append("rotation determinant is not close to 1")
    if not (0 < principal_x < expected_size[0] and 0 < principal_y < expected_size[1]):
        warnings.append("principal point is outside image")
    if np.any(distortion != 0):
        warnings.append("distortion is not zero")

    return {
        "ok": not warnings,
        "warnings": warnings,
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
        "cx": principal_x,
        "cy": principal_y,
        "translation": extrinsic[:3, 3].tolist(),
        "rotation_determinant": float(np.linalg.det(rotation)),
    }


def uniform_sample(count: int, total: int) -> list[int]:
    if total == 0:
        return []
    if total <= count:
        return list(range(total))
    return sorted(
        {
            int(round(index * (total - 1) / (count - 1)))
            for index in range(count)
        }
    )


def make_contact_sheet(
    image_paths: list[Path],
    labels: list[str],
    output_path: Path,
    columns: int,
    thumbnail_width: int,
) -> None:
    if not image_paths:
        return
    with Image.open(image_paths[0]) as first:
        aspect = first.size[1] / first.size[0]
    thumbnail_height = int(thumbnail_width * aspect)
    label_height = 54
    rows = math.ceil(len(image_paths) / columns)
    sheet = Image.new(
        "RGB",
        (columns * thumbnail_width, rows * (thumbnail_height + label_height)),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)

    for index, (path, label) in enumerate(zip(image_paths, labels)):
        row = index // columns
        column = index % columns
        x = column * thumbnail_width
        y = row * (thumbnail_height + label_height)
        with Image.open(path) as image:
            image = image.resize((thumbnail_width, thumbnail_height), Image.Resampling.LANCZOS)
            sheet.paste(image, (x, y + label_height))
        draw.text((x + 6, y + 4), label, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-per-scene", type=int, default=None)
    parser.add_argument("--no-point-data", action="store_true")
    args = parser.parse_args()

    config = load_config()
    root = raw_data_dir(config)
    expected_size = (
        int(config["image_width"]),
        int(config["image_height"]),
    )
    frames_per_scene = args.frames_per_scene or int(
        config["sampling"]["frames_per_scene"]
    )

    index_path = CALIBRATION_DIR / "index" / "frame_index.csv"
    with index_path.open("r", newline="", encoding="utf-8") as handle:
        frame_rows = list(csv.DictReader(handle))

    sample_rows: list[dict[str, Any]] = []

    for scene in config["scenes"]:
        scene_rows = [row for row in frame_rows if row["scene"] == scene]
        selected = uniform_sample(frames_per_scene, len(scene_rows))
        contact_paths: list[Path] = []
        contact_labels: list[str] = []

        for sample_position, frame_number in enumerate(selected):
            frame = scene_rows[frame_number]
            timestamp = frame["timestamp"]
            scene_dir = root / scene
            row: dict[str, Any] = {
                "sample_id": len(sample_rows),
                "scene": scene,
                "sample_position_in_scene": sample_position,
                "frame_index": int(frame["frame_index"]),
                "timestamp": timestamp,
                "issues": [],
                "images": {},
                "calibrations": {},
            }

            pcd_path = scene_dir / "lidar" / f"{timestamp}.pcd"
            try:
                if args.no_point_data:
                    row["lidar"] = {"ok": pcd_path.is_file(), "loaded": False}
                else:
                    header, points = parse_pcd(pcd_path)
                    xyz = points[:, :3]
                    finite = np.isfinite(xyz)
                    row["lidar"] = {
                        "ok": bool(finite.all()),
                        "loaded": True,
                        "point_count": int(points.shape[0]),
                        "fields": header["fields"],
                        "finite_xyz": bool(finite.all()),
                        "xyz_min": np.nanmin(xyz, axis=0).tolist(),
                        "xyz_max": np.nanmax(xyz, axis=0).tolist(),
                    }
                    if not finite.all():
                        row["issues"].append("non-finite lidar points")
            except Exception as exc:
                row["lidar"] = {"ok": False, "error": str(exc)}
                row["issues"].append("lidar parse failed")

            for camera in config["cameras"]:
                image_path = (
                    scene_dir
                    / "camera"
                    / camera["directory"]
                    / f"{timestamp}.jpg"
                )
                calibration_path = (
                    scene_dir
                    / "calib"
                    / "camera"
                    / camera["calibration"]
                )
                try:
                    image_report = check_image(image_path, expected_size)
                    row["images"][camera["name"]] = image_report
                    if not image_report["ok"]:
                        row["issues"].append(f"{camera['name']} image mismatch")
                except Exception as exc:
                    row["images"][camera["name"]] = {
                        "ok": False,
                        "error": str(exc),
                    }
                    row["issues"].append(f"{camera['name']} image unreadable")

                try:
                    calibration_report = check_calibration(
                        calibration_path,
                        expected_size,
                    )
                    row["calibrations"][camera["name"]] = calibration_report
                    if not calibration_report["ok"]:
                        row["issues"].extend(
                            f"{camera['name']}: {warning}"
                            for warning in calibration_report["warnings"]
                        )
                except Exception as exc:
                    row["calibrations"][camera["name"]] = {
                        "ok": False,
                        "error": str(exc),
                    }
                    row["issues"].append(f"{camera['name']} calibration unreadable")

                if camera["name"] == config["sampling"]["contact_sheet_camera"]:
                    if image_path.exists():
                        contact_paths.append(image_path)
                        contact_labels.append(
                            f"{scene} #{frame_number} ts={timestamp}"
                        )

            sample_rows.append(row)

        contact_sheet_path = (
            CALIBRATION_DIR
            / "overlays"
            / f"sample_contact_{scene}.jpg"
        )
        make_contact_sheet(
            contact_paths,
            contact_labels,
            contact_sheet_path,
            int(config["sampling"]["contact_sheet_columns"]),
            int(config["sampling"]["thumbnail_width"]),
        )

    sample_csv = CALIBRATION_DIR / "index" / "sample_frames.csv"
    csv_fields = [
        "sample_id",
        "scene",
        "frame_index",
        "timestamp",
        "issue_count",
        "issues",
    ]
    with sample_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in sample_rows:
            issues = row["issues"]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "scene": row["scene"],
                    "frame_index": row["frame_index"],
                    "timestamp": row["timestamp"],
                    "issue_count": len(issues),
                    "issues": "; ".join(issues),
                }
            )

    summary = {
        "raw_data_dir": str(root),
        "expected_image_size": list(expected_size),
        "frames_per_scene_requested": frames_per_scene,
        "total_samples": len(sample_rows),
        "samples_with_issues": sum(1 for row in sample_rows if row["issues"]),
        "issue_frequency": {},
        "by_scene": {},
    }

    issue_counts: dict[str, int] = {}
    for row in sample_rows:
        for issue in row["issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    summary["issue_frequency"] = issue_counts

    for scene in config["scenes"]:
        rows = [row for row in sample_rows if row["scene"] == scene]
        summary["by_scene"][scene] = {
            "samples": len(rows),
            "samples_with_issues": sum(1 for row in rows if row["issues"]),
            "lidar_point_counts": [
                row["lidar"].get("point_count")
                for row in rows
                if row["lidar"].get("point_count") is not None
            ],
        }

    summary_path = CALIBRATION_DIR / "index" / "sample_diagnostic_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    detailed_path = (
        CALIBRATION_DIR
        / "index"
        / "sample_diagnostic_detailed.json"
    )
    detailed_path.write_text(
        json.dumps(sample_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(sample_csv)
    print(summary_path)
    print(detailed_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
