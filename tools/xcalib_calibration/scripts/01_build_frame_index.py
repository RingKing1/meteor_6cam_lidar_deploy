#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from common_paths import CALIBRATION_DIR, load_config, raw_data_dir


def main() -> int:
    config = load_config()
    root = raw_data_dir(config)
    cameras = config["cameras"]
    scenes = config["scenes"]

    rows: list[dict[str, object]] = []
    scene_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for scene_index, scene in enumerate(scenes):
        scene_dir = root / scene
        lidar_dir = scene_dir / "lidar"
        pcd_paths = sorted(lidar_dir.glob("*.pcd"))
        scene_counts[scene]["lidar_pcd"] = len(pcd_paths)

        for frame_index, pcd_path in enumerate(pcd_paths):
            timestamp = pcd_path.stem
            row: dict[str, object] = {
                "global_frame_id": len(rows),
                "scene_index": scene_index,
                "scene": scene,
                "frame_index": frame_index,
                "timestamp": timestamp,
                "lidar_file": str(pcd_path.relative_to(root)),
                "lidar_exists": pcd_path.is_file(),
                "lidar_size_bytes": pcd_path.stat().st_size if pcd_path.exists() else 0,
            }

            for camera in cameras:
                image_path = (
                    scene_dir
                    / "camera"
                    / camera["directory"]
                    / f"{timestamp}.jpg"
                )
                column = f"{camera['name']}_exists"
                row[column] = image_path.is_file()
                row[f"{camera['name']}_file"] = (
                    str(image_path.relative_to(root)) if image_path.exists() else ""
                )
                row[f"{camera['name']}_size_bytes"] = (
                    image_path.stat().st_size if image_path.exists() else 0
                )
                if image_path.exists():
                    scene_counts[scene][f"{camera['name']}_jpg"] += 1
                else:
                    scene_counts[scene][f"{camera['name']}_missing"] += 1

            rows.append(row)

    index_dir = CALIBRATION_DIR / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    csv_path = index_dir / "frame_index.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    missing_camera_rows = {
        camera["name"]: sum(
            1
            for row in rows
            if not row[f"{camera['name']}_exists"]
        )
        for camera in cameras
    }
    complete_rows = sum(
        1
        for row in rows
        if all(row[f"{camera['name']}_exists"] for camera in cameras)
    )

    summary = {
        "raw_data_dir": str(root),
        "total_lidar_frames": len(rows),
        "complete_six_camera_frames": complete_rows,
        "missing_image_count_by_camera": missing_camera_rows,
        "scenes": {
            scene: {
                "lidar_pcd": counts.get("lidar_pcd", 0),
                "complete_camera_frames": min(
                    (counts.get(f"{camera['name']}_jpg", 0) for camera in cameras),
                    default=0,
                ),
                "missing_by_camera": {
                    camera["name"]: counts.get(f"{camera['name']}_missing", 0)
                    for camera in cameras
                },
            }
            for scene, counts in scene_counts.items()
        },
    }

    summary_path = index_dir / "frame_index_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(csv_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
