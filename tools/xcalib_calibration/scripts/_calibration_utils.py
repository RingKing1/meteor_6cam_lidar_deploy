from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CAMERA_TO_MANIFEST = {
    "front_wide": "CAM_FRONT_WIDE",
    "front_left": "CAM_FRONT_LEFT",
    "front_right": "CAM_FRONT_RIGHT",
    "back_wide": "CAM_BACK_WIDE",
    "back_left": "CAM_BACK_LEFT",
    "back_right": "CAM_BACK_RIGHT",
}

VEHICLE_CLASSES = {1, 2, 3}
VRU_CLASSES = {4, 5, 6}


def load_manifest(scene_dir: Path) -> dict:
    return json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))


def macro_class(label: float) -> int:
    label_int = int(round(label))
    if label_int in VEHICLE_CLASSES:
        return 1
    if label_int in VRU_CLASSES:
        return 2
    return 0


def box3d_corners(box: np.ndarray) -> np.ndarray:
    _, cx, cy, zc, length, width, height, yaw = box
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    dx = np.array(
        [length / 2, length / 2, -length / 2, -length / 2,
         length / 2, length / 2, -length / 2, -length / 2]
    )
    dy = np.array(
        [width / 2, -width / 2, -width / 2, width / 2,
         width / 2, -width / 2, -width / 2, width / 2]
    )
    dz = np.array(
        [height / 2, height / 2, height / 2, height / 2,
         -height / 2, -height / 2, -height / 2, -height / 2]
    )
    x = cx + dx * cos_yaw - dy * sin_yaw
    y = cy + dx * sin_yaw + dy * cos_yaw
    z = zc + dz
    return np.stack([x, y, z], axis=1)


def rotated_box_to_aabb(box: np.ndarray) -> np.ndarray:
    _, cx, cy, zc, length, width, height, _ = box
    yaw = float(box[7])
    abs_cos_yaw = abs(np.cos(yaw))
    abs_sin_yaw = abs(np.sin(yaw))
    half_x = (length * abs_cos_yaw + width * abs_sin_yaw) / 2.0
    half_y = (length * abs_sin_yaw + width * abs_cos_yaw) / 2.0
    half_z = height / 2.0
    return np.array(
        [
            cx - half_x,
            cy - half_y,
            zc - half_z,
            cx + half_x,
            cy + half_y,
            zc + half_z,
        ],
        dtype=np.float64,
    )


def read_pcd_binary(path: str | Path) -> np.ndarray:
    path = Path(path)
    raw = path.read_bytes()
    marker = b"DATA binary\n"
    marker_position = raw.find(marker)
    if marker_position < 0:
        raise ValueError(f"Binary PCD marker not found: {path}")

    header = raw[:marker_position].decode("ascii").splitlines()
    fields: list[str] = []
    sizes: list[int] = []
    field_types: list[str] = []
    counts: list[int] = []
    point_count = 0

    for line in header:
        key, *values = line.split()
        if key == "FIELDS":
            fields = values
        elif key == "SIZE":
            sizes = [int(value) for value in values]
        elif key == "TYPE":
            field_types = values
        elif key == "COUNT":
            counts = [int(value) for value in values]
        elif key == "POINTS":
            point_count = int(values[0])

    if not fields or not sizes or not field_types or not counts:
        raise ValueError(f"Incomplete PCD header: {path}")

    type_map = {
        "F": {4: "f4", 8: "f8"},
        "I": {1: "i1", 2: "i2", 4: "i4", 8: "i8"},
        "U": {1: "u1", 2: "u2", 4: "u4", 8: "u8"},
    }
    dtype_fields = []
    for field, size, field_type, count in zip(fields, sizes, field_types, counts):
        numpy_type = type_map[field_type][size]
        if count == 1:
            dtype_fields.append((field, numpy_type))
        else:
            dtype_fields.append((field, numpy_type, count))

    offset = marker_position + len(marker)
    dtype = np.dtype(dtype_fields)
    expected_size = point_count * dtype.itemsize
    if len(raw) - offset < expected_size:
        raise ValueError(f"Truncated binary PCD data: {path}")

    data = np.frombuffer(raw, dtype=dtype, count=point_count, offset=offset)
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_points(points_camera: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = points_camera[:, 2]
    safe_depth = np.maximum(depth, 1e-6)
    u = K[0, 0] * points_camera[:, 0] / safe_depth + K[0, 2]
    v = K[1, 1] * points_camera[:, 1] / safe_depth + K[1, 2]
    return np.stack([u, v], axis=1), depth


def cxcywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    _, cx, cy, width, height = box
    return np.array(
        [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
        dtype=np.float64,
    )


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def uniform_sample(count: int, total: int) -> list[int]:
    if total <= 0:
        return []
    if total <= count:
        return list(range(total))
    return sorted(
        {int(round(index * (total - 1) / (count - 1))) for index in range(count)}
    )
