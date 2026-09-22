#!/usr/bin/env python3
"""SimpleTrack (ICRA 2022) 3D MOT engine adapter for METEOR autolabel pipeline.

Bridges the FocalFormer bev_box detections (ego frame) into SimpleTrack's
`mot_3d` library running in WORLD ENU coordinates, and returns per-frame
persistent track ids aligned with the raw detections.

Design notes (parity with the previous custom MOTTracker3D):
  - Detections are transformed ego -> world via ego_motion poses first.
  - One SimpleTrack MOTModel per object class (official per-class pattern),
    preventing cross-class association.
  - Detections carry a synthetic score of 1.0 (bev_box has no score field;
    upstream already filtered by score >= 0.15).
  - Every frame is fed as a key frame at 10 Hz (time_stamp = frame_idx * 0.1).
  - Output: for each frame, a list of (det_index, track_id) matching THIS
    frame's detections only (redundancy-predicted tracks with no raw det are
    excluded, matching the old engine's behavior of emitting only present
    detections).
"""
import sys
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIMPLETRACK_ROOT = os.path.abspath(
    os.path.join(_THIS_DIR, "..", "..", "tools", "SimpleTrack-main"))
if _SIMPLETRACK_ROOT not in sys.path:
    sys.path.insert(0, _SIMPLETRACK_ROOT)

from mot_3d.mot import MOTModel            # noqa: E402
from mot_3d.frame_data import FrameData    # noqa: E402
from mot_3d.data_protos import BBox        # noqa: E402

# ---------------------------------------------------------------------------
# SimpleTrack config (high-recall autolabel GT orientation)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "running": {
        "covariance": "default",
        "score_threshold": 0.01,        # all autolabel boxes pass
        "max_age_since_update": 10,     # frames a track survives w/o assoc (10 Hz)
        "min_hits_to_birth": 1,         # reject single-frame flicker tracks
        "match_type": "bipartite",      # Hungarian
        "asso": "giou",                 # geometry-aware association
        "has_velo": False,
        "nms_thres": 0.1,
        "motion_model": "kf",           # 10-dim Kalman filter
        "asso_thres": {"giou": 1.5, "iou": 0.9},
    },
    "redundancy": {
        "mode": "mm",                   # two-stage association (occlusion recovery)
        "det_score_threshold": {"iou": 0.01, "giou": 0.01},
        "det_dist_threshold": {"iou": 0.1, "giou": -0.5},
    },
    "data_loader": {"pc": False, "nms": False, "nms_thres": 0.1},
}

DT_FRAME = 0.1          # 10 Hz
ASSOC_MATCH_DIST = 10.0  # candidate outputs are 1:1 with dets (mode-1/birth
                         # only); generous threshold only disambiguates order.
                         # KF posteriors can transiently sit >2 m off the
                         # measurement (new tracks, high velocity covariance).


def build_config(asso="giou", max_age=10, min_hits=1, redundancy_mode="mm"):
    cfg = {
        "running": {
            "covariance": "default",
            "score_threshold": 0.01,
            "max_age_since_update": max_age,
            "min_hits_to_birth": min_hits,
            "match_type": "bipartite",
            "asso": asso,
            "has_velo": False,
            "nms_thres": 0.1,
            "motion_model": "kf",
            "asso_thres": {"giou": 1.5, "iou": 0.9},
        },
        "redundancy": {
            "mode": redundancy_mode,
            "det_score_threshold": {"iou": 0.01, "giou": 0.01},
            "det_dist_threshold": {"iou": 0.1, "giou": -0.5},
        },
        "data_loader": {"pc": False, "nms": False, "nms_thres": 0.1},
    }
    return cfg


class SimpleTrackMOTEngine:
    """Per-class SimpleTrack tracker digesting world-frame detections."""

    def __init__(self, configs=None):
        self.configs = configs or build_config()
        self.trackers = {}          # cls_id -> MOTModel
        self.next_frame = 0

    def _engine(self, cls_id):
        if cls_id not in self.trackers:
            self.trackers[cls_id] = MOTModel(self.configs)
        return self.trackers[cls_id]

    def update_frame(self, dets_world, frame_idx):
        """dets_world: list of dicts:
            {'cls': int, 'pos_w': [x,y,z], 'size': [l,w,h], 'yaw_w': float,
             'ego_box': (cls, cx, cy, l, w, yaw_ego)}
        Returns: dict id(det) -> track_id (persistent across frames).
        """
        matched_track_ids = {}
        time_stamp = frame_idx * DT_FRAME

        # group detections per class
        by_cls = {}
        for det in dets_world:
            by_cls.setdefault(det["cls"], []).append(det)

        for cls_id, dets in by_cls.items():
            engine = self._engine(cls_id)
            # world-frame [x, y, z, yaw, l, w, h, score]
            det_arrays = [
                [det["pos_w"][0], det["pos_w"][1], det["pos_w"][2],
                 det["yaw_w"], det["size"][0], det["size"][1], det["size"][2], 1.0]
                for det in dets
            ]
            frame_data = FrameData(
                dets=det_arrays,
                ego=np.eye(4, dtype=np.float64),
                time_stamp=time_stamp,
                det_types=[str(cls_id)] * len(dets),
                aux_info={"is_key_frame": True},
            )
            tracks = engine.frame_mot(frame_data)
            if not tracks or not dets:
                continue

            # Only tracks that were ASSOCIATED with a detection this frame
            # (state 'alive_1_*' = stage-1 matched, 'birth_*' = newly created)
            # correspond 1:1 to this frame's raw detections. Stage-0/3 outputs
            # are motion-model predictions without a raw det and must NOT be
            # matched (they could shadow a new track near a stale prediction).
            def _state_tokens(st):
                return st.split("_")

            candidates = []
            for b, tid, st, _ in tracks:
                toks = _state_tokens(st)
                if toks[0] == "alive" and len(toks) >= 2 and int(toks[1]) == 1:
                    candidates.append((b, tid))
                elif toks[0] == "birth":
                    candidates.append((b, tid))
            if not candidates:
                continue

            out_centers = np.array([[BBox.bbox2array(b)[0],
                                     BBox.bbox2array(b)[1]] for b, _ in candidates])
            det_centers = np.array([d["pos_w"][:2] for d in dets])

            # Hungarian map associated output tracks -> raw detections
            cost = np.linalg.norm(
                out_centers[:, None, :] - det_centers[None, :, :], axis=2)
            cost[cost > ASSOC_MATCH_DIST] = 1e6
            row_ind, col_ind = linear_sum_assignment(cost)
            used_cols = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < ASSOC_MATCH_DIST and c not in used_cols:
                    used_cols.add(c)
                    matched_track_ids[id(dets[c])] = candidates[r][1]

        self.next_frame = frame_idx + 1
        return matched_track_ids
