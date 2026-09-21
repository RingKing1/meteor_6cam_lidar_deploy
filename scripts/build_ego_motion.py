#!/usr/bin/env python3
"""Compute full E2E ego-motion GT from raw localization files for METEOR.

Converts WGS84 GPS degrees (lon, lat) to local metric ENU (East, North, Up) in metres,
and computes waypoints in ego coordinates (x-fwd, y-left) [metres].

Produces:
  <scene>/ego_motion.npz containing:
    - wp:    float32 [F, 6, 2]  future waypoints (+0.5..+3.0s) in ego frame (x-fwd, y-left) [m]
    - v0:    float32 [F]        smoothed forward speed [m/s]
    - acc:   float32 [F]        longitudinal acceleration [m/s^2]
    - steer: float32 [F]        bicycle model steering angle [rad]
    - brake: float32 [F]        brake flag (acc < -0.5 m/s^2)
    - valid: float32 [F]        1.0 if full 3s future exists, else 0.0
    - pose:  float32 [F, 3]     global 2D ENU pose (x_m, y_m, yaw_rad) for temporal BEV alignment
"""
import argparse
import os
import re
import sys

import numpy as np

WHEELBASE = 2.8        # [m] bicycle model wheelbase
HORIZON = 6            # 6 future waypoints
WP_DT = 0.5            # 0.5s per waypoint (3.0s total)
V_MIN_STEER = 0.5      # [m/s] minimum speed for steering calculation
R_EARTH = 6378137.0    # [m] WGS84 Earth major radius


def parse_loc_yaml(path):
    """Fast regex parse of position, orientation, velocity, and timestamp from yaml."""
    txt = open(path).read()
    
    # timestamp
    m_ts = re.search(r"timestampSec:\s*([-\d.eE+]+)", txt)
    ts = float(m_ts.group(1)) if m_ts else 0.0

    # position (lon, lat, alt in degrees / metres)
    m_px = re.search(r"position:\s*\n\s*x:\s*([-\d.eE+]+)", txt)
    m_py = re.search(r"\n\s*y:\s*([-\d.eE+]+)", txt[m_px.start():])
    m_pz = re.search(r"\n\s*z:\s*([-\d.eE+]+)", txt[m_px.start():])
    lon = float(m_px.group(1))
    lat = float(m_py.group(1))
    alt = float(m_pz.group(1))

    # orientation quaternion (w, x, y, z)
    m_ow = re.search(r"orientation:\s*\n\s*w:\s*([-\d.eE+]+)", txt)
    m_ox = re.search(r"\n\s*x:\s*([-\d.eE+]+)", txt[m_ow.start():])
    m_oy = re.search(r"\n\s*y:\s*([-\d.eE+]+)", txt[m_ow.start():])
    m_oz = re.search(r"\n\s*z:\s*([-\d.eE+]+)", txt[m_ow.start():])
    w = float(m_ow.group(1))
    qx = float(m_ox.group(1))
    qy = float(m_oy.group(1))
    qz = float(m_oz.group(1))

    # yaw angle from quaternion (standard ENU: 0=East, pi/2=North)
    yaw = np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    # body/sensor frame velocity
    m_vx = re.search(r"vel:\s*\n\s*x:\s*([-\d.eE+]+)", txt)
    m_vy = re.search(r"\n\s*y:\s*([-\d.eE+]+)", txt[m_vx.start():])
    vx = float(m_vx.group(1)) if m_vx else 0.0
    vy = float(m_vy.group(1)) if m_vy else 0.0
    v = float(np.hypot(vx, vy))

    return ts, lon, lat, alt, yaw, v


def wgs84_to_enu(lons, lats, lon0, lat0):
    """Convert WGS84 lon, lat in degrees to local East-North-Up (ENU) in metres."""
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)

    east = (lon_rad - lon0_rad) * R_EARTH * np.cos(lat0_rad)
    north = (lat_rad - lat0_rad) * R_EARTH
    return east, north


def process_scene(src_dir, dst_scene_dir):
    loc_dir = os.path.join(src_dir, "localization")
    if not os.path.exists(loc_dir):
        raise RuntimeError(f"Missing localization directory in {src_dir}")

    # Read timestamps matching lidar frames
    lidar_dir = os.path.join(src_dir, "lidar")
    pcd_files = sorted(f[:-4] for f in os.listdir(lidar_dir) if f.endswith(".pcd"))
    F = len(pcd_files)
    print(f"[*] Processing {src_dir}: {F} frames ...", flush=True)

    ts_list, lons, lats, alts, yaws, vs = [], [], [], [], [], []
    for t_str in pcd_files:
        loc_path = os.path.join(loc_dir, f"{t_str}.yaml")
        ts, lon, lat, alt, yaw, v = parse_loc_yaml(loc_path)
        ts_list.append(ts)
        lons.append(lon)
        lats.append(lat)
        alts.append(alt)
        yaws.append(yaw)
        vs.append(v)

    ts = np.array(ts_list)
    lons = np.array(lons)
    lats = np.array(lats)
    yaws = np.unwrap(np.array(yaws))
    vs = np.array(vs)

    # Convert WGS84 coordinates to local metric ENU relative to frame 0
    lon0, lat0 = lons[0], lats[0]
    xs, ys = wgs84_to_enu(lons, lats, lon0, lat0)

    # Gradient & smoothing
    dt = np.gradient(ts)
    dt[dt <= 0] = 0.1  # safety default 10Hz

    # Box-smooth velocity (5 frames ~ 0.5s)
    k = 5
    ker = np.ones(k) / k
    v_smooth = np.convolve(vs, ker, "same")
    acc = np.convolve(np.gradient(v_smooth) / dt, ker, "same")
    yaw_rate = np.convolve(np.gradient(yaws) / dt, ker, "same")

    # Arrays
    wp = np.zeros((F, HORIZON, 2), np.float32)
    v0 = np.zeros(F, np.float32)
    ac = np.zeros(F, np.float32)
    st = np.zeros(F, np.float32)
    br = np.zeros(F, np.float32)
    valid = np.zeros(F, np.float32)
    pose = np.zeros((F, 3), np.float32)

    for i in range(F):
        v0[i] = v_smooth[i]
        ac[i] = acc[i]
        st[i] = (np.arctan(WHEELBASE * yaw_rate[i] / max(v_smooth[i], 1e-4))
                 if v_smooth[i] > V_MIN_STEER else 0.0)
        br[i] = 1.0 if acc[i] < -0.5 else 0.0
        pose[i] = [xs[i], ys[i], yaws[i]]

        # Future waypoints
        t0 = ts[i]
        t_targets = t0 + WP_DT * np.arange(1, HORIZON + 1)
        if t_targets[-1] > ts[-1]:
            # Scene end: not enough future poses
            continue

        xq = np.interp(t_targets, ts, xs)
        yq = np.interp(t_targets, ts, ys)
        
        # Transform global ENU waypoints to ego coordinate frame (x-fwd, y-left) [metres]
        dx = xq - xs[i]
        dy = yq - ys[i]
        c, s = np.cos(yaws[i]), np.sin(yaws[i])
        wp[i, :, 0] = c * dx + s * dy     # ego x (forward [m])
        wp[i, :, 1] = -s * dx + c * dy    # ego y (left [m])
        valid[i] = 1.0

    out_npz = os.path.join(dst_scene_dir, "ego_motion.npz")
    np.savez_compressed(
        out_npz,
        wp=wp.astype(np.float32),
        v0=v0.astype(np.float32),
        acc=ac.astype(np.float32),
        steer=st.astype(np.float32),
        brake=br.astype(np.float32),
        valid=valid.astype(np.float32),
        pose=pose.astype(np.float32)
    )
    total_dist = np.hypot(xs[-1], ys[-1])
    print(f"[+] Saved {out_npz} (Valid frames: {int(valid.sum())}/{F}, Traveled distance: {total_dist:.1f} m)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Raw data directory (containing localization/)")
    ap.add_argument("--dst", required=True, help="Target scene directory (where ego_motion.npz is saved)")
    args = ap.parse_args()
    process_scene(args.src, args.dst)


if __name__ == "__main__":
    main()
