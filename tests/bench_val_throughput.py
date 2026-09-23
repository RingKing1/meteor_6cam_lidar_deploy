#!/usr/bin/env python3
"""Throughput/memory sweep for a FULL 2970-frame val evaluation.

`train.py`'s val path is throttled in three places, so a naive "whole scene"
setting silently becomes ~165 frames:
  1. dataset:  frames[:max_per_scene]      (max_per_scene=8 -> first 8 frames)
  2. dv_ep:    _st = len(va)//160          (stride subsample to ~160 samples)
  3. evaluate: max_batches=vcap(n)         (batch caps of 20..166)
This script measures the real per-frame cost of the heavy val step (3 history
BEV + all-head forward) at several batch sizes, so a full-scene run can be
costed before committing to it.

Run: docker run --rm --gpus all -v <ws>:/work meteor_training:v1 \
       python3 /work/meteor_6cam_lidar_deploy/tests/bench_val_throughput.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, "/work/METEOR")
from bevlane.model import MODELS, enable_depth_slim  # noqa: E402
from bevlane.train import _temporal_inputs          # noqa: E402

DEV = "cuda"
CKPT = os.environ.get("BENCH_CKPT", "/work/METEOR/models/meteor_v157.pt")
N_HIST = 3
TMP_IDX = 4 + 1 + 4 + 1 + 1 + 1
N_VAL_FRAMES = 2970
# evaluators that run a full all-head forward per frame in our flag set
N_EVALS = 9   # seg, class_pr x2, seg2d, occ, det3d, stat, traj, tl, ego(+hs)


def mb(x):
    return x / 2 ** 20


def load_net():
    torch.manual_seed(0)
    net = MODELS["v52"](n_seg=21).to(DEV).eval()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    if "depth_head.0.0.weight" in sd and hasattr(net, "depth_head"):
        w = tuple(sd[f"depth_head.{i}.0.weight"].shape[0] for i in range(4)
                  if f"depth_head.{i}.0.weight" in sd)
        wc = tuple(m[0].out_channels for m in net.depth_head[:-1])
        if len(w) == 4 and w != wc:
            enable_depth_slim(net, widths=w)
    net.load_state_dict(sd, strict=False)
    del ck, sd
    for p in net.parameters():
        p.requires_grad_(False)
    torch.cuda.empty_cache()
    return net


def make_case(B):
    N, H, W = 8, 432, 768
    g = torch.Generator().manual_seed(1)
    imgs = torch.randn(B, N, 3, H, W, generator=g).to(DEV)
    K = torch.zeros(B, N, 3, 3, device=DEV)
    K[:, :, 0, 0], K[:, :, 1, 1] = 700.0, 700.0
    K[:, :, 0, 2], K[:, :, 1, 2] = 384.0, 216.0
    K[:, :, 2, 2] = 1.0
    Tc = torch.eye(4, device=DEV).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    Tc[:, :, 2, 3] = 1.5
    hist = torch.randn(B, N_HIST, N, 3, H, W, generator=g).to(DEV)
    rel = torch.zeros(B, N_HIST, 3, device=DEV)
    pv = torch.ones(B, N_HIST, device=DEV)
    v0 = torch.full((B,), 6.0, device=DEV)
    batch = [None] * (TMP_IDX + 3)
    batch[0], batch[1], batch[2] = imgs, K, Tc
    batch[TMP_IDX], batch[TMP_IDX + 1], batch[TMP_IDX + 2] = hist, rel, pv
    return imgs, K, Tc, v0, batch


def main():
    print("=" * 78)
    print(f"throughput sweep for a FULL {N_VAL_FRAMES}-frame val "
          f"({N_EVALS} evaluators)")
    print("=" * 78, flush=True)
    net = load_net()
    print(f"{'batch':>6s} {'ms/iter':>9s} {'ms/frame':>9s} {'peak MB':>9s} "
          f"{'full-val est':>14s}")
    print("-" * 78)
    for B in (1, 2, 4, 8, 16):
        try:
            imgs, K, Tc, v0, batch = make_case(B)
            # warmup
            with torch.no_grad():
                pb, th = _temporal_inputs(net, batch, DEV, TMP_IDX)
                _ = net(imgs, K, Tc, v0, pb, th)
            del pb, th, _
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            iters = 3
            t0 = time.time()
            with torch.no_grad():
                for _ in range(iters):
                    pb, th = _temporal_inputs(net, batch, DEV, TMP_IDX)
                    out = net(imgs, K, Tc, v0, pb, th)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / iters
            peak = torch.cuda.max_memory_allocated()
            ms_f = dt * 1000 / B
            # full scene: frames*N_EVALS steps of this cost
            est_min = ms_f * N_VAL_FRAMES * N_EVALS / 1000 / 60
            print(f"{B:6d} {dt * 1000:9.1f} {ms_f:9.1f} {mb(peak):9.1f} "
                  f"{est_min:11.0f} min")
            del pb, th, out, imgs, K, Tc, batch
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"{B:6d}   OOM at this batch size")
            torch.cuda.empty_cache()
            break
    print("-" * 78)
    print(f"note: est = frames({N_VAL_FRAMES}) x evaluators({N_EVALS}) x "
          f"ms/frame, i.e. ONE full-scene val pass")
    print("=" * 78)


if __name__ == "__main__":
    main()
