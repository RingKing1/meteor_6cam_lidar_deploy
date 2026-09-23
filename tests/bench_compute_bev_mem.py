#!/usr/bin/env python3
"""Benchmark: `_temporal_inputs` history-BEV pass in fp32 vs autocast(fp16).

This is the val-OOM root cause probe:
  evaluate_ego() (and every other evaluate_*) builds the 3-slot history BEV via
  _temporal_inputs(), which calls model.compute_bev() 3x OUTSIDE any autocast
  block (train.py:633-657) -> fp32 activations on top of the training
  reservation. This script measures the memory/time cost of that exact pattern
  and the fp16 alternative, plus the numerical delta of the BEV feature.

Run (GPU):  docker run --rm --gpus all -v <workspace>:/work meteor_training:v1 \
              python3 /work/meteor_6cam_lidar_deploy/tests/bench_compute_bev_mem.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, "/work/METEOR")
from bevlane.model import MODELS  # noqa: E402

DEV = "cuda"
CKPT = os.environ.get(
    "BENCH_CKPT",
    "/work/METEOR/models/meteor_v157.pt")
N_HIST = 3          # v29 memory queue slots (t-2, t-6, t-14)


def mb(x):
    return x / 2 ** 20


def load_net():
    torch.manual_seed(0)
    net = MODELS["v52"](n_seg=21).to(DEV).eval()
    try:
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck) if isinstance(ck, dict) else ck
        miss, unexp = net.load_state_dict(sd, strict=False)
        print(f"[ckpt] {os.path.basename(CKPT)} loaded "
              f"missing={len(miss)} unexpected={len(unexp)}", flush=True)
        del ck, sd
    except Exception as e:                      # noqa: BLE001
        print(f"[ckpt] load failed ({e}); random init", flush=True)
    for p in net.parameters():
        p.requires_grad_(False)                 # val = inference only
    torch.cuda.empty_cache()
    return net


def make_inputs():
    B, N = 1, 8
    g = torch.Generator(device="cpu").manual_seed(1)
    imgs = torch.randn(B, N, 3, 432, 768, generator=g).to(DEV)
    K = torch.zeros(B, N, 3, 3, device=DEV)
    K[:, :, 0, 0], K[:, :, 1, 1] = 700.0, 700.0
    K[:, :, 0, 2], K[:, :, 1, 2] = 384.0, 216.0
    K[:, :, 2, 2] = 1.0
    Tc = torch.eye(4, device=DEV).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    Tc[:, :, 2, 3] = 1.5                       # camera 1.5 m in front
    pv = torch.ones(B, N_HIST, device=DEV)
    return imgs, K, Tc, pv


def run_history_pass(net, imgs, K, Tc, pv, mode):
    """Exact pattern of _temporal_inputs for v29+ (tfuse3): 3x compute_bev,
    each scaled by its validity mask, then stacked and kept alive."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        if mode == "fp16":
            ctx = torch.autocast("cuda", torch.float16)
        else:
            ctx = torch.autocast("cuda", enabled=False)
        with ctx:
            pbs = []
            for hi in range(N_HIST):
                pbs.append(net.compute_bev(imgs, K, Tc)
                           * pv[:, hi].view(-1, 1, 1, 1))
            pb = torch.stack(pbs, 1)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated()
    return pb, peak, dt


def run_single(net, imgs, K, Tc, mode):
    """One compute_bev (what the training forward pays per step, in autocast)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        if mode == "fp16":
            with torch.autocast("cuda", torch.float16):
                out = net.compute_bev(imgs, K, Tc)
        else:
            with torch.autocast("cuda", enabled=False):
                out = net.compute_bev(imgs, K, Tc)
    torch.cuda.synchronize()
    return out, torch.cuda.max_memory_allocated(), time.time() - t0


def main():
    print("=" * 78)
    print("compute_bev memory benchmark (val _temporal_inputs root-cause probe)")
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"total={mb(torch.cuda.get_device_properties(0).total_memory):.0f} MB")
    print("=" * 78, flush=True)
    net = load_net()
    imgs, K, Tc, pv = make_inputs()

    print("\n--- single compute_bev (1 frame) ---")
    o32, p32, t32 = run_single(net, imgs, K, Tc, "fp32")
    o16, p16, t16 = run_single(net, imgs, K, Tc, "fp16")
    print(f"  fp32 : peak={mb(p32):8.1f} MB  time={t32 * 1000:7.1f} ms")
    print(f"  fp16 : peak={mb(p16):8.1f} MB  time={t16 * 1000:7.1f} ms")
    print(f"  saving: {mb(p32 - p16):.1f} MB  ({100 * (p32 - p16) / p32:.1f} %)")

    print(f"\n--- history pass: {N_HIST}x compute_bev + stack (val path) ---")
    del o32, o16
    hb32, hp32, ht32 = run_history_pass(net, imgs, K, Tc, pv, "fp32")
    numerr = (hb32.float() - hb32.float()).abs().max().item()
    hb16, hp16, ht16 = run_history_pass(net, imgs, K, Tc, pv, "fp16")
    print(f"  fp32 : peak={mb(hp32):8.1f} MB  time={ht32 * 1000:7.1f} ms")
    print(f"  fp16 : peak={mb(hp16):8.1f} MB  time={ht16 * 1000:7.1f} ms")
    print(f"  saving: {mb(hp32 - hp16):.1f} MB "
          f"({100 * (hp32 - hp16) / hp32:.1f} %)")

    # numerical delta fp32 vs fp16 on the same weights
    d = (hb32.float() - hb16.float()).abs()
    denom = hb32.float().abs().clamp(min=1e-3)
    print(f"\n--- fp16 vs fp32 BEV feature ---")
    print(f"  max |diff|={d.max().item():.4f}  "
          f"mean |diff|={d.mean().item():.5f}  "
          f"max rel={((d / denom).max().item() * 100):.2f}%  "
          f"finite(fp16)={bool(torch.isfinite(hb16).all())}")

    print("\n--- projected: val evaluate_ego peak estimate ---")
    print(f"  history pass fp32 = {mb(hp32):.0f} MB, fp16 = {mb(hp16):.0f} MB "
          f"-> the val spike shrinks by ~{mb(hp32 - hp16):.0f} MB")
    print("=" * 78)


if __name__ == "__main__":
    main()
