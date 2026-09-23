#!/usr/bin/env python3
"""Val-OOM probe: memory of the full `evaluate_ego` step under 4 precision modes.

The epoch-end val OOM happens inside evaluate_ego(), whose per-sample cost is
  _temporal_inputs()  -> 3x model.compute_bev() (history slots) + stack
  model(imgs,K,Tc,v0,pb,th) -> the FULL 19-head forward
This script reproduces exactly that pair and reports peak allocated memory for:
  A. fp32          + torch.no_grad()        (current val path)
  B. fp32          + torch.inference_mode()
  C. autocast fp16 + torch.no_grad()        (the proposed "wrap in autocast")
  D. net.half()    + torch.no_grad()        (true fp16 model)
plus the plain compute_bev history pass, and fp16 numerics vs fp32.

Run: docker run --rm --gpus all -v <ws>:/work meteor_training:v1 \
       python3 /work/meteor_6cam_lidar_deploy/tests/bench_val_memory.py
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
# evaluate_ego's tmp_idx for our flag set (seg2d, traj, ego, occ, tl all on)
TMP_IDX = 4 + 1 + 4 + 1 + 1 + 1


def mb(x):
    return x / 2 ** 20


def load_net():
    torch.manual_seed(0)
    net = MODELS["v52"](n_seg=21).to(DEV).eval()
    try:
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck) if isinstance(ck, dict) else ck
        # match the init's slimmed depth head exactly like train.py does
        if "depth_head.0.0.weight" in sd and hasattr(net, "depth_head"):
            w_ck = tuple(sd[f"depth_head.{i}.0.weight"].shape[0]
                         for i in range(4)
                         if f"depth_head.{i}.0.weight" in sd)
            w_cur = tuple(m[0].out_channels for m in net.depth_head[:-1])
            if len(w_ck) == 4 and w_ck != w_cur:
                enable_depth_slim(net, widths=w_ck)
                print(f"[depth-slim] rebuilt at width {w_ck} to match init",
                      flush=True)
        miss, unexp = net.load_state_dict(sd, strict=False)
        print(f"[ckpt] {os.path.basename(CKPT)} loaded missing={len(miss)} "
              f"unexpected={len(unexp)}", flush=True)
        del ck, sd
    except Exception as e:                       # noqa: BLE001
        print(f"[ckpt] load failed ({e}); random init", flush=True)
    for p in net.parameters():
        p.requires_grad_(False)
    torch.cuda.empty_cache()
    return net


def make_case():
    """imgs/K/Tc + a fake dataset batch carrying the 3 history slots so the
    real _temporal_inputs() can be called unchanged."""
    B, N, H, W = 1, 8, 432, 768
    g = torch.Generator().manual_seed(1)
    imgs = torch.randn(B, N, 3, H, W, generator=g).to(DEV)
    K = torch.zeros(B, N, 3, 3, device=DEV)
    K[:, :, 0, 0], K[:, :, 1, 1] = 700.0, 700.0
    K[:, :, 0, 2], K[:, :, 1, 2] = 384.0, 216.0
    K[:, :, 2, 2] = 1.0
    Tc = torch.eye(4, device=DEV).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    Tc[:, :, 2, 3] = 1.5
    hist = torch.randn(B, N_HIST, N, 3, H, W, generator=g).to(DEV)   # 6-D
    rel = torch.zeros(B, N_HIST, 3, device=DEV)
    pv = torch.ones(B, N_HIST, device=DEV)
    v0 = torch.full((B,), 6.0, device=DEV)                            # 6 m/s
    batch = [None] * (TMP_IDX + 3)
    batch[0], batch[1], batch[2] = imgs, K, Tc
    batch[TMP_IDX], batch[TMP_IDX + 1], batch[TMP_IDX + 2] = hist, rel, pv
    return imgs, K, Tc, v0, batch


def peak_of(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = fn()
    torch.cuda.synchronize()
    dt = time.time() - t0
    return out, torch.cuda.max_memory_allocated(), dt


def main():
    prop = torch.cuda.get_device_properties(0)
    print("=" * 78)
    print(f"val-step memory probe  |  GPU {torch.cuda.get_device_name(0)} "
          f"total={mb(prop.total_memory):.0f} MB")
    print("=" * 78, flush=True)
    net = load_net()
    imgs, K, Tc, v0, batch = make_case()

    def run_history(mode):
        pv = batch[TMP_IDX + 2]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            if mode == "fp16":
                with torch.autocast("cuda", torch.float16):
                    pbs = [net.compute_bev(imgs, K, Tc)
                           * pv[:, i].view(-1, 1, 1, 1) for i in range(N_HIST)]
                    pb = torch.stack(pbs, 1)
            else:
                with torch.autocast("cuda", enabled=False):
                    pbs = [net.compute_bev(imgs, K, Tc)
                           * pv[:, i].view(-1, 1, 1, 1) for i in range(N_HIST)]
                    pb = torch.stack(pbs, 1)
        torch.cuda.synchronize()
        return pb, torch.cuda.max_memory_allocated()

    print("\n--- 3x compute_bev history pass ---")
    pb32, p32 = run_history("fp32")
    print(f"  fp32 peak={mb(p32):8.1f} MB")
    del pb32
    pb16, p16 = run_history("fp16")
    print(f"  fp16 peak={mb(p16):8.1f} MB   ({mb(p16 - p32):+.1f} MB vs fp32)")
    del pb16

    def val_step(mode, use_inference=False, half_model=False):
        net_ = net.half() if half_model else net
        def fn():
            if use_inference:
                cm = torch.inference_mode()
            else:
                cm = torch.no_grad()
            with cm:
                pb, th = _temporal_inputs(net_, batch, DEV, TMP_IDX)
                if mode == "fp16" and not half_model:
                    with torch.autocast("cuda", torch.float16):
                        out = net_(imgs, K, Tc, v0, pb, th)
                else:
                    out = net_(imgs, K, Tc, v0, pb, th)
            return out
        out, pk, dt = peak_of(fn)
        if half_model:
            net.float()
        return out, pk, dt

    print("\n--- FULL evaluate_ego-style step (3 history + all-head forward) ---")
    results = {}
    for name, kw in (
        ("A fp32 no_grad   (current)", dict(mode="fp32")),
        ("B fp32 infer_mode", dict(mode="fp32", use_inference=True)),
        ("C autocast fp16  (proposed)", dict(mode="fp16")),
        ("D true fp16 model", dict(mode="fp16", half_model=True)),
    ):
        try:
            out, pk, dt = val_step(**kw)
            results[name] = (pk, dt)
            print(f"  {name:28s} peak={mb(pk):8.1f} MB  time={dt * 1000:7.1f} ms")
            del out
        except Exception as e:                   # noqa: BLE001
            print(f"  {name:28s} FAILED: {type(e).__name__}: {e}")
        torch.cuda.empty_cache()

    base = results.get("A fp32 no_grad   (current)", (0, 0))[0]
    print("\n--- deltas vs current (A) ---")
    for name, (pk, _) in results.items():
        print(f"  {name:28s} {mb(pk - base):+8.1f} MB "
              f"({100 * (pk - base) / max(base, 1):+.1f} %)")
    print("=" * 78)


if __name__ == "__main__":
    main()
