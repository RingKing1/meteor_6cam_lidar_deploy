#!/usr/bin/env python3
"""Fake-INT8 proxy for the Orin lane-thinning problem (PyTorch, no TensorRT required).

ADAPTED FOR THE 6-CAMERA METEOR RIG (2026-09-23). Upstream origin:
  tier4/METEOR, branch `vla-overlay-int8-probes`, file `bevlane/probe_int8_lane.py`
  (Apache-2.0; see ../meteor_train/LICENSE-METEOR-Apache-2.0).
Changes vs upstream are marked with `# [6cam]`:
  - --n-cams (default 6) instead of a hard-coded n_cams=8;
  - --model (default v52) instead of a hard-coded MODELS["v52"];
  - --root / --list default to this repo's layout (was /data/dataset/bevlane + ~/work/BevLane/val.lst).

What it does: the training host has no TensorRT, so this simulates post-training INT8 in PyTorch —
per-output-channel symmetric 8-bit weights for every Conv2d, per-tensor symmetric 8-bit activations
at every Conv2d output (scale = absmax over calibration frames, like the entropy/minmax calibrators
to first order) — and reports BEV road / lane IoU and the lane-pixel ratio fp32 -> int8 on val frames.

  python3 patches/meteor_deploy/probe_int8_lane.py \
      --ckpt checkpoints/meteor_custom_v52_st5ep_20260923/ep2.pt --tag ep2

Output: tag  frames  road_fp32  road_int8  lane_fp32  lane_int8  lane_px_ratio  lane_iou_ratio

NOTE: needs a CUDA GPU and imports `bevlane.*` -> run from a METEOR tree on PYTHONPATH.
      It allocates a second model copy on the GPU: do NOT run it while training holds the card.
"""
import argparse, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset          # noqa: E402
from bevlane.model import MODELS                    # noqa: E402
from bevlane.ckpt_load import load_net              # noqa: E402

ROAD = (1, 3, 4, 5); LANE = (2, 6)
B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# [6cam] this repo's layout
DEF_ROOT = os.environ.get("METEOR_SCENES_ROOT", os.path.join(B, "meteor_6cam_lidar_deploy", "scenes"))
DEF_LIST = os.environ.get("METEOR_VAL_LIST", os.path.join(B, "meteor_6cam_lidar_deploy", "val_scenes.txt"))


class FakeInt8:
    """Attach: quantise conv weights in place (per out-channel) and register activation hooks."""

    def __init__(self, net, mode="absmax", skip=("hm", "reg", "tl_fc", "ego_mlp", "lg_", "intent")):
        self.hooks, self.scales, self.calib, self.mode, self.hist = [], {}, True, mode, {}
        self.mods = [(n, m) for n, m in net.named_modules()
                     if isinstance(m, nn.Conv2d) and not n.startswith(skip)]
        for n, m in self.mods:
            self.hooks.append(m.register_forward_hook(self._hook(n)))

    def quantize_weights(self):
        with torch.no_grad():
            for n, m in self.mods:
                w = m.weight
                s = w.abs().flatten(1).max(1)[0].clamp_min(1e-8) / 127.0
                s = s.view(-1, *([1] * (w.dim() - 1)))
                m.weight.copy_((w / s).round().clamp(-127, 127) * s)

    def _hook(self, name):
        def f(mod, inp, out):
            if self.calib:
                a = out.detach().abs().float().flatten()
                if self.mode == "absmax":
                    self.scales[name] = max(self.scales.get(name, 0.0), float(a.max()))
                else:
                    # keep a histogram (2048 bins over a running absmax) for percentile / KL calibration
                    amax = float(a.max()); h = self.hist.get(name)
                    if h is None or amax > h[1]:
                        nb = 2048; edges_max = amax * 1.05
                        if h is not None:   # re-bin the old histogram onto the wider range
                            old, omax = h; centers = (torch.arange(nb, device=a.device) + 0.5) * (omax / nb)
                            newh = torch.zeros(nb, device=a.device); idx = (centers / edges_max * nb).long().clamp(max=nb - 1)
                            newh.index_add_(0, idx, old); old = newh
                        else:
                            old = torch.zeros(nb, device=a.device)
                        h = [old, edges_max]; self.hist[name] = h
                    h[0] += torch.histc(a, bins=2048, min=0.0, max=h[1])
                return None
            s = self.scales.get(name, 0.0)
            if s <= 0:
                return None
            q = s / 127.0
            return (out / q).round().clamp(-127, 127) * q
        return f

    def finalize(self):
        """Turn histograms into scales: pct:<p> = percentile of |x| mass, kl = TensorRT-style entropy calibration."""
        if self.mode == "absmax":
            return
        for name, (h, amax) in self.hist.items():
            h = h.cpu().double(); nb = len(h); bw = amax / nb
            if self.mode.startswith("pct:"):
                p = float(self.mode[4:]) / 100.0; c = torch.cumsum(h, 0) / h.sum()
                self.scales[name] = float((int((c >= p).nonzero()[0]) + 1) * bw)
            else:   # kl (IInt8EntropyCalibrator2): threshold minimising KL(ref || quantised) over 128 bins
                best, bestd = nb, 1e30; hn = h + 1e-9
                for t in range(128, nb + 1, 16):
                    ref = hn[:t].clone(); ref[-1] += hn[t:].sum()
                    q = torch.zeros(t, dtype=torch.float64); step = t / 128.0
                    for b in range(128):
                        lo, hi = int(b * step), int((b + 1) * step) if b < 127 else t
                        seg = ref[lo:hi]; nz = (seg > 1e-8).double(); tot = seg.sum()
                        q[lo:hi] = (tot / max(float(nz.sum()), 1.0)) * nz
                    q += 1e-9; pr = ref / ref.sum(); qr = q / q.sum()
                    d = float((pr * (pr / qr).log()).sum())
                    if d < bestd: bestd, best = d, t
                self.scales[name] = float(best * bw)

    def remove(self):
        for h in self.hooks:
            h.remove()


def run(net, ds, idxs, fq=None):
    inter = {k: 0 for k in ("road", "lane")}; union = dict(inter); lane_px = 0; n = 0
    for i in idxs:
        x = ds[i]
        if x is None:
            continue
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = net(x[0][None].cuda(), x[1][None].cuda(), x[2][None].cuda())
        p = out[0].float().argmax(1)[0].cpu().numpy(); g = x[3].numpy(); valid = g != 255
        for nm, ks in (("road", ROAD), ("lane", LANE)):
            pm = np.isin(p, ks) & valid; gm = np.isin(g, ks) & valid
            inter[nm] += int((pm & gm).sum()); union[nm] += int((pm | gm).sum())
        lane_px += int((np.isin(p, LANE) & valid).sum()); n += 1
    return {k: inter[k] / max(union[k], 1) for k in inter}, lane_px, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", default="")
    ap.add_argument("--root", default=DEF_ROOT, help="[6cam] scene tree root")
    ap.add_argument("--list", default=DEF_LIST, help="[6cam] scene-name list")
    ap.add_argument("--n-scenes", type=int, default=6); ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--calib-frames", type=int, default=24); ap.add_argument("--gt-key", default="gt")
    ap.add_argument("--calib", default="absmax", help="absmax | pct:99.99 | pct:99.9 | kl")
    ap.add_argument("--model", default="v52", help="[6cam] model id in MODELS")
    ap.add_argument("--n-cams", type=int, default=6, help="[6cam] rig camera count (was hard-coded 8)")
    a = ap.parse_args()
    os.environ.setdefault("METEOR_ZERO_HIST", "1")
    scenes = [l.strip() for l in open(a.list) if l.strip()]
    ev_sc, cal_sc = scenes[:a.n_scenes], scenes[a.n_scenes:a.n_scenes + 4]
    net = MODELS[a.model](n_seg=21).cuda().eval(); load_net(net, a.ckpt, verbose=False)
    ds = BevLaneDataset(a.root, ev_sc, gt_key=a.gt_key, n_cams=a.n_cams, trim_start=3, trim_end=5)
    idxs = list(range(0, len(ds), a.stride))
    r32, px32, n = run(net, ds, idxs)
    fq = FakeInt8(net, mode=a.calib)
    cds = BevLaneDataset(a.root, cal_sc, gt_key=a.gt_key, n_cams=a.n_cams, trim_start=3, trim_end=5)
    cidx = np.linspace(0, len(cds) - 1, a.calib_frames).astype(int).tolist()
    fq.calib = True; run(net, cds, cidx); fq.calib = False; fq.finalize()
    fq.quantize_weights()
    r8, px8, _ = run(net, ds, idxs)
    print(f"{a.tag}/{a.calib}\t{n}\t{r32['road']:.4f}\t{r8['road']:.4f}\t{r32['lane']:.4f}\t{r8['lane']:.4f}\t"
          f"{px8 / max(px32, 1):.3f}\t{r8['lane'] / max(r32['lane'], 1e-6):.3f}", flush=True)


if __name__ == "__main__":
    main()
