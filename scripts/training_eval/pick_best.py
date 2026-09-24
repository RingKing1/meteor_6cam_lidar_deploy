#!/usr/bin/env python3
"""Pick the best epoch checkpoint by PARSING the validation output in train logs.

Every epoch-end val has already run during training (full set with --val-full),
so re-running inference per checkpoint would waste ~47 min x3. This parses the
[valX epN] lines straight out of the training logs, compares epochs, and writes
a verdict: lower score = better.

  python3 scripts/training_eval/pick_best.py \
      --epoch-logs '2:logs/train_ep5_fullval_20260923.log' \
                   '3:logs/train_ep5_resume_ep2_ipc_20260923.log' \
                   '4:logs/train_ep5_fullval_XXXX.log' \
      --ckpt-dir checkpoints/meteor_custom_v52_st5ep_20260923 \
      --score-json checkpoints/meteor_custom_v52_st5ep_20260923/pick_best_scores.json
"""
import argparse, json, os, re, sys

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--epoch-logs", nargs="+", required=True,
                help="'<epoch>:<path to train log>' pairs, in epoch order")
ap.add_argument("--ckpt-dir", required=True)
ap.add_argument("--score-json", default="")
a = ap.parse_args()


def parse_val(epoch, path):
    """Extract the [valX epN] metric lines for one epoch into a flat dict."""
    txt = open(os.path.join(BASE_DIR, path) if not os.path.isabs(path) else path,
                errors="ignore").read()
    out = {}
    patterns = {
        "bev_miou": (rf"\[val ep{epoch}\] mIoU=([0-9.]+)", float),
        "seg2d_miou": (rf"\[val2d ep{epoch}\] mIoU=([0-9.]+)", float),
        "det_veh_P": (rf"\[val3D ep{epoch}\] veh P=([0-9.]+)", float),
        "det_veh_R": (rf"\[val3D ep{epoch}\] veh P=[0-9.]+ R=([0-9.]+)", float),
        "det_veh_Rn": (rf"\[val3D ep{epoch}\].* Rn=([0-9.]+)", float),
        "stat_acc": (rf"\[valStat ep{epoch}\].* acc=([0-9.]+)", float),
        "traj_ade": (rf"\[valTraj ep{epoch}\] agentADE=([0-9.]+)", float),
        "tl_acc": (rf"\[valTL ep{epoch}\] acc=([0-9.]+)", float),
        "e2e_ade": (rf"\[valE2E ep{epoch}\] ADE=([0-9.]+)", float),
        "e2e_fde": (rf"\[valE2E ep{epoch}\].* FDE=([0-9.]+)", float),
        "e2e_adec": (rf"\[valE2E ep{epoch}\].* ADEc=([0-9.]+)", float),
        "e2e_oracle": (rf"\[valE2Ed ep{epoch}\] oracle=([0-9.]+)", float),
    }
    for k, (pat, conv) in patterns.items():
        m = re.search(pat, txt)
        out[k] = conv(m.group(1)) if m else None
    return out


epochs = []
for spec in a.epoch_logs:
    ep, path = spec.split(":", 1)
    epochs.append((int(ep), path))

raw = {}
for ep, path in epochs:
    r = parse_val(ep, path)
    raw[ep] = r
    missing = [k for k, v in r.items() if v is None]
    print(f"[pick] epoch {ep} ({os.path.basename(path)}): "
          f"{json.dumps(r)}" + (f"  MISSING: {missing}" if missing else ""),
          flush=True)

# ---- score: error metrics normalised across epochs; IoU/accuracy inverted ----
ERROR_KEYS = ["e2e_ade", "e2e_fde", "e2e_adec", "e2e_oracle", "traj_ade"]
GOOD_KEYS = ["bev_miou", "seg2d_miou", "det_veh_R", "det_veh_Rn", "stat_acc",
             "tl_acc"]
WEIGHTS = {"e2e_ade": 0.25, "e2e_adec": 0.20, "e2e_oracle": 0.10,
           "traj_ade": 0.10, "bev_miou": 0.15, "seg2d_miou": 0.10,
           "det_veh_Rn": 0.05, "stat_acc": 0.05}

eps = [ep for ep, _ in epochs]


def norm(k):
    vals = [raw[ep][k] for ep in eps if raw[ep][k] is not None]
    if not vals:
        return None
    return min(vals), max(vals)


final = {ep: 0.0 for ep in eps}
used_w = 0.0
for k, w in WEIGHTS.items():
    n = norm(k)
    if n is None:
        continue
    lo, hi = n
    for ep in eps:
        v = raw[ep][k]
        if v is None:
            continue
        x = 0.5 if hi == lo else (v - lo) / (hi - lo)
        if k in GOOD_KEYS:
            x = 1.0 - x          # higher-is-better
        final[ep] += w * x
    used_w += w

print("\n[pick] per-epoch score (lower = better):")
best_ep = min(eps, key=lambda e: final[e])
for ep in eps:
    print(f"  ep{ep}: {final[ep]:.4f}" + ("  <-- BEST" if ep == best_ep else ""))

best_ckpt = os.path.join(a.ckpt_dir, f"ep{best_ep}.pt")
print(f"\n[pick] BEST checkpoint = {best_ckpt}", flush=True)

# point a stable name at the best epoch checkpoint (copy; ep{N}.pt stays as-is)
picked = os.path.join(a.ckpt_dir, "picked_best.pt")
import shutil
shutil.copy(best_ckpt, picked)
print(f"[pick] copied -> {picked}")

if a.score_json:
    os.makedirs(os.path.dirname(a.score_json) or ".", exist_ok=True)
    json.dump({"best_epoch": best_ep, "best_ckpt": best_ckpt,
               "scores": {f"ep{e}": final[e] for e in eps},
               "raw": {str(e): raw[e] for e in eps}},
              open(a.score_json, "w"), indent=2)
    print(f"[pick] scores -> {a.score_json}")
