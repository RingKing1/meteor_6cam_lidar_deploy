#!/usr/bin/env python3
import os, subprocess, time, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://huggingface.co/datasets/AutowareFoundation/meteor-demo-scenes/resolve/main"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = [l.strip() for l in open("/tmp/dl_list.txt") if l.strip()]

def dl(f):
    dst = os.path.join(ROOT, "data", f)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = f"{BASE}/{f}"
    for _ in range(50):
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return f, True, os.path.getsize(dst)
        r = subprocess.run(
            ["curl", "-sS", "-L", "-C", "-", "--connect-timeout", "15",
             "-m", "300", "-o", dst, url],
            capture_output=True)
        if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
            return f, True, os.path.getsize(dst)
        time.sleep(1.5)
    return f, False, 0

done = fail = 0
t0 = time.time()
with ThreadPoolExecutor(16) as ex:
    futs = {ex.submit(dl, f): f for f in files}
    for fut in as_completed(futs):
        f, ok, sz = fut.result()
        if ok: done += 1
        else:
            fail += 1
            print("FAIL", f, flush=True)
        n = done + fail
        if n % 200 == 0:
            print(f"progress {n}/{len(files)} ok={done} fail={fail} "
                  f"{n/(time.time()-t0):.1f} f/s", flush=True)
print(f"DONE ok={done} fail={fail} elapsed={time.time()-t0:.0f}s", flush=True)
sys.exit(1 if fail else 0)
