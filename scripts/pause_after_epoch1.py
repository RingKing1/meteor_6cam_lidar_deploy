#!/usr/bin/env python3
"""Monitors training log and automatically pauses training when Epoch 1 completes."""
import os
import subprocess
import time

LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../train_meteor.log"))
CKPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../checkpoints/meteor_custom_v52/last.pt"))


def main():
    print(f"[*] Watcher active. Monitoring for Epoch 1 completion in {LOG_PATH} ...", flush=True)
    while True:
        time.sleep(10)
        if os.path.exists(LOG_PATH):
            try:
                with open(LOG_PATH, "r", errors="ignore") as f:
                    lines = f.readlines()
                # Check for [val ep1] in log
                val_ep1_found = any("[val ep1]" in l for l in lines)
                if val_ep1_found:
                    # Allow 5s for checkpoint file flush
                    time.sleep(5)
                    print("[+] Detected Epoch 1 completion and validation in log!", flush=True)
                    print("[*] Terminating training process to pause cleanly ...", flush=True)
                    subprocess.run(["docker", "exec", "meteor_run", "pkill", "-f", "run_finetune.py"])
                    subprocess.run(["docker", "exec", "meteor_run", "pkill", "-f", "bevlane/train.py"])
                    print("[+] Training successfully paused at the end of Epoch 1!", flush=True)
                    break
            except Exception as e:
                print(f"[!] Warning reading log: {e}", flush=True)


if __name__ == "__main__":
    main()
