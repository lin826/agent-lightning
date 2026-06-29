#!/usr/bin/env python3
"""
Maintain steady low GPU utilization on each visible device.

BM25 retrieval servers have idle gaps between search requests; cluster schedulers
reap jobs whose GPUs stay at 0% utilization. This script runs a small matmul on a
fixed duty cycle per GPU so utilization stays near a target (default 5%) instead of
spiking to 100% in bursts.

Tunable via env:
  KEEPALIVE_TARGET_UTIL   target utilization fraction or percent (default 0.05 = 5%)
  KEEPALIVE_MATMUL_N      NxN bf16 matmul size                    (default 2048)
  KEEPALIVE_STARTUP_DELAY seconds to hold off before starting     (default 0)
  KEEPALIVE_SKIP_DEVICES  comma/space-separated GPU ids to skip   (default none)
"""

from __future__ import annotations

import os
import threading
import time
from typing import List, Set

import torch

def parse_target_util(raw: str) -> float:
    """Parse KEEPALIVE_TARGET_UTIL as a fraction (0.05) or legacy percent (5, 35)."""
    val = float(raw.strip().rstrip("%"))
    if val > 1.0:
        return val / 100.0
    return val


def duty_cycle_sleep_seconds(work_seconds: float, target_util: float) -> float:
    """Sleep duration so work_seconds / (work + sleep) ≈ target_util."""
    if target_util <= 0.0:
        return 0.0
    if target_util >= 1.0:
        return 0.0
    return work_seconds * (1.0 / target_util - 1.0)


def parse_skip_devices(raw: str) -> Set[int]:
    return {int(x) for x in raw.replace(",", " ").split() if x.strip().isdigit()}


TARGET_UTIL = parse_target_util(os.environ.get("KEEPALIVE_TARGET_UTIL", "0.05"))
MATMUL_N = int(os.environ.get("KEEPALIVE_MATMUL_N", "2048"))
STARTUP_DELAY = int(os.environ.get("KEEPALIVE_STARTUP_DELAY", "0"))
SKIP_DEVICES = parse_skip_devices(os.environ.get("KEEPALIVE_SKIP_DEVICES", ""))


def gpu_worker(device_id: int, stop_event: threading.Event, target_util: float) -> None:
    """Run steady low-intensity matmuls on one GPU."""
    device = torch.device(f"cuda:{device_id}")
    a = torch.randn(MATMUL_N, MATMUL_N, device=device, dtype=torch.bfloat16)
    b = torch.randn(MATMUL_N, MATMUL_N, device=device, dtype=torch.bfloat16)
    grace = f"deferred {STARTUP_DELAY}s" if STARTUP_DELAY > 0 else "starting immediately"
    pct = target_util * 100.0
    print(
        f"GPU {device_id}: keepalive started ({MATMUL_N}x{MATMUL_N} bf16, "
        f"steady ~{pct:.1f}% util, {grace})",
        flush=True,
    )

    if STARTUP_DELAY > 0:
        grace_end = time.monotonic() + STARTUP_DELAY
        while not stop_event.is_set() and time.monotonic() < grace_end:
            time.sleep(min(5.0, max(0.5, grace_end - time.monotonic())))
        if not stop_event.is_set():
            print(f"GPU {device_id}: keepalive active (grace elapsed)", flush=True)

    while not stop_event.is_set():
        t0 = time.monotonic()
        c = torch.matmul(a, b)
        a = torch.nn.functional.relu(c * 1e-4 + 0.01)
        torch.cuda.synchronize(device)
        work_elapsed = time.monotonic() - t0

        sleep_for = duty_cycle_sleep_seconds(work_elapsed, target_util)
        if sleep_for > 0.0:
            stop_event.wait(timeout=sleep_for)

    print(f"GPU {device_id}: keepalive stopped", flush=True)


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available!", flush=True)
        return

    num_gpus = torch.cuda.device_count()
    covered = [g for g in range(num_gpus) if g not in SKIP_DEVICES]
    skipped = sorted(SKIP_DEVICES & set(range(num_gpus)))
    mem_mb = len(covered) * (MATMUL_N * MATMUL_N * 2 * 3) // (1024**2)
    pct = TARGET_UTIL * 100.0
    print(
        f"keepalive: {len(covered)}/{num_gpus} GPU(s), {MATMUL_N}x{MATMUL_N} bf16 "
        f"(~{mem_mb}MB total), steady target ~{pct:.1f}%, skip={skipped or 'none'}",
        flush=True,
    )

    stop_event = threading.Event()
    threads: List[threading.Thread] = []
    try:
        for gpu_id in covered:
            t = threading.Thread(
                target=gpu_worker,
                args=(gpu_id, stop_event, TARGET_UTIL),
                daemon=True,
            )
            t.start()
            threads.append(t)
        print(
            f"keepalive: {len(covered)} GPU(s) covered (skipped {skipped or 'none'}). "
            f"Ctrl+C to stop.",
            flush=True,
        )
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("keepalive: stopping...", flush=True)
        stop_event.set()
        for t in threads:
            t.join(timeout=3)
        print("keepalive: stopped.", flush=True)


if __name__ == "__main__":
    main()
