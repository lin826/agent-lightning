#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Strip optimizer shards from non-latest VERL checkpoints under Search-R1 roots.

Usage:
    python strip_stale_checkpoint_optim.py checkpoints/searchr1_qwen7b
    python strip_stale_checkpoint_optim.py checkpoints/searchr1_qwen7b --keep-step 30
    python strip_stale_checkpoint_optim.py checkpoints/searchr1_* --dry-run

Training resume from the latest ``global_step_N`` still loads actor weights plus
optimizer state. Older checkpoints keep model/extra shards for eval and export.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentlightning.verl.checkpoint_utils import (
    OPTIM_SHARD_GLOB,
    resolve_latest_global_step,
    strip_stale_optimizer_shards,
)

_RECIPE_DIR = Path(__file__).resolve().parent.parent


def _format_bytes(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.2f} {unit}" if unit != "B" else f"{num_bytes} B"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def _count_optim_shards(checkpoint_root: Path, *, keep_step: int) -> tuple[int, int]:
    stale_bytes = 0
    stale_files = 0
    for child in sorted(checkpoint_root.iterdir()):
        if not child.is_dir() or not child.name.startswith("global_step_"):
            continue
        step_text = child.name.removeprefix("global_step_")
        if not step_text.isdigit() or int(step_text) == keep_step:
            continue
        actor_dir = child / "actor"
        if not actor_dir.is_dir():
            continue
        for optim_path in actor_dir.glob(OPTIM_SHARD_GLOB):
            stale_files += 1
            stale_bytes += optim_path.stat().st_size
    return stale_files, stale_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_roots",
        nargs="+",
        help="Checkpoint root directory or glob (e.g. checkpoints/searchr1_qwen7b)",
    )
    parser.add_argument(
        "--keep-step",
        type=int,
        default=None,
        help="Global step to keep optimizer shards on (default: latest tracker step)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report stale optimizer size without deleting files",
    )
    args = parser.parse_args()

    expanded_roots: list[Path] = []
    for pattern in args.checkpoint_roots:
        path = Path(pattern)
        if path.is_absolute():
            candidates = [path]
        elif any(ch in pattern for ch in "*?["):
            candidates = sorted(_RECIPE_DIR.glob(pattern))
        else:
            candidates = [_RECIPE_DIR / path]
        expanded_roots.extend(candidates)

    if not expanded_roots:
        raise SystemExit("No checkpoint roots matched.")

    total_removed = 0
    for root in expanded_roots:
        if not root.is_dir():
            print(f"Skipping missing directory: {root}")
            continue
        keep_step = args.keep_step if args.keep_step is not None else resolve_latest_global_step(root)
        if keep_step is None:
            print(f"[{root.name}] No global_step_* checkpoints found")
            continue

        stale_files, stale_bytes = _count_optim_shards(root, keep_step=keep_step)
        print(f"[{root.name}] keep global_step_{keep_step}; stale optim: {stale_files} files, {_format_bytes(stale_bytes)}")
        if args.dry_run or stale_bytes == 0:
            continue

        removed = strip_stale_optimizer_shards(root, keep_step=keep_step)
        variant_bytes = sum(removed.values())
        total_removed += variant_bytes
        print(f"[{root.name}] removed {_format_bytes(variant_bytes)} from {len(removed)} checkpoint(s)")

    if not args.dry_run:
        print(f"Total removed: {_format_bytes(total_removed)}")


if __name__ == "__main__":
    main()
