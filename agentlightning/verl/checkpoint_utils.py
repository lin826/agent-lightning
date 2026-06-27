# Copyright (c) Microsoft. All rights reserved.

"""Helpers for VERL FSDP checkpoint optimizer retention."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

OPTIM_SHARD_GLOB = "optim_world_size_*_rank_*.pt"


def parse_global_step_dir(path: Path) -> int | None:
    """Return the training step encoded in a ``global_step_N`` directory name."""
    if not path.is_dir() or not path.name.startswith("global_step_"):
        return None
    suffix = path.name.removeprefix("global_step_")
    if not suffix.isdigit():
        return None
    return int(suffix)


def strip_optimizer_shards(actor_dir: Path) -> int:
    """Delete FSDP optimizer shard files under an actor checkpoint directory."""
    if not actor_dir.is_dir():
        return 0
    removed_bytes = 0
    for optim_path in actor_dir.glob(OPTIM_SHARD_GLOB):
        removed_bytes += optim_path.stat().st_size
        optim_path.unlink()
        logger.info("Removed optimizer shard %s", optim_path)
    return removed_bytes


def resolve_latest_global_step(checkpoint_root: Path) -> int | None:
    """Resolve the latest saved global step from the VERL tracker file or directories."""
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        text = tracker.read_text().strip()
        if text.isdigit():
            return int(text)
    steps = [
        step
        for child in checkpoint_root.iterdir()
        if (step := parse_global_step_dir(child)) is not None
    ]
    return max(steps) if steps else None


def strip_stale_optimizer_shards(
    checkpoint_root: str | Path,
    *,
    keep_step: int | None = None,
) -> dict[str, int]:
    """Remove optimizer shards from all ``global_step_*`` dirs except ``keep_step``.

    Only the latest checkpoint needs optimizer state for training resume. Older
    checkpoints remain usable for eval/export via actor model weights.

    Args:
        checkpoint_root: Experiment checkpoint root (contains ``global_step_N/``).
        keep_step: Global step whose optimizer shards are retained. Defaults to
            the value in ``latest_checkpointed_iteration.txt``, or the highest
            ``global_step_N`` directory when the tracker is missing.

    Returns:
        Mapping from ``global_step_N`` directory names to bytes removed.
    """
    root = Path(checkpoint_root)
    if not root.is_dir():
        return {}

    if keep_step is None:
        keep_step = resolve_latest_global_step(root)
    if keep_step is None:
        return {}

    removed: dict[str, int] = {}
    for child in sorted(root.iterdir()):
        step = parse_global_step_dir(child)
        if step is None or step == keep_step:
            continue
        actor_dir = child / "actor"
        nbytes = strip_optimizer_shards(actor_dir)
        if nbytes:
            removed[child.name] = nbytes
            logger.info(
                "Stripped %d bytes of optimizer shards from %s (keeping global_step_%d)",
                nbytes,
                child.name,
                keep_step,
            )
    return removed
