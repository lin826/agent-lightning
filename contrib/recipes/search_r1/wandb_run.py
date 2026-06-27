"""Helpers to persist and resume WandB runs across training and full-test eval."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

WANDB_RUN_ID_FILENAME = "wandb_run_id.txt"


def checkpoint_root_from_actor(actor_path: Path) -> Path:
    """Return the experiment checkpoint root from an actor path (``.../global_step_N/actor``)."""
    return actor_path.resolve().parent.parent


def save_wandb_run_id(directory: Path, run_id: str) -> Path:
    """Write ``wandb_run_id.txt`` under ``directory`` and return the file path."""
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / WANDB_RUN_ID_FILENAME
    normalized = run_id.strip()
    if path.exists() and path.read_text().strip() == normalized:
        return path
    path.write_text(normalized + "\n")
    return path


def load_wandb_run_id(directory: Path) -> Optional[str]:
    """Read ``wandb_run_id.txt`` from ``directory`` if present."""
    path = (directory / WANDB_RUN_ID_FILENAME).resolve()
    if not path.is_file():
        return None
    run_id = path.read_text().strip()
    return run_id or None


def find_wandb_run_id_from_local(wandb_dir: Path, experiment_name: str) -> Optional[str]:
    """Find the newest local WandB run id whose debug log matches ``experiment_name``."""
    if not wandb_dir.is_dir():
        return None

    candidates: list[tuple[float, str]] = []
    for run_dir in wandb_dir.glob("run-*"):
        run_id = run_dir.name.rsplit("-", 1)[-1]
        debug_log = run_dir / "logs" / "debug.log"
        if not debug_log.is_file():
            continue
        try:
            content = debug_log.read_text(errors="replace")
        except OSError:
            continue
        if f"Syncing run {experiment_name}" in content:
            candidates.append((run_dir.stat().st_mtime, run_id))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def resolve_wandb_run_id(
    *,
    checkpoint_dir: Path | None = None,
    run_dir: Path | None = None,
    experiment_name: str | None = None,
    wandb_dir: Path | None = None,
) -> Optional[str]:
    """Resolve a WandB run id from env, saved file, or local run directories."""
    env_run_id = os.environ.get("WANDB_RUN_ID", "").strip()
    if env_run_id:
        return env_run_id

    for directory in (checkpoint_dir, run_dir):
        if directory is not None:
            run_id = load_wandb_run_id(directory)
            if run_id:
                return run_id

    if experiment_name and wandb_dir is not None:
        return find_wandb_run_id_from_local(wandb_dir, experiment_name)

    return None


def setup_wandb_resume(run_id: str, *, resume: str = "allow") -> None:
    """Set env vars so VERL / wandb.init resume the original run."""
    os.environ["WANDB_RUN_ID"] = run_id.strip()
    os.environ["WANDB_RESUME"] = resume
