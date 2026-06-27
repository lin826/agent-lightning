"""Helpers to persist and resume WandB runs across training and full-test eval."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

WANDB_RUN_ID_FILENAME = "wandb_run_id.txt"


def checkpoint_root_from_actor(actor_path: Path) -> Path:
    """Return the experiment checkpoint root from an actor path (``.../global_step_N/actor``)."""
    return actor_path.resolve().parent.parent


def resolve_actor_checkpoint(checkpoint_path: Path | str) -> tuple[Path, Path]:
    """Resolve and validate a VERL actor checkpoint path.

    Accepts either ``.../global_step_N/actor`` or ``.../global_step_N``.

    Returns:
        ``(actor_dir, global_step_dir)`` where ``actor_dir`` holds FSDP shards and
        ``global_step_dir`` is the path VERL expects for ``resume_from_path``.
    """
    path = Path(checkpoint_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    else:
        path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    if path.name == "actor" and path.parent.name.startswith("global_step_"):
        actor_dir = path
        global_step_dir = path.parent
    elif path.name.startswith("global_step_") and (path / "actor").is_dir():
        global_step_dir = path
        actor_dir = path / "actor"
    else:
        raise ValueError(
            "Expected a VERL checkpoint directory "
            "(.../global_step_N/actor or .../global_step_N), "
            f"got: {path}"
        )

    hf_dir = actor_dir / "huggingface"
    if not hf_dir.is_dir():
        raise FileNotFoundError(
            f"Missing HuggingFace tokenizer export at {hf_dir}. "
            "Ensure the checkpoint was saved by VERL with tokenizer artifacts."
        )

    if not any(actor_dir.glob("model_world_size_*_rank_*.pt")):
        raise FileNotFoundError(f"No FSDP model shards found under {actor_dir}")

    return actor_dir, global_step_dir


def resolve_tokenizer_path(actor_dir: Path) -> Path:
    """Return the HuggingFace export directory for tokenizer loading."""
    hf_dir = (actor_dir / "huggingface").resolve()
    if not hf_dir.is_dir():
        raise FileNotFoundError(
            f"Missing HuggingFace tokenizer export at {hf_dir}. "
            "Ensure the checkpoint was saved by VERL with tokenizer artifacts."
        )
    return hf_dir


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


def build_gepa_wandb_init_kwargs(
    *,
    project: str,
    name: str,
    config: dict[str, Any],
    run_dir: Path,
    wandb_dir: Path | None = None,
) -> dict[str, Any]:
    """Build ``wandb.init`` kwargs for a GEPA run, resuming when ``wandb_run_id.txt`` exists."""
    kwargs: dict[str, Any] = {"project": project, "name": name, "config": config}
    if os.environ.get("WANDB_ENTITY"):
        kwargs["entity"] = os.environ["WANDB_ENTITY"]
    run_id = resolve_wandb_run_id(run_dir=run_dir, experiment_name=name, wandb_dir=wandb_dir)
    if run_id:
        kwargs["id"] = run_id
        kwargs["resume"] = "allow"
    return kwargs


def install_gepa_wandb_grpo_compat_patch(*, reflection_minibatch_size: int) -> None:
    """Mirror GEPA val/train scores into GRPO-style ``val/reward`` and ``training/reward`` keys."""
    from gepa.logging.experiment_tracker import ExperimentTracker

    if getattr(ExperimentTracker, "_agl_grpo_compat_patched", False):
        return

    original_log_metrics = ExperimentTracker.log_metrics
    minibatch_size = max(1, reflection_minibatch_size)

    def log_metrics(self: ExperimentTracker, metrics: dict[str, Any], step: int | None = None) -> None:
        original_log_metrics(self, metrics, step=step)
        if not self.use_wandb or step is None:
            return

        extra: dict[str, float] = {}
        for key in ("best_valset_agg_score", "best_score_on_valset", "base_program_full_valset_score"):
            if key in metrics:
                score = float(metrics[key])
                extra["val/reward"] = score
                extra["val/em"] = score
                break

        if "subsample_score" in metrics:
            train_mean = float(metrics["subsample_score"]) / minibatch_size
            extra["training/reward"] = train_mean
            extra["training/em"] = train_mean
        elif "new_subsample_score" in metrics:
            train_mean = float(metrics["new_subsample_score"]) / minibatch_size
            extra["training/reward"] = train_mean
            extra["training/em"] = train_mean

        if extra:
            original_log_metrics(self, extra, step=step)

    ExperimentTracker.log_metrics = log_metrics  # type: ignore[method-assign]
    ExperimentTracker._agl_grpo_compat_patched = True


def log_gepa_wandb_metrics(
    metrics: dict[str, float],
    *,
    step: int,
    project: str,
    experiment_name: str,
    run_dir: Path,
    config: dict[str, Any] | None = None,
    wandb_dir: Path | None = None,
    finish: bool = False,
) -> None:
    """Resume (if needed), log GRPO-comparable GEPA metrics, and optionally finish the run."""
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; skipping WandB logging")
        return

    if wandb.run is None:
        init_kwargs = build_gepa_wandb_init_kwargs(
            project=project,
            name=experiment_name,
            config=config or {},
            run_dir=run_dir,
            wandb_dir=wandb_dir,
        )
        if init_kwargs.get("id"):
            init_kwargs["resume"] = "must"
        wandb.init(**init_kwargs)

    assert wandb.run is not None
    wandb.log(metrics, step=step)
    save_wandb_run_id(run_dir, wandb.run.id)
    if finish:
        wandb.finish()
