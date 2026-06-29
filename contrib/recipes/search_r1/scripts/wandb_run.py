"""Helpers to persist and resume WandB runs across training and full-test eval."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

WANDB_RUN_ID_FILENAME = "wandb_run_id.txt"
WANDB_EVAL_RUN_ID_FILENAME = "wandb_eval_run_id.txt"

# WandB display names for dedicated full-test eval runs (``eval_search_r1_agent.py``).
# Training runs use ``searchr1_*``; eval runs use the ``eval_*`` prefix below.
EVAL_WANDB_RUN_NAMES: dict[str, str] = {
    "qwen7b": "eval_baseline",
    "qwen3_8b": "eval_baseline_a",
    "qwen3_8b_rewrite": "eval_rewrite",
    "qwen3_8b_rewrite_em": "eval_rewrite_em",
    "qwen3_8b_shaped": "eval_shaped",
}

# WandB display names for GEPA full-test eval runs (``eval_gepa_prompt.py``).
GEPA_EVAL_WANDB_RUN_NAMES: dict[str, str] = {
    "baseline": "eval_gepa",
    "rewrite": "eval_gepa_rewrite",
}

def resolve_eval_wandb_run_name(config_key: str) -> str:
    """Return the WandB run name for a GRPO full-test eval variant.

    Args:
        config_key: CLI/config key passed to ``eval_search_r1_agent.py`` (e.g.
            ``qwen3_8b_rewrite_em``).

    Raises:
        KeyError: If ``config_key`` is not a known eval variant.
    """
    try:
        return EVAL_WANDB_RUN_NAMES[config_key]
    except KeyError as exc:
        known = ", ".join(sorted(EVAL_WANDB_RUN_NAMES))
        raise KeyError(f"Unknown eval config key {config_key!r}; expected one of: {known}") from exc


def resolve_gepa_eval_wandb_run_name(variant_name: str) -> str:
    """Return the WandB run name for a GEPA full-test eval variant.

    Override with ``WANDB_EVAL_RUN_NAME`` (e.g. ``eval_gepa``).

    Args:
        variant_name: GEPA variant key (``baseline`` or ``rewrite``).

    Raises:
        KeyError: If ``variant_name`` is not a known GEPA eval variant.
    """
    override = os.environ.get("WANDB_EVAL_RUN_NAME", "").strip()
    if override:
        return override
    try:
        return GEPA_EVAL_WANDB_RUN_NAMES[variant_name]
    except KeyError as exc:
        known = ", ".join(sorted(GEPA_EVAL_WANDB_RUN_NAMES))
        raise KeyError(f"Unknown GEPA eval variant {variant_name!r}; expected one of: {known}") from exc


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


def save_wandb_eval_run_id(directory: Path, run_id: str) -> Path:
    """Write ``wandb_eval_run_id.txt`` under ``directory`` and return the file path."""
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / WANDB_EVAL_RUN_ID_FILENAME
    normalized = run_id.strip()
    if path.exists() and path.read_text().strip() == normalized:
        return path
    path.write_text(normalized + "\n")
    return path


def load_wandb_eval_run_id(directory: Path) -> Optional[str]:
    """Read ``wandb_eval_run_id.txt`` from ``directory`` if present."""
    path = (directory / WANDB_EVAL_RUN_ID_FILENAME).resolve()
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


def verify_wandb_run_exists(
    run_id: str,
    *,
    project: str,
    entity: str | None = None,
) -> bool:
    """Return False when the WandB run is missing or deleted."""
    normalized = run_id.strip()
    if not normalized:
        return False
    try:
        from wandb.apis.public import Api

        api = Api()
        resolved_entity = (entity or os.environ.get("WANDB_ENTITY", "")).strip()
        if not resolved_entity:
            logger.warning(
                "WANDB_ENTITY unset; cannot verify run %s — allowing resume attempt",
                normalized,
            )
            return True
        api.run(f"{resolved_entity}/{project}/{normalized}")
        return True
    except Exception as exc:
        message = str(exc).lower()
        if "404" in message or "not found" in message or "could not find run" in message:
            return False
        logger.warning("Could not verify WandB run %s: %s", normalized, exc)
        return True


def clear_wandb_run_id_file(directory: Path, *, kind: str = "train") -> None:
    """Remove a stale WandB run id file under ``directory``."""
    filename = WANDB_EVAL_RUN_ID_FILENAME if kind == "eval" else WANDB_RUN_ID_FILENAME
    path = (directory / filename).resolve()
    if path.is_file():
        path.unlink()
        logger.info("Removed stale WandB id file %s", path)


def validate_wandb_run_id(
    run_id: str | None,
    *,
    project: str,
    directory: Path | None = None,
    kind: str = "train",
) -> str | None:
    """Return ``run_id`` when the run exists; otherwise clear saved id files and return None."""
    if not run_id:
        return None
    if verify_wandb_run_exists(run_id, project=project):
        return run_id.strip()
    logger.warning("WandB run %s is missing or deleted; starting a fresh run", run_id)
    if directory is not None:
        clear_wandb_run_id_file(directory, kind=kind)
    os.environ.pop("WANDB_RUN_ID", None)
    os.environ.pop("WANDB_RESUME", None)
    return None


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


def resolve_wandb_eval_run_id(
    *,
    checkpoint_dir: Path | None = None,
    run_dir: Path | None = None,
) -> Optional[str]:
    """Resolve a dedicated full-test eval WandB run id (never the training run file)."""
    env_run_id = os.environ.get("WANDB_RUN_ID", "").strip()
    if env_run_id:
        return env_run_id

    for directory in (checkpoint_dir, run_dir):
        if directory is not None:
            run_id = load_wandb_eval_run_id(directory)
            if run_id:
                return run_id

    return None


def setup_wandb_resume(run_id: str, *, resume: str = "allow", run_name: str | None = None) -> None:
    """Set env vars so VERL / wandb.init resume the original run.

    When ``run_name`` is set it is exported as ``WANDB_NAME`` for new runs. Resuming an
    existing run id keeps the name stored in WandB; delete ``wandb_eval_run_id.txt`` to
    start a fresh eval run with the desired name.
    """
    os.environ["WANDB_RUN_ID"] = run_id.strip()
    os.environ["WANDB_RESUME"] = resume
    if run_name:
        os.environ["WANDB_NAME"] = run_name
    else:
        os.environ.pop("WANDB_NAME", None)


def resolve_gepa_wandb_run_id(
    *,
    project: str,
    experiment_name: str,
    run_dir: Path,
    wandb_dir: Path | None = None,
) -> str | None:
    """Resolve a GEPA training run id from env, saved file, or local ``wandb/`` dirs."""
    run_id = resolve_wandb_run_id(
        run_dir=run_dir,
        experiment_name=experiment_name,
        wandb_dir=wandb_dir,
    )
    validated = validate_wandb_run_id(run_id, project=project, directory=run_dir, kind="train")
    if validated:
        return validated
    if wandb_dir is None:
        return None
    local_id = find_wandb_run_id_from_local(wandb_dir, experiment_name)
    return validate_wandb_run_id(local_id, project=project, directory=None, kind="train")


def build_gepa_wandb_init_kwargs(
    *,
    project: str,
    name: str,
    config: dict[str, Any],
    run_dir: Path,
    wandb_dir: Path | None = None,
) -> dict[str, Any]:
    """Build ``wandb.init`` kwargs for a GEPA run, resuming only when the saved run still exists."""
    kwargs: dict[str, Any] = {"project": project, "name": name, "config": config}
    if os.environ.get("WANDB_ENTITY"):
        kwargs["entity"] = os.environ["WANDB_ENTITY"]
    run_id = resolve_gepa_wandb_run_id(
        project=project,
        experiment_name=name,
        run_dir=run_dir,
        wandb_dir=wandb_dir,
    )
    if run_id:
        kwargs["id"] = run_id
        kwargs["resume"] = "allow"
    return kwargs


def resolve_gepa_iteration_for_metric_calls(
    run_dir: Path,
    metric_calls: int,
    *,
    project: str = "AgentLightning",
    wandb_dir: Path | None = None,
) -> int:
    """Map a rollout budget (``metric_calls``) to the GEPA iteration for WandB ``_step``.

    Full-test eval logs ``test/reward`` on the training run at the iteration that
    produced the prompt, not at ``metric_calls``, so ``_step`` stays aligned with
    per-iteration training curves.
    """
    if metric_calls <= 0:
        return 0

    try:
        from gepa_full_eval import load_full_eval_state

        fe_state = load_full_eval_state(run_dir)
        iteration = fe_state.iteration_by_metric_calls.get(metric_calls)
        if iteration is not None:
            return int(iteration)
    except Exception as exc:
        logger.debug("Could not read iteration map from full_eval_state: %s", exc)

    run_id = load_wandb_run_id(run_dir)
    if run_id:
        try:
            from wandb.apis.public import Api

            entity = os.environ.get("WANDB_ENTITY", "ibm-bv")
            run = Api().run(f"{entity}/{project}/{run_id}")
            for row in run.scan_history(keys=["rollouts", "total_metric_calls", "iteration", "_step"]):
                rollouts = row.get("rollouts", row.get("total_metric_calls"))
                if rollouts is None or int(rollouts) != metric_calls:
                    continue
                iteration = row.get("iteration", row.get("_step"))
                if iteration is not None:
                    return int(iteration)
        except Exception as exc:
            logger.debug("Could not resolve iteration for metric_calls=%s from WandB API: %s", metric_calls, exc)

    if wandb_dir is not None and wandb_dir.is_dir():
        experiment_names = (
            "searchr1_qwen25_3b_gepa",
            "searchr1_qwen25_3b_gepa_rewrite",
        )
        for name in experiment_names:
            local_id = find_wandb_run_id_from_local(wandb_dir, name)
            if local_id != run_id:
                continue

    logger.warning(
        "Could not map metric_calls=%d to a GEPA iteration in %s; using metric_calls as WandB step",
        metric_calls,
        run_dir,
    )
    return metric_calls


def install_gepa_wandb_grpo_compat_patch(
    *,
    reflection_minibatch_size: int,
    run_dir: Path | None = None,
    eval_job_tag: str = "qwen25_3b_gepa",
    eval_addr_file: str = "bm25_server_addr_eval_gepa.txt",
    run_dir_rel: str = "outputs/gepa_qwen25_3b",
    use_rewrite: bool = False,
) -> None:
    """Mirror GEPA mean EM scores into GRPO-style ``val/reward`` and ``training/reward`` keys.

    GEPA adapter scores are per-example EM (0/1); ``val_program_average`` and
    ``subsample_score`` / ``new_subsample_score`` are sums or means over those scores.

    When ``run_dir`` is set, a new dev-subset historical best
    (``best_score_on_valset`` / ``best_valset_agg_score``) also submits full-test eval
    via :func:`gepa_full_eval.maybe_trigger_full_eval_from_metrics`.
    """
    from gepa.logging.experiment_tracker import ExperimentTracker

    if getattr(ExperimentTracker, "_agl_grpo_compat_patched", False):
        return

    original_log_metrics = ExperimentTracker.log_metrics
    minibatch_size = max(1, reflection_minibatch_size)
    best_val_trigger_keys = (
        "best_score_on_valset",
        "best_valset_agg_score",
        "base_program_full_valset_score",
    )

    def log_metrics(self: ExperimentTracker, metrics: dict[str, Any], step: int | None = None) -> None:
        original_log_metrics(self, metrics, step=step)
        if not self.use_wandb or step is None:
            return

        extra: dict[str, float | int] = {}
        # Per-step validation signal: mean EM of the program evaluated on valset this iteration
        # (new candidate after acceptance, or the base seed at iteration 1).
        if "val_program_average" in metrics:
            val_mean = float(metrics["val_program_average"])
            extra["val/reward"] = val_mean
            extra["val/em"] = val_mean
        elif "base_program_full_valset_score" in metrics:
            val_mean = float(metrics["base_program_full_valset_score"])
            extra["val/reward"] = val_mean
            extra["val/em"] = val_mean

        # Historical / monitoring metrics — keep separate from per-step val/reward.
        if "valset_pareto_front_agg" in metrics:
            extra["val/pareto_front_agg"] = float(metrics["valset_pareto_front_agg"])
        for key in ("best_valset_agg_score", "best_score_on_valset"):
            if key in metrics:
                extra["val/best_single_program_em"] = float(metrics[key])
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
            if "total_metric_calls" in metrics:
                extra["total_metric_calls"] = int(metrics["total_metric_calls"])
            original_log_metrics(self, extra, step=step)

        if run_dir is not None and any(key in metrics for key in best_val_trigger_keys):
            try:
                from gepa_full_eval import maybe_trigger_full_eval_from_metrics

                maybe_trigger_full_eval_from_metrics(
                    metrics,
                    run_dir,
                    step=step,
                    eval_job_tag=eval_job_tag,
                    addr_file=eval_addr_file,
                    run_dir_rel=run_dir_rel,
                    use_rewrite=use_rewrite,
                )
            except Exception as exc:
                logger.warning("GEPA full-test eval trigger failed at step %s: %s", step, exc)

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
