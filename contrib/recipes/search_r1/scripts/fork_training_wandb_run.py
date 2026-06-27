"""Fork a polluted GRPO training WandB run and backfill metrics from job logs.

W&B does not support deleting history rows or logging to steps below the run's
current max step. Eval jobs that logged into training runs can push the internal
step counter ahead of training (e.g. to 14) so resumed training metrics at steps
11–13 are rejected. The fix is:

1. Copy history from the source run through ``--clip-step`` (default 10).
2. Append ``val/reward`` and ``training/reward`` parsed from the training log
   for steps ``clip-step + 1 .. max-backfill-step``.
3. Write the new run id to ``checkpoints/<variant>/wandb_run_id.txt``.

**Fork/backfill must match the last checkpointed step**, not the last step
completed in training logs. If a job ran steps 12–13 but only saved
``global_step_11``, backfill only through step 11 so resumed training logs
step 12+ live without duplicating or skipping ahead. When ``--max-backfill-step``
is omitted and ``--checkpoint-dir`` is set, the script reads
``latest_checkpointed_iteration.txt`` from that directory.

**Do not hot-swap ``WANDB_RUN_ID`` on a running job.** VERL calls ``wandb.init``
once at startup; env changes mid-run have no effect. Update the id file and let
the next preemption/resume pick up the clean run.

Usage (from ``contrib/recipes/search_r1``, with ``.env`` containing ``WANDB_API_KEY``)::

    python scripts/fork_training_wandb_run.py --preset rewrite --dry-run
    python scripts/fork_training_wandb_run.py --preset rewrite_em
    python scripts/fork_training_wandb_run.py \\
        --source-run mo2pvowi --clip-step 10 \\
        --training-log outputs/train_qwen25_3b_rewrite.1806007.out \\
        --checkpoint-dir checkpoints/searchr1_qwen3_8b_rewrite \\
        --name searchr1_qwen3_8b_rewrite_clean
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import wandb
from wandb.apis.public import Api

logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parent
_RECIPE_DIR = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from wandb_run import save_wandb_run_id  # noqa: E402

DEFAULT_ENTITY = "ibm-bv"
DEFAULT_PROJECT = "AgentLightning"

PRESETS: dict[str, dict[str, Any]] = {
    "rewrite": {
        "source_run": "mo2pvowi",
        "clip_step": 10,
        "training_log": "outputs/train_qwen25_3b_rewrite.1806007.out",
        "checkpoint_dir": "checkpoints/searchr1_qwen3_8b_rewrite",
        "name": "searchr1_qwen3_8b_rewrite_clean",
        "notes": (
            "Fork of mo2pvowi: history through step 10, backfill 11+ from training logs "
            "(eval pollution at steps 11–13 removed)."
        ),
    },
    "rewrite_em": {
        "source_run": "l1ybdy1b",
        "clip_step": 10,
        "training_log": "outputs/train_qwen25_3b_rewrite_em.1805890.out",
        "checkpoint_dir": "checkpoints/searchr1_qwen3_8b_rewrite_em",
        "name": "searchr1_qwen3_8b_rewrite_em_clean",
        "notes": (
            "Fork of l1ybdy1b: history through step 10, backfill 11+ from training logs "
            "(eval pollution at steps 11–13 removed)."
        ),
    },
}

_STEP_LINE = re.compile(r"step:(\d+) - ([^\n]+)")
_METRIC = re.compile(r"(val/reward|training/reward|training/em):([0-9.eE+-]+)")


def _load_dotenv_if_present() -> None:
    dotenv = _RECIPE_DIR / ".env"
    if not dotenv.is_file():
        return
    for line in dotenv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _scan_history_rows(api: Api, entity: str, project: str, run_id: str) -> list[dict[str, Any]]:
    run = api.run(f"{entity}/{project}/{run_id}")
    rows: list[dict[str, Any]] = []
    for row in run.scan_history(page_size=1000):
        step = row.get("_step")
        if step is None:
            continue
        rows.append(dict(row))
    rows.sort(key=lambda r: int(r["_step"]))
    return rows


def _payload_from_row(row: dict[str, Any]) -> dict[str, Any]:
    skip = {"_timestamp", "_runtime", "_step", "run_id"}
    return {k: v for k, v in row.items() if not k.startswith("_") and k not in skip and v is not None}


def read_latest_checkpoint_step(checkpoint_dir: Path) -> int | None:
    """Return ``latest_checkpointed_iteration.txt`` value if present."""
    marker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        return None
    text = marker.read_text().strip()
    if not text:
        return None
    return int(text)


def parse_training_log_metrics(log_path: Path) -> dict[int, dict[str, float]]:
    """Parse ``step:N - ... val/reward:... training/reward:...`` lines from a VERL log."""
    if not log_path.is_file():
        raise FileNotFoundError(f"Training log not found: {log_path}")

    metrics: dict[int, dict[str, float]] = {}
    text = log_path.read_text(errors="replace")
    for match in _STEP_LINE.finditer(text):
        step = int(match.group(1))
        line = match.group(2)
        row: dict[str, float] = {}
        for key, value in _METRIC.findall(line):
            row[key] = float(value)
        if row:
            metrics[step] = row
    return metrics


def fork_training_run(
    *,
    entity: str,
    project: str,
    source_run: str,
    clip_step: int,
    training_log: Path,
    checkpoint_dir: Path | None,
    name: str,
    notes: str,
    tags: list[str],
    max_backfill_step: int | None,
    dry_run: bool,
) -> str | None:
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY is not set (source contrib/recipes/search_r1/.env)")

    api = Api(api_key=api_key)
    source_path = f"{entity}/{project}/{source_run}"
    source = api.run(source_path)
    logger.info(
        "Source %s state=%s lastHistoryStep=%s",
        source_path,
        source.state,
        source.lastHistoryStep,
    )

    all_rows = _scan_history_rows(api, entity, project, source_run)
    if not all_rows:
        raise RuntimeError(f"No history rows found for {source_path}")

    clipped = [row for row in all_rows if int(row["_step"]) <= clip_step]
    log_metrics = parse_training_log_metrics(training_log)
    backfill_steps = sorted(step for step in log_metrics if step > clip_step)
    if max_backfill_step is not None:
        backfill_steps = [s for s in backfill_steps if s <= max_backfill_step]

    logger.info(
        "Clipping source history to step %s (%s rows); backfill steps from log: %s",
        clip_step,
        len(clipped),
        backfill_steps,
    )

    merged: dict[int, dict[str, Any]] = {}
    for row in clipped:
        merged[int(row["_step"])] = _payload_from_row(row)

    for step in backfill_steps:
        payload = {k: v for k, v in log_metrics[step].items()}
        merged[step] = payload
        logger.info(
            "  backfill step %s: val/reward=%s training/reward=%s",
            step,
            payload.get("val/reward"),
            payload.get("training/reward"),
        )

    for step in sorted(merged):
        if step <= clip_step:
            continue
        row = merged[step]
        logger.info(
            "  final step %s: val/reward=%s training/reward=%s",
            step,
            row.get("val/reward"),
            row.get("training/reward"),
        )

    if dry_run:
        logger.info("Dry run — no new run created")
        return None

    source_config = dict(source.config)
    run = wandb.init(
        entity=entity,
        project=project,
        name=name,
        notes=notes,
        tags=tags,
        config={
            **source_config,
            "forked_from": source_run,
            "clip_step": clip_step,
            "backfill_log": str(training_log),
        },
        settings=wandb.Settings(_disable_stats=True, silent=False),
    )
    assert run is not None

    logged = 0
    for step in sorted(merged):
        payload = merged[step]
        if not payload:
            continue
        run.log(payload, step=step)
        logged += 1

    run.finish()
    new_id = run.id
    logger.info("Created run %s (%s steps logged)", new_id, logged)
    logger.info("View: https://wandb.ai/%s/%s/runs/%s", entity, project, new_id)

    if checkpoint_dir is not None:
        path = save_wandb_run_id(checkpoint_dir.resolve(), new_id)
        logger.info("Updated %s (old run %s was left unchanged on W&B)", path, source_run)

    return new_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--source-run", help="Polluted source W&B run id")
    parser.add_argument("--clip-step", type=int, default=10, help="Keep source history with _step <= this value")
    parser.add_argument("--training-log", type=Path, help="VERL training .out log for backfill metrics")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Write new run id to <dir>/wandb_run_id.txt (e.g. checkpoints/searchr1_qwen3_8b_rewrite)",
    )
    parser.add_argument("--name", help="Display name for the forked run")
    parser.add_argument("--notes", default="")
    parser.add_argument("--tag", action="append", default=[], dest="tags")
    parser.add_argument(
        "--max-backfill-step",
        type=int,
        default=None,
        help="Only backfill log steps up to this value (default: all steps > clip-step in log)",
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Built-in rewrite / rewrite_em config")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _load_dotenv_if_present()
    args = build_parser().parse_args(argv)

    source_run = args.source_run
    clip_step = args.clip_step
    training_log = args.training_log
    checkpoint_dir = args.checkpoint_dir
    name = args.name
    notes = args.notes

    if args.preset:
        preset = PRESETS[args.preset]
        source_run = source_run or preset["source_run"]
        clip_step = preset.get("clip_step", clip_step)
        training_log = training_log or (_RECIPE_DIR / preset["training_log"])
        checkpoint_dir = checkpoint_dir or (_RECIPE_DIR / preset["checkpoint_dir"])
        name = name or preset["name"]
        notes = notes or preset["notes"]

    if not source_run or training_log is None or not name:
        logger.error("Provide --source-run, --training-log, and --name (or use --preset)")
        return 2

    if not training_log.is_absolute():
        training_log = (_RECIPE_DIR / training_log).resolve()
    if checkpoint_dir is not None and not checkpoint_dir.is_absolute():
        checkpoint_dir = (_RECIPE_DIR / checkpoint_dir).resolve()

    max_backfill_step = args.max_backfill_step
    if max_backfill_step is None and checkpoint_dir is not None:
        ckpt_step = read_latest_checkpoint_step(checkpoint_dir)
        if ckpt_step is not None:
            max_backfill_step = ckpt_step
            logger.info(
                "Using latest_checkpointed_iteration.txt=%s as --max-backfill-step",
                ckpt_step,
            )

    tags = list(args.tags)
    tags.extend(["history_forked", f"from_{source_run}"])

    try:
        fork_training_run(
            entity=args.entity,
            project=args.project,
            source_run=source_run,
            clip_step=clip_step,
            training_log=training_log,
            checkpoint_dir=checkpoint_dir,
            name=name,
            notes=notes,
            tags=tags,
            max_backfill_step=max_backfill_step,
            dry_run=args.dry_run,
        )
    except Exception:
        logger.exception("Failed to fork training run")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
