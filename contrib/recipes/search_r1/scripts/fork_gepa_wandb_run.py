"""Fork a polluted GEPA WandB run and re-log only healthy iteration history.

W&B cannot delete history rows or log below the run's current max ``_step``.
When ``train_gepa.py`` or ``backfill_gepa_wandb.py`` logs final metrics at
``step=total_metric_calls`` (e.g. 2569) instead of the last iteration (63),
the run's step counter jumps ahead and resumed training metrics are rejected.

This script copies ``_step`` 0 through ``--max-iteration`` (default 63) from
the source run into a **new** WandB run, preserving ``rollouts`` /
``total_metric_calls`` alignment, and writes the new id to
``outputs/gepa_qwen25_3b/wandb_run_id.txt``.

Usage (from ``contrib/recipes/search_r1``, with ``.env`` containing ``WANDB_API_KEY``)::

    python scripts/fork_gepa_wandb_run.py --source-run sq5hdz51 --dry-run
    python scripts/fork_gepa_wandb_run.py --source-run sq5hdz51 --preset gepa_qwen25_3b
"""

from __future__ import annotations

import argparse
import logging
import os
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
    "gepa_qwen25_3b": {
        "source_run": "sq5hdz51",
        "max_iteration": 63,
        "run_dir": "outputs/gepa_qwen25_3b",
        "name": "searchr1_qwen25_3b_gepa_clean",
        "notes": (
            "Fork of sq5hdz51: iteration history 0-63 only; orphan final point at "
            "_step=2569 (total_metric_calls) removed."
        ),
    },
}


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
    payload = {k: v for k, v in row.items() if not k.startswith("_") and k not in skip and v is not None}
    # Drop legacy dual-axis keys from polluted source runs.
    return {k: v for k, v in payload.items() if not k.endswith("@rollouts") and k != "rollouts"}


def fork_gepa_run(
    *,
    entity: str,
    project: str,
    source_run: str,
    max_iteration: int,
    run_dir: Path,
    name: str,
    notes: str,
    tags: list[str],
    dry_run: bool,
) -> str | None:
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY is not set (source contrib/recipes/search_r1/.env)")

    api = Api(api_key=api_key)
    source_path = f"{entity}/{project}/{source_run}"
    source = api.run(source_path)
    all_rows = _scan_history_rows(api, entity, project, source_run)
    if not all_rows:
        raise RuntimeError(f"No history rows found for {source_path}")

    max_step = max(int(r["_step"]) for r in all_rows)
    clipped = [row for row in all_rows if int(row["_step"]) <= max_iteration]
    dropped = [int(row["_step"]) for row in all_rows if int(row["_step"]) > max_iteration]

    logger.info(
        "Source %s state=%s history_rows=%d max_step=%d clip<=%d kept=%d dropped_steps=%s",
        source_path,
        source.state,
        len(all_rows),
        max_step,
        max_iteration,
        len(clipped),
        sorted(set(dropped)),
    )

    merged: dict[int, dict[str, Any]] = {}
    for row in clipped:
        step = int(row["_step"])
        payload = _payload_from_row(row)
        merged[step] = payload
        logger.info(
            "  keep step %s: total_metric_calls=%s val/em=%s",
            step,
            payload.get("total_metric_calls"),
            payload.get("val/em"),
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
            "max_iteration_clipped": max_iteration,
            "dropped_orphan_steps": sorted(set(dropped)),
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

    last_payload = merged[max(merged)] if merged else {}
    summary_keys = (
        "val/em",
        "val/reward",
        "train/em",
        "training/em",
        "training/reward",
        "seed/val_em",
        "total_metric_calls",
        "iteration",
        "val_program_average",
        "best_valset_agg_score",
        "best_score_on_valset",
    )
    summary_update = {k: last_payload[k] for k in summary_keys if k in last_payload}
    if summary_update:
        run.summary.update(summary_update)

    run.finish()
    new_id = run.id
    logger.info("Created run %s (%s steps logged, max _step=%s)", new_id, logged, max(merged) if merged else None)
    logger.info("View: https://wandb.ai/%s/%s/runs/%s", entity, project, new_id)

    run_dir.mkdir(parents=True, exist_ok=True)
    path = save_wandb_run_id(run_dir.resolve(), new_id)
    logger.info("Updated %s (source run %s left unchanged on W&B)", path, source_run)
    return new_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--source-run", help="Polluted source W&B run id")
    parser.add_argument(
        "--max-iteration",
        type=int,
        default=63,
        help="Keep source history with _step <= this iteration (default: 63)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=_RECIPE_DIR / "outputs" / "gepa_qwen25_3b",
        help="GEPA output dir for wandb_run_id.txt",
    )
    parser.add_argument("--name", help="Display name for the forked run")
    parser.add_argument("--notes", default="")
    parser.add_argument("--tag", action="append", default=[], dest="tags")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Built-in gepa_qwen25_3b config")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _load_dotenv_if_present()
    args = build_parser().parse_args(argv)

    source_run = args.source_run
    max_iteration = args.max_iteration
    run_dir = args.run_dir
    name = args.name
    notes = args.notes

    if args.preset:
        preset = PRESETS[args.preset]
        source_run = source_run or preset["source_run"]
        max_iteration = preset.get("max_iteration", max_iteration)
        run_dir = run_dir if args.run_dir != (_RECIPE_DIR / "outputs" / "gepa_qwen25_3b") else (_RECIPE_DIR / preset["run_dir"])
        name = name or preset["name"]
        notes = notes or preset["notes"]

    if not source_run or not name:
        logger.error("Provide --source-run and --name (or use --preset)")
        return 2

    if not run_dir.is_absolute():
        run_dir = (_RECIPE_DIR / run_dir).resolve()

    tags = list(args.tags)
    tags.extend(["gepa_history_forked", f"from_{source_run}"])

    try:
        fork_gepa_run(
            entity=args.entity,
            project=args.project,
            source_run=source_run,
            max_iteration=max_iteration,
            run_dir=run_dir,
            name=name,
            notes=notes,
            tags=tags,
            dry_run=args.dry_run,
        )
    except Exception:
        logger.exception("Failed to fork GEPA run")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
