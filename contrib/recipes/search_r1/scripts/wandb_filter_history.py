"""Export W&B run history and re-log a truncated copy to a new run.

W&B does not support deleting individual history rows on cloud (rewind/fork are
private-preview on many teams). This script is the reliable workaround: scan
history from the public API, keep rows up to ``--max-step`` (and optionally drop
steps whose ``val/reward`` is below a threshold), then ``run.log(..., step=...)``
into a fresh run.

Usage (from ``contrib/recipes/search_r1``, with ``.env`` containing ``WANDB_API_KEY``)::

    python scripts/wandb_filter_history.py \\
        --entity ibm-bv --project AgentLightning \\
        --source-run vb39gplb --max-step 34 \\
        --name searchr1_qwen3_8b_shaped_clean

    python scripts/wandb_filter_history.py \\
        --entity ibm-bv --project AgentLightning \\
        --source-run ccp3q821 --max-step 36 \\
        --name searchr1_qwen7b_clean --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import wandb
from wandb.apis.public import Api

logger = logging.getLogger(__name__)

DEFAULT_ENTITY = "ibm-bv"
DEFAULT_PROJECT = "AgentLightning"

# Preset cutoffs for the Search-R1 Qwen2.5 3B runs (bad val/reward tail steps).
PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {
        "source_run": "ccp3q821",
        "max_step": 36,
        "name": "searchr1_qwen7b_clean",
        "notes": "Filtered copy of ccp3q821 through step 36 (drops bad step 37 val/reward≈0.005).",
    },
    "shaped": {
        "source_run": "vb39gplb",
        "max_step": 34,
        "name": "searchr1_qwen3_8b_shaped_clean",
        "notes": "Filtered copy of vb39gplb through step 34 (drops bad steps 35–36).",
    },
}


def _load_dotenv_if_present() -> None:
    dotenv = Path(__file__).resolve().parents[1] / ".env"
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


def _filter_rows(
    rows: Iterable[dict[str, Any]],
    *,
    max_step: int,
    min_val_reward: float | None,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in rows:
        step = int(row["_step"])
        if step > max_step:
            continue
        if min_val_reward is not None:
            val_reward = row.get("val/reward")
            if val_reward is not None and float(val_reward) < min_val_reward:
                logger.info("Skipping step %s: val/reward=%s < %s", step, val_reward, min_val_reward)
                continue
        kept.append(row)
    return kept


def _payload_from_row(row: dict[str, Any]) -> dict[str, Any]:
    skip = {"_timestamp", "_runtime", "_step", "run_id"}
    return {k: v for k, v in row.items() if not k.startswith("_") and k not in skip and v is not None}


def _summarize_val_reward(rows: list[dict[str, Any]]) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for row in rows:
        val = row.get("val/reward")
        if val is not None:
            out.append((int(row["_step"]), float(val)))
    return out


def create_filtered_run(
    *,
    entity: str,
    project: str,
    source_run: str,
    max_step: int,
    name: str,
    notes: str,
    tags: list[str],
    min_val_reward: float | None,
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

    kept = _filter_rows(all_rows, max_step=max_step, min_val_reward=min_val_reward)
    val_tail = _summarize_val_reward(kept)[-5:]
    logger.info("Keeping %s/%s rows (max_step=%s)", len(kept), len(all_rows), max_step)
    for step, val in val_tail:
        logger.info("  step %s val/reward=%.6f", step, val)

    if dry_run:
        logger.info("Dry run — no new run created")
        return None

    source_config = dict(source.config)
    source_summary = dict(source.summary or {})

    run = wandb.init(
        entity=entity,
        project=project,
        name=name,
        notes=notes,
        tags=tags,
        config=source_config,
        settings=wandb.Settings(_disable_stats=True, silent=False),
    )
    assert run is not None

    logged = 0
    for row in kept:
        payload = _payload_from_row(row)
        if not payload:
            continue
        run.log(payload, step=int(row["_step"]))
        logged += 1

    # Copy summary keys that still apply (best-effort).
    for key, value in source_summary.items():
        if key.startswith("_"):
            continue
        try:
            run.summary[key] = value
        except Exception:
            logger.debug("Could not set summary key %s", key, exc_info=True)

    run.finish()
    logger.info("Created run %s (%s steps logged)", run.id, logged)
    logger.info("View: https://wandb.ai/%s/%s/runs/%s", entity, project, run.id)
    return run.id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--source-run", help="Source W&B run id (e.g. ccp3q821)")
    parser.add_argument("--max-step", type=int, help="Keep history rows with _step <= this value")
    parser.add_argument("--name", help="Display name for the new run")
    parser.add_argument("--notes", default="", help="Notes on the new run")
    parser.add_argument("--tag", action="append", default=[], dest="tags")
    parser.add_argument(
        "--min-val-reward",
        type=float,
        default=None,
        help="Additionally drop steps whose val/reward is below this value",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="Use built-in cutoff for baseline (ccp3q821) or shaped (vb39gplb)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan only; do not create a run")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _load_dotenv_if_present()
    args = build_parser().parse_args(argv)

    source_run = args.source_run
    max_step = args.max_step
    name = args.name
    notes = args.notes

    if args.preset:
        preset = PRESETS[args.preset]
        source_run = source_run or preset["source_run"]
        max_step = max_step if max_step is not None else preset["max_step"]
        name = name or preset["name"]
        notes = notes or preset["notes"]

    if not source_run or max_step is None or not name:
        logging.error("Provide --source-run, --max-step, and --name (or use --preset)")
        return 2

    tags = list(args.tags)
    tags.append("history_filtered")
    if args.preset:
        tags.append(f"from_{args.preset}")

    try:
        create_filtered_run(
            entity=args.entity,
            project=args.project,
            source_run=source_run,
            max_step=max_step,
            name=name,
            notes=notes,
            tags=tags,
            min_val_reward=args.min_val_reward,
            dry_run=args.dry_run,
        )
    except Exception:
        logger.exception("Failed to create filtered run")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
