"""Backfill full-test GEPA eval metrics onto the dedicated eval WandB run.

Reads ``test_eval_m{N}.json`` files under a GEPA run directory and logs
``test/em`` and ``test/reward`` at step ``metric_calls`` on the eval run
(``eval_gepa`` by default; see ``resolve_gepa_eval_wandb_run_name``).

Usage:
    python scripts/backfill_gepa_eval_wandb.py
    python scripts/backfill_gepa_eval_wandb.py --metric-calls 0 1213 1631
    python scripts/backfill_gepa_eval_wandb.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

_RECIPE_DIR = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (_RECIPE_DIR, _SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from search_r1_gepa.train_gepa import WANDB_PROJECT, default_run_dir, resolve_gepa_variant  # noqa: E402
from wandb_run import (  # noqa: E402
    resolve_gepa_eval_wandb_run_name,
    resolve_wandb_eval_run_id,
    save_wandb_eval_run_id,
    setup_wandb_resume,
    validate_wandb_run_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_TEST_EVAL_RE = re.compile(r"^test_eval_m(\d+)\.json$")


def _discover_metric_calls(run_dir: Path) -> list[int]:
    steps: list[int] = []
    for path in sorted(run_dir.glob("test_eval_m*.json")):
        match = _TEST_EVAL_RE.match(path.name)
        if match:
            steps.append(int(match.group(1)))
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill GEPA full-test eval metrics to WandB")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="GEPA run directory (default: outputs/gepa_qwen25_3b)",
    )
    parser.add_argument(
        "--metric-calls",
        type=int,
        nargs="*",
        default=None,
        help="Metric-call steps to backfill (default: all test_eval_m*.json)",
    )
    parser.add_argument("--rewrite", action="store_true", help="GEPA rewrite variant")
    parser.add_argument("--wandb-run-id", default=None, help="Override eval WandB run id")
    parser.add_argument("--dry-run", action="store_true", help="Print planned logs without writing to WandB")
    args = parser.parse_args()

    variant = resolve_gepa_variant(rewrite=args.rewrite)
    run_dir = args.run_dir or default_run_dir(_RECIPE_DIR, rewrite=args.rewrite)
    metric_calls_list = args.metric_calls if args.metric_calls is not None else _discover_metric_calls(run_dir)
    if not metric_calls_list:
        raise SystemExit(f"No test_eval_m*.json files found under {run_dir}")

    eval_run_name = resolve_gepa_eval_wandb_run_name(variant.name)
    payloads: list[tuple[int, float]] = []
    for step in sorted(metric_calls_list):
        summary_path = run_dir / f"test_eval_m{step}.json"
        if not summary_path.is_file():
            logger.warning("Skipping metric_calls=%d — missing %s", step, summary_path)
            continue
        summary = json.loads(summary_path.read_text())
        test_em = float(summary["test_em"])
        payloads.append((step, test_em))
        logger.info("metric_calls=%d test/em=%.4f from %s", step, test_em, summary_path.name)

    if not payloads:
        raise SystemExit("No eval summaries to backfill")

    if args.dry_run:
        logger.info(
            "[DRY RUN] Would log %d points to WandB run %r under %s",
            len(payloads),
            eval_run_name,
            run_dir,
        )
        return

    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb not installed") from None

    run_id = args.wandb_run_id or resolve_wandb_eval_run_id(run_dir=run_dir)
    run_id = validate_wandb_run_id(run_id, project=WANDB_PROJECT, directory=run_dir, kind="eval")
    if run_id:
        setup_wandb_resume(run_id, run_name=eval_run_name)
    else:
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_RESUME", None)
        os.environ["WANDB_NAME"] = eval_run_name

    init_kwargs: dict[str, object] = {
        "project": WANDB_PROJECT,
        "name": eval_run_name,
        "config": {"eval_type": "full_test_backfill", "variant": variant.name},
    }
    if os.environ.get("WANDB_ENTITY"):
        init_kwargs["entity"] = os.environ["WANDB_ENTITY"]
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"
    wandb.init(**init_kwargs)
    assert wandb.run is not None

    for step, test_em in payloads:
        wandb.log(
            {
                "test/em": test_em,
                "test/reward": test_em,
                "eval/metric_calls": step,
            },
            step=step,
        )

    save_wandb_eval_run_id(run_dir, wandb.run.id)
    run_url = getattr(wandb.run, "url", None) or f"https://wandb.ai/{WANDB_PROJECT}/runs/{wandb.run.id}"
    logger.info("Backfilled %d eval points to WandB run %s (%s): %s", len(payloads), wandb.run.id, eval_run_name, run_url)
    wandb.finish()


if __name__ == "__main__":
    main()
