"""Backfill GRPO-comparable WandB metrics onto a completed GEPA run.

Appends ``val/reward``, ``val/em``, ``seed/val_em``, and optional per-iteration
``val/reward`` derived from GEPA's ``best_valset_agg_score`` history. Does not
rewind or delete existing WandB steps.

Usage:
    python scripts/backfill_gepa_wandb.py --run-id youriju4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_RECIPE_DIR = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (_RECIPE_DIR, _SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from search_r1_gepa.train_gepa import WANDB_EXPERIMENT, WANDB_PROJECT  # noqa: E402
from wandb_run import log_gepa_wandb_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill GRPO-style metrics onto a GEPA WandB run")
    parser.add_argument("--run-id", required=True, help="Existing WandB run id (e.g. youriju4)")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=_RECIPE_DIR / "outputs" / "gepa_qwen25_3b",
        help="GEPA run directory with gepa_summary.json",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned metrics without logging")
    args = parser.parse_args()

    summary_path = args.run_dir / "gepa_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}")
    summary = json.loads(summary_path.read_text())

    seed_val_em = float(summary["seed_val_em"])
    best_val_em = float(summary["best_val_em"])
    best_train_em = float(summary.get("best_train_em_sample500", summary.get("best_train_em", 0.0)))
    total_metric_calls = int(summary["total_metric_calls"])

    os.environ["WANDB_RUN_ID"] = args.run_id.strip()
    os.environ["WANDB_RESUME"] = "must"

    seed_metrics = {
        "seed/val_em": seed_val_em,
        "val/em": seed_val_em,
        "val/reward": seed_val_em,
    }
    final_metrics = {
        "val/em": best_val_em,
        "val/reward": best_val_em,
        "train/em": best_train_em,
        "training/reward": best_train_em,
        "training/em": best_train_em,
        "seed/val_em": seed_val_em,
    }

    iteration_metrics: list[tuple[int, dict[str, float]]] = []
    try:
        import wandb
        from wandb.apis.public import Api

        api = Api()
        entity = os.environ.get("WANDB_ENTITY", "ibm-bv")
        run = api.run(f"{entity}/{WANDB_PROJECT}/{args.run_id}")
        history = run.history(keys=["best_valset_agg_score"], pandas=True)
        if not history.empty and "best_valset_agg_score" in history.columns:
            for _, row in history.iterrows():
                step = int(row["_step"])
                score = float(row["best_valset_agg_score"])
                iteration_metrics.append(
                    (
                        step,
                        {
                            "val/reward": score,
                            "val/em": score,
                        },
                    )
                )
    except Exception as exc:
        logger.warning("Could not load iteration history from WandB API: %s", exc)

    if args.dry_run:
        logger.info("Would log seed metrics at step 0: %s", seed_metrics)
        for step, metrics in iteration_metrics:
            logger.info("Would log iteration metrics at step %s: %s", step, metrics)
        logger.info("Would log final metrics at step %s: %s", total_metric_calls, final_metrics)
        return

    log_gepa_wandb_metrics(
        seed_metrics,
        step=0,
        project=WANDB_PROJECT,
        experiment_name=WANDB_EXPERIMENT,
        run_dir=args.run_dir,
        config={"backfill": True},
        wandb_dir=_RECIPE_DIR / "wandb",
        finish=True,
    )

    for step, metrics in iteration_metrics:
        log_gepa_wandb_metrics(
            metrics,
            step=step,
            project=WANDB_PROJECT,
            experiment_name=WANDB_EXPERIMENT,
            run_dir=args.run_dir,
            config={"backfill": True},
            wandb_dir=_RECIPE_DIR / "wandb",
            finish=True,
        )

    log_gepa_wandb_metrics(
        final_metrics,
        step=total_metric_calls,
        project=WANDB_PROJECT,
        experiment_name=WANDB_EXPERIMENT,
        run_dir=args.run_dir,
        config={"backfill": True},
        wandb_dir=_RECIPE_DIR / "wandb",
        finish=True,
    )
    logger.info(
        "Backfilled run %s: seed step 0, %d iteration steps, final step %s",
        args.run_id,
        len(iteration_metrics),
        total_metric_calls,
    )


if __name__ == "__main__":
    main()
