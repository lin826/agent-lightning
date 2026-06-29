"""Backfill GRPO-comparable WandB metrics onto a completed GEPA run.

Appends ``val/reward``, ``val/em``, ``seed/val_em``, and optional per-iteration
``val/reward`` derived from GEPA's per-step ``val_program_average`` (mean val EM).
Does not rewind or delete existing WandB steps.

Usage:
    python scripts/backfill_gepa_wandb.py --run-id <wandb_run_id>
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
    parser.add_argument("--run-id", required=True, help="Existing WandB run id")
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
    final_iteration = int(summary.get("final_iteration", 0))

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
    rollout_metrics: list[tuple[int, dict[str, float]]] = []
    try:
        import wandb
        from wandb.apis.public import Api

        api = Api()
        entity = os.environ.get("WANDB_ENTITY", "ibm-bv")
        run = api.run(f"{entity}/{WANDB_PROJECT}/{args.run_id}")
        history = run.history(keys=["val_program_average", "base_program_full_valset_score"], pandas=True)
        val_col = (
            "val_program_average"
            if "val_program_average" in history.columns
            else "base_program_full_valset_score"
        )
        if not history.empty and val_col in history.columns:
            for _, row in history.iterrows():
                step = int(row["_step"])
                raw = row[val_col]
                if raw != raw:  # NaN
                    continue
                score = float(raw)
                iteration_metrics.append(
                    (
                        step,
                        {
                            "val/reward": score,
                            "val/em": score,
                        },
                    )
                )
        for row in run.scan_history():
            step = int(row.get("_step", 0))
            calls = row.get("total_metric_calls")
            if calls is None or calls != calls:
                continue
            rollout_metrics.append((step, {"rollouts": int(calls), "total_metric_calls": int(calls)}))
    except Exception as exc:
        logger.warning("Could not load iteration history from WandB API: %s", exc)

    if final_iteration <= 0:
        try:
            from wandb.apis.public import Api

            api = Api()
            entity = os.environ.get("WANDB_ENTITY", "ibm-bv")
            run = api.run(f"{entity}/{WANDB_PROJECT}/{args.run_id}")
            iterations = [
                int(row["iteration"])
                for row in run.scan_history()
                if row.get("iteration") is not None and int(row.get("iteration", 0)) > 0
            ]
            final_iteration = max(iterations) if iterations else 0
        except Exception as exc:
            logger.warning("Could not infer final_iteration from WandB history: %s", exc)
            final_iteration = 0
        if final_iteration <= 0:
            final_iteration = total_metric_calls

    if args.dry_run:
        logger.info("Would log seed metrics at step 0: %s", seed_metrics)
        for step, metrics in iteration_metrics:
            logger.info("Would log iteration metrics at step %s: %s", step, metrics)
        for step, metrics in rollout_metrics:
            logger.info("Would backfill rollouts at step %s: %s", step, metrics)
        logger.info(
            "Would log final metrics at step %s (rollouts=%s): %s",
            final_iteration,
            total_metric_calls,
            final_metrics,
        )
        logger.info("Would update run summary with final + seed metrics")
        return

    try:
        from wandb.apis.public import Api

        api = Api()
        entity = os.environ.get("WANDB_ENTITY", "ibm-bv")
        run = api.run(f"{entity}/{WANDB_PROJECT}/{args.run_id}")
        run.summary.update(
            {
                **final_metrics,
            }
        )
        run.summary.save()
        logger.info("Updated run summary for %s", args.run_id)
    except Exception as exc:
        logger.warning("Could not update WandB summary via API: %s", exc)

    # Append-only history: steps at or below the run's current max step are ignored by WandB.
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

    for step, metrics in rollout_metrics:
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
        {
            **final_metrics,
            "total_metric_calls": total_metric_calls,
            "iteration": final_iteration,
        },
        step=final_iteration,
        project=WANDB_PROJECT,
        experiment_name=WANDB_EXPERIMENT,
        run_dir=args.run_dir,
        config={"backfill": True},
        wandb_dir=_RECIPE_DIR / "wandb",
        finish=True,
    )
    logger.info(
        "Backfilled run %s: seed step 0, %d iteration steps, %d rollout steps, final step %s (rollouts=%s)",
        args.run_id,
        len(iteration_metrics),
        len(rollout_metrics),
        final_iteration,
        total_metric_calls,
    )


if __name__ == "__main__":
    main()
