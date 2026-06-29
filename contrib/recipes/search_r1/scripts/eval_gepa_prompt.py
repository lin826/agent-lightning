"""Evaluate a GEPA-optimized instruction prompt on full test.parquet (hotpotqa).

Usage:
    python eval_gepa_prompt.py <prompt_file> [--metric-calls N]

Requires OPENAI_API_BASE (vLLM) and RETRIEVAL_SERVER_URL (BM25). Logs test/em and
test/reward to the dedicated eval WandB run (``wandb_eval_run_id.txt``) and, when
``wandb_run_id.txt`` exists, to the GEPA training run at ``metric_calls`` step.

Environment:
    GEPA_ROLLOUT_CONCURRENCY — parallel Search-R1 rollouts (default: 8)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

_RECIPE_DIR = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (_RECIPE_DIR, _SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from search_r1_gepa.search_r1_gepa_adapter import (  # noqa: E402
    INSTRUCTION_COMPONENT,
    SearchR1DataInst,
    SearchR1GEPAAdapter,
    make_openai_llm_call,
)
from search_r1_gepa.train_gepa import (  # noqa: E402
    DATA_SOURCE,
    MODEL_NAME,
    WANDB_PROJECT,
    default_run_dir,
    evaluate_split,
    resolve_gepa_variant,
    resolve_rollout_concurrency,
)
from wandb_run import (  # noqa: E402
    load_wandb_run_id,
    log_gepa_wandb_metrics,
    resolve_gepa_eval_wandb_run_name,
    resolve_gepa_iteration_for_metric_calls,
    resolve_wandb_eval_run_id,
    save_wandb_eval_run_id,
    setup_wandb_resume,
    validate_wandb_run_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TEST_FILE = "data/test.parquet"


def load_test_dataset(data_dir: Path) -> list[SearchR1DataInst]:
    test_df = pd.read_parquet(data_dir / TEST_FILE)
    test_df = test_df[test_df["data_source"] == DATA_SOURCE]
    records: list[SearchR1DataInst] = []
    for row in test_df.to_dict(orient="records"):
        golden = row["golden_answers"]
        if hasattr(golden, "tolist"):
            golden = golden.tolist()
        records.append(
            {
                "question": str(row["question"]),
                "golden_answers": [str(a) for a in list(golden)],
                "data_id": str(row.get("id", row["question"])),
            }
        )
    logger.info("Loaded hotpotqa test split: n=%d", len(records))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GEPA prompt on full test.parquet")
    parser.add_argument("prompt_file", type=Path, help="Path to best instruction prompt text file")
    parser.add_argument("--metric-calls", type=int, default=0, help="GEPA metric_calls tag for this eval")
    parser.add_argument("--data-dir", type=Path, default=_RECIPE_DIR, help="Recipe directory containing data/")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="GEPA run directory containing wandb_eval_run_id.txt",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Use <rewrite> turn during rollouts (must match training variant)",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--rollout-concurrency",
        type=int,
        default=None,
        help="Parallel Search-R1 rollouts per batch (default: GEPA_ROLLOUT_CONCURRENCY or 8)",
    )
    parser.add_argument("--wandb-run-id", default=None, help="Override eval WandB run id (default: auto-resolve)")
    args = parser.parse_args()

    variant = resolve_gepa_variant(rewrite=args.rewrite)
    if args.run_dir is None:
        args.run_dir = default_run_dir(_RECIPE_DIR, rewrite=args.rewrite)

    api_base = os.environ.get("OPENAI_API_BASE", "")
    if not api_base:
        raise RuntimeError("OPENAI_API_BASE must point to the vLLM OpenAI server.")
    if not os.environ.get("RETRIEVAL_SERVER_URL") and not (
        os.environ.get("RETRIEVAL_SERVER_ADDR_FILE") or os.environ.get("ADDR_FILE")
    ):
        raise RuntimeError(
            "RETRIEVAL_SERVER_URL or RETRIEVAL_SERVER_ADDR_FILE/ADDR_FILE must be set (BM25 retrieval server)."
        )

    instruction = args.prompt_file.read_text()
    candidate = {INSTRUCTION_COMPONENT: instruction}
    test_data = load_test_dataset(args.data_dir)

    rollout_concurrency = resolve_rollout_concurrency(args.rollout_concurrency)
    logger.info("Rollout concurrency=%d", rollout_concurrency)

    llm_call = make_openai_llm_call(base_url=api_base, model=MODEL_NAME, default_temperature=0.0)
    adapter = SearchR1GEPAAdapter(
        llm_call,
        eval_mode="val",
        rollout_concurrency=rollout_concurrency,
        use_rewrite=variant.use_rewrite,
    )
    test_em = evaluate_split(adapter, candidate, test_data, batch_size=args.batch_size)

    summary = {
        "test_em": test_em,
        "metric_calls": args.metric_calls,
        "prompt_file": str(args.prompt_file),
        "n_test": len(test_data),
        "variant": variant.name,
        "use_rewrite": variant.use_rewrite,
    }
    summary_path = args.run_dir / f"test_eval_m{args.metric_calls}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Full test EM=%.4f (n=%d), saved to %s", test_em, len(test_data), summary_path)

    try:
        import wandb

        eval_run_name = resolve_gepa_eval_wandb_run_name(variant.name)
        run_id = args.wandb_run_id or resolve_wandb_eval_run_id(run_dir=args.run_dir)
        run_id = validate_wandb_run_id(
            run_id,
            project=WANDB_PROJECT,
            directory=args.run_dir,
            kind="eval",
        )
        if run_id:
            setup_wandb_resume(run_id, run_name=eval_run_name)
            logger.info(
                "Resuming WandB eval run %s (%s) for full-test eval at metric_calls=%d",
                run_id,
                eval_run_name,
                args.metric_calls,
            )
        else:
            os.environ.pop("WANDB_RUN_ID", None)
            os.environ.pop("WANDB_RESUME", None)
            os.environ["WANDB_NAME"] = eval_run_name
            logger.info("No eval WandB run id found; creating eval run %r", eval_run_name)
        wandb_init_kwargs: dict[str, object] = {
            "project": WANDB_PROJECT,
            "name": eval_run_name,
            "config": {
                "eval_type": "full_test",
                "metric_calls": args.metric_calls,
                "variant": variant.name,
                "use_rewrite": variant.use_rewrite,
            },
        }
        if os.environ.get("WANDB_ENTITY"):
            wandb_init_kwargs["entity"] = os.environ["WANDB_ENTITY"]
        if run_id:
            wandb_init_kwargs["id"] = run_id
            wandb_init_kwargs["resume"] = "allow"
        wandb.init(**wandb_init_kwargs)
        assert wandb.run is not None
        wandb.log(
            {
                "test/em": test_em,
                "test/reward": test_em,
                "eval/metric_calls": args.metric_calls,
            },
            step=args.metric_calls,
        )
        save_wandb_eval_run_id(args.run_dir, wandb.run.id)
        wandb.finish()
    except ImportError:
        logger.warning("wandb not installed; skipping WandB logging")

    if load_wandb_run_id(args.run_dir):
        train_iteration = resolve_gepa_iteration_for_metric_calls(
            args.run_dir,
            args.metric_calls,
            project=WANDB_PROJECT,
            wandb_dir=_RECIPE_DIR / "wandb",
        )
        log_gepa_wandb_metrics(
            {
                "test/em": test_em,
                "test/reward": test_em,
                "total_metric_calls": args.metric_calls,
            },
            rollouts=args.metric_calls,
            iteration=train_iteration,
            project=WANDB_PROJECT,
            experiment_name=variant.wandb_experiment,
            run_dir=args.run_dir,
            finish=True,
        )
        logger.info(
            "Logged test/em=%.4f to GEPA training WandB run at rollouts=%d (iteration=%d)",
            test_em,
            args.metric_calls,
            train_iteration,
        )


if __name__ == "__main__":
    main()
