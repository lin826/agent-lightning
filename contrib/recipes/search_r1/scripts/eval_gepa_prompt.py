"""Evaluate a GEPA-optimized instruction prompt on full test.parquet (hotpotqa).

Usage:
    python eval_gepa_prompt.py <prompt_file> [--metric-calls N]

Requires OPENAI_API_BASE (vLLM) and RETRIEVAL_SERVER_URL (BM25). Logs test/em and
test/reward to the original searchr1_qwen25_3b_gepa WandB run (resume via
WANDB_RUN_ID or outputs/gepa_qwen25_3b/wandb_run_id.txt).
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
    WANDB_EXPERIMENT,
    WANDB_PROJECT,
    evaluate_split,
)
from wandb_run import resolve_wandb_run_id  # noqa: E402

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
        default=_RECIPE_DIR / "outputs" / "gepa_qwen25_3b",
        help="GEPA run directory containing wandb_run_id.txt",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--wandb-run-id", default=None, help="Override WandB run id (default: auto-resolve)")
    args = parser.parse_args()

    api_base = os.environ.get("OPENAI_API_BASE", "")
    if not api_base:
        raise RuntimeError("OPENAI_API_BASE must point to the vLLM OpenAI server.")
    if not os.environ.get("RETRIEVAL_SERVER_URL"):
        raise RuntimeError("RETRIEVAL_SERVER_URL must be set (BM25 retrieval server).")

    instruction = args.prompt_file.read_text()
    candidate = {INSTRUCTION_COMPONENT: instruction}
    test_data = load_test_dataset(args.data_dir)

    llm_call = make_openai_llm_call(base_url=api_base, model=MODEL_NAME, default_temperature=0.0)
    adapter = SearchR1GEPAAdapter(llm_call, eval_mode="val")
    test_em = evaluate_split(adapter, candidate, test_data, batch_size=args.batch_size)

    summary = {
        "test_em": test_em,
        "metric_calls": args.metric_calls,
        "prompt_file": str(args.prompt_file),
        "n_test": len(test_data),
    }
    summary_path = args.data_dir / "outputs" / "gepa_qwen25_3b" / f"test_eval_m{args.metric_calls}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Full test EM=%.4f (n=%d), saved to %s", test_em, len(test_data), summary_path)

    try:
        import wandb

        run_id = args.wandb_run_id or resolve_wandb_run_id(
            run_dir=args.run_dir,
            experiment_name=WANDB_EXPERIMENT,
            wandb_dir=_RECIPE_DIR / "wandb",
        )
        wandb_init_kwargs: dict[str, object] = {
            "project": WANDB_PROJECT,
            "name": WANDB_EXPERIMENT,
            "config": {"eval_type": "full_test", "metric_calls": args.metric_calls},
        }
        if os.environ.get("WANDB_ENTITY"):
            wandb_init_kwargs["entity"] = os.environ["WANDB_ENTITY"]
        if run_id:
            wandb_init_kwargs["id"] = run_id
            wandb_init_kwargs["resume"] = "must"
            logger.info("Resuming WandB run %s for full-test eval at metric_calls=%d", run_id, args.metric_calls)
        else:
            logger.warning("WandB run id not found; full-test eval may create a separate run")
        wandb.init(**wandb_init_kwargs)
        wandb.log(
            {
                "test/em": test_em,
                "test/reward": test_em,
                "eval/metric_calls": args.metric_calls,
            },
            step=args.metric_calls,
        )
        wandb.finish()
    except ImportError:
        logger.warning("wandb not installed; skipping WandB logging")


if __name__ == "__main__":
    main()
