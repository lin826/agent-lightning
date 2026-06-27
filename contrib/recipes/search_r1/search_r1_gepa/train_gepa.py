# Copyright (c) Microsoft. All rights reserved.

"""Run GEPA prompt optimization for Search-R1 on HotpotQA.

Baseline to compare frozen-weight GEPA prompt evolution against GRPO weight
updates (including Search-R1 ``<rewrite>`` variants). Uses the same data split,
BM25 retrieval, EM metric, and Qwen2.5-3B-Instruct task model as GRPO jobs.

Usage:
    python search_r1_gepa/train_gepa.py

Environment:
    RETRIEVAL_SERVER_URL or RETRIEVAL_SERVER_ADDR_FILE — BM25 server (required)
    OPENAI_API_BASE — vLLM OpenAI endpoint for Qwen2.5-3B-Instruct (required)
    GEPA_MAX_METRIC_CALLS — optimization budget (default: 1500)
    GEPA_REFLECTION_LM — litellm model id for reflection (default: same as task LM)
    GEPA_TRAIN_SUBSET — optional cap on hotpotqa train examples for smoke tests
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import gepa
import pandas as pd

# Allow imports from recipe root and scripts/.
_RECIPE_DIR = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _RECIPE_DIR / "scripts"
for _path in (_RECIPE_DIR, _SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from search_r1_gepa.search_r1_gepa_adapter import (  # noqa: E402
    INSTRUCTION_COMPONENT,
    SearchR1DataInst,
    SearchR1GEPAAdapter,
    default_seed_candidate,
    make_openai_llm_call,
)
from search_r1_agent import INSTRUCTION_FORMAT  # noqa: E402
from wandb_run import save_wandb_run_id  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

WANDB_PROJECT = "AgentLightning"
WANDB_EXPERIMENT = "searchr1_qwen25_3b_gepa"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_SOURCE = "hotpotqa"
TRAIN_FILE = "data/train.parquet"
VAL_FILE = "data/test_dev.parquet"
DEFAULT_MAX_METRIC_CALLS = 1500


def load_dataset(data_dir: Path, *, train_subset: int | None = None) -> tuple[list[SearchR1DataInst], list[SearchR1DataInst]]:
    train_df = pd.read_parquet(data_dir / TRAIN_FILE)
    val_df = pd.read_parquet(data_dir / VAL_FILE)
    train_df = train_df[train_df["data_source"] == DATA_SOURCE]
    val_df = val_df[val_df["data_source"] == DATA_SOURCE]

    if train_subset is not None and train_subset > 0:
        train_df = train_df.head(train_subset)

    def _to_records(df: pd.DataFrame) -> list[SearchR1DataInst]:
        records: list[SearchR1DataInst] = []
        for row in df.to_dict(orient="records"):
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
        return records

    train_data = _to_records(train_df)
    val_data = _to_records(val_df)
    logger.info("Loaded hotpotqa split: train=%d val=%d", len(train_data), len(val_data))
    return train_data, val_data


def evaluate_split(
    adapter: SearchR1GEPAAdapter,
    candidate: dict[str, str],
    dataset: list[SearchR1DataInst],
    *,
    batch_size: int = 20,
) -> float:
    """Compute mean EM on a dataset (for logging train/em and val/em)."""
    scores: list[float] = []
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]
        result = adapter.evaluate(batch, candidate, capture_traces=False)
        scores.extend(result.scores)
    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="GEPA prompt optimization baseline for Search-R1")
    parser.add_argument("--data-dir", type=Path, default=_RECIPE_DIR, help="Recipe directory containing data/")
    parser.add_argument("--run-dir", type=Path, default=_RECIPE_DIR / "outputs" / "gepa_qwen25_3b", help="GEPA state dir")
    parser.add_argument("--max-metric-calls", type=int, default=None, help="GEPA evaluation budget")
    parser.add_argument("--train-subset", type=int, default=None, help="Cap train examples (smoke tests)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    max_metric_calls = args.max_metric_calls
    if max_metric_calls is None:
        max_metric_calls = int(os.environ.get("GEPA_MAX_METRIC_CALLS", str(DEFAULT_MAX_METRIC_CALLS)))

    train_subset = args.train_subset
    if train_subset is None and os.environ.get("GEPA_TRAIN_SUBSET"):
        train_subset = int(os.environ["GEPA_TRAIN_SUBSET"])

    api_base = os.environ.get("OPENAI_API_BASE", "")
    if not api_base:
        raise RuntimeError("OPENAI_API_BASE must point to the vLLM OpenAI server.")

    reflection_lm = os.environ.get("GEPA_REFLECTION_LM")
    if not reflection_lm:
        reflection_lm = f"openai/{MODEL_NAME}"

    train_data, val_data = load_dataset(args.data_dir, train_subset=train_subset)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    llm_call = make_openai_llm_call(base_url=api_base, model=MODEL_NAME, default_temperature=1.0)
    adapter = SearchR1GEPAAdapter(llm_call, eval_mode="train")

    seed_candidate = default_seed_candidate()
    wandb_init_kwargs: dict[str, Any] = {
        "project": WANDB_PROJECT,
        "name": WANDB_EXPERIMENT,
        "config": {
            "baseline": "gepa",
            "model": MODEL_NAME,
            "data_source": DATA_SOURCE,
            "train_file": TRAIN_FILE,
            "val_file": VAL_FILE,
            "max_metric_calls": max_metric_calls,
            "seed_instruction": INSTRUCTION_FORMAT[:200],
        },
    }
    if os.environ.get("WANDB_ENTITY"):
        wandb_init_kwargs["entity"] = os.environ["WANDB_ENTITY"]

    logger.info("Evaluating seed prompt on val (n=%d)...", len(val_data))
    val_adapter = SearchR1GEPAAdapter(llm_call, eval_mode="val")
    seed_val_em = evaluate_split(val_adapter, seed_candidate, val_data)
    logger.info("Seed val/em=%.4f", seed_val_em)

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train_data,
        valset=val_data,
        adapter=adapter,
        reflection_lm=reflection_lm,
        candidate_selection_strategy="pareto",
        reflection_minibatch_size=3,
        max_metric_calls=max_metric_calls,
        use_merge=False,
        use_wandb=True,
        wandb_api_key=os.environ.get("WANDB_API_KEY"),
        wandb_init_kwargs=wandb_init_kwargs,
        run_dir=str(args.run_dir),
        seed=args.seed,
        display_progress_bar=True,
    )

    best_candidate = result.best_candidate
    best_val_em = evaluate_split(val_adapter, best_candidate, val_data)
    train_adapter = SearchR1GEPAAdapter(llm_call, eval_mode="train")
    best_train_em = evaluate_split(train_adapter, best_candidate, train_data[: min(500, len(train_data))])

    summary = {
        "best_val_em": best_val_em,
        "best_train_em_sample500": best_train_em,
        "seed_val_em": seed_val_em,
        "total_metric_calls": result.total_metric_calls,
        "best_candidate_keys": list(best_candidate.keys()),
    }
    summary_path = args.run_dir / "gepa_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    prompt_path = args.run_dir / "best_instruction_prompt.txt"
    prompt_path.write_text(best_candidate[INSTRUCTION_COMPONENT])

    try:
        import wandb

        if wandb.run is not None:
            save_wandb_run_id(args.run_dir, wandb.run.id)
            wandb.log(
                {
                    "val/em": best_val_em,
                    "train/em": best_train_em,
                    "seed/val_em": seed_val_em,
                }
            )
    except ImportError:
        pass

    logger.info("GEPA complete. best val/em=%.4f train/em(sample500)=%.4f", best_val_em, best_train_em)
    logger.info("Best prompt saved to %s", prompt_path)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
