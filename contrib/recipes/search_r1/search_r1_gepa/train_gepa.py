# Copyright (c) Microsoft. All rights reserved.

"""Run GEPA prompt optimization for Search-R1 on HotpotQA.

Baseline to compare frozen-weight GEPA prompt evolution against GRPO weight
updates (including Search-R1 ``<rewrite>`` variants). Uses the same data split,
BM25 retrieval, EM metric, and Qwen2.5-3B-Instruct task model as GRPO jobs.

Usage:
    python search_r1_gepa/train_gepa.py [--rewrite]

Environment:
    RETRIEVAL_SERVER_URL or RETRIEVAL_SERVER_ADDR_FILE — BM25 server (required)
    OPENAI_API_BASE — vLLM OpenAI endpoint for Qwen2.5-3B-Instruct (required)
    GEPA_MAX_METRIC_CALLS — absolute optimization budget (default: 60000; train_gepa.bsub sets 60000)
    GEPA_CHUNK_METRIC_CALLS — incremental budget per job/chunk (default when set: 9965 ≈ one GRPO step)
    GEPA_ROLLOUT_CONCURRENCY — parallel Search-R1 rollouts (default: 8)
    GEPA_REFLECTION_MINIBATCH_SIZE — reflection minibatch for gepa.optimize (default: 16)
    GEPA_REFLECTION_LM — litellm model id for reflection (default: same as task LM)
    GEPA_TRAIN_SUBSET — optional cap on hotpotqa train examples for smoke tests
    GEPA_TRAIN_TEMPERATURE — rollout temperature during gepa.optimize (default: 1.0)
    GEPA_VAL_TEMPERATURE — rollout temperature for seed/final val eval (default: 0.0)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
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
    resolve_reward_mode,
)
from search_r1_agent import INSTRUCTION_FORMAT, INSTRUCTION_FORMAT_REWRITE  # noqa: E402
from gepa_full_eval import (  # noqa: E402
    clear_program_prompt_cache,
    install_gepa_program_prompt_cache,
    maybe_trigger_full_eval,
    register_program_candidates,
    save_seed_instruction_prompt,
    start_training_session,
)
from wandb_run import (  # noqa: E402
    build_gepa_wandb_init_kwargs,
    install_gepa_wandb_grpo_compat_patch,
    log_gepa_wandb_metrics,
    resolve_gepa_wandb_run_id,
    save_wandb_run_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

WANDB_PROJECT = "AgentLightning"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_SOURCE = "hotpotqa"
TRAIN_FILE = "data/train.parquet"
VAL_FILE = "data/test.parquet"
DEFAULT_MAX_METRIC_CALLS = 60000
DEFAULT_ROLLOUT_CONCURRENCY = 8
DEFAULT_REFLECTION_MINIBATCH_SIZE = 16

# GRPO parity: one global_step ≈ train rollouts + val rollouts (RL_TRAINING_CONFIG in train_search_r1_agent.py).
GRPO_TRAIN_BATCH_SIZE = 512
GRPO_ROLLOUT_N = 5
GRPO_VAL_EXAMPLES = 7405  # hotpotqa rows in data/test.parquet (full val set)
DEFAULT_GRPO_STEP_ROLLOUTS = GRPO_TRAIN_BATCH_SIZE * GRPO_ROLLOUT_N + GRPO_VAL_EXAMPLES


@dataclass(frozen=True)
class GepaVariantConfig:
    """Paths and flags for a GEPA experiment variant."""

    name: str
    run_dir_name: str
    wandb_experiment: str
    eval_job_tag: str
    eval_addr_file: str
    use_rewrite: bool
    reward_mode: str = "em"


GEPA_VARIANTS: dict[str, GepaVariantConfig] = {
    "baseline": GepaVariantConfig(
        name="baseline",
        run_dir_name="gepa_qwen25_3b",
        wandb_experiment="searchr1_qwen25_3b_gepa",
        eval_job_tag="qwen25_3b_gepa",
        eval_addr_file="bm25_server_addr_eval_gepa.txt",
        use_rewrite=False,
        reward_mode="em",
    ),
    "rewrite": GepaVariantConfig(
        name="rewrite",
        run_dir_name="gepa_qwen25_3b_rewrite",
        wandb_experiment="searchr1_qwen25_3b_gepa_rewrite",
        eval_job_tag="qwen25_3b_gepa_rewrite",
        eval_addr_file="bm25_server_addr_eval_gepa_rewrite.txt",
        use_rewrite=True,
        reward_mode="em",
    ),
    "shaped": GepaVariantConfig(
        name="shaped",
        run_dir_name="gepa_qwen25_3b_shaped",
        wandb_experiment="searchr1_qwen25_3b_gepa_shaped",
        eval_job_tag="qwen25_3b_gepa_shaped",
        eval_addr_file="bm25_server_addr_eval_gepa_shaped.txt",
        use_rewrite=False,
        reward_mode="shaped",
    ),
    "rewrite_shaped": GepaVariantConfig(
        name="rewrite_shaped",
        run_dir_name="gepa_qwen25_3b_rewrite_shaped",
        wandb_experiment="searchr1_qwen25_3b_gepa_rewrite_shaped",
        eval_job_tag="qwen25_3b_gepa_rewrite_shaped",
        eval_addr_file="bm25_server_addr_eval_gepa_rewrite_shaped.txt",
        use_rewrite=True,
        reward_mode="shaped",
    ),
}


def resolve_gepa_variant(*, rewrite: bool = False, shaped: bool = False) -> GepaVariantConfig:
    if rewrite and shaped:
        return GEPA_VARIANTS["rewrite_shaped"]
    if shaped:
        return GEPA_VARIANTS["shaped"]
    if rewrite:
        return GEPA_VARIANTS["rewrite"]
    return GEPA_VARIANTS["baseline"]


def default_run_dir(recipe_dir: Path, *, rewrite: bool = False, shaped: bool = False) -> Path:
    return recipe_dir / "outputs" / resolve_gepa_variant(rewrite=rewrite, shaped=shaped).run_dir_name


# Backward-compatible alias for scripts that target the baseline GEPA run.
WANDB_EXPERIMENT = GEPA_VARIANTS["baseline"].wandb_experiment


def resolve_rollout_concurrency(cli_value: int | None) -> int:
    if cli_value is not None:
        return max(1, cli_value)
    if os.environ.get("GEPA_ROLLOUT_CONCURRENCY"):
        return max(1, int(os.environ["GEPA_ROLLOUT_CONCURRENCY"]))
    return DEFAULT_ROLLOUT_CONCURRENCY


def resolve_reflection_minibatch_size(cli_value: int | None) -> int:
    if cli_value is not None:
        return max(1, cli_value)
    if os.environ.get("GEPA_REFLECTION_MINIBATCH_SIZE"):
        return max(1, int(os.environ["GEPA_REFLECTION_MINIBATCH_SIZE"]))
    return DEFAULT_REFLECTION_MINIBATCH_SIZE


def resolve_train_temperature() -> float:
    if os.environ.get("GEPA_TRAIN_TEMPERATURE"):
        return float(os.environ["GEPA_TRAIN_TEMPERATURE"])
    return 1.0


def resolve_val_temperature() -> float:
    if os.environ.get("GEPA_VAL_TEMPERATURE"):
        return float(os.environ["GEPA_VAL_TEMPERATURE"])
    return 0.0


def compute_grpo_step_rollouts(
    *,
    train_batch_size: int = GRPO_TRAIN_BATCH_SIZE,
    rollout_n: int = GRPO_ROLLOUT_N,
    val_examples: int = GRPO_VAL_EXAMPLES,
) -> int:
    """Return rollout budget for one GRPO ``global_step`` (train batch × n + val set)."""
    return train_batch_size * rollout_n + val_examples


def load_gepa_total_evals(run_dir: Path) -> int:
    """Return ``total_num_evals`` from ``gepa_state.bin``, or 0 when missing."""
    state_path = run_dir / "gepa_state.bin"
    if not state_path.is_file():
        return 0
    from gepa.core.state import GEPAState

    state = GEPAState.load(str(run_dir))
    return int(state.total_num_evals)


def resolve_chunk_metric_calls(cli_value: int | None) -> int | None:
    """Parse incremental chunk size from CLI or ``GEPA_CHUNK_METRIC_CALLS``."""
    if cli_value is not None:
        return max(1, cli_value)
    env_val = os.environ.get("GEPA_CHUNK_METRIC_CALLS")
    if env_val is not None and env_val.strip() != "":
        return max(1, int(env_val))
    return None


def resolve_max_metric_calls(
    *,
    cli_max: int | None,
    cli_chunk: int | None,
    run_dir: Path,
    resuming_gepa: bool,
) -> tuple[int, int | None, int]:
    """Return ``(effective_max, chunk_size_or_none, prior_total_evals)``."""
    chunk = resolve_chunk_metric_calls(cli_chunk)
    prior_total = load_gepa_total_evals(run_dir) if resuming_gepa else 0

    if chunk is not None:
        effective = prior_total + chunk
        logger.info(
            "GEPA chunk budget: prior_total=%d chunk=%d effective_max=%d (≈ one GRPO step when chunk=%d)",
            prior_total,
            chunk,
            effective,
            DEFAULT_GRPO_STEP_ROLLOUTS,
        )
        return effective, chunk, prior_total

    if cli_max is not None:
        return cli_max, None, prior_total

    absolute = int(os.environ.get("GEPA_MAX_METRIC_CALLS", str(DEFAULT_MAX_METRIC_CALLS)))
    return absolute, None, prior_total


def load_seed_val_em(run_dir: Path) -> float | None:
    """Load cached seed val/em from a prior ``gepa_summary.json``."""
    summary_path = run_dir / "gepa_summary.json"
    if not summary_path.is_file():
        return None
    try:
        with open(summary_path) as f:
            summary = json.load(f)
        if "seed_val_em" in summary:
            return float(summary["seed_val_em"])
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Failed to read seed_val_em from %s: %s", summary_path, exc)
    return None


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
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="GEPA state dir (default: outputs/gepa_qwen25_3b or outputs/gepa_qwen25_3b_rewrite)",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Use <rewrite> seed instruction and rewrite turn during rollouts (GRPO rewrite variant)",
    )
    parser.add_argument(
        "--shaped",
        action="store_true",
        help="Optimize with shaped reward (EM + retrieval-hit) during training rollouts",
    )
    parser.add_argument("--max-metric-calls", type=int, default=None, help="GEPA absolute evaluation budget")
    parser.add_argument(
        "--chunk-metric-calls",
        type=int,
        default=None,
        help="Incremental budget for this run (default: GEPA_CHUNK_METRIC_CALLS or GRPO-step parity)",
    )
    parser.add_argument(
        "--reflection-minibatch-size",
        type=int,
        default=None,
        help="Reflection minibatch size (default: GEPA_REFLECTION_MINIBATCH_SIZE or 16)",
    )
    parser.add_argument("--train-subset", type=int, default=None, help="Cap train examples (smoke tests)")
    parser.add_argument(
        "--rollout-concurrency",
        type=int,
        default=None,
        help="Parallel Search-R1 rollouts per batch (default: GEPA_ROLLOUT_CONCURRENCY or 8)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    variant = resolve_gepa_variant(rewrite=args.rewrite, shaped=args.shaped or resolve_reward_mode() == "shaped")
    if args.run_dir is None:
        args.run_dir = default_run_dir(_RECIPE_DIR, rewrite=variant.use_rewrite, shaped=variant.reward_mode == "shaped")

    train_subset = args.train_subset
    reflection_minibatch_size = resolve_reflection_minibatch_size(args.reflection_minibatch_size)

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

    wandb_dir = _RECIPE_DIR / "wandb"
    fresh_session = os.environ.get("GEPA_FRESH_SESSION", "").lower() in {"1", "true", "yes"}
    gepa_state_path = args.run_dir / "gepa_state.bin"
    resuming_gepa = gepa_state_path.is_file() and not fresh_session

    max_metric_calls, chunk_metric_calls, prior_total_evals = resolve_max_metric_calls(
        cli_max=args.max_metric_calls,
        cli_chunk=args.chunk_metric_calls,
        run_dir=args.run_dir,
        resuming_gepa=resuming_gepa,
    )

    if not fresh_session:
        run_id = resolve_gepa_wandb_run_id(
            project=WANDB_PROJECT,
            experiment_name=variant.wandb_experiment,
            run_dir=args.run_dir,
            wandb_dir=wandb_dir,
        )
        if run_id:
            save_wandb_run_id(args.run_dir, run_id)
        elif resuming_gepa:
            logger.warning(
                "Resuming gepa_state.bin but no valid WandB run id was found; "
                "gepa.optimize may create a new WandB run"
            )
    start_training_session(args.run_dir, fresh=fresh_session)
    if fresh_session:
        clear_program_prompt_cache(args.run_dir)

    rollout_concurrency = resolve_rollout_concurrency(args.rollout_concurrency)
    logger.info("Rollout concurrency=%d", rollout_concurrency)

    train_temperature = resolve_train_temperature()
    val_temperature = resolve_val_temperature()
    llm_call = make_openai_llm_call(base_url=api_base, model=MODEL_NAME, default_temperature=train_temperature)
    # train=stochastic rollouts during gepa.optimize; val=deterministic for seed/final eval
    # so logged val/em matches eval_gepa_prompt.py and in-process full-test eval.
    train_adapter = SearchR1GEPAAdapter(
        llm_call,
        eval_mode="train",
        train_temperature=train_temperature,
        val_temperature=val_temperature,
        rollout_concurrency=rollout_concurrency,
        use_rewrite=variant.use_rewrite,
        reward_mode=variant.reward_mode,
    )
    val_adapter = SearchR1GEPAAdapter(
        llm_call,
        eval_mode="val",
        train_temperature=train_temperature,
        val_temperature=val_temperature,
        rollout_concurrency=rollout_concurrency,
        use_rewrite=variant.use_rewrite,
        reward_mode=variant.reward_mode,
    )

    seed_candidate = default_seed_candidate(use_rewrite=variant.use_rewrite)
    seed_instruction = INSTRUCTION_FORMAT_REWRITE if variant.use_rewrite else INSTRUCTION_FORMAT
    save_seed_instruction_prompt(args.run_dir, seed_candidate)
    register_program_candidates(args.run_dir, [seed_candidate])
    install_gepa_program_prompt_cache(args.run_dir)
    wandb_config: dict[str, Any] = {
        "baseline": "gepa",
        "variant": variant.name,
        "use_rewrite": variant.use_rewrite,
        "reward_mode": variant.reward_mode,
        "model": MODEL_NAME,
        "data_source": DATA_SOURCE,
        "train_file": TRAIN_FILE,
        "val_file": VAL_FILE,
        "max_metric_calls": max_metric_calls,
        "chunk_metric_calls": chunk_metric_calls,
        "prior_total_evals": prior_total_evals,
        "grpo_step_rollouts": DEFAULT_GRPO_STEP_ROLLOUTS,
        "rollout_concurrency": rollout_concurrency,
        "reflection_minibatch_size": reflection_minibatch_size,
        "seed_instruction": seed_instruction[:200],
    }
    wandb_init_kwargs = build_gepa_wandb_init_kwargs(
        project=WANDB_PROJECT,
        name=variant.wandb_experiment,
        config=wandb_config,
        run_dir=args.run_dir,
        wandb_dir=wandb_dir,
    )

    if resuming_gepa:
        cached_seed = load_seed_val_em(args.run_dir)
        if cached_seed is not None:
            seed_val_em = cached_seed
            logger.info(
                "Resuming: using cached seed val/em=%.4f (skipped redundant seed evaluate_split)",
                seed_val_em,
            )
        else:
            logger.warning("Resuming without gepa_summary.json seed_val_em; re-evaluating seed prompt")
            logger.info("Evaluating seed prompt on val (n=%d)...", len(val_data))
            seed_val_em = evaluate_split(val_adapter, seed_candidate, val_data)
            logger.info("Seed val/em=%.4f", seed_val_em)
    else:
        logger.info("Evaluating seed prompt on val (n=%d)...", len(val_data))
        seed_val_em = evaluate_split(val_adapter, seed_candidate, val_data)
        logger.info("Seed val/em=%.4f", seed_val_em)

    install_gepa_wandb_grpo_compat_patch(
        reflection_minibatch_size=reflection_minibatch_size,
        run_dir=args.run_dir,
        eval_job_tag=variant.eval_job_tag,
        eval_addr_file=variant.eval_addr_file,
        run_dir_rel=f"outputs/{variant.run_dir_name}",
        use_rewrite=variant.use_rewrite,
        reward_mode=variant.reward_mode,
    )
    if not resuming_gepa:
        maybe_trigger_full_eval(
            run_dir=args.run_dir,
            dev_score=seed_val_em,
            metric_calls=0,
            program_idx=0,
            prompt=seed_candidate,
            eval_job_tag=variant.eval_job_tag,
            addr_file=variant.eval_addr_file,
            run_dir_rel=f"outputs/{variant.run_dir_name}",
            use_rewrite=variant.use_rewrite,
            iteration=0,
        )
        # Fresh runs only: seed metrics before gepa.optimize. On resume, gepa re-logs from
        # gepa_state.bin and a separate init+finish would fork a new WandB run.
        log_gepa_wandb_metrics(
            {
                "seed/val_em": seed_val_em,
                "val/em": seed_val_em,
                "val/reward": seed_val_em,
            },
            rollouts=0,
            iteration=0,
            project=WANDB_PROJECT,
            experiment_name=variant.wandb_experiment,
            run_dir=args.run_dir,
            config=wandb_config,
            wandb_dir=wandb_dir,
            finish=True,
        )

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train_data,
        valset=val_data,
        adapter=train_adapter,
        reflection_lm=reflection_lm,
        candidate_selection_strategy="pareto",
        reflection_minibatch_size=reflection_minibatch_size,
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
    best_train_em = evaluate_split(train_adapter, best_candidate, train_data[: min(500, len(train_data))])

    from gepa.core.state import GEPAState

    gepa_state = GEPAState.load(str(args.run_dir))
    final_iteration = gepa_state.i + 1
    final_rollouts = max(int(result.total_metric_calls or 0), 1)

    summary = {
        "best_val_em": best_val_em,
        "best_train_em_sample500": best_train_em,
        "seed_val_em": seed_val_em,
        "total_metric_calls": final_rollouts,
        "final_iteration": final_iteration,
        "best_candidate_keys": list(best_candidate.keys()),
    }
    summary_path = args.run_dir / "gepa_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    prompt_path = args.run_dir / "best_instruction_prompt.txt"
    prompt_path.write_text(best_candidate[INSTRUCTION_COMPONENT])
    save_seed_instruction_prompt(args.run_dir, seed_candidate)

    # Final summary on the rollout-budget x-axis; Step kept as a reference field.
    log_gepa_wandb_metrics(
        {
            "val/em": best_val_em,
            "val/reward": best_val_em,
            "train/em": best_train_em,
            "training/reward": best_train_em,
            "training/em": best_train_em,
            "seed/val_em": seed_val_em,
            "total_metric_calls": final_rollouts,
        },
        rollouts=final_rollouts,
        iteration=final_iteration,
        project=WANDB_PROJECT,
        experiment_name=variant.wandb_experiment,
        run_dir=args.run_dir,
        config=wandb_config,
        wandb_dir=_RECIPE_DIR / "wandb",
        finish=True,
    )

    logger.info("GEPA complete. best val/em=%.4f train/em(sample500)=%.4f", best_val_em, best_train_em)
    logger.info("Best prompt saved to %s", prompt_path)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
