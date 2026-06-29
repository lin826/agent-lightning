"""Standalone evaluation script for Search-R1 checkpoints.

Usage:
    python eval_search_r1_agent.py <config> <checkpoint_path> [--step N]

Full test.parquet metrics (``test/em``, ``test/reward``) are logged to a dedicated
eval WandB run at the given global step. Eval runs are named ``eval_*`` (e.g.
``eval_rewrite_em``), not the training ``searchr1_*`` names. Resolve the run id from
``WANDB_RUN_ID``, ``{checkpoint_root}/wandb_eval_run_id.txt``, or let VERL create
and persist a new eval run on first launch. Existing ``wandb_eval_run_id.txt`` files
resume the stored run (display name unchanged in WandB); delete the file to start a
fresh eval run with the correct ``eval_*`` name.

Examples:
    python eval_search_r1_agent.py qwen7b checkpoints/searchr1_qwen7b/global_step_10/actor --step 10
    python eval_search_r1_agent.py qwen3_8b_rewrite checkpoints/searchr1_qwen3_8b_rewrite/global_step_20/actor --step 20
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import pandas as pd

import agentlightning as agl

from train_search_r1_agent import (
    build_agent,
    config_train_qwen3_8b,
    config_train_qwen3_8b_rewrite,
    config_train_qwen3_8b_rewrite_em,
    config_train_qwen3_8b_shaped,
    config_train_qwen7b,
)
from wandb_run import (
    checkpoint_root_from_actor,
    resolve_actor_checkpoint,
    resolve_eval_wandb_run_name,
    resolve_wandb_eval_run_id,
    setup_wandb_resume,
)

_RECIPE_DIR = Path(__file__).resolve().parent.parent


def make_eval_config(
    base_config: Dict[str, Any],
    checkpoint_path: str,
    step: int,
    *,
    config_key: str,
    n_gpus: int | None = None,
    sanity: bool = False,
) -> Dict[str, Any]:
    """Modify a training config into an eval-only config pointing at a checkpoint."""
    config = deepcopy(base_config)
    config["trainer"]["experiment_name"] = resolve_eval_wandb_run_name(config_key)
    _actor_dir, global_step_dir = resolve_actor_checkpoint(checkpoint_path)
    # Keep model.path as the base HF model for tokenizer init. VERL entrypoint calls
    # hf_tokenizer(model.path); FSDP actor dirs only store shards under actor/, with
    # tokenizer files in actor/huggingface/. Load trained weights via resume instead.
    config["trainer"]["resume_mode"] = "resume_path"
    config["trainer"]["resume_from_path"] = str(global_step_dir)
    # Eval only needs actor weights; stale checkpoints may omit optimizer shards.
    config["actor_rollout_ref"]["actor"]["checkpoint"] = {"load_contents": ["model"]}
    config["data"]["val_files"] = "data/test.parquet"
    config["data_source_filter"] = "hotpotqa"
    config["trainer"]["val_only"] = True
    config["trainer"]["val_before_train"] = True
    config["trainer"]["eval_global_step"] = step
    config["trainer"]["val_metric_prefix"] = "test"
    if n_gpus is not None:
        config["trainer"]["n_gpus_per_node"] = n_gpus
    if sanity:
        # Single-GPU smoke test: skip ref worker (KL not used in val_only) and shrink batches.
        config["actor_rollout_ref"]["actor"]["use_kl_loss"] = False
        config["actor_rollout_ref"]["actor"]["fsdp_config"]["param_offload"] = True
        config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.45
        config["data"]["train_batch_size"] = 32
        config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 32
        config["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] = 1
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Search-R1 checkpoint on full test.parquet (hotpotqa)")
    parser.add_argument(
        "config",
        choices=["qwen7b", "qwen3_8b", "qwen3_8b_rewrite", "qwen3_8b_rewrite_em", "qwen3_8b_shaped"],
        help="Which experiment config to base evaluation on",
    )
    parser.add_argument("checkpoint_path", help="Path to the actor checkpoint directory (e.g. global_step_N/actor)")
    parser.add_argument("--step", type=int, required=True, help="Training global step to log full-test metrics at")
    parser.add_argument("--wandb-run-id", default=None, help="Override eval WandB run id (default: auto-resolve)")
    parser.add_argument("--n-gpus", type=int, default=None, help="Override trainer.n_gpus_per_node (default: from training config)")
    parser.add_argument("--n-runners", type=int, default=None, help="Parallel agent rollout workers (default: 32, or 4 with --sanity)")
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Cap validation set size for smoke tests (default: full test.parquet hotpotqa split)",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="1-GPU smoke-test overrides: disable ref/KL, lower vLLM memory, smaller batches",
    )
    args = parser.parse_args()

    config_functions = {
        "qwen7b": config_train_qwen7b,
        "qwen3_8b": config_train_qwen3_8b,
        "qwen3_8b_rewrite": config_train_qwen3_8b_rewrite,
        "qwen3_8b_rewrite_em": config_train_qwen3_8b_rewrite_em,
        "qwen3_8b_shaped": config_train_qwen3_8b_shaped,
    }

    base_config = config_functions[args.config]()
    try:
        actor_dir, _global_step_dir = resolve_actor_checkpoint(args.checkpoint_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Invalid checkpoint path: {exc}") from exc
    checkpoint_path = actor_dir
    checkpoint_root = checkpoint_root_from_actor(checkpoint_path)
    eval_run_name = resolve_eval_wandb_run_name(args.config)
    run_id = args.wandb_run_id or resolve_wandb_eval_run_id(checkpoint_dir=checkpoint_root)
    if run_id:
        setup_wandb_resume(run_id, run_name=eval_run_name)
        print(
            f"Resuming WandB eval run {run_id} ({eval_run_name}) for full-test eval at step {args.step}"
        )
    else:
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_RESUME", None)
        os.environ["WANDB_NAME"] = eval_run_name
        print(
            f"No eval WandB run id found; eval will create run {eval_run_name!r} and persist its id"
        )

    config = make_eval_config(
        base_config,
        str(checkpoint_path),
        args.step,
        config_key=args.config,
        n_gpus=1 if args.sanity and args.n_gpus is None else args.n_gpus,
        sanity=args.sanity,
    )

    agent = build_agent(args.config)

    n_runners = args.n_runners
    if n_runners is None:
        n_runners = 4 if args.sanity else 32
    print(f"Using n_gpus_per_node={config['trainer']['n_gpus_per_node']}, n_runners={n_runners}, sanity={args.sanity}")

    algorithm = agl.VERL(config)
    trainer = agl.Trainer(n_runners=n_runners, algorithm=algorithm)

    val_df = pd.read_parquet(config["data"]["val_files"])
    if config.get("data_source_filter"):
        source = config["data_source_filter"]
        val_df = val_df[val_df["data_source"] == source]
        print(f"Eval on data_source='{source}': n={len(val_df)}")
    if args.max_val_samples is not None:
        val_df = val_df.head(args.max_val_samples)
        print(f"Capped validation set to max_val_samples={args.max_val_samples}")

    val_data = val_df.to_dict(orient="records")
    trainer.fit(agent, train_dataset=[], val_dataset=val_data)


if __name__ == "__main__":
    main()
