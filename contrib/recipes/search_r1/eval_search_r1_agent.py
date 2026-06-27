"""Standalone evaluation script for Search-R1 checkpoints.

Usage:
    python eval_search_r1_agent.py <config> <checkpoint_path> [--step N]

Full test.parquet metrics (``test/em``, ``test/reward``) are logged to the original
training WandB run at the given global step. Resolve the run id from
``WANDB_RUN_ID``, ``{checkpoint_root}/wandb_run_id.txt``, or local ``wandb/`` logs.

Examples:
    python eval_search_r1_agent.py qwen7b checkpoints/searchr1_qwen7b/global_step_10/actor --step 10
    python eval_search_r1_agent.py qwen3_8b_rewrite checkpoints/searchr1_qwen3_8b_rewrite/global_step_20/actor --step 20
"""

from __future__ import annotations

import argparse
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
from wandb_run import checkpoint_root_from_actor, find_wandb_run_id_from_local, resolve_wandb_run_id, setup_wandb_resume

_RECIPE_DIR = Path(__file__).resolve().parent
_GRPO_TRAIN_WANDB_DIR = (
    Path("/proj/inf-scaling/zwhong/projs/asmi/agent-lightning/.claude/worktrees")
    / "feature+searchr1-qwen25-repro-qwen3-eval/contrib/recipes/search_r1/wandb"
)


def make_eval_config(base_config: Dict[str, Any], checkpoint_path: str, step: int) -> Dict[str, Any]:
    """Modify a training config into an eval-only config pointing at a checkpoint."""
    config = deepcopy(base_config)
    config["actor_rollout_ref"]["model"]["path"] = checkpoint_path
    config["data"]["val_files"] = "data/test.parquet"
    config["data_source_filter"] = "hotpotqa"
    config["trainer"]["val_only"] = True
    config["trainer"]["val_before_train"] = True
    config["trainer"]["eval_global_step"] = step
    config["trainer"]["val_metric_prefix"] = "test"
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
    parser.add_argument("--wandb-run-id", default=None, help="Override WandB run id (default: auto-resolve)")
    args = parser.parse_args()

    config_functions = {
        "qwen7b": config_train_qwen7b,
        "qwen3_8b": config_train_qwen3_8b,
        "qwen3_8b_rewrite": config_train_qwen3_8b_rewrite,
        "qwen3_8b_rewrite_em": config_train_qwen3_8b_rewrite_em,
        "qwen3_8b_shaped": config_train_qwen3_8b_shaped,
    }

    base_config = config_functions[args.config]()
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_root = checkpoint_root_from_actor(checkpoint_path)
    run_id = args.wandb_run_id or resolve_wandb_run_id(
        checkpoint_dir=checkpoint_root,
        experiment_name=base_config["trainer"]["experiment_name"],
        wandb_dir=_RECIPE_DIR / "wandb",
    )
    if not run_id:
        for wandb_dir in (_RECIPE_DIR / "wandb", _GRPO_TRAIN_WANDB_DIR):
            run_id = find_wandb_run_id_from_local(wandb_dir, base_config["trainer"]["experiment_name"])
            if run_id:
                break
    if run_id:
        setup_wandb_resume(run_id)
        print(f"Resuming WandB run {run_id} for full-test eval at step {args.step}")
    else:
        print(
            "WARNING: WandB run id not found (set WANDB_RUN_ID or save wandb_run_id.txt under checkpoint root); "
            "eval may create a separate run"
        )

    config = make_eval_config(base_config, str(checkpoint_path), args.step)

    agent = build_agent(args.config)

    algorithm = agl.VERL(config)
    trainer = agl.Trainer(n_runners=32, algorithm=algorithm)

    val_df = pd.read_parquet(config["data"]["val_files"])
    if config.get("data_source_filter"):
        source = config["data_source_filter"]
        val_df = val_df[val_df["data_source"] == source]
        print(f"Eval on data_source='{source}': n={len(val_df)}")

    val_data = val_df.to_dict(orient="records")
    trainer.fit(agent, train_dataset=[], val_dataset=val_data)


if __name__ == "__main__":
    main()
