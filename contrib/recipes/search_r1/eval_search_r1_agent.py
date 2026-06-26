"""Standalone evaluation script for Search-R1 checkpoints.

Usage:
    python eval_search_r1_agent.py <config> <checkpoint_path>

Examples:
    python eval_search_r1_agent.py qwen7b checkpoints/searchr1_checkpoints/global_step_10/actor
    python eval_search_r1_agent.py qwen3_8b_rewrite checkpoints/searchr1_checkpoints/global_step_20/actor
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from typing import Any, Dict

import pandas as pd
from search_r1_agent import SearchR1Agent, SearchR1RewriteAgent

import agentlightning as agl

from train_search_r1_agent import (
    RL_TRAINING_CONFIG,
    config_train_qwen3_8b,
    config_train_qwen3_8b_rewrite,
    config_train_qwen7b,
)


def make_eval_config(base_config: Dict[str, Any], checkpoint_path: str, step: int) -> Dict[str, Any]:
    """Modify a training config into an eval-only config pointing at a checkpoint."""
    config = deepcopy(base_config)
    config["actor_rollout_ref"]["model"]["path"] = checkpoint_path
    config["data"]["val_files"] = "data/test.parquet"
    config["data_source_filter"] = "hotpotqa"
    config["trainer"]["val_only"] = True
    config["trainer"]["val_before_train"] = True
    config["trainer"]["experiment_name"] = config["trainer"]["experiment_name"] + f"_eval_step{step}"
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Search-R1 checkpoint on full test.parquet (hotpotqa)")
    parser.add_argument(
        "config",
        choices=["qwen7b", "qwen3_8b", "qwen3_8b_rewrite"],
        help="Which experiment config to base evaluation on",
    )
    parser.add_argument("checkpoint_path", help="Path to the actor checkpoint directory (e.g. global_step_N/actor)")
    parser.add_argument("--step", type=int, default=0, help="Training step (used for WandB run naming)")
    args = parser.parse_args()

    config_functions = {
        "qwen7b": config_train_qwen7b,
        "qwen3_8b": config_train_qwen3_8b,
        "qwen3_8b_rewrite": config_train_qwen3_8b_rewrite,
    }

    base_config = config_functions[args.config]()
    config = make_eval_config(base_config, args.checkpoint_path, args.step)

    use_rewrite = args.config == "qwen3_8b_rewrite"
    if use_rewrite:
        agent = SearchR1RewriteAgent()
    else:
        agent = SearchR1Agent()

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
