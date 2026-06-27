# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

SEARCH_R1_SCRIPTS = Path(__file__).resolve().parents[4] / "contrib" / "recipes" / "search_r1" / "scripts"
sys.path.insert(0, str(SEARCH_R1_SCRIPTS))

import wandb_run  # noqa: E402


def test_save_and_load_wandb_run_id(tmp_path: Path) -> None:
    path = wandb_run.save_wandb_run_id(tmp_path, "abc123")
    assert path.read_text().strip() == "abc123"
    assert wandb_run.load_wandb_run_id(tmp_path) == "abc123"


def test_resolve_wandb_run_id_prefers_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wandb_run.save_wandb_run_id(tmp_path, "from_file")
    monkeypatch.setenv("WANDB_RUN_ID", "from_env")
    assert wandb_run.resolve_wandb_run_id(run_dir=tmp_path) == "from_env"


def test_save_and_load_wandb_eval_run_id(tmp_path: Path) -> None:
    path = wandb_run.save_wandb_eval_run_id(tmp_path, "eval456")
    assert path.read_text().strip() == "eval456"
    assert wandb_run.load_wandb_eval_run_id(tmp_path) == "eval456"


def test_resolve_wandb_eval_run_id_ignores_training_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wandb_run.save_wandb_run_id(tmp_path, "train_run")
    wandb_run.save_wandb_eval_run_id(tmp_path, "eval_run")
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    assert wandb_run.resolve_wandb_eval_run_id(run_dir=tmp_path) == "eval_run"


def test_resolve_wandb_eval_run_id_skips_training_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wandb_run.save_wandb_run_id(tmp_path, "train_run")
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    assert wandb_run.resolve_wandb_eval_run_id(checkpoint_dir=tmp_path) is None


def test_resolve_wandb_eval_run_id_isolated_per_variant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    baseline_dir = tmp_path / "searchr1_qwen7b"
    shaped_dir = tmp_path / "searchr1_qwen3_8b_shaped"
    wandb_run.save_wandb_eval_run_id(baseline_dir, "eval_baseline")
    wandb_run.save_wandb_eval_run_id(shaped_dir, "eval_shaped")
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    assert wandb_run.resolve_wandb_eval_run_id(checkpoint_dir=baseline_dir) == "eval_baseline"
    assert wandb_run.resolve_wandb_eval_run_id(checkpoint_dir=shaped_dir) == "eval_shaped"


def test_build_gepa_wandb_init_kwargs_resumes_from_file(tmp_path: Path) -> None:
    wandb_run.save_wandb_run_id(tmp_path, "gepa_run_1")
    kwargs = wandb_run.build_gepa_wandb_init_kwargs(
        project="AgentLightning",
        name="searchr1_qwen25_3b_gepa",
        config={"baseline": "gepa"},
        run_dir=tmp_path,
    )
    assert kwargs["id"] == "gepa_run_1"
    assert kwargs["resume"] == "allow"


def test_install_gepa_wandb_grpo_compat_patch_mirrors_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("gepa")
    from gepa.logging.experiment_tracker import ExperimentTracker

    # Reset patch flag so the test can re-install.
    if hasattr(ExperimentTracker, "_agl_grpo_compat_patched"):
        delattr(ExperimentTracker, "_agl_grpo_compat_patched")

    logged: list[tuple[dict[str, Any], int | None]] = []

    def fake_log_metrics(self: ExperimentTracker, metrics: dict[str, Any], step: int | None = None) -> None:
        logged.append((metrics, step))

    monkeypatch.setattr(ExperimentTracker, "log_metrics", fake_log_metrics)
    wandb_run.install_gepa_wandb_grpo_compat_patch(reflection_minibatch_size=3)

    tracker = ExperimentTracker(use_wandb=True)
    tracker.log_metrics({"best_valset_agg_score": 0.24, "subsample_score": 2.0}, step=42)

    assert logged[0][0]["best_valset_agg_score"] == 0.24
    assert logged[1][0]["val/reward"] == pytest.approx(0.24)
    assert logged[1][0]["val/em"] == pytest.approx(0.24)
    assert logged[2][0]["training/reward"] == pytest.approx(2.0 / 3.0)
    assert logged[2][0]["training/em"] == pytest.approx(2.0 / 3.0)
