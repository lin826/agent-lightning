# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("gepa")

_RECIPE_DIR = Path(__file__).resolve().parents[4] / "contrib" / "recipes" / "search_r1"
_SCRIPTS_DIR = _RECIPE_DIR / "scripts"
for _path in (_RECIPE_DIR, _SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from search_r1_gepa.train_gepa import (  # noqa: E402
    DEFAULT_GRPO_STEP_ROLLOUTS,
    DEFAULT_REFLECTION_MINIBATCH_SIZE,
    DEFAULT_VAL_FILE,
    GRPO_ROLLOUT_N,
    GRPO_TRAIN_BATCH_SIZE,
    GRPO_VAL_EXAMPLES,
    compute_grpo_step_rollouts,
    load_seed_val_em,
    resolve_chunk_metric_calls,
    resolve_max_metric_calls,
    resolve_reflection_minibatch_size,
    resolve_val_file,
)


def test_default_grpo_step_rollouts_matches_formula() -> None:
    assert compute_grpo_step_rollouts() == DEFAULT_GRPO_STEP_ROLLOUTS
    assert DEFAULT_GRPO_STEP_ROLLOUTS == GRPO_TRAIN_BATCH_SIZE * GRPO_ROLLOUT_N + GRPO_VAL_EXAMPLES
    assert DEFAULT_GRPO_STEP_ROLLOUTS == 9965


def test_resolve_chunk_metric_calls_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEPA_CHUNK_METRIC_CALLS", raising=False)
    assert resolve_chunk_metric_calls(None) is None
    monkeypatch.setenv("GEPA_CHUNK_METRIC_CALLS", "9965")
    assert resolve_chunk_metric_calls(None) == 9965
    assert resolve_chunk_metric_calls(1024) == 1024


def test_resolve_max_metric_calls_chunk_fresh_start(tmp_path: Path) -> None:
    effective, chunk, prior = resolve_max_metric_calls(
        cli_max=None,
        cli_chunk=9965,
        run_dir=tmp_path,
        resuming_gepa=False,
    )
    assert prior == 0
    assert chunk == 9965
    assert effective == 9965


def test_resolve_max_metric_calls_chunk_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "search_r1_gepa.train_gepa.load_gepa_total_evals",
        lambda _run_dir: 9965,
    )
    effective, chunk, prior = resolve_max_metric_calls(
        cli_max=None,
        cli_chunk=9965,
        run_dir=tmp_path,
        resuming_gepa=True,
    )
    assert prior == 9965
    assert chunk == 9965
    assert effective == 19930


def test_resolve_max_metric_calls_absolute_without_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEPA_CHUNK_METRIC_CALLS", raising=False)
    monkeypatch.setenv("GEPA_MAX_METRIC_CALLS", "60000")
    effective, chunk, prior = resolve_max_metric_calls(
        cli_max=None,
        cli_chunk=None,
        run_dir=tmp_path,
        resuming_gepa=False,
    )
    assert chunk is None
    assert prior == 0
    assert effective == 60000


def test_load_seed_val_em_from_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "gepa_summary.json"
    summary_path.write_text('{"seed_val_em": 0.315}')
    assert load_seed_val_em(tmp_path) == pytest.approx(0.315)
    assert load_seed_val_em(tmp_path / "missing") is None


def test_default_reflection_minibatch_size() -> None:
    assert DEFAULT_REFLECTION_MINIBATCH_SIZE == 8
    assert resolve_reflection_minibatch_size(None) == 8


def test_resolve_val_file_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEPA_VAL_DATA", raising=False)
    assert resolve_val_file() == DEFAULT_VAL_FILE
    assert DEFAULT_VAL_FILE == "data/test.parquet"
    monkeypatch.setenv("GEPA_VAL_DATA", "data/test_dev.parquet")
    assert resolve_val_file() == "data/test_dev.parquet"
