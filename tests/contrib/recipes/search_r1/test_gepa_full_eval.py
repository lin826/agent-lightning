# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

SEARCH_R1_RECIPE = Path(__file__).resolve().parents[4] / "contrib" / "recipes" / "search_r1"
SEARCH_R1_SCRIPTS = SEARCH_R1_RECIPE / "scripts"
for _path in (SEARCH_R1_RECIPE, SEARCH_R1_SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gepa_full_eval  # noqa: E402

INSTRUCTION_COMPONENT = "instruction_prompt"


@pytest.fixture(autouse=True)
def mock_save_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_save(run_dir: Path, metric_calls: int, program_idx: int, prompt: dict[str, str]) -> Path:
        path = run_dir / "monitored_prompts" / f"best_m{metric_calls}_idx{program_idx}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(prompt.get(INSTRUCTION_COMPONENT, ""))
        return path

    monkeypatch.setattr(gepa_full_eval, "save_gepa_prompt", fake_save)


@pytest.fixture(autouse=True)
def mock_submit_gepa_eval_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent accidental real ``bsub`` when tests seed state before per-test mocks."""

    def fake_submit(*, dry_run: bool = False, **kwargs: Any) -> str | None:
        if dry_run:
            return None
        return "test-job"

    monkeypatch.setattr(gepa_full_eval, "submit_gepa_eval_job", fake_submit)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    session = gepa_full_eval.start_training_session(tmp_path, fresh=True)
    assert session.session_id
    return tmp_path


def test_maybe_trigger_full_eval_on_new_best(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    submitted: list[int] = []

    def fake_submit(**kwargs: Any) -> str:
        submitted.append(kwargs["metric_calls"])
        return "12345"

    monkeypatch.setattr(gepa_full_eval, "submit_gepa_eval_job", fake_submit)
    prompt = {INSTRUCTION_COMPONENT: "test prompt"}
    assert gepa_full_eval.maybe_trigger_full_eval(
        run_dir=run_dir,
        dev_score=0.30,
        metric_calls=42,
        program_idx=1,
        prompt=prompt,
        dry_run=False,
    )
    assert submitted == [42]
    state = gepa_full_eval.load_full_eval_state(run_dir)
    assert state.best_dev_score == pytest.approx(0.30)
    assert state.submitted_metric_calls == [42]


def test_maybe_trigger_full_eval_skips_non_record(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gepa_full_eval, "submit_gepa_eval_job", MagicMock(return_value="1"))
    prompt = {INSTRUCTION_COMPONENT: "test prompt"}
    gepa_full_eval.maybe_trigger_full_eval(
        run_dir=run_dir,
        dev_score=0.30,
        metric_calls=10,
        program_idx=0,
        prompt=prompt,
    )
    assert not gepa_full_eval.maybe_trigger_full_eval(
        run_dir=run_dir,
        dev_score=0.28,
        metric_calls=20,
        program_idx=0,
        prompt=prompt,
    )


def test_maybe_trigger_full_eval_skips_tie_score(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    submitted: list[int] = []

    def fake_submit(**kwargs: Any) -> str:
        submitted.append(kwargs["metric_calls"])
        return "1"

    monkeypatch.setattr(gepa_full_eval, "submit_gepa_eval_job", fake_submit)
    prompt = {INSTRUCTION_COMPONENT: "test prompt"}
    gepa_full_eval.maybe_trigger_full_eval(
        run_dir=run_dir,
        dev_score=0.30,
        metric_calls=10,
        program_idx=0,
        prompt=prompt,
    )
    assert not gepa_full_eval.maybe_trigger_full_eval(
        run_dir=run_dir,
        dev_score=0.30,
        metric_calls=20,
        program_idx=1,
        prompt=prompt,
    )
    assert submitted == [10]


def test_maybe_trigger_from_metrics_skips_val_program_average_only(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[int] = []

    def fake_submit(**kwargs: Any) -> str:
        submitted.append(kwargs["metric_calls"])
        return "99"

    monkeypatch.setattr(gepa_full_eval, "submit_gepa_eval_job", fake_submit)
    monkeypatch.setattr(
        gepa_full_eval,
        "_resolve_program_prompt",
        lambda *_args, **_kwargs: {INSTRUCTION_COMPONENT: "candidate"},
    )

    metrics = {
        "val_program_average": 0.31,
        "new_program_idx": 2,
        "total_metric_calls": 500,
    }
    assert not gepa_full_eval.maybe_trigger_full_eval_from_metrics(
        metrics,
        run_dir,
        step=500,
        dry_run=False,
    )
    assert submitted == []


def test_maybe_trigger_from_metrics_best_score_on_record(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = {INSTRUCTION_COMPONENT: "seed"}
    gepa_full_eval.maybe_trigger_full_eval(
        run_dir=run_dir,
        dev_score=0.30,
        metric_calls=10,
        program_idx=0,
        prompt=prompt,
        dry_run=True,
    )
    submitted: list[int] = []

    def fake_submit(**kwargs: Any) -> str:
        submitted.append(kwargs["metric_calls"])
        return "88"

    monkeypatch.setattr(gepa_full_eval, "submit_gepa_eval_job", fake_submit)
    monkeypatch.setattr(
        gepa_full_eval,
        "_resolve_program_prompt",
        lambda *_args, **_kwargs: {INSTRUCTION_COMPONENT: "best"},
    )

    metrics = {
        "val_program_average": 0.28,
        "best_score_on_valset": 0.32,
        "best_program_as_per_agg_score_valset": 1,
        "total_metric_calls": 600,
    }
    assert gepa_full_eval.maybe_trigger_full_eval_from_metrics(
        metrics,
        run_dir,
        step=600,
        dry_run=False,
    )
    assert submitted == [600]


def test_collect_full_eval_trigger_candidates_uses_best_keys_only(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = {
        "val_program_average": 0.22,
        "new_program_idx": 3,
        "best_score_on_valset": 0.24,
        "best_program_as_per_agg_score_valset": 0,
        "total_metric_calls": 100,
    }
    monkeypatch.setattr(
        gepa_full_eval,
        "load_best_from_gepa_state",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        gepa_full_eval,
        "_resolve_program_prompt",
        lambda *_args, **_kwargs: {INSTRUCTION_COMPONENT: "p"},
    )
    candidates = gepa_full_eval.collect_full_eval_trigger_candidates(
        metrics,
        run_dir,
        step=100,
        use_rewrite=False,
    )
    assert len(candidates) == 1
    assert candidates[0][0] == pytest.approx(0.24)


def test_sync_monitor_state_from_full_eval(run_dir: Path) -> None:
    gepa_full_eval.maybe_trigger_full_eval(
        run_dir=run_dir,
        dev_score=0.31,
        metric_calls=120,
        program_idx=2,
        prompt={INSTRUCTION_COMPONENT: "best"},
        dry_run=True,
    )
    score, step, prog, submitted = gepa_full_eval.sync_monitor_state_from_full_eval(
        -1.0,
        -1,
        -1,
        [],
        run_dir,
    )
    assert score == pytest.approx(0.31)
    assert step == 120
    assert prog == 2
    assert submitted == [120]


def test_resolve_best_metric_calls_prefers_full_eval_state(run_dir: Path) -> None:
    gepa_full_eval.save_full_eval_state(
        run_dir,
        gepa_full_eval.FullEvalState(
            best_dev_score=0.29,
            best_metric_calls=1631,
            best_program_idx=4,
            submitted_metric_calls=[1631],
            training_session_id=gepa_full_eval.load_training_session(run_dir).session_id,
        ),
    )
    assert gepa_full_eval.resolve_best_metric_calls(run_dir, 0.29, 2569) == 1631
    assert gepa_full_eval.resolve_best_metric_calls(run_dir, 0.30, 2569) == 2569


def test_maybe_trigger_resolves_negative_program_idx(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved: list[int] = []

    def fake_save(run_dir: Path, metric_calls: int, program_idx: int, prompt: dict[str, str]) -> Path:
        saved.append(program_idx)
        path = run_dir / "monitored_prompts" / f"best_m{metric_calls}_idx{program_idx}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(prompt.get(INSTRUCTION_COMPONENT, ""))
        return path

    monkeypatch.setattr(gepa_full_eval, "save_gepa_prompt", fake_save)
    monkeypatch.setattr(gepa_full_eval, "submit_gepa_eval_job", lambda **_kwargs: "1")
    monkeypatch.setattr(
        gepa_full_eval,
        "load_best_from_gepa_state",
        lambda *_args, **_kwargs: (0.33, 200, 5, {INSTRUCTION_COMPONENT: "from-state"}),
    )

    assert gepa_full_eval.maybe_trigger_full_eval(
        run_dir=run_dir,
        dev_score=0.33,
        metric_calls=200,
        program_idx=-1,
        prompt={INSTRUCTION_COMPONENT: "ignored"},
    )
    assert saved == [5]


def test_bootstrap_full_eval_state_requires_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "gepa_run"
    run_dir.mkdir()
    scores_path = tmp_path / "best_val_scores.json"
    scores_path.write_text(
        json.dumps(
            {
                "gepa_qwen25_3b": {
                    "best_score": 0.29,
                    "best_step": 12,
                    "eval_submitted_steps": [12],
                }
            }
        )
    )
    monkeypatch.delenv("GEPA_BOOTSTRAP_FULL_EVAL_STATE", raising=False)
    assert gepa_full_eval.bootstrap_full_eval_state(run_dir, monitor_scores_path=scores_path) is None

    monkeypatch.setenv("GEPA_BOOTSTRAP_FULL_EVAL_STATE", "1")
    state = gepa_full_eval.bootstrap_full_eval_state(run_dir, monitor_scores_path=scores_path)
    assert state is not None
    assert state.best_dev_score == pytest.approx(0.29)
    assert state.submitted_metric_calls == [12]
