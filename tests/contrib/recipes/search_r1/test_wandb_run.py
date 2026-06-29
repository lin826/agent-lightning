# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import os
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
    wandb_run.save_wandb_eval_run_id(baseline_dir, "eval_baseline_id")
    wandb_run.save_wandb_eval_run_id(shaped_dir, "eval_shaped_id")
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    assert wandb_run.resolve_wandb_eval_run_id(checkpoint_dir=baseline_dir) == "eval_baseline_id"
    assert wandb_run.resolve_wandb_eval_run_id(checkpoint_dir=shaped_dir) == "eval_shaped_id"


@pytest.mark.parametrize(
    ("config_key", "expected_name"),
    [
        ("qwen7b", "eval_baseline"),
        ("qwen3_8b", "eval_baseline_a"),
        ("qwen3_8b_rewrite", "eval_rewrite"),
        ("qwen3_8b_rewrite_em", "eval_rewrite_em"),
        ("qwen3_8b_shaped", "eval_shaped"),
    ],
)
def test_resolve_eval_wandb_run_name(config_key: str, expected_name: str) -> None:
    assert wandb_run.resolve_eval_wandb_run_name(config_key) == expected_name


def test_resolve_eval_wandb_run_name_unknown_key() -> None:
    with pytest.raises(KeyError, match="unknown_variant"):
        wandb_run.resolve_eval_wandb_run_name("unknown_variant")


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


def test_resolve_gepa_wandb_run_id_falls_back_to_local_wandb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wandb_run.save_wandb_run_id(tmp_path, "deleted_run")
    wandb_dir = tmp_path / "wandb"
    run_dir = wandb_dir / "run-20260628_043054-sq5hdz51"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "debug.log").write_text("Syncing run searchr1_qwen25_3b_gepa\n")

    monkeypatch.setattr(
        wandb_run,
        "validate_wandb_run_id",
        lambda run_id, **kwargs: run_id if run_id == "sq5hdz51" else None,
    )

    resolved = wandb_run.resolve_gepa_wandb_run_id(
        project="AgentLightning",
        experiment_name="searchr1_qwen25_3b_gepa",
        run_dir=tmp_path,
        wandb_dir=wandb_dir,
    )
    assert resolved == "sq5hdz51"


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
    tracker.log_metrics({"val_program_average": 0.24, "subsample_score": 2.0}, step=42)

    assert logged[0][0]["val_program_average"] == 0.24
    assert logged[1][0]["val/reward"] == pytest.approx(0.24)
    assert logged[1][0]["val/em"] == pytest.approx(0.24)
    assert logged[1][0]["training/reward"] == pytest.approx(2.0 / 3.0)
    assert logged[1][0]["training/em"] == pytest.approx(2.0 / 3.0)


def test_install_gepa_wandb_grpo_compat_patch_maps_base_program_val(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("gepa")
    from gepa.logging.experiment_tracker import ExperimentTracker

    if hasattr(ExperimentTracker, "_agl_grpo_compat_patched"):
        delattr(ExperimentTracker, "_agl_grpo_compat_patched")

    logged: list[tuple[dict[str, Any], int | None]] = []

    def fake_log_metrics(self: ExperimentTracker, metrics: dict[str, Any], step: int | None = None) -> None:
        logged.append((metrics, step))

    monkeypatch.setattr(ExperimentTracker, "log_metrics", fake_log_metrics)
    wandb_run.install_gepa_wandb_grpo_compat_patch(reflection_minibatch_size=3)

    tracker = ExperimentTracker(use_wandb=True)
    tracker.log_metrics({"base_program_full_valset_score": 0.26}, step=1)

    val_logs = [entry for entry, _ in logged if "val/em" in entry]
    assert val_logs[-1]["val/em"] == pytest.approx(0.26)
    assert val_logs[-1]["val/reward"] == pytest.approx(0.26)


def test_install_gepa_wandb_grpo_compat_patch_prefers_per_step_val_over_pareto(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("gepa")
    from gepa.logging.experiment_tracker import ExperimentTracker

    if hasattr(ExperimentTracker, "_agl_grpo_compat_patched"):
        delattr(ExperimentTracker, "_agl_grpo_compat_patched")

    logged: list[tuple[dict[str, Any], int | None]] = []

    def fake_log_metrics(self: ExperimentTracker, metrics: dict[str, Any], step: int | None = None) -> None:
        logged.append((metrics, step))

    monkeypatch.setattr(ExperimentTracker, "log_metrics", fake_log_metrics)
    wandb_run.install_gepa_wandb_grpo_compat_patch(reflection_minibatch_size=3)

    tracker = ExperimentTracker(use_wandb=True)
    tracker.log_metrics(
        {
            "val_program_average": 0.22,
            "valset_pareto_front_agg": 0.385,
            "best_score_on_valset": 0.235,
            "subsample_score": 1.0,
        },
        step=100,
    )

    val_em_logs = [entry for entry, _ in logged if "val/em" in entry]
    assert val_em_logs[-1]["val/em"] == pytest.approx(0.22)
    assert val_em_logs[-1]["val/reward"] == pytest.approx(0.22)
    assert val_em_logs[-1]["val/pareto_front_agg"] == pytest.approx(0.385)
    assert val_em_logs[-1]["val/best_single_program_em"] == pytest.approx(0.235)


def test_ensure_gepa_wandb_rollouts_axis_defined_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    wandb_run._gepa_wandb_rollouts_axis_defined_run_ids.clear()
    define_calls: list[tuple[str, ...]] = []

    fake_run = MagicMock()
    fake_run.id = "run_abc"

    fake_wandb = MagicMock()
    fake_wandb.run = fake_run

    def fake_define_metric(name: str, **kwargs: object) -> None:
        define_calls.append((name, *sorted(kwargs.items())))

    fake_wandb.define_metric = fake_define_metric
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    wandb_run.ensure_gepa_wandb_rollouts_axis_defined()
    wandb_run.ensure_gepa_wandb_rollouts_axis_defined()

    expected = [("rollouts", ("summary", "max")), ("iteration", ("summary", "max"))]
    expected.extend((metric, ("step_metric", "rollouts")) for metric in wandb_run.GEPA_ROLLOUT_AXIS_METRICS)
    assert define_calls == expected
    assert not any(name.endswith("@rollouts") for name, *_ in define_calls)

    wandb_run._gepa_wandb_rollouts_axis_defined_run_ids.clear()


def test_log_gepa_wandb_metrics_logs_on_rollouts_axis(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wandb_run._gepa_wandb_rollouts_axis_defined_run_ids.clear()
    logged: list[dict[str, float]] = []
    define_calls: list[str] = []

    fake_run = MagicMock()
    fake_run.id = "gepa_run_rollouts"

    fake_wandb = MagicMock()
    fake_wandb.run = fake_run

    def fake_init(**kwargs: object) -> None:
        fake_wandb.run = fake_run

    def fake_log(metrics: dict[str, float], **kwargs: object) -> None:
        logged.append(metrics)

    def fake_define_metric(name: str, **kwargs: object) -> None:
        define_calls.append(name)

    fake_wandb.init = fake_init
    fake_wandb.log = fake_log
    fake_wandb.define_metric = fake_define_metric
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    wandb_run.log_gepa_wandb_metrics(
        {"val/em": 0.5, "total_metric_calls": 120},
        rollouts=120,
        iteration=3,
        project="AgentLightning",
        experiment_name="searchr1_gepa_test",
        run_dir=tmp_path,
    )

    assert "rollouts" in define_calls
    assert "iteration" in define_calls
    assert "val/em" in define_calls
    assert not any(name.endswith("@rollouts") for name in define_calls)
    assert logged[0]["rollouts"] == 120
    assert logged[0]["iteration"] == 3
    assert logged[0]["val/em"] == 0.5
    assert "val/em@rollouts" not in logged[0]

    wandb_run._gepa_wandb_rollouts_axis_defined_run_ids.clear()


def test_install_gepa_wandb_grpo_compat_patch_logs_grpo_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("gepa")
    from gepa.logging.experiment_tracker import ExperimentTracker

    wandb_run._gepa_wandb_rollouts_axis_defined_run_ids.clear()
    if hasattr(ExperimentTracker, "_agl_grpo_compat_patched"):
        delattr(ExperimentTracker, "_agl_grpo_compat_patched")

    logged: list[tuple[dict[str, Any], int | None]] = []
    rollout_logs: list[dict[str, Any]] = []

    def fake_log_metrics(self: ExperimentTracker, metrics: dict[str, Any], step: int | None = None) -> None:
        logged.append((metrics, step))

    fake_wandb = MagicMock()
    fake_wandb.run = MagicMock(id="compat_run")
    fake_wandb.define_metric = MagicMock()
    fake_wandb.log = lambda payload, **kwargs: rollout_logs.append(payload)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(ExperimentTracker, "log_metrics", fake_log_metrics)
    wandb_run.install_gepa_wandb_grpo_compat_patch(reflection_minibatch_size=3)

    tracker = ExperimentTracker(use_wandb=True)
    tracker.log_metrics(
        {"val_program_average": 0.24, "subsample_score": 2.0, "total_metric_calls": 87},
        step=42,
    )

    assert logged[0][0]["rollouts"] == 87
    assert logged[0][0]["iteration"] == 42
    compat_payload = rollout_logs[0]
    assert compat_payload["val/reward"] == pytest.approx(0.24)
    assert compat_payload["training/reward"] == pytest.approx(2.0 / 3.0)
    assert compat_payload["rollouts"] == 87
    assert compat_payload["iteration"] == 42
    assert compat_payload["total_metric_calls"] == 87
    assert "val/reward@rollouts" not in compat_payload

    wandb_run._gepa_wandb_rollouts_axis_defined_run_ids.clear()


def test_install_gepa_wandb_grpo_compat_patch_historical_best_does_not_clobber_per_step_val(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("gepa")
    from gepa.logging.experiment_tracker import ExperimentTracker

    if hasattr(ExperimentTracker, "_agl_grpo_compat_patched"):
        delattr(ExperimentTracker, "_agl_grpo_compat_patched")

    logged: list[tuple[dict[str, Any], int | None]] = []

    def fake_log_metrics(self: ExperimentTracker, metrics: dict[str, Any], step: int | None = None) -> None:
        logged.append((metrics, step))

    monkeypatch.setattr(ExperimentTracker, "log_metrics", fake_log_metrics)
    wandb_run.install_gepa_wandb_grpo_compat_patch(reflection_minibatch_size=3)

    tracker = ExperimentTracker(use_wandb=True)
    tracker.log_metrics({"val_program_average": 0.22, "valset_pareto_front_agg": 0.385}, step=100)
    tracker.log_metrics({"best_score_on_valset": 0.235}, step=100)

    val_em_logs = [entry for entry, _ in logged if "val/em" in entry]
    assert len(val_em_logs) == 1
    assert val_em_logs[0]["val/em"] == pytest.approx(0.22)
    assert val_em_logs[0]["val/pareto_front_agg"] == pytest.approx(0.385)

    best_logs = [entry for entry, _ in logged if "val/best_single_program_em" in entry]
    assert best_logs[-1]["val/best_single_program_em"] == pytest.approx(0.235)
    assert "val/em" not in best_logs[-1]


def test_install_gepa_wandb_grpo_compat_patch_triggers_full_eval_on_new_best(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("gepa")
    from gepa.logging.experiment_tracker import ExperimentTracker
    from gepa_full_eval import maybe_trigger_full_eval, start_training_session

    if hasattr(ExperimentTracker, "_agl_grpo_compat_patched"):
        delattr(ExperimentTracker, "_agl_grpo_compat_patched")

    run_dir = tmp_path / "gepa_run"
    start_training_session(run_dir, fresh=True)
    maybe_trigger_full_eval(
        run_dir=run_dir,
        dev_score=0.30,
        metric_calls=10,
        program_idx=0,
        prompt={"instruction_prompt": "seed"},
        dry_run=True,
    )

    triggered: list[dict[str, Any]] = []

    def fake_trigger(metrics: dict[str, Any], run_dir_arg: Path, **kwargs: Any) -> bool:
        triggered.append({"metrics": metrics, "run_dir": run_dir_arg, **kwargs})
        return True

    monkeypatch.setattr("gepa_full_eval.maybe_trigger_full_eval_from_metrics", fake_trigger)

    def fake_log_metrics(self: ExperimentTracker, metrics: dict[str, Any], step: int | None = None) -> None:
        return None

    monkeypatch.setattr(ExperimentTracker, "log_metrics", fake_log_metrics)
    wandb_run.install_gepa_wandb_grpo_compat_patch(
        reflection_minibatch_size=3,
        run_dir=run_dir,
        eval_job_tag="qwen25_3b_gepa",
    )

    tracker = ExperimentTracker(use_wandb=True)
    tracker.log_metrics(
        {
            "val_program_average": 0.28,
            "best_score_on_valset": 0.32,
            "total_metric_calls": 600,
        },
        step=42,
    )

    assert len(triggered) == 1
    assert triggered[0]["metrics"]["best_score_on_valset"] == pytest.approx(0.32)
    assert triggered[0]["run_dir"] == run_dir
    assert triggered[0]["step"] == 42

    tracker.log_metrics({"val_program_average": 0.27}, step=43)
    assert len(triggered) == 1


def test_resolve_gepa_iteration_for_metric_calls_uses_full_eval_state(tmp_path: Path) -> None:
    from gepa_full_eval import FullEvalState, save_full_eval_state

    run_dir = tmp_path / "gepa_run"
    run_dir.mkdir()
    save_full_eval_state(
        run_dir,
        FullEvalState(iteration_by_metric_calls={1631: 45}),
    )
    assert wandb_run.resolve_gepa_iteration_for_metric_calls(run_dir, 1631) == 45
    assert wandb_run.resolve_gepa_iteration_for_metric_calls(run_dir, 0) == 0


def test_setup_wandb_resume_sets_run_name() -> None:
    saved = {key: os.environ.get(key) for key in ("WANDB_NAME", "WANDB_RUN_ID", "WANDB_RESUME")}
    try:
        for key in saved:
            os.environ.pop(key, None)
        wandb_run.setup_wandb_resume("run123", run_name="eval_rewrite_em")
        assert os.environ["WANDB_RUN_ID"] == "run123"
        assert os.environ["WANDB_RESUME"] == "allow"
        assert os.environ["WANDB_NAME"] == "eval_rewrite_em"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
