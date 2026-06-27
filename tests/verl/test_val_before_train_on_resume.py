# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from omegaconf import OmegaConf

from agentlightning.verl.trainer import AgentLightningTrainer


class _TrainerStub(AgentLightningTrainer):
    """Minimal stub to exercise val-before-train gating without Ray workers."""

    def __init__(self, trainer_cfg: dict | None = None, *, global_steps: int = 0) -> None:
        self.config = OmegaConf.create({"trainer": trainer_cfg or {}})
        self.global_steps = global_steps
        self.val_reward_fn = object()


def test_val_before_train_runs_on_fresh_start() -> None:
    trainer = _TrainerStub({"val_before_train": True}, global_steps=0)
    assert trainer._should_run_val_before_train() is True


def test_val_before_train_skipped_when_resuming_training() -> None:
    trainer = _TrainerStub({"val_before_train": True}, global_steps=10)
    assert trainer._should_run_val_before_train() is False


def test_val_before_train_runs_for_val_only_eval() -> None:
    trainer = _TrainerStub({"val_before_train": True, "val_only": True}, global_steps=30)
    assert trainer._should_run_val_before_train() is True


def test_val_before_train_respects_explicit_disable() -> None:
    trainer = _TrainerStub({"val_before_train": False}, global_steps=0)
    assert trainer._should_run_val_before_train() is False
