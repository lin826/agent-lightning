# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from omegaconf import OmegaConf

from agentlightning.verl.trainer import AgentLightningTrainer


class _TrainerStub(AgentLightningTrainer):
    """Minimal stub to exercise best-val helpers without Ray workers."""

    def __init__(self, trainer_cfg: dict | None = None) -> None:
        self.config = OmegaConf.create({"trainer": trainer_cfg or {}})


def test_get_val_reward_reads_default_metric() -> None:
    trainer = _TrainerStub()
    assert trainer._get_val_reward({"val/reward": 0.31}) == 0.31


def test_get_val_reward_reads_prefixed_metric() -> None:
    trainer = _TrainerStub({"val_metric_prefix": "test", "save_on_best_val_metric": "val/reward"})
    assert trainer._get_val_reward({"test/reward": 0.42}) == 0.42


def test_get_val_reward_returns_none_when_missing() -> None:
    trainer = _TrainerStub()
    assert trainer._get_val_reward({"val/em": 0.5}) is None
