# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from omegaconf import OmegaConf

from agentlightning.verl.trainer import AgentLightningTrainer


class _TrainerStub(AgentLightningTrainer):
    """Minimal stub to exercise metric-prefix rewriting without Ray workers."""

    def __init__(self, prefix: str | None) -> None:
        self.config = OmegaConf.create({"trainer": {"val_metric_prefix": prefix}})


def test_val_metric_prefix_rewrites_val_to_test() -> None:
    trainer = _TrainerStub("test")
    metrics = {
        "val/reward": 0.5,
        "val/em": 0.4,
        "val/hotpotqa/reward": 0.5,
        "val/hotpotqa/em": 0.4,
        "training/reward": 0.3,
    }

    result = trainer._apply_val_metric_prefix(metrics)

    assert result == {
        "test/reward": 0.5,
        "test/em": 0.4,
        "test/hotpotqa/reward": 0.5,
        "test/hotpotqa/em": 0.4,
        "training/reward": 0.3,
    }


def test_val_metric_prefix_keeps_val_when_unset() -> None:
    trainer = _TrainerStub(None)
    metrics = {"val/reward": 0.5, "val/em": 0.4}

    assert trainer._apply_val_metric_prefix(metrics) == metrics


def test_val_metric_prefix_keeps_val_when_explicit_val() -> None:
    trainer = _TrainerStub("val")
    metrics = {"val/reward": 0.5}

    assert trainer._apply_val_metric_prefix(metrics) == metrics
