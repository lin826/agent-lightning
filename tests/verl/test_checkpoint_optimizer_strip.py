# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from pathlib import Path

from agentlightning.verl.checkpoint_utils import (
    parse_global_step_dir,
    resolve_latest_global_step,
    strip_optimizer_shards,
    strip_stale_optimizer_shards,
)


def test_parse_global_step_dir(tmp_path: Path) -> None:
    step_dir = tmp_path / "global_step_10"
    step_dir.mkdir()
    assert parse_global_step_dir(step_dir) == 10
    assert parse_global_step_dir(tmp_path / "global_step_x") is None
    assert parse_global_step_dir(tmp_path / "actor") is None


def test_strip_optimizer_shards(tmp_path: Path) -> None:
    actor = tmp_path / "actor"
    actor.mkdir()
    optim = actor / "optim_world_size_8_rank_0.pt"
    optim.write_bytes(b"x" * 100)
    model = actor / "model_world_size_8_rank_0.pt"
    model.write_bytes(b"y" * 50)

    nbytes = strip_optimizer_shards(actor)
    assert nbytes == 100
    assert not optim.exists()
    assert model.exists()


def test_strip_stale_keeps_latest(tmp_path: Path) -> None:
    for step in (10, 20):
        actor = tmp_path / f"global_step_{step}" / "actor"
        actor.mkdir(parents=True)
        (actor / "optim_world_size_8_rank_0.pt").write_bytes(b"o" * 10)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("20\n")

    removed = strip_stale_optimizer_shards(tmp_path)
    assert removed == {"global_step_10": 10}
    assert (tmp_path / "global_step_20/actor/optim_world_size_8_rank_0.pt").exists()
    assert not (tmp_path / "global_step_10/actor/optim_world_size_8_rank_0.pt").exists()


def test_resolve_latest_global_step_from_tracker(tmp_path: Path) -> None:
    (tmp_path / "global_step_5").mkdir()
    (tmp_path / "global_step_12").mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("5\n")
    assert resolve_latest_global_step(tmp_path) == 5
