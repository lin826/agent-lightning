# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SEARCH_R1_DIR = Path(__file__).resolve().parents[4] / "contrib" / "recipes" / "search_r1"
sys.path.insert(0, str(SEARCH_R1_DIR))

from qa_em import compute_score_em, compute_shaped_reward  # noqa: E402


def test_compute_score_em_matches_normalized_answer() -> None:
    solution = "<answer> Beijing </answer>"
    assert compute_score_em(solution, ["Beijing"]) == 1.0
    assert compute_score_em(solution, ["Shanghai"]) == 0.0


def test_shaped_reward_keeps_em_separate_from_retrieval_hit() -> None:
    solution = "<answer> Beijing </answer>"
    passages = [["Shanghai is a large city in China"]]

    em = compute_score_em(solution, ["Beijing"])
    shaped = compute_shaped_reward(solution, ["Beijing"], passages, alpha=0.7, beta=0.3)

    assert em == 1.0
    assert shaped == pytest.approx(0.7)
    assert shaped != em
