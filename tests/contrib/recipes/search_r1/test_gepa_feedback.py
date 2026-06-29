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

from search_r1_agent import INSTRUCTION_FORMAT, INSTRUCTION_FORMAT_REWRITE  # noqa: E402
from search_r1_gepa.search_r1_gepa_adapter import (  # noqa: E402
    INSTRUCTION_COMPONENT,
    _build_feedback,
    classify_failure_category,
    default_seed_candidate,
)


@pytest.mark.parametrize(
    ("rollout", "expected"),
    [
        ("<search>q</search><information>x</information>", "no_answer_tag"),
        (
            "<search>q</search><answer>The capital is Paris.</answer>",
            "verbose_answer",
        ),
        ("<search>q</search><answer>Sorry, I cannot answer.</answer>", "refusal"),
        ("<search>q</search><answer>London</answer>", "wrong_entity"),
    ],
)
def test_classify_failure_category(rollout: str, expected: str) -> None:
    from qa_em import extract_solution

    extracted = extract_solution(rollout)
    assert classify_failure_category(extracted, rollout, ["Paris"]) == expected


def test_build_feedback_includes_category_and_extracted_vs_gold() -> None:
    rollout = "<search>capital</search><answer>The capital is Paris.</answer>"
    feedback = _build_feedback("What is the capital of France?", ["Paris"], rollout, 0.0)
    assert "Failure category: verbose_answer" in feedback
    assert "Extracted answer: 'The capital is Paris.'" in feedback
    assert "Expected one of: 'Paris'" in feedback
    assert "Diagnosis:" in feedback


def test_build_feedback_success_mentions_concise_answer() -> None:
    rollout = "<search>capital</search><answer>Paris</answer>"
    feedback = _build_feedback("What is the capital of France?", ["Paris"], rollout, 1.0)
    assert "EM=1" in feedback
    assert "Extracted answer: 'Paris'" in feedback
    assert "concise <answer> tag" in feedback


@pytest.mark.parametrize(
    "instruction",
    [INSTRUCTION_FORMAT, INSTRUCTION_FORMAT_REWRITE],
)
def test_seed_prompts_include_em_answer_rules(instruction: str) -> None:
    assert "MUST" in instruction and "<answer>" in instruction
    assert "shortest" in instruction.lower()
    assert "Good:" in instruction and "Bad:" in instruction


def test_default_seed_candidate_baseline_has_em_rules() -> None:
    candidate = default_seed_candidate(use_rewrite=False)
    assert candidate[INSTRUCTION_COMPONENT] == INSTRUCTION_FORMAT
    assert "no sentences" in candidate[INSTRUCTION_COMPONENT].lower()


def test_default_seed_candidate_rewrite_has_em_rules_and_rewrite() -> None:
    candidate = default_seed_candidate(use_rewrite=True)
    assert candidate[INSTRUCTION_COMPONENT] == INSTRUCTION_FORMAT_REWRITE
    assert "<rewrite>" in candidate[INSTRUCTION_COMPONENT]
    assert "Good:" in candidate[INSTRUCTION_COMPONENT]
