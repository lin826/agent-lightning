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
    default_seed_candidate,
    run_search_r1_rollout,
)
from search_r1_gepa.train_gepa import GEPA_VARIANTS, resolve_gepa_variant  # noqa: E402


def test_default_seed_candidate_rewrite_uses_rewrite_instruction() -> None:
    candidate = default_seed_candidate(use_rewrite=True)
    assert candidate[INSTRUCTION_COMPONENT] == INSTRUCTION_FORMAT_REWRITE
    assert "<rewrite>" in candidate[INSTRUCTION_COMPONENT]


def test_default_seed_candidate_baseline_uses_standard_instruction() -> None:
    candidate = default_seed_candidate(use_rewrite=False)
    assert candidate[INSTRUCTION_COMPONENT] == INSTRUCTION_FORMAT
    assert "<rewrite>" not in candidate[INSTRUCTION_COMPONENT]


def test_resolve_gepa_variant_rewrite() -> None:
    variant = resolve_gepa_variant(rewrite=True)
    assert variant.name == "rewrite"
    assert variant.use_rewrite is True
    assert variant.run_dir_name == "gepa_qwen25_3b_rewrite"
    assert variant.wandb_experiment == "searchr1_qwen25_3b_gepa_rewrite"
    assert GEPA_VARIANTS["rewrite"].eval_addr_file == "bm25_server_addr_eval_gepa_rewrite.txt"


def test_run_search_r1_rollout_rewrite_turn_prepended() -> None:
    calls: list[str] = []

    def fake_llm(prompt: str, temperature: float | None) -> str:
        del temperature
        calls.append(prompt)
        if len(calls) == 1:
            return "<rewrite> clearer question </rewrite>"
        return "<answer> Paris </answer>"

    content = run_search_r1_rollout(
        fake_llm,
        INSTRUCTION_FORMAT_REWRITE,
        "Where is France?",
        max_turns=1,
        use_rewrite=True,
    )
    assert content.startswith("<rewrite> clearer question </rewrite>")
    assert "<answer> Paris </answer>" in content
    assert len(calls) >= 2
