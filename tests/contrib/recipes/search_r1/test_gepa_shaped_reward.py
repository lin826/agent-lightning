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

from qa_em import compute_shaped_reward  # noqa: E402
from search_r1_gepa.search_r1_gepa_adapter import (  # noqa: E402
    SearchR1GEPAAdapter,
    extract_retrieved_passages_from_rollout,
    resolve_reward_mode,
)
from search_r1_gepa.train_gepa import GEPA_VARIANTS, resolve_gepa_variant  # noqa: E402


def test_extract_retrieved_passages_from_rollout_parses_information_blocks() -> None:
    rollout = (
        "<search>capital france</search>"
        "\n\n<information>Doc 1(Title: Paris) Paris is the capital of France.\n"
        "Doc 2(Title: Lyon) Lyon is a city in France.\n</information>\n\n"
        "<answer>Paris</answer>"
    )
    passages = extract_retrieved_passages_from_rollout(rollout)
    assert len(passages) == 1
    assert len(passages[0]) == 2
    assert "Paris is the capital of France" in passages[0][0]


def test_shaped_reward_from_parsed_passages_matches_qa_em() -> None:
    rollout = (
        "<search>capital france</search>"
        "\n\n<information>Doc 1(Title: Paris) Paris is the capital of France.\n</information>\n\n"
        "<answer>London</answer>"
    )
    passages = extract_retrieved_passages_from_rollout(rollout)
    shaped = compute_shaped_reward(rollout, ["Paris"], passages, alpha=0.7, beta=0.3)
    assert shaped == pytest.approx(0.3)


def test_resolve_reward_mode_env_and_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEPA_USE_SHAPED_REWARD", raising=False)
    monkeypatch.delenv("GEPA_REWARD_MODE", raising=False)
    assert resolve_reward_mode() == "em"
    monkeypatch.setenv("GEPA_USE_SHAPED_REWARD", "1")
    assert resolve_reward_mode() == "shaped"
    monkeypatch.delenv("GEPA_USE_SHAPED_REWARD", raising=False)
    monkeypatch.setenv("GEPA_REWARD_MODE", "shaped")
    assert resolve_reward_mode() == "shaped"
    assert resolve_reward_mode(cli_shaped=True) == "shaped"


def test_resolve_gepa_variant_shaped_entries() -> None:
    shaped = resolve_gepa_variant(shaped=True)
    assert shaped.name == "shaped"
    assert shaped.reward_mode == "shaped"
    assert shaped.run_dir_name == "gepa_qwen25_3b_shaped"
    assert shaped.wandb_experiment == "searchr1_qwen25_3b_gepa_shaped"

    rewrite_shaped = resolve_gepa_variant(rewrite=True, shaped=True)
    assert rewrite_shaped.name == "rewrite_shaped"
    assert rewrite_shaped.use_rewrite is True
    assert rewrite_shaped.reward_mode == "shaped"
    assert GEPA_VARIANTS["rewrite_shaped"].eval_addr_file == "bm25_server_addr_eval_gepa_rewrite_shaped.txt"


def test_adapter_shaped_train_mode_uses_shaped_score() -> None:
    rollout = (
        "<search>capital france</search>"
        "\n\n<information>Doc 1(Title: Paris) Paris is the capital of France.\n</information>\n\n"
        "<answer>London</answer>"
    )

    def fake_llm(_prompt: str, _temperature: float | None) -> str:
        return rollout

    adapter = SearchR1GEPAAdapter(fake_llm, eval_mode="train", reward_mode="shaped", max_turns=1)
    shaped = adapter._optimization_score(rollout, ["Paris"], em_score=0.0)
    assert shaped == pytest.approx(0.3)

    em_only = SearchR1GEPAAdapter(fake_llm, eval_mode="train", reward_mode="em", max_turns=1)
    assert em_only._optimization_score(rollout, ["Paris"], em_score=0.0) == 0.0


def test_adapter_val_mode_uses_em_even_when_shaped() -> None:
    rollout = (
        "<search>capital france</search>"
        "\n\n<information>Doc 1(Title: Paris) Paris is the capital of France.\n</information>\n\n"
        "<answer>London</answer>"
    )

    def fake_llm(_prompt: str, _temperature: float | None) -> str:
        return rollout

    adapter = SearchR1GEPAAdapter(fake_llm, eval_mode="val", reward_mode="shaped", max_turns=1)
    assert adapter._optimization_score(rollout, ["Paris"], em_score=0.0) == 0.0
    assert adapter._optimization_score(rollout, ["Paris"], em_score=1.0) == 1.0
