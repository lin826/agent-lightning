# Copyright (c) Microsoft. All rights reserved.

"""GEPA adapter for Search-R1 multi-turn retrieval QA.

Optimizes the Search-R1 instruction prompt (same tag format as GRPO baseline)
while keeping Qwen2.5-3B-Instruct weights frozen. Uses BM25 retrieval and the
same EM metric as ``qa_em.compute_score_em``.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, TypedDict

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from qa_em import compute_score_em, extract_solution
from search_r1_agent import (
    call_llm,
    execute_response,
    INSTRUCTION_FORMAT,
    postprocess_response,
)

logger = logging.getLogger(__name__)

INSTRUCTION_COMPONENT = "instruction_prompt"


class SearchR1DataInst(TypedDict):
    question: str
    golden_answers: list[str]
    data_id: str


class SearchR1RolloutOutput(TypedDict):
    rollout_content: str
    extracted_answer: str | None


class SearchR1Trajectory(TypedDict):
    data: SearchR1DataInst
    rollout_content: str
    extracted_answer: str | None
    feedback: str


SearchR1ReflectiveRecord = TypedDict(
    "SearchR1ReflectiveRecord",
    {
        "Inputs": str,
        "Generated Outputs": str,
        "Feedback": str,
    },
)


def _format_golden_answers(golden_answers: list[str]) -> str:
    if len(golden_answers) == 1:
        return golden_answers[0]
    return ", ".join(golden_answers)


def _build_feedback(question: str, golden_answers: list[str], rollout_content: str, em_score: float) -> str:
    extracted = extract_solution(rollout_content)
    gold_str = _format_golden_answers(golden_answers)
    if em_score >= 1.0:
        return (
            f"The model answered correctly with EM=1. Question: {question!r}. "
            f"Extracted answer: {extracted!r}. Expected: {gold_str!r}. "
            "The instruction prompt successfully guided search-then-answer behavior."
        )
    return (
        f"The model answered incorrectly with EM=0. Question: {question!r}. "
        f"Extracted answer: {extracted!r}. Expected one of: {gold_str!r}. "
        f"Full rollout (truncated): {rollout_content[:2000]!r}. "
        "Improve the instruction so the model searches relevant queries and puts the "
        "final answer inside <answer>...</answer> tags."
    )


def run_search_r1_rollout(
    llm_call: Callable[[str, float | None], str],
    instruction_prompt: str,
    question: str,
    *,
    max_turns: int = 4,
    temperature: float = 1.0,
    max_tokens: int = 500,
) -> str:
    """Run a multi-turn Search-R1 rollout with a custom instruction prompt."""
    del max_tokens  # reserved for future per-turn token limits
    prompt = instruction_prompt + question
    rollout_content = ""
    finished = False
    turn_id = 0

    while turn_id < max_turns and not finished:
        turn_id += 1
        turn_response = llm_call(prompt + rollout_content, temperature)
        valid_turn_response = postprocess_response(turn_response)
        rollout_content += valid_turn_response
        env_feedback = execute_response(valid_turn_response)
        if not env_feedback:
            finished = True
        else:
            rollout_content += env_feedback

    if not finished:
        rollout_content += llm_call(prompt + rollout_content, temperature)

    return rollout_content


class SearchR1GEPAAdapter(GEPAAdapter[SearchR1DataInst, SearchR1Trajectory, SearchR1RolloutOutput]):
    """GEPA adapter wiring Search-R1 rollouts to reflective prompt evolution."""

    def __init__(
        self,
        llm_call: Callable[[str, float | None], str],
        *,
        max_turns: int = 4,
        train_temperature: float = 1.0,
        val_temperature: float = 0.0,
        eval_mode: str = "train",
        rollout_concurrency: int = 1,
    ) -> None:
        self.llm_call = llm_call
        self.max_turns = max_turns
        self.train_temperature = train_temperature
        self.val_temperature = val_temperature
        self.eval_mode = eval_mode
        self.rollout_concurrency = max(1, rollout_concurrency)

    def _temperature(self) -> float:
        return self.val_temperature if self.eval_mode == "val" else self.train_temperature

    def _evaluate_one(
        self,
        data: SearchR1DataInst,
        instruction: str,
    ) -> tuple[SearchR1RolloutOutput, float, SearchR1Trajectory]:
        try:
            rollout_content = run_search_r1_rollout(
                self.llm_call,
                instruction,
                data["question"],
                max_turns=self.max_turns,
                temperature=self._temperature(),
            )
            em_score = float(compute_score_em(rollout_content, data["golden_answers"]))
            extracted = extract_solution(rollout_content)
            output: SearchR1RolloutOutput = {
                "rollout_content": rollout_content,
                "extracted_answer": extracted,
            }
            feedback = _build_feedback(data["question"], data["golden_answers"], rollout_content, em_score)
        except Exception as exc:
            logger.exception("Rollout failed for %s: %s", data.get("data_id", "?"), exc)
            em_score = 0.0
            rollout_content = ""
            extracted = None
            output = {"rollout_content": rollout_content, "extracted_answer": extracted}
            feedback = f"Rollout failed with error: {exc}"

        trajectory = {
            "data": data,
            "rollout_content": rollout_content,
            "extracted_answer": extracted,
            "feedback": feedback,
        }
        return output, em_score, trajectory

    def evaluate(
        self,
        batch: list[SearchR1DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[SearchR1Trajectory, SearchR1RolloutOutput]:
        instruction = candidate[INSTRUCTION_COMPONENT]
        outputs: list[SearchR1RolloutOutput] = []
        scores: list[float] = []
        trajectories: list[SearchR1Trajectory] | None = [] if capture_traces else None

        if self.rollout_concurrency <= 1 or len(batch) <= 1:
            for data in batch:
                output, em_score, trajectory = self._evaluate_one(data, instruction)
                outputs.append(output)
                scores.append(em_score)
                if trajectories is not None:
                    trajectories.append(trajectory)
        else:
            workers = min(self.rollout_concurrency, len(batch))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(lambda data: self._evaluate_one(data, instruction), batch))
            for output, em_score, trajectory in results:
                outputs.append(output)
                scores.append(em_score)
                if trajectories is not None:
                    trajectories.append(trajectory)

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[SearchR1Trajectory, SearchR1RolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        del candidate
        assert len(components_to_update) == 1
        component = components_to_update[0]
        trajectories = eval_batch.trajectories
        assert trajectories is not None

        records: list[SearchR1ReflectiveRecord] = []
        for traj in trajectories:
            records.append(
                {
                    "Inputs": traj["data"]["question"],
                    "Generated Outputs": traj["rollout_content"][:4000],
                    "Feedback": traj["feedback"],
                }
            )

        if not records:
            raise RuntimeError("No trajectories available for reflective dataset.")

        return {component: records}


def make_openai_llm_call(
    *,
    base_url: str,
    model: str,
    api_key: str | None = None,
    default_temperature: float = 1.0,
    max_tokens: int = 500,
) -> Callable[[str, float | None], str]:
    """Build a user-message LLM callable backed by OpenAI-compatible API."""
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "token-abc123"),
    )

    def _call(user_content: str, temperature: float | None = None) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_content}],
            temperature=temperature if temperature is not None else default_temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    return _call


def default_seed_candidate() -> dict[str, str]:
    return {INSTRUCTION_COMPONENT: INSTRUCTION_FORMAT}
