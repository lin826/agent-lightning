# Copyright (c) Microsoft. All rights reserved.

"""GEPA adapter for Search-R1 multi-turn retrieval QA.

Optimizes the Search-R1 instruction prompt (same tag format as GRPO baseline)
while keeping Qwen2.5-3B-Instruct weights frozen. Uses BM25 retrieval and EM
or shaped (EM + retrieval-hit) rewards matching GRPO variants.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, TypedDict

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from qa_em import compute_score_em, compute_shaped_reward, extract_solution
from search_r1_agent import (
    execute_response,
    INSTRUCTION_FORMAT,
    INSTRUCTION_FORMAT_REWRITE,
    postprocess_response,
)

logger = logging.getLogger(__name__)

INSTRUCTION_COMPONENT = "instruction_prompt"
DEFAULT_SHAPED_ALPHA = 0.7
DEFAULT_SHAPED_BETA = 0.3
_INFORMATION_PATTERN = re.compile(r"<information>(.*?)</information>", re.DOTALL)

# Set by evaluate() when reward_mode=shaped so WandB can log strict EM separately.
_last_eval_em_mean: float | None = None


def resolve_reward_mode(*, cli_shaped: bool = False) -> str:
    """Resolve GEPA reward mode from CLI, ``GEPA_REWARD_MODE``, or ``GEPA_USE_SHAPED_REWARD``."""
    if cli_shaped or os.environ.get("GEPA_USE_SHAPED_REWARD", "").strip().lower() in {"1", "true", "yes"}:
        return "shaped"
    env_mode = os.environ.get("GEPA_REWARD_MODE", "").strip().lower()
    if env_mode in {"shaped", "em"}:
        return env_mode
    return "em"


def get_last_eval_em_mean() -> float | None:
    """Return mean strict EM from the most recent shaped train-mode evaluate() call."""
    return _last_eval_em_mean


def extract_retrieved_passages_from_rollout(rollout_content: str) -> list[list[str]]:
    """Parse ``<information>`` blocks into per-search passage lists for shaped reward."""
    passages_per_search: list[list[str]] = []
    for match in _INFORMATION_PATTERN.finditer(rollout_content):
        block = match.group(1).strip()
        if not block:
            passages_per_search.append([])
            continue
        docs = re.split(r"(?=Doc \d+\(Title:)", block)
        passages = [doc.strip() for doc in docs if doc.strip()]
        passages_per_search.append(passages)
    return passages_per_search


class SearchR1DataInst(TypedDict):
    question: str
    golden_answers: list[str]
    data_id: str


class SearchR1RolloutOutput(TypedDict):
    rollout_content: str
    extracted_answer: str | None


class SearchR1Trajectory(TypedDict, total=False):
    data: SearchR1DataInst
    rollout_content: str
    extracted_answer: str | None
    feedback: str
    em_score: float
    reward_score: float


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


_REFUSAL_RE = re.compile(
    r"\b(sorry|cannot|can't|can not|don't know|do not know|unable to|i am not|i'm not|no answer|not sure)\b",
    re.IGNORECASE,
)

_FAILURE_CATEGORY_HINTS: dict[str, str] = {
    "no_answer_tag": (
        "The rollout never contained a valid <answer>...</answer> tag. "
        "Require the model to always end with a concise <answer>."
    ),
    "verbose_answer": (
        "The <answer> tag contained a full sentence or explanation instead of the shortest exact entity/phrase."
    ),
    "refusal": "The <answer> tag contained a refusal or hedge instead of the factual answer.",
    "wrong_entity": "The <answer> tag had the correct format but the entity/phrase did not match any gold answer.",
}


def _looks_like_verbose_answer(answer: str) -> bool:
    stripped = answer.strip()
    if not stripped:
        return False
    if stripped.endswith((".", "!", "?")):
        return True
    if len(stripped.split()) > 6:
        return True
    lower = f" {stripped.lower()} "
    return any(phrase in lower for phrase in (" is ", " are ", " was ", " the ", " a ", " an "))


def classify_failure_category(
    extracted: str | None,
    rollout_content: str,
    golden_answers: list[str],
) -> str | None:
    """Classify EM failures for GEPA reflection feedback."""
    del golden_answers
    if extracted is None:
        return "no_answer_tag"
    if _REFUSAL_RE.search(extracted):
        return "refusal"
    if _looks_like_verbose_answer(extracted):
        return "verbose_answer"
    return "wrong_entity"


def _build_feedback(question: str, golden_answers: list[str], rollout_content: str, em_score: float) -> str:
    extracted = extract_solution(rollout_content)
    gold_str = _format_golden_answers(golden_answers)
    if em_score >= 1.0:
        return (
            f"The model answered correctly with EM=1. Question: {question!r}. "
            f"Extracted answer: {extracted!r}. Expected: {gold_str!r}. "
            "The instruction prompt successfully guided search-then-answer behavior with a concise <answer> tag."
        )

    category = classify_failure_category(extracted, rollout_content, golden_answers)
    hint = _FAILURE_CATEGORY_HINTS[category or "wrong_entity"]
    extracted_repr = extracted if extracted is not None else "(none)"
    return (
        f"The model answered incorrectly with EM=0. Failure category: {category}. Question: {question!r}. "
        f"Extracted answer: {extracted_repr!r}. Expected one of: {gold_str!r}. "
        f"Diagnosis: {hint} "
        f"Rollout excerpt: {rollout_content[:1500]!r}."
    )


def run_search_r1_rollout(
    llm_call: Callable[[str, float | None], str],
    instruction_prompt: str,
    question: str,
    *,
    max_turns: int = 4,
    temperature: float = 1.0,
    max_tokens: int = 500,
    use_rewrite: bool = False,
) -> str:
    """Run a multi-turn Search-R1 rollout with a custom instruction prompt."""
    del max_tokens  # reserved for future per-turn token limits
    prompt = instruction_prompt + question
    rollout_content = ""
    finished = False
    turn_id = 0

    if use_rewrite:
        rewrite_response = llm_call(prompt, temperature)
        if "</rewrite>" in rewrite_response:
            rewrite_response = rewrite_response.split("</rewrite>")[0] + "</rewrite>"
        rollout_content += rewrite_response

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
        use_rewrite: bool = False,
        reward_mode: str = "em",
        alpha: float = DEFAULT_SHAPED_ALPHA,
        beta: float = DEFAULT_SHAPED_BETA,
    ) -> None:
        self.llm_call = llm_call
        self.max_turns = max_turns
        self.train_temperature = train_temperature
        self.val_temperature = val_temperature
        self.eval_mode = eval_mode
        self.rollout_concurrency = max(1, rollout_concurrency)
        self.use_rewrite = use_rewrite
        self.reward_mode = reward_mode
        self.alpha = alpha
        self.beta = beta

    def _temperature(self) -> float:
        return self.val_temperature if self.eval_mode == "val" else self.train_temperature

    def _optimization_score(self, rollout_content: str, golden_answers: list[str], em_score: float) -> float:
        if self.reward_mode != "shaped" or self.eval_mode != "train":
            return em_score
        retrieved_passages = extract_retrieved_passages_from_rollout(rollout_content)
        return float(
            compute_shaped_reward(
                rollout_content,
                golden_answers,
                retrieved_passages,
                alpha=self.alpha,
                beta=self.beta,
            )
        )

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
                use_rewrite=self.use_rewrite,
            )
            em_score = float(compute_score_em(rollout_content, data["golden_answers"]))
            reward_score = self._optimization_score(rollout_content, data["golden_answers"], em_score)
            extracted = extract_solution(rollout_content)
            output: SearchR1RolloutOutput = {
                "rollout_content": rollout_content,
                "extracted_answer": extracted,
            }
            feedback = _build_feedback(data["question"], data["golden_answers"], rollout_content, em_score)
        except Exception as exc:
            logger.exception("Rollout failed for %s: %s", data.get("data_id", "?"), exc)
            em_score = 0.0
            reward_score = 0.0
            rollout_content = ""
            extracted = None
            output = {"rollout_content": rollout_content, "extracted_answer": extracted}
            feedback = f"Rollout failed with error: {exc}"

        trajectory: SearchR1Trajectory = {
            "data": data,
            "rollout_content": rollout_content,
            "extracted_answer": extracted,
            "feedback": feedback,
            "em_score": em_score,
            "reward_score": reward_score,
        }
        return output, reward_score, trajectory

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

        em_scores: list[float] = []
        if self.rollout_concurrency <= 1 or len(batch) <= 1:
            for data in batch:
                output, reward_score, trajectory = self._evaluate_one(data, instruction)
                outputs.append(output)
                scores.append(reward_score)
                em_scores.append(float(trajectory.get("em_score", reward_score)))
                if trajectories is not None:
                    trajectories.append(trajectory)
        else:
            workers = min(self.rollout_concurrency, len(batch))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(lambda data: self._evaluate_one(data, instruction), batch))
            for output, reward_score, trajectory in results:
                outputs.append(output)
                scores.append(reward_score)
                em_scores.append(float(trajectory.get("em_score", reward_score)))
                if trajectories is not None:
                    trajectories.append(trajectory)

        global _last_eval_em_mean
        if self.reward_mode == "shaped" and self.eval_mode == "train" and em_scores:
            _last_eval_em_mean = sum(em_scores) / len(em_scores)
        else:
            _last_eval_em_mean = None

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


def default_seed_candidate(*, use_rewrite: bool = False) -> dict[str, str]:
    instruction = INSTRUCTION_FORMAT_REWRITE if use_rewrite else INSTRUCTION_FORMAT
    return {INSTRUCTION_COMPONENT: instruction}
