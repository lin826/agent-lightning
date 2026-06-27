# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd
import requests
from openai import OpenAI
from qa_em import compute_score_em, compute_shaped_reward

from agentlightning import LLM, LitAgent, NamedResources, Rollout, Trainer, configure_logger, emit_reward, setup_logging

setup_logging()
logger = configure_logger(name=__name__)

# Copied and adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/data_process/nq_search.py
INSTRUCTION_FORMAT = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: """



def eval(prediction: str, ground_truth: List[str]) -> float:
    reward_score = float(compute_score_em(prediction, ground_truth))
    print(f"pred: {prediction} | {type(ground_truth)} gold_answer: {ground_truth} | res: {reward_score}")
    return reward_score


def _emit_rollout_scores(reward_score: float, em_score: float) -> None:
    """Emit composite reward and pure EM for VERL optimization and WandB logging."""
    emit_reward({"reward": reward_score, "em": em_score}, primary_key="reward")


def postprocess_response(response: str) -> str:
    """Process responses to stop at search operation or answer operation."""
    if "</search>" in response:
        response = response.split("</search>")[0] + "</search>"
    elif "</answer>" in response:
        response = response.split("</answer>")[0] + "</answer>"
    return response


def extract_action(response: str) -> Tuple[Optional[str], str]:
    """Process (text-based) predictions from llm into actions and validity flags."""
    pattern = r"<(search|answer)>(.*?)</\1>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        content = match.group(2).strip()  # Return only the content inside the tags
        action: Optional[str] = match.group(1)
    else:
        content = ""
        action = None
    return action, content


def execute_response(response: str, do_search: bool = True) -> str:
    """
    Execute predictions across multiple environments.
    """
    action, content = extract_action(response)
    if action == "answer":
        return ""
    elif action == "search":
        search_result = retrieve_doc(content) if do_search else ""
        return f"\n\n<information>{search_result}</information>\n\n"
    else:
        return (
            "\nMy previous action is invalid. If I want to search, I should put the query between <search> and </search>. "
            "If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
        )


_RETRIEVAL_CONNECT_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ConnectTimeout,
    requests.exceptions.ReadTimeout,
)
_RETRIEVAL_MAX_RETRIES = 3
_RETRIEVAL_BACKOFF_BASE_S = 1.0
_RETRIEVAL_REQUEST_TIMEOUT_S = 30.0

_cached_retrieval_url: Optional[str] = None
_cached_addr_file: Optional[str] = None


def _recipe_outputs_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "outputs"


def _read_url_from_addr_file(addr_file: str) -> Optional[str]:
    if addr_file and os.path.isfile(addr_file):
        with open(addr_file) as f:
            url = f.read().strip()
            if url:
                return url.rstrip("/")
    return None


def _resolve_addr_file_path() -> Optional[str]:
    for key in ("RETRIEVAL_SERVER_ADDR_FILE", "ADDR_FILE"):
        path = os.environ.get(key, "")
        if path and os.path.isfile(path):
            return path
    return None


def _find_addr_file_matching_url(url: str) -> Optional[str]:
    explicit = _resolve_addr_file_path()
    if explicit:
        return explicit
    outputs = _recipe_outputs_dir()
    if not outputs.is_dir():
        return None
    normalized = url.rstrip("/")
    for path in sorted(outputs.glob("bm25_server_addr_*.txt")):
        try:
            content = path.read_text().strip().rstrip("/")
            if content == normalized:
                return str(path)
        except OSError:
            continue
    return None


def _ensure_retrieval_cache() -> str:
    global _cached_retrieval_url, _cached_addr_file
    if _cached_retrieval_url is not None:
        return _cached_retrieval_url

    url = os.environ.get("RETRIEVAL_SERVER_URL", "").rstrip("/")
    if not url:
        addr_file = _resolve_addr_file_path()
        if addr_file:
            url = _read_url_from_addr_file(addr_file) or ""
            _cached_addr_file = addr_file
    if not url:
        url = "http://127.0.0.1:8000"

    if _cached_addr_file is None:
        _cached_addr_file = _find_addr_file_matching_url(url)

    _cached_retrieval_url = url
    return url


def _refresh_retrieval_url() -> Optional[str]:
    """Re-read the BM25 addr file and update the cached retrieval URL."""
    global _cached_retrieval_url, _cached_addr_file
    addr_file = _cached_addr_file or _resolve_addr_file_path()
    if not addr_file:
        logger.warning("Cannot refresh retrieval URL: no addr file path known")
        return None
    new_url = _read_url_from_addr_file(addr_file)
    if not new_url:
        logger.warning("Cannot refresh retrieval URL: addr file %s empty or missing", addr_file)
        return None
    old_url = _cached_retrieval_url
    _cached_retrieval_url = new_url
    _cached_addr_file = addr_file
    os.environ["RETRIEVAL_SERVER_URL"] = new_url
    if new_url != old_url:
        logger.info("Refreshed retrieval URL from %s: %s -> %s", addr_file, old_url, new_url)
    else:
        logger.info("Retrieval URL unchanged after re-read from %s: %s", addr_file, new_url)
    return new_url


def _retrieval_url() -> str:
    """Return the retrieval server base URL.

    Resolution order: cached URL → ``RETRIEVAL_SERVER_URL`` env var → URL read
    from ``RETRIEVAL_SERVER_ADDR_FILE`` / ``ADDR_FILE`` → localhost fallback.
    """
    return _ensure_retrieval_cache()


def _retrieval_post(query: str) -> requests.Response:
    url = _retrieval_url()
    last_exc: Optional[Exception] = None

    for attempt in range(_RETRIEVAL_MAX_RETRIES):
        try:
            response = requests.post(
                f"{url}/search",
                json={"query": query},
                timeout=_RETRIEVAL_REQUEST_TIMEOUT_S,
            )
            response.raise_for_status()
            return response
        except _RETRIEVAL_CONNECT_ERRORS as exc:
            last_exc = exc
            if attempt + 1 < _RETRIEVAL_MAX_RETRIES:
                sleep_time = _RETRIEVAL_BACKOFF_BASE_S * (2**attempt)
                logger.warning(
                    "Retrieval request failed (attempt %d/%d) at %s: %s; retrying in %.1fs",
                    attempt + 1,
                    _RETRIEVAL_MAX_RETRIES,
                    url,
                    exc,
                    sleep_time,
                )
                time.sleep(sleep_time)
                continue
            break

    logger.warning("Retrieval retries exhausted at %s; re-reading addr file", url)
    _refresh_retrieval_url()
    retry_url = _retrieval_url()
    try:
        response = requests.post(
            f"{retry_url}/search",
            json={"query": query},
            timeout=_RETRIEVAL_REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        return response
    except Exception as exc:
        logger.error("Retrieval request failed after addr-file refresh at %s: %s", retry_url, exc)
        if last_exc is not None:
            raise last_exc from exc
        raise


def retrieve_doc(query: str) -> str:
    response = _retrieval_post(query)
    json_resp: Dict[str, Any] = cast(Dict[str, Any], response.json())
    passages: List[str] = cast(List[str], json_resp["passages"])
    # Drop the trailing "Other retrieved pages have titles: …" summary entry.
    full_passages = [p for p in passages if not p.startswith("Other retrieved pages")]
    return passages2string(full_passages[:3])


def passages2string(passages: List[str]) -> str:
    """Format a list of ``"Title | content"`` passage strings for the prompt."""
    format_reference = ""
    for idx, passage in enumerate(passages):
        if " | " in passage:
            title, text = passage.split(" | ", 1)
        else:
            title, text = f"Doc {idx + 1}", passage
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference


def call_llm(
    llm_client: OpenAI,
    model_name: str,
    content: str,
    temperature: float = 1.0,
    max_tokens: int = 500,
) -> str:
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


class SearchR1Agent(LitAgent[Dict[str, Any]]):

    def __init__(
        self,
        val_temperature: Optional[float] = 0.0,
        max_turns: int = 4,
    ) -> None:
        super().__init__()
        self.val_temperature = val_temperature
        self.data_dir = os.environ.get("VERL_SEARCHR1_DATA_DIR", "data")
        self.max_turns = max_turns

    def rollout(
        self,
        task: Dict[str, Any],
        resources: NamedResources,
        rollout: Rollout,
    ) -> float | None:
        prompt = INSTRUCTION_FORMAT + task["question"]
        answer_list: List[str] = cast(List[str], task["golden_answers"])
        rollout_id = rollout.rollout_id
        logger.info(f"[Rollout {rollout_id}] Question: {task['question']}")
        logger.info(f"[Rollout {rollout_id}] Ground Truth: {answer_list}")

        start_time = time.time()
        llm: LLM = cast(LLM, resources["main_llm"])
        client = OpenAI(
            base_url=llm.get_base_url(rollout_id, rollout.attempt.attempt_id),  # type: ignore
            api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
        )

        if rollout.mode == "train":
            temperature = llm.sampling_parameters.get("temperature", 1.0)
        else:
            temperature = self.val_temperature if self.val_temperature is not None else 0.0

        turn_id = 0
        finished_flag = False
        rollout_content: str = ""

        try:
            while turn_id < self.max_turns and not finished_flag:
                turn_id += 1
                turn_response = call_llm(
                    client, llm.model, prompt + rollout_content, temperature=temperature, max_tokens=500
                )
                valid_turn_response = postprocess_response(turn_response)
                rollout_content += valid_turn_response
                turn_env_feedback = execute_response(valid_turn_response)
                if len(turn_env_feedback) == 0:
                    finished_flag = True
                else:
                    rollout_content += turn_env_feedback
                logger.info(f"TURN ID {turn_id} | RESP: {turn_response} | ENV FEEDBACK: {turn_env_feedback}")

            if not finished_flag:
                turn_response = call_llm(
                    client, llm.model, prompt + rollout_content, temperature=temperature, max_tokens=500
                )
                rollout_content += turn_response
                logger.info(f"LAST TURN GENERATE | RESP: {turn_response}")

        except Exception as e:
            logger.exception(f"[Rollout {rollout_id}] Error during rollout: {e}")
            return None

        end_time_rollout = time.time()
        em_score = eval(rollout_content, answer_list)
        reward_score = em_score
        logger.info("[Rollout %s] Reward: %s | EM: %s", rollout_id, reward_score, em_score)
        end_time_eval = time.time()

        logger.info("[Rollout %s] Time taken for rollout: %.2f seconds", rollout_id, end_time_rollout - start_time)
        logger.info(
            "[Rollout %s] Time taken for evaluation: %.2f seconds", rollout_id, end_time_eval - end_time_rollout
        )
        logger.info(
            "question: {} answer: {} ground_truth: {} reward: {} em: {}".format(
                task["question"], rollout_content, answer_list, reward_score, em_score
            )
        )
        _emit_rollout_scores(reward_score, em_score)
        return None


INSTRUCTION_FORMAT_REWRITE = """You will answer a question using search.

First, rewrite the question to make it clearer and easier to search for. Put your rewritten question inside <rewrite> and </rewrite> tags. For multi-hop questions, decompose into sub-questions.

Then conduct reasoning inside <think> and </think> every time you get new information. Search by writing <search> query </search>. When ready, answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.

Question: """


def extract_rewrite(response: str) -> Optional[str]:
    """Extract the content of the first <rewrite>...</rewrite> tag."""
    match = re.search(r"<rewrite>(.*?)</rewrite>", response, re.DOTALL)
    return match.group(1).strip() if match else None


class SearchR1RewriteAgent(LitAgent[Dict[str, Any]]):
    """Search-R1 agent supporting the question-rewrite ablation variants.

    A single class covers three of the four experiment variants via the
    ``use_rewrite`` flag and the ``alpha``/``beta`` shaped-reward weights:

    - **B (rewrite + EM):** ``use_rewrite=True, alpha=1.0, beta=0.0``
    - **C (rewrite + shaped):** ``use_rewrite=True, alpha=0.7, beta=0.3``
    - **D (no rewrite + shaped):** ``use_rewrite=False, alpha=0.7, beta=0.3``

    When ``use_rewrite`` is False the rewrite turn is skipped and the base
    Search-R1 prompt is used, but retrieved passages are still tracked so the
    shaped reward can credit retrieval hits. Validation always reports pure EM
    so ``val/reward`` is comparable across every variant.
    """

    def __init__(
        self,
        val_temperature: Optional[float] = 0.0,
        max_turns: int = 4,
        alpha: float = 0.7,
        beta: float = 0.3,
        use_rewrite: bool = True,
    ) -> None:
        super().__init__()
        self.val_temperature = val_temperature
        self.max_turns = max_turns
        self.alpha = alpha
        self.beta = beta
        self.use_rewrite = use_rewrite

    def rollout(
        self,
        task: Dict[str, Any],
        resources: NamedResources,
        rollout: Rollout,
    ) -> float | None:
        instruction = INSTRUCTION_FORMAT_REWRITE if self.use_rewrite else INSTRUCTION_FORMAT
        prompt = instruction + task["question"]
        answer_list: List[str] = cast(List[str], task["golden_answers"])
        rollout_id = rollout.rollout_id
        logger.info(f"[Rollout {rollout_id}] Question: {task['question']}")
        logger.info(f"[Rollout {rollout_id}] Ground Truth: {answer_list}")

        start_time = time.time()
        llm: LLM = cast(LLM, resources["main_llm"])
        client = OpenAI(
            base_url=llm.get_base_url(rollout_id, rollout.attempt.attempt_id),
            api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
        )

        if rollout.mode == "train":
            temperature = llm.sampling_parameters.get("temperature", 1.0)
        else:
            temperature = self.val_temperature if self.val_temperature is not None else 0.0

        rollout_content: str = ""
        retrieved_passages: List[List[str]] = []

        try:
            # Turn 0: Rewrite stage (skipped for the no-rewrite variant D)
            if self.use_rewrite:
                rewrite_response = call_llm(client, llm.model, prompt, temperature=temperature, max_tokens=300)
                if "</rewrite>" in rewrite_response:
                    rewrite_response = rewrite_response.split("</rewrite>")[0] + "</rewrite>"
                rollout_content += rewrite_response
                rewritten = extract_rewrite(rewrite_response)
                if rewritten:
                    logger.info(f"[Rollout {rollout_id}] Rewritten question: {rewritten}")
                else:
                    logger.info(f"[Rollout {rollout_id}] No rewrite tag found, continuing with raw response")

            # Turns 1-N: Search/Think/Answer loop (same as baseline)
            turn_id = 0
            finished_flag = False

            while turn_id < self.max_turns and not finished_flag:
                turn_id += 1
                turn_response = call_llm(
                    client, llm.model, prompt + rollout_content, temperature=temperature, max_tokens=500
                )
                valid_turn_response = postprocess_response(turn_response)
                rollout_content += valid_turn_response

                action, content = extract_action(valid_turn_response)
                if action == "answer":
                    finished_flag = True
                elif action == "search":
                    passages = _retrieve_passages(content)
                    retrieved_passages.append(passages)
                    formatted = passages2string(passages[:3])
                    turn_env_feedback = f"\n\n<information>{formatted}</information>\n\n"
                    rollout_content += turn_env_feedback
                else:
                    turn_env_feedback = (
                        "\nMy previous action is invalid. If I want to search, I should put the query between "
                        "<search> and </search>. If I want to give the final answer, I should put the answer "
                        "between <answer> and </answer>. Let me try again.\n"
                    )
                    rollout_content += turn_env_feedback
                logger.info(f"TURN ID {turn_id} | RESP: {turn_response}")

            if not finished_flag:
                turn_response = call_llm(
                    client, llm.model, prompt + rollout_content, temperature=temperature, max_tokens=500
                )
                rollout_content += turn_response
                logger.info(f"LAST TURN GENERATE | RESP: {turn_response}")

        except Exception as e:
            logger.exception(f"[Rollout {rollout_id}] Error during rollout: {e}")
            return None

        end_time_rollout = time.time()
        em_score = float(compute_score_em(rollout_content, answer_list))

        if rollout.mode == "train":
            reward_score = compute_shaped_reward(
                rollout_content, answer_list, retrieved_passages, alpha=self.alpha, beta=self.beta
            )
        else:
            reward_score = em_score

        logger.info(
            "[Rollout %s] Reward: %s | EM: %s (mode=%s)", rollout_id, reward_score, em_score, rollout.mode
        )
        logger.info("[Rollout %s] Time taken for rollout: %.2f seconds", rollout_id, end_time_rollout - start_time)
        logger.info(
            "question: {} answer: {} ground_truth: {} reward: {} em: {}".format(
                task["question"], rollout_content, answer_list, reward_score, em_score
            )
        )
        _emit_rollout_scores(reward_score, em_score)
        return None


def _retrieve_passages(query: str) -> List[str]:
    """Retrieve raw passages (before formatting) for reward computation."""
    response = _retrieval_post(query)
    json_resp: Dict[str, Any] = cast(Dict[str, Any], response.json())
    passages: List[str] = cast(List[str], json_resp["passages"])
    return [p for p in passages if not p.startswith("Other retrieved pages")][:3]


def debug_search_r1_agent():
    searchr1_dev_data_path = os.path.join(os.environ.get("VERL_SEARCHR1_DATA_DIR", "data"), "test.parquet")
    if not os.path.exists(searchr1_dev_data_path):
        raise FileNotFoundError(f"Search_R1 dev data file {searchr1_dev_data_path} does not exist.")
    df = pd.read_parquet(searchr1_dev_data_path).head(10)  # type: ignore
    df = cast(List[Dict[str, Any]], df.to_dict(orient="records"))  # type: ignore
    print("Debug data:", df)

    trainer = Trainer(
        n_workers=1,
        initial_resources={
            "main_llm": LLM(
                endpoint=os.environ["OPENAI_API_BASE"],
                model="gpt-4.1-nano",
                sampling_parameters={"temperature": 0.0},
            )
        },
    )
    trainer.dev(SearchR1Agent(), df)


if __name__ == "__main__":
    debug_search_r1_agent()
