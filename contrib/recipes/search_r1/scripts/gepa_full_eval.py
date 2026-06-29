"""Submit full test.parquet eval when GEPA dev-subset validation hits a new best.

GEPA optimizes against ``test_dev.parquet`` (200 hotpotqa examples). Whenever the
historical dev-subset best (``best_score_on_valset`` / ``best_valset_agg_score``)
improves, persist the best program prompt from ``gepa_state.bin`` and submit an LSF
eval job (``eval/eval_gepa_prompt.bsub``) to score it on ``test.parquet`` (7405 examples).

Per-candidate ``val_program_average`` scores do not trigger eval — only record-breaking
aggregate dev bests, matching GRPO's ``val/reward`` record semantics.

Used from ``train_gepa.py`` (in-process trigger) and ``monitor_best_and_eval.py``
(fallback polling).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RECIPE_DIR = Path(__file__).resolve().parent.parent
EVAL_GENERATED_DIR = RECIPE_DIR / "eval" / "generated"
GEPA_EVAL_TEMPLATE = RECIPE_DIR / "eval" / "eval_gepa_prompt.bsub"
FULL_EVAL_STATE_FILENAME = "full_eval_state.json"
TRAINING_SESSION_FILENAME = "training_session.json"

DEFAULT_EVAL_JOB_TAG = "qwen25_3b_gepa"
DEFAULT_ADDR_FILE = "bm25_server_addr_eval_gepa.txt"


@dataclass
class FullEvalState:
    """Tracks dev-subset bests and submitted full-test eval jobs for one GEPA run."""

    best_dev_score: float = -1.0
    best_metric_calls: int = -1
    best_program_idx: int = -1
    submitted_metric_calls: list[int] = field(default_factory=list)
    training_session_id: str = ""


@dataclass
class TrainingSession:
    """Identifies the active GEPA training run; full-test eval only tracks scores from this session."""

    session_id: str
    started_at: float


def full_eval_state_path(run_dir: Path) -> Path:
    return run_dir / FULL_EVAL_STATE_FILENAME


def training_session_path(run_dir: Path) -> Path:
    return run_dir / TRAINING_SESSION_FILENAME


def load_training_session(run_dir: Path) -> TrainingSession | None:
    path = training_session_path(run_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Could not parse training session file %s", path)
        return None
    session_id = str(data.get("session_id", "")).strip()
    if not session_id:
        return None
    return TrainingSession(session_id=session_id, started_at=float(data.get("started_at", 0.0)))


def save_training_session(run_dir: Path, session: TrainingSession) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    training_session_path(run_dir).write_text(
        json.dumps({"session_id": session.session_id, "started_at": session.started_at}, indent=2)
    )


def start_training_session(run_dir: Path, *, fresh: bool = False) -> TrainingSession:
    """Begin or resume a training session and reset full-eval tracking when starting fresh."""
    existing = None if fresh else load_training_session(run_dir)
    if existing is not None:
        return existing

    session = TrainingSession(session_id=str(uuid.uuid4()), started_at=time.time())
    save_training_session(run_dir, session)
    save_full_eval_state(
        run_dir,
        FullEvalState(training_session_id=session.session_id),
    )
    logger.info("Started new GEPA training session %s", session.session_id)
    return session


def gepa_state_from_current_session(run_dir: Path) -> bool:
    """Return True when ``gepa_state.bin`` was updated during the active training session."""
    session = load_training_session(run_dir)
    if session is None:
        return False
    state_path = run_dir / "gepa_state.bin"
    if not state_path.is_file():
        return False
    return state_path.stat().st_mtime >= session.started_at


def bootstrap_full_eval_state(
    run_dir: Path,
    monitor_key: str = "gepa_qwen25_3b",
    monitor_scores_path: Path | None = None,
) -> FullEvalState | None:
    """Seed ``full_eval_state.json`` from ``best_val_scores.json`` (opt-in via ``GEPA_BOOTSTRAP_FULL_EVAL_STATE``)."""
    if os.environ.get("GEPA_BOOTSTRAP_FULL_EVAL_STATE", "").lower() not in {"1", "true", "yes"}:
        return None
    if full_eval_state_path(run_dir).is_file():
        return None
    scores_path = monitor_scores_path or (RECIPE_DIR / "outputs" / "best_val_scores.json")
    if not scores_path.is_file():
        return None
    try:
        all_scores = json.loads(scores_path.read_text())
    except json.JSONDecodeError:
        logger.warning("Could not parse monitor scores file %s", scores_path)
        return None
    entry = all_scores.get(monitor_key)
    if not entry:
        return None
    submitted = list(entry.get("eval_submitted_steps", []))
    if not submitted and float(entry.get("best_score", -1.0)) < 0:
        return None
    session = load_training_session(run_dir)
    session_id = session.session_id if session is not None else ""
    state = FullEvalState(
        best_dev_score=float(entry.get("best_score", -1.0)),
        best_metric_calls=int(entry.get("best_step", -1)),
        best_program_idx=int(entry.get("best_program_idx", -1)),
        submitted_metric_calls=submitted,
        training_session_id=session_id,
    )
    save_full_eval_state(run_dir, state)
    logger.info(
        "Bootstrapped full_eval_state from monitor (%s): best=%.4f submitted=%s",
        monitor_key,
        state.best_dev_score,
        state.submitted_metric_calls,
    )
    return state


def load_full_eval_state(run_dir: Path) -> FullEvalState:
    session = load_training_session(run_dir)
    path = full_eval_state_path(run_dir)
    if not path.is_file():
        bootstrap_full_eval_state(run_dir)
        path = full_eval_state_path(run_dir)
    if not path.is_file():
        session_id = session.session_id if session is not None else ""
        return FullEvalState(training_session_id=session_id)
    data = json.loads(path.read_text())
    state = FullEvalState(
        best_dev_score=float(data.get("best_dev_score", -1.0)),
        best_metric_calls=int(data.get("best_metric_calls", -1)),
        best_program_idx=int(data.get("best_program_idx", -1)),
        submitted_metric_calls=list(data.get("submitted_metric_calls", [])),
        training_session_id=str(data.get("training_session_id", "")),
    )
    if session is not None and state.training_session_id != session.session_id:
        logger.info(
            "Resetting full_eval_state (session %s != current %s)",
            state.training_session_id or "<none>",
            session.session_id,
        )
        return FullEvalState(training_session_id=session.session_id)
    return state


def save_full_eval_state(run_dir: Path, state: FullEvalState) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    session = load_training_session(run_dir)
    session_id = state.training_session_id or (session.session_id if session is not None else "")
    payload = {
        "best_dev_score": state.best_dev_score,
        "best_metric_calls": state.best_metric_calls,
        "best_program_idx": state.best_program_idx,
        "submitted_metric_calls": state.submitted_metric_calls,
        "training_session_id": session_id,
    }
    full_eval_state_path(run_dir).write_text(json.dumps(payload, indent=2))


def save_gepa_prompt(
    run_dir: Path,
    metric_calls: int,
    program_idx: int,
    prompt: dict[str, str],
) -> Path | None:
    """Persist a prompt snapshot for full-test eval submission."""
    try:
        from search_r1_gepa.search_r1_gepa_adapter import INSTRUCTION_COMPONENT
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning("GEPA deps unavailable, cannot save prompt: %s", exc)
        return None

    prompt_dir = run_dir / "monitored_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    instruction = prompt.get(INSTRUCTION_COMPONENT, "")
    prompt_path = prompt_dir / f"best_m{metric_calls}_idx{program_idx}.txt"
    prompt_path.write_text(instruction)
    latest = run_dir / "best_instruction_prompt.txt"
    latest.write_text(instruction)
    return prompt_path


def submit_gepa_eval_job(
    *,
    eval_job_tag: str,
    metric_calls: int,
    prompt_path: Path,
    addr_file: str,
    run_dir_rel: str = "outputs/gepa_qwen25_3b",
    rewrite_flag: str = "",
    dry_run: bool = False,
) -> str | None:
    """Generate and submit a bsub GEPA full-test eval job. Returns LSF job id."""
    from monitor_retrieval_servers import ensure_eval_retrieval_server

    ensure_eval_retrieval_server(addr_file, dry_run=dry_run)

    template = GEPA_EVAL_TEMPLATE.read_text()
    script = template.replace("%EVAL_TAG%", eval_job_tag)
    script = script.replace("%METRIC_CALLS%", str(metric_calls))
    script = script.replace("%PROMPT_PATH%", str(prompt_path))
    script = script.replace("%ADDR_FILE%", addr_file)
    script = script.replace("%RUN_DIR_REL%", run_dir_rel)
    script = script.replace("%REWRITE_FLAG%", rewrite_flag)

    tmp_bsub = EVAL_GENERATED_DIR / f"eval_{eval_job_tag}_m{metric_calls}.bsub"
    EVAL_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    tmp_bsub.write_text(script)

    if dry_run:
        logger.info("[DRY RUN] Would submit GEPA full-test eval: %s", tmp_bsub)
        return None

    result = subprocess.run(f"bsub < {tmp_bsub}", capture_output=True, text=True, shell=True)
    job_match = re.search(r"Job <(\d+)>", result.stdout)
    if job_match:
        job_id = job_match.group(1)
        logger.info(
            "Submitted GEPA full-test eval job %s for metric_calls=%d prompt=%s",
            job_id,
            metric_calls,
            prompt_path,
        )
        return job_id
    logger.warning("Unexpected bsub output: stdout=%r stderr=%r", result.stdout, result.stderr)
    return None


def resolve_best_metric_calls(run_dir: Path, best_score: float, fallback: int) -> int:
    """Return the WandB ``metric_calls`` step for a dev-subset best score.

  Prefer the step recorded in ``full_eval_state.json`` when the score matches so
  monitor/end-of-run triggers log ``test/em`` at the same step as in-process triggers.
    """
    fe_state = load_full_eval_state(run_dir)
    if fe_state.best_metric_calls >= 0 and abs(fe_state.best_dev_score - best_score) < 1e-9:
        return fe_state.best_metric_calls
    return fallback


def load_best_from_gepa_state(run_dir: Path) -> tuple[float, int, int, dict[str, str]] | None:
    """Load best prompt/score from ``gepa_state.bin`` if present."""
    state_path = run_dir / "gepa_state.bin"
    if not state_path.is_file():
        return None
    try:
        from gepa.core.state import GEPAState
        from gepa.strategies.eval_policy import FullEvaluationPolicy

        state = GEPAState.load(str(run_dir))
        policy = FullEvaluationPolicy()
        best_idx = policy.get_best_program(state)
        best_score = policy.get_valset_score(best_idx, state)
        metric_calls = resolve_best_metric_calls(run_dir, best_score, state.total_num_evals)
        prompt = state.program_candidates[best_idx]
        return best_score, metric_calls, best_idx, prompt
    except Exception as exc:
        logger.warning("Failed to load gepa_state.bin from %s: %s", run_dir, exc)
        return None


def sync_monitor_state_from_full_eval(
    monitor_best_score: float,
    monitor_best_step: int,
    monitor_best_program_idx: int,
    monitor_submitted_steps: list[int],
    run_dir: Path,
) -> tuple[float, int, int, list[int]]:
    """Align monitor tracking with ``full_eval_state.json`` for one GEPA run."""
    fe_state = load_full_eval_state(run_dir)
    best_score = monitor_best_score
    best_step = monitor_best_step
    best_program_idx = monitor_best_program_idx
    submitted = list(monitor_submitted_steps)
    if fe_state.best_dev_score > best_score:
        best_score = fe_state.best_dev_score
        best_step = fe_state.best_metric_calls
        best_program_idx = fe_state.best_program_idx
    if fe_state.submitted_metric_calls:
        submitted = list(fe_state.submitted_metric_calls)
    return best_score, best_step, best_program_idx, submitted


_BEST_VAL_SCORE_KEYS = ("best_score_on_valset", "best_valset_agg_score", "base_program_full_valset_score")


def _resolve_program_prompt(
    run_dir: Path,
    program_idx: int,
    *,
    use_rewrite: bool,
    prefer_best_from_state: bool = False,
) -> dict[str, str] | None:
    """Resolve the instruction prompt to evaluate for a full-test job."""
    if prefer_best_from_state:
        loaded = load_best_from_gepa_state(run_dir)
        if loaded is not None:
            return loaded[3]

    state_path = run_dir / "gepa_state.bin"
    if state_path.is_file():
        try:
            from gepa.core.state import GEPAState

            state = GEPAState.load(str(run_dir))
            candidates = state.program_candidates
            if 0 <= program_idx < len(candidates):
                return candidates[program_idx]
            if candidates:
                return candidates[0]
        except Exception as exc:
            logger.warning("Failed to read program %d from gepa_state.bin: %s", program_idx, exc)

    try:
        from search_r1_gepa.search_r1_gepa_adapter import default_seed_candidate

        return default_seed_candidate(use_rewrite=use_rewrite)
    except (ImportError, ModuleNotFoundError):
        logger.warning("Cannot resolve prompt for full-test eval trigger")
        return None


def collect_full_eval_trigger_candidates(
    metrics: dict[str, Any],
    run_dir: Path,
    *,
    step: int | None,
    use_rewrite: bool,
) -> list[tuple[float, int, int, dict[str, str]]]:
    """Return dev-subset historical-best candidates that may warrant full-test eval.

    Mirrors GRPO's ``val/reward`` record-break semantics: only aggregate dev-subset
    bests (``best_score_on_valset`` and related keys) trigger eval, not per-candidate
    ``val_program_average`` scores for non-record candidates.
    """
    metric_calls = int(metrics.get("total_metric_calls", step if step is not None else 0))
    candidates: list[tuple[float, int, int, dict[str, str]]] = []

    for key in _BEST_VAL_SCORE_KEYS:
        if key not in metrics:
            continue
        dev_score = float(metrics[key])
        loaded = load_best_from_gepa_state(run_dir)
        if loaded is not None:
            program_idx = loaded[2]
            prompt = loaded[3]
        else:
            program_idx = int(
                metrics.get(
                    "best_program_as_per_agg_score_valset",
                    metrics.get("new_program_idx", 0),
                )
            )
            prompt = _resolve_program_prompt(
                run_dir,
                program_idx,
                use_rewrite=use_rewrite,
                prefer_best_from_state=True,
            )
        if prompt is not None:
            candidates.append((dev_score, metric_calls, program_idx, prompt))
        break

    return candidates


def maybe_trigger_full_eval_from_metrics(
    metrics: dict[str, Any],
    run_dir: Path,
    *,
    step: int | None,
    eval_job_tag: str = DEFAULT_EVAL_JOB_TAG,
    addr_file: str = DEFAULT_ADDR_FILE,
    run_dir_rel: str = "outputs/gepa_qwen25_3b",
    use_rewrite: bool = False,
    dry_run: bool = False,
) -> bool:
    """Submit full-test eval when logged metrics include a new dev-subset EM record."""
    triggered = False
    for dev_score, metric_calls, program_idx, prompt in collect_full_eval_trigger_candidates(
        metrics, run_dir, step=step, use_rewrite=use_rewrite
    ):
        if maybe_trigger_full_eval(
            run_dir=run_dir,
            dev_score=dev_score,
            metric_calls=metric_calls,
            program_idx=program_idx,
            prompt=prompt,
            eval_job_tag=eval_job_tag,
            addr_file=addr_file,
            run_dir_rel=run_dir_rel,
            use_rewrite=use_rewrite,
            dry_run=dry_run,
        ):
            triggered = True
    return triggered


def maybe_trigger_full_eval(
    *,
    run_dir: Path,
    dev_score: float,
    metric_calls: int,
    program_idx: int,
    prompt: dict[str, str],
    eval_job_tag: str = DEFAULT_EVAL_JOB_TAG,
    addr_file: str = DEFAULT_ADDR_FILE,
    run_dir_rel: str = "outputs/gepa_qwen25_3b",
    use_rewrite: bool = False,
    dry_run: bool = False,
) -> bool:
    """Submit full-test eval when ``dev_score`` beats the tracked dev-subset best."""
    session = load_training_session(run_dir)
    if session is None:
        logger.debug("Skipping full-test eval trigger — no active training session in %s", run_dir)
        return False

    state = load_full_eval_state(run_dir)
    if state.training_session_id != session.session_id:
        state = FullEvalState(training_session_id=session.session_id)
    if program_idx < 0:
        loaded = load_best_from_gepa_state(run_dir)
        if loaded is not None:
            program_idx = loaded[2]
            prompt = loaded[3]
        else:
            program_idx = 0

    if dev_score <= state.best_dev_score:
        return False
    if metric_calls in state.submitted_metric_calls:
        logger.info(
            "Dev-subset new best %.4f at metric_calls=%d but eval already submitted",
            dev_score,
            metric_calls,
        )
        state.best_dev_score = dev_score
        state.best_metric_calls = metric_calls
        state.best_program_idx = program_idx
        save_full_eval_state(run_dir, state)
        return False

    logger.info(
        "Dev-subset new best %.4f (prev %.4f) at metric_calls=%d program_idx=%d — triggering full-test eval",
        dev_score,
        state.best_dev_score,
        metric_calls,
        program_idx,
    )

    prompt_path = save_gepa_prompt(run_dir, metric_calls, program_idx, prompt)
    if prompt_path is None:
        return False

    job_id = submit_gepa_eval_job(
        eval_job_tag=eval_job_tag,
        metric_calls=metric_calls,
        prompt_path=prompt_path,
        addr_file=addr_file,
        run_dir_rel=run_dir_rel,
        rewrite_flag="--rewrite" if use_rewrite else "",
        dry_run=dry_run,
    )
    if job_id or dry_run:
        state.best_dev_score = dev_score
        state.best_metric_calls = metric_calls
        state.best_program_idx = program_idx
        state.submitted_metric_calls.append(metric_calls)
        save_full_eval_state(run_dir, state)
        return True
    return False


def maybe_trigger_full_eval_from_state_file(
    run_dir: Path,
    *,
    eval_job_tag: str = DEFAULT_EVAL_JOB_TAG,
    addr_file: str = DEFAULT_ADDR_FILE,
    run_dir_rel: str = "outputs/gepa_qwen25_3b",
    use_rewrite: bool = False,
    dry_run: bool = False,
) -> bool:
    """Load ``gepa_state.bin`` and trigger full-test eval if dev-subset best improved."""
    loaded = load_best_from_gepa_state(run_dir)
    if loaded is None:
        return False
    score, metric_calls, program_idx, prompt = loaded
    return maybe_trigger_full_eval(
        run_dir=run_dir,
        dev_score=score,
        metric_calls=metric_calls,
        program_idx=program_idx,
        prompt=prompt,
        eval_job_tag=eval_job_tag,
        addr_file=addr_file,
        run_dir_rel=run_dir_rel,
        use_rewrite=use_rewrite,
        dry_run=dry_run,
    )


def install_gepa_full_eval_trigger(
    *,
    run_dir: Path,
    eval_job_tag: str = DEFAULT_EVAL_JOB_TAG,
    addr_file: str = DEFAULT_ADDR_FILE,
    run_dir_rel: str = "outputs/gepa_qwen25_3b",
    use_rewrite: bool = False,
    dry_run: bool | None = None,
) -> None:
    """Patch GEPA ``ExperimentTracker.log_metrics`` to submit full-test eval on dev-subset records."""
    from gepa.logging.experiment_tracker import ExperimentTracker

    if getattr(ExperimentTracker, "_agl_full_eval_trigger_installed", False):
        return

    if dry_run is None:
        dry_run = os.environ.get("GEPA_FULL_EVAL_DRY_RUN", "").lower() in {"1", "true", "yes"}

    original_log_metrics = ExperimentTracker.log_metrics

    def log_metrics(self: ExperimentTracker, metrics: dict[str, Any], step: int | None = None) -> None:
        original_log_metrics(self, metrics, step=step)
        maybe_trigger_full_eval_from_metrics(
            metrics,
            run_dir,
            step=step,
            eval_job_tag=eval_job_tag,
            addr_file=addr_file,
            run_dir_rel=run_dir_rel,
            use_rewrite=use_rewrite,
            dry_run=dry_run,
        )

    ExperimentTracker.log_metrics = log_metrics  # type: ignore[method-assign]
    ExperimentTracker._agl_full_eval_trigger_installed = True
    logger.info(
        "Installed GEPA full-test eval trigger (run_dir=%s dry_run=%s)",
        run_dir,
        dry_run,
    )
