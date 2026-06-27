"""Monitor training logs and submit full eval when val score breaks the record.

Polls GRPO output logs for val/reward lines and GEPA state/logs for
best_score_on_valset, tracks best score per experiment, and submits full
test.parquet eval jobs whenever a new best is detected.

Usage:
    python scripts/monitor_best_and_eval.py [--poll-interval 60] [--dry-run]
    python scripts/monitor_best_and_eval.py --grpo-outputs-dir /path/to/outputs
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

RECIPE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = RECIPE_DIR / "outputs"
EVAL_GENERATED_DIR = RECIPE_DIR / "eval" / "generated"
BEST_SCORES_FILE = OUTPUTS_DIR / "best_val_scores.json"
EVAL_TEMPLATE = RECIPE_DIR / "eval" / "eval_checkpoint.bsub"
GEPA_EVAL_TEMPLATE = RECIPE_DIR / "eval" / "eval_gepa_prompt.bsub"

# Absolute checkpoint root — must match CHECKPOINT_ROOT in train_search_r1_agent.py.
CHECKPOINT_ROOT = "/proj/inf-scaling/zwhong/projs/asmi/agent-lightning/contrib/recipes/search_r1/checkpoints"

# GRPO worktree outputs (training jobs run from feature worktree).
DEFAULT_GRPO_OUTPUTS_DIR = (
    "/proj/inf-scaling/zwhong/projs/asmi/agent-lightning/.claude/worktrees/"
    "feature+searchr1-qwen25-repro-qwen3-eval/contrib/recipes/search_r1/outputs"
)

EXPERIMENTS = {
    "qwen7b": {
        "job_prefix": "train_qwen7b",
        "eval_job_tag": "qwen25_3b_baseline",
        "config": "qwen7b",
        "addr_file": "bm25_server_addr_baseline.txt",
        "ckpt_dir": f"{CHECKPOINT_ROOT}/searchr1_qwen7b",
    },
    "qwen3_8b": {
        "job_prefix": "train_qwen3_8b",
        "eval_job_tag": "qwen25_3b_baseline_a",
        "config": "qwen3_8b",
        "addr_file": "bm25_server_addr_baseline_a.txt",
        "ckpt_dir": f"{CHECKPOINT_ROOT}/searchr1_qwen3_8b",
    },
    "qwen3_8b_rewrite": {
        "job_prefix": "train_qwen3_8b_rewrite",
        "eval_job_tag": "qwen25_3b_rewrite",
        "config": "qwen3_8b_rewrite",
        "addr_file": "bm25_server_addr_rewrite.txt",
        "ckpt_dir": f"{CHECKPOINT_ROOT}/searchr1_qwen3_8b_rewrite",
    },
    "qwen3_8b_rewrite_em": {
        "job_prefix": "train_qwen3_8b_rewrite_em",
        "eval_job_tag": "qwen25_3b_rewrite_em",
        "config": "qwen3_8b_rewrite_em",
        "addr_file": "bm25_server_addr_rewrite_em.txt",
        "ckpt_dir": f"{CHECKPOINT_ROOT}/searchr1_qwen3_8b_rewrite_em",
    },
    "qwen3_8b_shaped": {
        "job_prefix": "train_qwen3_8b_shaped",
        "eval_job_tag": "qwen25_3b_shaped",
        "config": "qwen3_8b_shaped",
        "addr_file": "bm25_server_addr_shaped.txt",
        "ckpt_dir": f"{CHECKPOINT_ROOT}/searchr1_qwen3_8b_shaped",
    },
}

GEPA_EXPERIMENT = {
    "name": "gepa_qwen25_3b",
    "job_prefix": "train_gepa_qwen25_3b",
    "eval_job_tag": "qwen25_3b_gepa",
    "run_dir": OUTPUTS_DIR / "gepa_qwen25_3b",
    "addr_file": "bm25_server_addr_gepa.txt",
}

VAL_SCORE_PATTERN = re.compile(r"step:(\d+)\s.*?val/reward:([\d.]+)")
GEPA_SEED_VAL_PATTERN = re.compile(r"Seed val/em=([\d.]+)")
GEPA_BEST_VAL_PATTERN = re.compile(r"Best score on valset: ([\d.]+)")


@dataclass
class ExperimentState:
    best_score: float = -1.0
    best_step: int = -1
    eval_submitted_steps: List[int] = field(default_factory=list)
    last_log_pos: int = 0
    best_program_idx: int = -1
    last_state_mtime: float = 0.0


def load_best_scores() -> Dict[str, ExperimentState]:
    """Load persisted best scores from disk."""
    states: Dict[str, ExperimentState] = {}
    if BEST_SCORES_FILE.exists():
        data = json.loads(BEST_SCORES_FILE.read_text())
        for name, s in data.items():
            states[name] = ExperimentState(
                best_score=s["best_score"],
                best_step=s["best_step"],
                eval_submitted_steps=s.get("eval_submitted_steps", []),
                last_log_pos=s.get("last_log_pos", 0),
                best_program_idx=s.get("best_program_idx", -1),
                last_state_mtime=s.get("last_state_mtime", 0.0),
            )
    return states


def save_best_scores(states: Dict[str, ExperimentState]) -> None:
    """Persist best scores to disk."""
    data = {}
    for name, s in states.items():
        data[name] = {
            "best_score": s.best_score,
            "best_step": s.best_step,
            "eval_submitted_steps": s.eval_submitted_steps,
            "last_log_pos": s.last_log_pos,
            "best_program_idx": s.best_program_idx,
            "last_state_mtime": s.last_state_mtime,
        }
    BEST_SCORES_FILE.write_text(json.dumps(data, indent=2))


def find_latest_log(outputs_dir: Path, prefix: str) -> Optional[Path]:
    """Find the most recently modified output log matching a job prefix."""
    logs = sorted(outputs_dir.glob(f"{prefix}.*.out"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def find_latest_err_log(outputs_dir: Path, prefix: str) -> Optional[Path]:
    """Find the most recently modified stderr log matching a job prefix."""
    logs = sorted(outputs_dir.glob(f"{prefix}.*.err"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def parse_new_val_scores(log_path: Path, last_pos: int) -> List[tuple[int, float]]:
    """Read new content from log and extract (step, val_reward) tuples."""
    scores = []
    with open(log_path) as f:
        f.seek(last_pos)
        new_content = f.read()
    for match in VAL_SCORE_PATTERN.finditer(new_content):
        step = int(match.group(1))
        score = float(match.group(2))
        scores.append((step, score))
    return scores


def parse_gepa_log_scores(log_path: Path, last_pos: int) -> List[tuple[int, float, int]]:
    """Parse GEPA stderr for (metric_calls_tag, score, program_idx) updates.

    Uses metric_calls=0 for seed val; for optimization iterations we rely on
    gepa_state.bin when available (log lines lack metric_calls).
    """
    del last_pos  # GEPA log parsing uses full-file scan for best lines only
    results: List[tuple[int, float, int]] = []
    content = log_path.read_text()
    seed_match = GEPA_SEED_VAL_PATTERN.search(content)
    if seed_match:
        results.append((0, float(seed_match.group(1)), 0))
    for match in GEPA_BEST_VAL_PATTERN.finditer(content):
        results.append((-1, float(match.group(1)), -1))
    return results


def load_gepa_state_best(run_dir: Path) -> Optional[tuple[float, int, int, dict[str, str]]]:
    """Load best prompt/score from gepa_state.bin if present."""
    state_path = run_dir / "gepa_state.bin"
    if not state_path.exists():
        return None
    try:
        from gepa.core.state import GEPAState
        from gepa.strategies.eval_policy import FullEvaluationPolicy

        state = GEPAState.load(str(run_dir))
        policy = FullEvaluationPolicy()
        best_idx = policy.get_best_program(state)
        best_score = policy.get_valset_score(best_idx, state)
        metric_calls = state.total_num_evals
        prompt = state.program_candidates[best_idx]
        return best_score, metric_calls, best_idx, prompt
    except Exception as exc:
        print(f"  WARNING: failed to load gepa_state.bin: {exc}", flush=True)
        return None


def save_gepa_prompt(run_dir: Path, metric_calls: int, program_idx: int, prompt: dict[str, str]) -> Path:
    """Persist monitored best prompt snapshot for eval submission."""
    from search_r1_gepa.search_r1_gepa_adapter import INSTRUCTION_COMPONENT

    prompt_dir = run_dir / "monitored_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / f"best_m{metric_calls}_idx{program_idx}.txt"
    instruction = prompt.get(INSTRUCTION_COMPONENT, "")
    prompt_path.write_text(instruction)
    latest = run_dir / "best_instruction_prompt.txt"
    latest.write_text(instruction)
    return prompt_path


def find_checkpoint(ckpt_dir: str, step: int) -> Optional[str]:
    """Find the actor checkpoint for a given step."""
    ckpt_root = Path(ckpt_dir)
    ckpt_path = ckpt_root / f"global_step_{step}" / "actor"
    if ckpt_path.exists():
        return str(ckpt_path)
    ckpt_path_alt = OUTPUTS_DIR / ckpt_dir / f"global_step_{step}" / "actor"
    if ckpt_path_alt.exists():
        return str(ckpt_path_alt)
    return None


def submit_eval_job(
    config: str,
    eval_job_tag: str,
    checkpoint_path: str,
    step: int,
    addr_file: str,
    dry_run: bool = False,
) -> Optional[str]:
    """Generate and submit a bsub GRPO eval job. Returns job ID or None."""
    template = EVAL_TEMPLATE.read_text()
    script = template.replace("%EVAL_TAG%", eval_job_tag)
    script = script.replace("%CONF%", config)
    script = script.replace("%STEP%", str(step))
    script = script.replace("%CKPT_PATH%", checkpoint_path)
    script = script.replace("%ADDR_FILE%", addr_file)

    tmp_bsub = EVAL_GENERATED_DIR / f"eval_{eval_job_tag}_step{step}.bsub"
    EVAL_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    tmp_bsub.write_text(script)

    if dry_run:
        print(f"  [DRY RUN] Would submit: {tmp_bsub}")
        return None

    result = subprocess.run(f"bsub < {tmp_bsub}", capture_output=True, text=True, shell=True)
    job_match = re.search(r"Job <(\d+)>", result.stdout)
    if job_match:
        job_id = job_match.group(1)
        print(f"  Submitted eval job {job_id} for {config} step {step}")
        return job_id
    print(f"  WARNING: bsub output unexpected: {result.stdout} {result.stderr}")
    return None


def submit_gepa_eval_job(
    eval_job_tag: str,
    metric_calls: int,
    prompt_path: Path,
    addr_file: str,
    dry_run: bool = False,
) -> Optional[str]:
    """Generate and submit a bsub GEPA full-test eval job."""
    template = GEPA_EVAL_TEMPLATE.read_text()
    script = template.replace("%EVAL_TAG%", eval_job_tag)
    script = script.replace("%METRIC_CALLS%", str(metric_calls))
    script = script.replace("%PROMPT_PATH%", str(prompt_path))
    script = script.replace("%ADDR_FILE%", addr_file)

    tmp_bsub = EVAL_GENERATED_DIR / f"eval_{eval_job_tag}_m{metric_calls}.bsub"
    EVAL_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    tmp_bsub.write_text(script)

    if dry_run:
        print(f"  [DRY RUN] Would submit GEPA eval: {tmp_bsub}")
        return None

    result = subprocess.run(f"bsub < {tmp_bsub}", capture_output=True, text=True, shell=True)
    job_match = re.search(r"Job <(\d+)>", result.stdout)
    if job_match:
        job_id = job_match.group(1)
        print(f"  Submitted GEPA eval job {job_id} for metric_calls={metric_calls}")
        return job_id
    print(f"  WARNING: bsub output unexpected: {result.stdout} {result.stderr}")
    return None


def monitor_grpo(states: Dict[str, ExperimentState], grpo_outputs_dir: Path, dry_run: bool) -> bool:
    """Poll GRPO training logs. Returns True if state was updated."""
    updated = False
    for name, exp in EXPERIMENTS.items():
        state = states[name]
        log_path = find_latest_log(grpo_outputs_dir, exp["job_prefix"])
        if log_path is None:
            continue

        file_size = log_path.stat().st_size
        if file_size <= state.last_log_pos:
            continue

        new_scores = parse_new_val_scores(log_path, state.last_log_pos)
        state.last_log_pos = file_size

        for step, score in new_scores:
            print(f"[{name}] step {step}: val/reward = {score:.4f} (best: {state.best_score:.4f})", flush=True)

            if score > state.best_score:
                state.best_score = score
                state.best_step = step
                updated = True
                print(f"  *** NEW BEST for {name}: {score:.4f} at step {step} ***", flush=True)

                if step in state.eval_submitted_steps:
                    print(f"  (eval already submitted for step {step}, skipping)")
                    continue

                ckpt_path = find_checkpoint(exp["ckpt_dir"], step)
                if ckpt_path:
                    job_id = submit_eval_job(
                        exp["config"], exp["eval_job_tag"], ckpt_path, step, exp["addr_file"], dry_run
                    )
                    if job_id or dry_run:
                        state.eval_submitted_steps.append(step)
                else:
                    print(f"  Checkpoint not found for step {step}, will retry next poll")
    return updated


def monitor_gepa(states: Dict[str, ExperimentState], dry_run: bool) -> bool:
    """Poll GEPA gepa_state.bin and training stderr. Returns True if state was updated."""
    name = GEPA_EXPERIMENT["name"]
    state = states[name]
    run_dir = Path(GEPA_EXPERIMENT["run_dir"])
    updated = False

    # Prefer authoritative gepa_state.bin when updated.
    state_path = run_dir / "gepa_state.bin"
    if state_path.exists():
        mtime = state_path.stat().st_mtime
        if mtime > state.last_state_mtime:
            loaded = load_gepa_state_best(run_dir)
            if loaded:
                score, metric_calls, program_idx, prompt = loaded
                state.last_state_mtime = mtime
                print(
                    f"[{name}] gepa_state: best_score={score:.4f} metric_calls={metric_calls} prog={program_idx}",
                    flush=True,
                )
                if score > state.best_score:
                    state.best_score = score
                    state.best_step = metric_calls
                    state.best_program_idx = program_idx
                    updated = True
                    print(
                        f"  *** NEW BEST for {name}: {score:.4f} at metric_calls={metric_calls} ***",
                        flush=True,
                    )
                    eval_key = metric_calls
                    if eval_key in state.eval_submitted_steps:
                        print(f"  (eval already submitted for metric_calls={eval_key}, skipping)")
                    else:
                        prompt_path = save_gepa_prompt(run_dir, metric_calls, program_idx, prompt)
                        job_id = submit_gepa_eval_job(
                            GEPA_EXPERIMENT["eval_job_tag"],
                            metric_calls,
                            prompt_path,
                            GEPA_EXPERIMENT["addr_file"],
                            dry_run,
                        )
                        if job_id or dry_run:
                            state.eval_submitted_steps.append(eval_key)

    # Fallback: seed val from stderr before gepa_state.bin exists.
    err_log = find_latest_err_log(OUTPUTS_DIR, GEPA_EXPERIMENT["job_prefix"])
    if err_log is None:
        return updated

    err_size = err_log.stat().st_size
    if err_size <= state.last_log_pos and state.best_score >= 0:
        return updated

    if err_size > state.last_log_pos:
        seed_scores = parse_gepa_log_scores(err_log, state.last_log_pos)
        state.last_log_pos = err_size
        for metric_calls, score, program_idx in seed_scores:
            if metric_calls < 0:
                continue  # log-only lines without metric_calls — state file handles these
            if score <= state.best_score:
                continue
            state.best_score = score
            state.best_step = metric_calls
            state.best_program_idx = program_idx
            updated = True
            print(f"  *** NEW BEST for {name} (log): {score:.4f} at metric_calls={metric_calls} ***", flush=True)
            if metric_calls in state.eval_submitted_steps:
                continue
            prompt_path = run_dir / "best_instruction_prompt.txt"
            if metric_calls == 0:
                try:
                    from search_r1_gepa.search_r1_gepa_adapter import default_seed_candidate

                    prompt_path = save_gepa_prompt(run_dir, 0, 0, default_seed_candidate())
                except ImportError as exc:
                    print(f"  WARNING: GEPA deps unavailable, skipping seed eval: {exc}", flush=True)
                    continue
            if prompt_path.exists():
                job_id = submit_gepa_eval_job(
                    GEPA_EXPERIMENT["eval_job_tag"],
                    metric_calls,
                    prompt_path,
                    GEPA_EXPERIMENT["addr_file"],
                    dry_run,
                )
                if job_id or dry_run:
                    state.eval_submitted_steps.append(metric_calls)

    return updated


def monitor_loop(poll_interval: int, grpo_outputs_dir: Path, dry_run: bool) -> None:
    """Main monitoring loop."""
    states = load_best_scores()

    for name in EXPERIMENTS:
        if name not in states:
            states[name] = ExperimentState()
    if GEPA_EXPERIMENT["name"] not in states:
        states[GEPA_EXPERIMENT["name"]] = ExperimentState()

    n_total = len(EXPERIMENTS) + 1
    print(f"Monitoring {n_total} experiments ({len(EXPERIMENTS)} GRPO + 1 GEPA), polling every {poll_interval}s", flush=True)
    print(f"GRPO outputs: {grpo_outputs_dir}", flush=True)
    print(f"GEPA outputs: {OUTPUTS_DIR}", flush=True)
    print(
        f"Current bests: {', '.join(f'{n}={s.best_score:.3f}@step{s.best_step}' for n, s in states.items())}",
        flush=True,
    )

    while True:
        updated = monitor_grpo(states, grpo_outputs_dir, dry_run)
        updated = monitor_gepa(states, dry_run) or updated

        if updated:
            save_best_scores(states)

        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor training and submit eval on new best val scores")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between log polls (default: 60)")
    parser.add_argument(
        "--grpo-outputs-dir",
        type=Path,
        default=Path(DEFAULT_GRPO_OUTPUTS_DIR),
        help="Directory containing GRPO train_*.out logs",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be submitted without actually submitting")
    args = parser.parse_args()

    monitor_loop(args.poll_interval, args.grpo_outputs_dir, args.dry_run)


if __name__ == "__main__":
    main()
