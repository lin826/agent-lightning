"""Monitor training logs and submit full eval when val score breaks the record.

Polls output logs for val/reward lines, tracks best score per experiment,
and submits an eval bsub job on test.parquet (hotpotqa, 7405 samples) whenever
a new best is detected.

Usage:
    python monitor_best_and_eval.py [--poll-interval 60] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

RECIPE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = RECIPE_DIR / "outputs"
BEST_SCORES_FILE = OUTPUTS_DIR / "best_val_scores.json"
EVAL_TEMPLATE = RECIPE_DIR / "eval_checkpoint.bsub"

EXPERIMENTS = {
    "qwen7b": {
        "job_prefix": "train_qwen7b",
        "config": "qwen7b",
        "addr_file": "bm25_server_addr.txt",
        "ckpt_dir": "checkpoints/searchr1_checkpoints",
    },
    "qwen3_8b": {
        "job_prefix": "train_qwen3_8b",
        "config": "qwen3_8b",
        "addr_file": "bm25_server_addr_qwen3.txt",
        "ckpt_dir": "checkpoints/searchr1_checkpoints",
    },
    "qwen3_8b_rewrite": {
        "job_prefix": "train_qwen3_8b_rewrite",
        "config": "qwen3_8b_rewrite",
        "addr_file": "bm25_server_addr_rewrite.txt",
        "ckpt_dir": "checkpoints/searchr1_checkpoints",
    },
}

VAL_SCORE_PATTERN = re.compile(r"step:(\d+)\s.*?val/reward:([\d.]+)")


@dataclass
class ExperimentState:
    best_score: float = -1.0
    best_step: int = -1
    eval_submitted_steps: List[int] = field(default_factory=list)
    last_log_pos: int = 0


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
        }
    BEST_SCORES_FILE.write_text(json.dumps(data, indent=2))


def find_latest_log(prefix: str) -> Optional[Path]:
    """Find the most recently modified output log matching a job prefix."""
    logs = sorted(OUTPUTS_DIR.glob(f"{prefix}.*.out"), key=lambda p: p.stat().st_mtime, reverse=True)
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


def find_checkpoint(ckpt_dir: str, step: int) -> Optional[str]:
    """Find the actor checkpoint for a given step."""
    ckpt_path = RECIPE_DIR / ckpt_dir / f"global_step_{step}" / "actor"
    if ckpt_path.exists():
        return str(ckpt_path)
    ckpt_path_alt = OUTPUTS_DIR / ckpt_dir / f"global_step_{step}" / "actor"
    if ckpt_path_alt.exists():
        return str(ckpt_path_alt)
    return None


def submit_eval_job(config: str, checkpoint_path: str, step: int, addr_file: str, dry_run: bool = False) -> Optional[str]:
    """Generate and submit a bsub eval job. Returns job ID or None."""
    template = EVAL_TEMPLATE.read_text()
    script = template.replace("%CONF%", config)
    script = script.replace("%STEP%", str(step))
    script = script.replace("%CKPT_PATH%", checkpoint_path)
    script = script.replace("%ADDR_FILE%", addr_file)

    tmp_bsub = OUTPUTS_DIR / f"eval_{config}_step{step}.bsub"
    tmp_bsub.write_text(script)

    if dry_run:
        print(f"  [DRY RUN] Would submit: {tmp_bsub}")
        return None

    result = subprocess.run(["bsub", "<", str(tmp_bsub)], capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        result = subprocess.run(f"bsub < {tmp_bsub}", capture_output=True, text=True, shell=True)

    job_match = re.search(r"Job <(\d+)>", result.stdout)
    if job_match:
        job_id = job_match.group(1)
        print(f"  Submitted eval job {job_id} for {config} step {step}")
        return job_id
    else:
        print(f"  WARNING: bsub output unexpected: {result.stdout} {result.stderr}")
        return None


def monitor_loop(poll_interval: int, dry_run: bool) -> None:
    """Main monitoring loop."""
    states = load_best_scores()

    for name in EXPERIMENTS:
        if name not in states:
            states[name] = ExperimentState()

    print(f"Monitoring {len(EXPERIMENTS)} experiments, polling every {poll_interval}s")
    print(f"Current bests: {', '.join(f'{n}={s.best_score:.3f}@step{s.best_step}' for n, s in states.items())}")

    while True:
        updated = False

        for name, exp in EXPERIMENTS.items():
            state = states[name]
            log_path = find_latest_log(exp["job_prefix"])
            if log_path is None:
                continue

            file_size = log_path.stat().st_size
            if file_size <= state.last_log_pos:
                continue

            new_scores = parse_new_val_scores(log_path, state.last_log_pos)
            state.last_log_pos = file_size

            for step, score in new_scores:
                print(f"[{name}] step {step}: val/reward = {score:.4f} (best: {state.best_score:.4f})")

                if score > state.best_score:
                    state.best_score = score
                    state.best_step = step
                    updated = True
                    print(f"  *** NEW BEST for {name}: {score:.4f} at step {step} ***")

                    if step in state.eval_submitted_steps:
                        print(f"  (eval already submitted for step {step}, skipping)")
                        continue

                    ckpt_path = find_checkpoint(exp["ckpt_dir"], step)
                    if ckpt_path:
                        job_id = submit_eval_job(exp["config"], ckpt_path, step, exp["addr_file"], dry_run)
                        if job_id or dry_run:
                            state.eval_submitted_steps.append(step)
                    else:
                        print(f"  Checkpoint not found for step {step}, will retry next poll")
                        state.best_score = score
                        state.best_step = step

        if updated:
            save_best_scores(states)

        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor training and submit eval on new best val scores")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between log polls (default: 60)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be submitted without actually submitting")
    args = parser.parse_args()

    monitor_loop(args.poll_interval, args.dry_run)


if __name__ == "__main__":
    main()
