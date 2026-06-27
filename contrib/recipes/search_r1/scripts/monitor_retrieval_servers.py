"""Monitor BM25 retrieval servers for all Search-R1 training and eval variants.

Checks LSF job status, addr files, and HTTP ``GET /health`` for each variant
defined in the README pairing table. Detects unexpected exits, tracks addr-file
updates when servers restart, and can optionally resubmit serve jobs.

Usage:
    python scripts/monitor_retrieval_servers.py              # one-shot status
    python scripts/monitor_retrieval_servers.py --watch      # poll loop
    python scripts/monitor_retrieval_servers.py --watch --resubmit --expect-train
    python scripts/monitor_retrieval_servers.py --watch --resubmit --expect-eval
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

RECIPE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = RECIPE_DIR / "outputs"
SERVE_DIR = RECIPE_DIR / "serve"
STATE_FILE = OUTPUTS_DIR / "retrieval_monitor_state.json"

HEALTH_TIMEOUT_S = 8.0
ACTIVE_LSF_STATS = frozenset({"RUN", "PEND", "PSUSP", "USUSP", "SSUSP"})


class HealthState(str, Enum):
    HEALTHY = "healthy"
    STARTING = "starting"
    UNHEALTHY = "unhealthy"
    IDLE = "idle"


@dataclass(frozen=True)
class RetrievalVariant:
    name: str
    lsf_job_name: str
    serve_script: str
    addr_file: str
    train_job_name: str


RETRIEVAL_VARIANTS: Dict[str, RetrievalVariant] = {
    "baseline": RetrievalVariant(
        name="baseline",
        lsf_job_name="serve_bm25_qwen25_3b_baseline",
        serve_script="serve/serve_retrieval_baseline.bsub",
        addr_file="bm25_server_addr_baseline.txt",
        train_job_name="train_searchr1_qwen25_3b_baseline",
    ),
    "baseline_a": RetrievalVariant(
        name="baseline_a",
        lsf_job_name="serve_bm25_qwen25_3b_baseline_a",
        serve_script="serve/serve_retrieval_baseline_a.bsub",
        addr_file="bm25_server_addr_baseline_a.txt",
        train_job_name="train_searchr1_qwen25_3b_baseline_a",
    ),
    "rewrite": RetrievalVariant(
        name="rewrite",
        lsf_job_name="serve_bm25_qwen25_3b_rewrite",
        serve_script="serve/serve_retrieval_rewrite.bsub",
        addr_file="bm25_server_addr_rewrite.txt",
        train_job_name="train_searchr1_qwen25_3b_rewrite",
    ),
    "rewrite_em": RetrievalVariant(
        name="rewrite_em",
        lsf_job_name="serve_bm25_qwen25_3b_rewrite_em",
        serve_script="serve/serve_retrieval_rewrite_em.bsub",
        addr_file="bm25_server_addr_rewrite_em.txt",
        train_job_name="train_searchr1_qwen25_3b_rewrite_em",
    ),
    "shaped": RetrievalVariant(
        name="shaped",
        lsf_job_name="serve_bm25_qwen25_3b_shaped",
        serve_script="serve/serve_retrieval_shaped.bsub",
        addr_file="bm25_server_addr_shaped.txt",
        train_job_name="train_searchr1_qwen25_3b_shaped",
    ),
    "gepa": RetrievalVariant(
        name="gepa",
        lsf_job_name="serve_bm25_qwen25_3b_gepa",
        serve_script="serve/serve_retrieval_gepa.bsub",
        addr_file="bm25_server_addr_gepa.txt",
        train_job_name="train_searchr1_gepa_qwen25_3b",
    ),
}

EVAL_RETRIEVAL_VARIANTS: Dict[str, RetrievalVariant] = {
    "eval_baseline": RetrievalVariant(
        name="eval_baseline",
        lsf_job_name="serve_bm25_eval_qwen25_3b_baseline",
        serve_script="serve/serve_retrieval_eval_baseline.bsub",
        addr_file="bm25_server_addr_eval_baseline.txt",
        train_job_name="eval_searchr1_qwen25_3b_baseline",
    ),
    "eval_baseline_a": RetrievalVariant(
        name="eval_baseline_a",
        lsf_job_name="serve_bm25_eval_qwen25_3b_baseline_a",
        serve_script="serve/serve_retrieval_eval_baseline_a.bsub",
        addr_file="bm25_server_addr_eval_baseline_a.txt",
        train_job_name="eval_searchr1_qwen25_3b_baseline_a",
    ),
    "eval_rewrite": RetrievalVariant(
        name="eval_rewrite",
        lsf_job_name="serve_bm25_eval_qwen25_3b_rewrite",
        serve_script="serve/serve_retrieval_eval_rewrite.bsub",
        addr_file="bm25_server_addr_eval_rewrite.txt",
        train_job_name="eval_searchr1_qwen25_3b_rewrite",
    ),
    "eval_rewrite_em": RetrievalVariant(
        name="eval_rewrite_em",
        lsf_job_name="serve_bm25_eval_qwen25_3b_rewrite_em",
        serve_script="serve/serve_retrieval_eval_rewrite_em.bsub",
        addr_file="bm25_server_addr_eval_rewrite_em.txt",
        train_job_name="eval_searchr1_qwen25_3b_rewrite_em",
    ),
    "eval_shaped": RetrievalVariant(
        name="eval_shaped",
        lsf_job_name="serve_bm25_eval_qwen25_3b_shaped",
        serve_script="serve/serve_retrieval_eval_shaped.bsub",
        addr_file="bm25_server_addr_eval_shaped.txt",
        train_job_name="eval_searchr1_qwen25_3b_shaped",
    ),
    "eval_gepa": RetrievalVariant(
        name="eval_gepa",
        lsf_job_name="serve_bm25_eval_qwen25_3b_gepa",
        serve_script="serve/serve_retrieval_eval_gepa.bsub",
        addr_file="bm25_server_addr_eval_gepa.txt",
        train_job_name="eval_searchr1_qwen25_3b_gepa",
    ),
}

ALL_RETRIEVAL_VARIANTS: Dict[str, RetrievalVariant] = {**RETRIEVAL_VARIANTS, **EVAL_RETRIEVAL_VARIANTS}


@dataclass
class LsfJobRecord:
    job_id: int
    stat: str
    exit_code: Optional[int]


@dataclass
class VariantSnapshot:
    variant: str
    health_state: HealthState
    expected: bool
    lsf_stat: str
    job_id: Optional[int]
    exit_code: Optional[int]
    url: Optional[str]
    addr_mtime: Optional[float]
    health_ok: bool
    note: str = ""


@dataclass
class MonitorState:
    last_job_id: Dict[str, Optional[int]] = field(default_factory=dict)
    last_url: Dict[str, Optional[str]] = field(default_factory=dict)
    last_health: Dict[str, str] = field(default_factory=dict)
    resubmit_cooldown_until: Dict[str, float] = field(default_factory=dict)


def load_monitor_state() -> MonitorState:
    if not STATE_FILE.exists():
        return MonitorState()
    data = json.loads(STATE_FILE.read_text())
    return MonitorState(
        last_job_id=data.get("last_job_id", {}),
        last_url=data.get("last_url", {}),
        last_health=data.get("last_health", {}),
        resubmit_cooldown_until=data.get("resubmit_cooldown_until", {}),
    )


def save_monitor_state(state: MonitorState) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(asdict(state), indent=2))


def _parse_exit_code(raw: str) -> Optional[int]:
    raw = raw.strip()
    if not raw or raw == "-":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def query_lsf_jobs(job_name: str) -> List[LsfJobRecord]:
    """Return all LSF jobs matching *job_name*, newest first."""
    try:
        result = subprocess.run(
            ["bjobs", "-a", "-o", "jobid:12 job_name:40 stat:8 exit_code:10", "-J", job_name],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("bjobs unavailable: %s", exc)
        return []

    if result.returncode != 0:
        return []

    records: List[LsfJobRecord] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("JOBID"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            job_id = int(parts[0])
        except ValueError:
            continue
        stat = parts[2]
        exit_code = _parse_exit_code(parts[3]) if len(parts) > 3 else None
        records.append(LsfJobRecord(job_id=job_id, stat=stat, exit_code=exit_code))

    records.sort(key=lambda r: r.job_id, reverse=True)
    return records


def query_active_train_jobs() -> Dict[str, bool]:
    """Map variant name -> True when its train job is RUN/PEND."""
    active: Dict[str, bool] = {name: False for name in RETRIEVAL_VARIANTS}
    for name, variant in RETRIEVAL_VARIANTS.items():
        jobs = query_lsf_jobs(variant.train_job_name)
        active[name] = any(j.stat in ACTIVE_LSF_STATS for j in jobs)
    return active


def query_lsf_jobs_by_prefix(job_prefix: str) -> List[LsfJobRecord]:
    """Return LSF jobs whose name equals *job_prefix* or starts with ``{prefix}_``."""
    try:
        result = subprocess.run(
            ["bjobs", "-w", "-u", os.environ.get("USER", "")],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("bjobs unavailable: %s", exc)
        return []

    if result.returncode != 0:
        return []

    records: List[LsfJobRecord] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("JOBID"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        job_name = parts[6] if len(parts) > 6 else ""
        if job_name != job_prefix and not job_name.startswith(f"{job_prefix}_"):
            continue
        try:
            job_id = int(parts[0])
        except ValueError:
            continue
        stat = parts[2]
        records.append(LsfJobRecord(job_id=job_id, stat=stat, exit_code=None))

    records.sort(key=lambda r: r.job_id, reverse=True)
    return records


def query_active_eval_jobs() -> Dict[str, bool]:
    """Map eval variant name -> True when its eval job is RUN/PEND."""
    active: Dict[str, bool] = {name: False for name in EVAL_RETRIEVAL_VARIANTS}
    for name, variant in EVAL_RETRIEVAL_VARIANTS.items():
        jobs = query_lsf_jobs_by_prefix(variant.train_job_name)
        active[name] = any(j.stat in ACTIVE_LSF_STATS for j in jobs)
    return active


def read_addr_file(addr_path: Path) -> Tuple[Optional[str], Optional[float]]:
    if not addr_path.is_file():
        return None, None
    try:
        url = addr_path.read_text(encoding="utf-8").strip().rstrip("/")
        mtime = addr_path.stat().st_mtime
    except OSError as exc:
        logger.debug("Failed to read addr file %s: %s", addr_path, exc)
        return None, None
    if not url:
        return None, mtime
    return url, mtime


def check_health(url: str) -> bool:
    health_url = f"{url}/health"
    req = Request(health_url, method="GET")
    try:
        with urlopen(req, timeout=HEALTH_TIMEOUT_S) as resp:
            if resp.status != 200:
                return False
            body = resp.read().decode("utf-8", errors="replace")
    except (URLError, TimeoutError, OSError, ValueError):
        return False
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "ok"


def _pick_active_job(jobs: List[LsfJobRecord]) -> Optional[LsfJobRecord]:
    for job in jobs:
        if job.stat in ACTIVE_LSF_STATS:
            return job
    return jobs[0] if jobs else None


def inspect_variant(variant: RetrievalVariant, expected: bool) -> VariantSnapshot:
    jobs = query_lsf_jobs(variant.lsf_job_name)
    active = _pick_active_job(jobs)
    addr_path = OUTPUTS_DIR / variant.addr_file
    url, addr_mtime = read_addr_file(addr_path)
    health_ok = check_health(url) if url else False

    lsf_stat = active.stat if active else "NONE"
    job_id = active.job_id if active else None
    exit_code = active.exit_code if active and active.stat not in ACTIVE_LSF_STATS else None

    note = ""
    if active and active.stat in ACTIVE_LSF_STATS and not url:
        health_state = HealthState.STARTING
        note = "LSF job active; waiting for addr file"
    elif active and active.stat in ACTIVE_LSF_STATS and url and not health_ok:
        health_state = HealthState.STARTING
        note = "LSF job active; addr present but /health not ready"
    elif health_ok:
        health_state = HealthState.HEALTHY
    elif expected:
        health_state = HealthState.UNHEALTHY
        if active and active.stat not in ACTIVE_LSF_STATS:
            if active.exit_code == 130:
                note = "Job exited (SIGINT/130 — may be user bkill)"
            elif active.exit_code is not None and active.exit_code != 0:
                note = f"Job exited unexpectedly (code {active.exit_code})"
            else:
                note = "Job exited; server unreachable"
        elif not active:
            note = "No LSF job; server unreachable"
        elif url:
            note = "Addr file present but /health failed"
        else:
            note = "Expected server missing"
    else:
        health_state = HealthState.IDLE
        if jobs and jobs[0].stat not in ACTIVE_LSF_STATS:
            note = f"Last job {jobs[0].job_id} {jobs[0].stat}"

    return VariantSnapshot(
        variant=variant.name,
        health_state=health_state,
        expected=expected,
        lsf_stat=lsf_stat,
        job_id=job_id,
        exit_code=exit_code,
        url=url,
        addr_mtime=addr_mtime,
        health_ok=health_ok,
        note=note,
    )


def eval_variant_for_addr_file(addr_file: str) -> Optional[str]:
    """Map an eval addr filename to its ``EVAL_RETRIEVAL_VARIANTS`` key."""
    for name, variant in EVAL_RETRIEVAL_VARIANTS.items():
        if variant.addr_file == addr_file:
            return name
    return None


def ensure_eval_retrieval_server(addr_file: str, *, dry_run: bool = False) -> Optional[str]:
    """Submit eval BM25 serve job when no active job and server is not healthy.

    Called before full-test eval submission so eval jobs poll the dedicated eval
    addr file instead of blocking forever or falling back to a training server.
    """
    variant_name = eval_variant_for_addr_file(addr_file)
    if variant_name is None:
        logger.warning("No eval retrieval variant for addr file %s; skipping serve ensure", addr_file)
        return None

    variant = EVAL_RETRIEVAL_VARIANTS[variant_name]
    snap = inspect_variant(variant, expected=True)

    if snap.health_state == HealthState.HEALTHY:
        logger.info("Eval retrieval server %s already healthy at %s", variant_name, snap.url)
        return str(snap.job_id) if snap.job_id is not None else None

    if snap.lsf_stat in ACTIVE_LSF_STATS:
        logger.info(
            "Eval retrieval server %s already %s (job %s); eval will poll addr file",
            variant_name,
            snap.lsf_stat,
            snap.job_id,
        )
        return str(snap.job_id) if snap.job_id is not None else None

    logger.info("No active/healthy eval retrieval server for %s; submitting serve job", variant_name)
    return submit_serve_job(variant, dry_run=dry_run)


def submit_serve_job(variant: RetrievalVariant, dry_run: bool) -> Optional[str]:
    serve_path = RECIPE_DIR / variant.serve_script
    if not serve_path.is_file():
        logger.error("Serve script missing: %s", serve_path)
        return None

    if dry_run:
        print(f"  [DRY RUN] Would resubmit: bsub < {serve_path}", flush=True)
        return None

    result = subprocess.run(
        f"bsub < {serve_path}",
        capture_output=True,
        text=True,
        shell=True,
        cwd=str(RECIPE_DIR),
    )
    job_match = re.search(r"Job <(\d+)>", result.stdout)
    if job_match:
        job_id = job_match.group(1)
        print(f"  Resubmitted {variant.name} serve job {job_id} via {serve_path.name}", flush=True)
        return job_id
    print(f"  WARNING: bsub failed for {variant.name}: {result.stdout} {result.stderr}", flush=True)
    return None


def _format_snapshot(s: VariantSnapshot) -> str:
    url_part = s.url or "-"
    job_part = str(s.job_id) if s.job_id is not None else "-"
    expect = "yes" if s.expected else "no"
    note = f" ({s.note})" if s.note else ""
    return (
        f"{s.variant:12} {s.health_state.value:10} expect={expect:3} "
        f"lsf={s.lsf_stat:6} job={job_part:8} health={'ok' if s.health_ok else 'fail':4} url={url_part}{note}"
    )


def _resolve_expected(
    expect_all: bool,
    expect_train: bool,
    expect_eval: bool,
    expect_variants: List[str],
    active_trains: Dict[str, bool],
    active_evals: Dict[str, bool],
) -> Dict[str, bool]:
    expected = {name: False for name in ALL_RETRIEVAL_VARIANTS}
    if expect_all:
        for name in expected:
            expected[name] = True
    if expect_train:
        for name, active in active_trains.items():
            if active:
                expected[name] = True
    if expect_eval:
        for name, active in active_evals.items():
            if active:
                expected[name] = True
    for name in expect_variants:
        if name in expected:
            expected[name] = True
    return expected


def run_check(
    *,
    expect_all: bool,
    expect_train: bool,
    expect_eval: bool,
    expect_variants: List[str],
    resubmit: bool,
    resubmit_cooldown_s: int,
    dry_run: bool,
    state: MonitorState,
) -> Tuple[List[VariantSnapshot], bool]:
    """Check all variants once. Returns snapshots and whether state changed."""
    active_trains = query_active_train_jobs() if expect_train else {}
    active_evals = query_active_eval_jobs() if expect_eval else {}
    expected_map = _resolve_expected(
        expect_all, expect_train, expect_eval, expect_variants, active_trains, active_evals
    )

    snapshots: List[VariantSnapshot] = []
    state_changed = False
    now = time.time()

    for name, variant in ALL_RETRIEVAL_VARIANTS.items():
        expected = expected_map[name]
        snap = inspect_variant(variant, expected=expected)
        snapshots.append(snap)

        prev_job = state.last_job_id.get(name)
        prev_url = state.last_url.get(name)
        prev_health = state.last_health.get(name)

        if snap.job_id is not None and snap.job_id != prev_job:
            print(f"  [{name}] NEW LSF job {snap.job_id} (was {prev_job})", flush=True)
            state.last_job_id[name] = snap.job_id
            state_changed = True
        elif snap.job_id is None and prev_job is not None and snap.lsf_stat == "NONE":
            state.last_job_id[name] = None
            state_changed = True

        if snap.url and snap.url != prev_url:
            print(f"  [{name}] NEW addr URL {snap.url} (was {prev_url})", flush=True)
            state.last_url[name] = snap.url
            state_changed = True
        elif snap.url is None and prev_url is not None:
            print(f"  [{name}] Addr file cleared (was {prev_url})", flush=True)
            state.last_url[name] = None
            state_changed = True

        if snap.health_state.value != prev_health:
            if prev_health == HealthState.HEALTHY.value and snap.health_state == HealthState.UNHEALTHY:
                print(f"  *** [{name}] UNEXPECTED EXIT / failure (was healthy) ***", flush=True)
            state.last_health[name] = snap.health_state.value
            state_changed = True

        should_resubmit = (
            resubmit
            and expected
            and snap.health_state == HealthState.UNHEALTHY
            and now >= state.resubmit_cooldown_until.get(name, 0.0)
        )
        if should_resubmit:
            # Avoid duplicate resubmit while a new job is already pending.
            if snap.lsf_stat in {"PEND", "RUN"}:
                print(f"  [{name}] Unhealthy but LSF job {snap.job_id} already {snap.lsf_stat}; skip resubmit", flush=True)
            else:
                print(f"  [{name}] Resubmitting serve job ...", flush=True)
                job_id = submit_serve_job(variant, dry_run=dry_run)
                if job_id or dry_run:
                    state.resubmit_cooldown_until[name] = now + resubmit_cooldown_s
                    state_changed = True

    return snapshots, state_changed


def print_report(snapshots: List[VariantSnapshot]) -> None:
    print("Retrieval server status:", flush=True)
    for snap in snapshots:
        print(f"  {_format_snapshot(snap)}", flush=True)

    unhealthy = [s for s in snapshots if s.health_state == HealthState.UNHEALTHY]
    starting = [s for s in snapshots if s.health_state == HealthState.STARTING]
    healthy = [s for s in snapshots if s.health_state == HealthState.HEALTHY]
    print(
        f"Summary: {len(healthy)} healthy, {len(starting)} starting, "
        f"{len(unhealthy)} unhealthy, {len(snapshots) - len(healthy) - len(starting) - len(unhealthy)} idle",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor Search-R1 BM25 retrieval servers")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll continuously (default: one-shot check and exit)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between checks in --watch mode (default: 60)",
    )
    parser.add_argument(
        "--expect-all",
        action="store_true",
        help="Treat every variant as required (alert on any unhealthy)",
    )
    parser.add_argument(
        "--expect-train",
        action="store_true",
        help="Expect serve jobs for variants whose train job is RUN/PEND",
    )
    parser.add_argument(
        "--expect-eval",
        action="store_true",
        help="Expect eval serve jobs for variants whose eval job is RUN/PEND",
    )
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="VARIANT",
        help="Mark a variant as required (repeatable; e.g. baseline, eval_rewrite, gepa)",
    )
    parser.add_argument(
        "--resubmit",
        action="store_true",
        help="Resubmit serve/*.bsub when an expected server is unhealthy",
    )
    parser.add_argument(
        "--resubmit-cooldown",
        type=int,
        default=300,
        help="Minimum seconds between resubmits per variant (default: 300)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without bsub")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    unknown = [v for v in args.expect if v not in ALL_RETRIEVAL_VARIANTS]
    if unknown:
        print(f"Unknown variants: {', '.join(unknown)}", file=sys.stderr)
        print(f"Valid: {', '.join(ALL_RETRIEVAL_VARIANTS)}", file=sys.stderr)
        return 2

    state = load_monitor_state()

    def once() -> int:
        snapshots, changed = run_check(
            expect_all=args.expect_all,
            expect_train=args.expect_train,
            expect_eval=args.expect_eval,
            expect_variants=args.expect,
            resubmit=args.resubmit,
            resubmit_cooldown_s=args.resubmit_cooldown,
            dry_run=args.dry_run,
            state=state,
        )
        print_report(snapshots)
        if changed:
            save_monitor_state(state)
        unhealthy_expected = [
            s for s in snapshots if s.expected and s.health_state == HealthState.UNHEALTHY
        ]
        return 1 if unhealthy_expected else 0

    if not args.watch:
        return once()

    print(
        f"Watching {len(RETRIEVAL_VARIANTS)} retrieval variants every {args.poll_interval}s "
        f"(resubmit={'on' if args.resubmit else 'off'})",
        flush=True,
    )
    while True:
        rc = once()
        if rc != 0 and not args.resubmit:
            logger.warning("One or more expected servers are unhealthy")
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
