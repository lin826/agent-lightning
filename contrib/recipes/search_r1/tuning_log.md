# Search-R1 Hyperparameter Tuning Log

## 2026-06-27 — Step-20 checkpoint review

### Job status snapshot

| Variant | Job ID | Status | Current step | Notes |
|---------|--------|--------|--------------|-------|
| qwen7b (baseline) | 1775713 | RUN | 0 | Fresh run after 1775292 EXIT |
| qwen3_8b (baseline_a) | 1776067 | RUN | 0 | Resubmitted after 1775714 LSF file error |
| qwen3_8b_rewrite | 1775715 | RUN | 0 | |
| qwen3_8b_rewrite_em | 1775716 | RUN | 0 | |
| qwen3_8b_shaped | 1775717 | RUN | 0 | |

Retrieval servers RUN: `serve_bm25_searchr1` (1709333), `serve_bm25_qwen3` (1715810), `serve_bm25_rewrite` (1749640).

### qwen7b — tuned (prior run job 1764997 reached step 20)

**Metrics (job 1764997 / `best_val_scores.json`):**

| Step | val/reward | Notes |
|------|------------|-------|
| 0 | 0.26 | Initial baseline |
| 10 | 0.375 | +0.115 |
| 20 | 0.38 | +0.005 (plateau) |

At step 20: `actor/lr≈6.7e-8` (7% of 1e-6), `pg_clipfrac≈0.0006`, `training/reward≈0.44`.

**Change applied** (`config_train_qwen7b`):

- `lr_warmup_steps_ratio`: 0.95 → **0.70** — LR was still in early warmup at step 20, limiting policy updates despite rising train reward.

**Not applied to running job 1775713** — config takes effect on next restart/resubmit only.

### Other variants — waiting for step 20

No step-20 val metrics yet in current run (all at step 0). Prior partial runs did not reach step 20:

- `qwen3_8b_rewrite_em`: max step 4 (1764998)
- `qwen3_8b_shaped`: max step 6 (1764999)

**Step-0 val baselines (current run 177571x):** rewrite 0.26, rewrite_em 0.255, shaped 0.255, qwen7b 0.26.

Tuning deferred until step ≥ 20.
