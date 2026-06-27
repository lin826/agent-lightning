# Search-R1 Example

## Overview

This example implements **Search R1** within Agent Lightning. It also serves as a demonstration of a **framework-free agent training pipeline**, showing how to run end-to-end RL training without relying on specialized frameworks. **It's tested and compatible with Agent-lightning v0.2.x**.

The example is designed to run on a single node with 8 GPUs, each having at least 40 GB of memory.

## Included Files

| Path | Description |
|------|-------------|
| `scripts/` | Python entrypoints: `train_search_r1_agent.py`, `eval_search_r1_agent.py`, `eval_gepa_prompt.py`, `monitor_best_and_eval.py`, `monitor_retrieval_servers.py`, `strip_stale_checkpoint_optim.py`, `search_r1_agent.py`, `qa_em.py`, `retrieval_server.py`, `wandb_run.py` |
| `train/` | LSF bsub scripts for GRPO and GEPA training (`train_qwen3b*.bsub`, `train_gepa.bsub`) |
| `serve/` | LSF bsub scripts for per-variant BM25 retrieval servers (`serve_retrieval_*.bsub` for training, `serve_retrieval_eval_*.bsub` for eval) and optional watchdog (`monitor_retrieval_servers.bsub`) |
| `eval/` | Eval bsub templates (`eval_checkpoint.bsub`, `eval_gepa_prompt.bsub`); `eval/generated/` holds monitor-generated one-off eval jobs |
| `outputs/` | LSF logs (`.out`/`.err`), BM25 addr files, GEPA run state, monitor state |
| `checkpoints/` | VERL checkpoint roots per experiment variant |
| `search_r1_gepa/` | GEPA prompt-optimization baseline (frozen Qwen2.5-3B) |
| `data/` | Wikipedia corpus and train/test parquet (from `data_process.sh`) |
| `data_process.sh` | Prepares corpus, datasets, and `retriever` conda environment |
| `retrieval_launch.sh` | Local retrieval service for interactive dev (not LSF) |

### Layout migration (2026-06)

Job scripts moved from the recipe root into `train/`, `serve/`, and `eval/`; Python entrypoints into `scripts/`. **Already-running LSF jobs are unaffected** (they use absolute paths baked in at submit time). After pulling, resubmit with the new paths, e.g. `bsub < serve/serve_retrieval_baseline.bsub` then `bsub < train/train_qwen3b.bsub`. Run the monitor from the recipe root: `python scripts/monitor_best_and_eval.py`.

---

## GEPA Baseline (prompt optimization vs GRPO)

To compare **Genetic-Pareto prompt evolution** (GEPA) against GRPO weight updates on the same Search-R1 task:

- **Task model:** `Qwen/Qwen2.5-3B-Instruct` (frozen weights; only the instruction prompt evolves)
- **Data:** `hotpotqa` filter on `data/train.parquet` / `data/test_dev.parquet` (same as `qwen7b` / `qwen3_8b` GRPO configs)
- **Retrieval:** training uses `serve/serve_retrieval_gepa.bsub` (`outputs/bm25_server_addr_gepa.txt`); full-test eval uses `serve/serve_retrieval_eval_gepa.bsub` (`outputs/bm25_server_addr_eval_gepa.txt`)
- **Metric:** exact match via `qa_em.py`
- **WandB:** project `AgentLightning`, run `searchr1_qwen25_3b_gepa`

1. Start the GEPA-dedicated BM25 retrieval server (`bsub < serve/serve_retrieval_gepa.bsub`).
2. Submit the GEPA job:

```bash
bsub < train/train_gepa.bsub
```

Optional env vars: `GEPA_MAX_METRIC_CALLS` (default 1500), `GEPA_REFLECTION_LM` (default: same local vLLM endpoint).

### Retrieval server pairing (GRPO + GEPA)

Each variant needs **separate** BM25 servers for training and full-test eval. Do not share `serve_retrieval_*.bsub`, `serve_retrieval_eval_*.bsub`, or addr files across variants, and do not point eval jobs at training addr files (or vice versa).

| Variant | Train LSF job | Train serve script | Train addr file | Train script | Eval LSF job | Eval serve script | Eval addr file | Eval |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| baseline (`qwen7b`) | `serve_bm25_qwen25_3b_baseline` | `serve/serve_retrieval_baseline.bsub` | `outputs/bm25_server_addr_baseline.txt` | `train/train_qwen3b.bsub` | `serve_bm25_eval_qwen25_3b_baseline` | `serve/serve_retrieval_eval_baseline.bsub` | `outputs/bm25_server_addr_eval_baseline.txt` | `eval/eval_checkpoint.bsub` (via `scripts/monitor_best_and_eval.py`) |
| baseline_a (`qwen3_8b`) | `serve_bm25_qwen25_3b_baseline_a` | `serve/serve_retrieval_baseline_a.bsub` | `outputs/bm25_server_addr_baseline_a.txt` | `train/train_qwen3b_a.bsub` | `serve_bm25_eval_qwen25_3b_baseline_a` | `serve/serve_retrieval_eval_baseline_a.bsub` | `outputs/bm25_server_addr_eval_baseline_a.txt` | same |
| rewrite | `serve_bm25_qwen25_3b_rewrite` | `serve/serve_retrieval_rewrite.bsub` | `outputs/bm25_server_addr_rewrite.txt` | `train/train_qwen3b_rewrite.bsub` | `serve_bm25_eval_qwen25_3b_rewrite` | `serve/serve_retrieval_eval_rewrite.bsub` | `outputs/bm25_server_addr_eval_rewrite.txt` | same |
| rewrite_em | `serve_bm25_qwen25_3b_rewrite_em` | `serve/serve_retrieval_rewrite_em.bsub` | `outputs/bm25_server_addr_rewrite_em.txt` | `train/train_qwen3b_rewrite_em.bsub` | `serve_bm25_eval_qwen25_3b_rewrite_em` | `serve/serve_retrieval_eval_rewrite_em.bsub` | `outputs/bm25_server_addr_eval_rewrite_em.txt` | same |
| shaped | `serve_bm25_qwen25_3b_shaped` | `serve/serve_retrieval_shaped.bsub` | `outputs/bm25_server_addr_shaped.txt` | `train/train_qwen3b_shaped.bsub` | `serve_bm25_eval_qwen25_3b_shaped` | `serve/serve_retrieval_eval_shaped.bsub` | `outputs/bm25_server_addr_eval_shaped.txt` | same |
| gepa | `serve_bm25_qwen25_3b_gepa` | `serve/serve_retrieval_gepa.bsub` | `outputs/bm25_server_addr_gepa.txt` | `train/train_gepa.bsub` | `serve_bm25_eval_qwen25_3b_gepa` | `serve/serve_retrieval_eval_gepa.bsub` | `outputs/bm25_server_addr_eval_gepa.txt` | `eval/eval_gepa_prompt.bsub` |

Submit the matching **train** `serve/serve_retrieval_<variant>.bsub` before training. For full-test eval, submit the matching **eval** `serve/serve_retrieval_eval_<variant>.bsub` manually if needed — GEPA eval triggers (`scripts/gepa_full_eval.py`, used from `train_gepa.py` and `scripts/monitor_best_and_eval.py`) auto-submit the eval serve job when missing. GRPO eval jobs poll until their eval addr file appears. Do not reuse legacy paths such as `bm25_server_addr.txt` or `bm25_server_addr_qwen3.txt`. `scripts/monitor_best_and_eval.py` fills `%ADDR_FILE%` with the eval addr file from the table above when it generates eval bsub scripts under `eval/generated/`.

### WandB run pairing (train vs full-test eval)

Training and full-test eval use **separate** WandB runs so eval can log at past training steps without hitting WandB's monotonic step constraint.

| Role | Run id file | Resume env |
| :--- | :--- | :--- |
| GRPO training | `checkpoints/<variant>/wandb_run_id.txt` | `WANDB_RUN_ID` + `WANDB_RESUME=allow` in `train/train_qwen3b_*.bsub` |
| GRPO full-test eval | `checkpoints/<variant>/wandb_eval_run_id.txt` | `WANDB_RUN_ID` + `WANDB_RESUME=allow` in `eval/eval_checkpoint.bsub` |
| GEPA training | `outputs/gepa_qwen25_3b/wandb_run_id.txt` | set in `train/train_gepa.bsub` |
| GEPA full-test eval | `outputs/gepa_qwen25_3b/wandb_eval_run_id.txt` | set in `eval/eval_gepa_prompt.bsub` |

Each variant keeps its **own** eval run id file (variants never share one eval run). On the first eval for a variant, WandB creates a new run and the job persists its id to that variant's `wandb_eval_run_id.txt`. Later eval jobs at other checkpoint steps for the **same** variant resume that eval run. Training jobs must **not** read `wandb_eval_run_id.txt`; eval jobs must **not** read `wandb_run_id.txt`.

| Variant | Eval run id file |
| :--- | :--- |
| baseline (`qwen7b`) | `checkpoints/searchr1_qwen7b/wandb_eval_run_id.txt` |
| baseline_a (`qwen3_8b`) | `checkpoints/searchr1_qwen3_8b/wandb_eval_run_id.txt` |
| rewrite | `checkpoints/searchr1_qwen3_8b_rewrite/wandb_eval_run_id.txt` |
| rewrite_em | `checkpoints/searchr1_qwen3_8b_rewrite_em/wandb_eval_run_id.txt` |
| shaped | `checkpoints/searchr1_qwen3_8b_shaped/wandb_eval_run_id.txt` |
| gepa | `outputs/gepa_qwen25_3b/wandb_eval_run_id.txt` |

Do **not** place a shared `wandb_eval_run_id.txt` under `outputs/` or the recipe root; eval bsub scripts derive `CKPT_DIR` from each job's checkpoint path so ids stay variant-local.

When eval pollution or a crashed job leaves the training WandB run ahead of the saved checkpoint, fork a clean run with `scripts/fork_training_wandb_run.py`. **Backfill only through `latest_checkpointed_iteration.txt`**, not through the last step line in the training log — otherwise resumed training re-logs steps that were already backfilled and the curve diverges. With `--checkpoint-dir`, the script defaults `--max-backfill-step` from that file. Update `wandb_run_id.txt` and restart (not hot-swap) the training job so VERL picks up the new id at `wandb.init`.

### Retrieval server health monitoring

Each serve job runs `torch_bm25_server.py`, which writes its URL to the addr file after the index loads and exposes `GET /health` → `{"status": "ok"}`. Use `scripts/monitor_retrieval_servers.py` to check **all twelve train + eval variants** (LSF job status + addr file + HTTP health), detect unexpected exits, and notice addr-file updates when a server restarts.

```bash
# One-shot status (from recipe root)
python scripts/monitor_retrieval_servers.py

# Watch active training variants; resubmit serve jobs after failures
python scripts/monitor_retrieval_servers.py --watch --expect-train --resubmit

# Watch active eval variants; resubmit eval serve jobs after failures
python scripts/monitor_retrieval_servers.py --watch --expect-eval --resubmit

# Or submit the LSF watchdog (same flags baked in)
bsub < serve/monitor_retrieval_servers.bsub
```

State is persisted in `outputs/retrieval_monitor_state.json` (last job id, URL, health per variant). Exit code is non-zero when any **expected** variant is unhealthy (`--expect-all`, `--expect-train`, or `--expect baseline`).

### Checkpoint optimizer retention

VERL FSDP checkpoints store actor model shards (~14 GiB), optimizer shards (~23 GiB), and small `extra_state` / `data.pt` files per `global_step_N`. After each save, `AgentLightningTrainer` strips optimizer shards from **older** steps so only the latest checkpoint keeps optimizer state (~50% disk savings when many steps are retained).

- **Training resume:** point `VERL_RESUME_FROM_PATH` at the **latest** `global_step_N` (the one named in `latest_checkpointed_iteration.txt`). That checkpoint retains optimizer + LR scheduler state for a true resume.
- **Eval / export:** older checkpoints still have actor model weights (and HuggingFace tokenizer artifacts). Full-test eval uses `load_contents: ["model"]` only.
- **One-time cleanup** of existing trees:

```bash
python scripts/strip_stale_checkpoint_optim.py checkpoints/searchr1_qwen7b --dry-run
python scripts/strip_stale_checkpoint_optim.py checkpoints/searchr1_*
```

Set `trainer.strip_stale_optimizer: false` in the VERL config to disable automatic stripping during training.

---

## Prepare Data and Environment

Run the following script once to prepare data and the retriever environment:

```bash
bash data_process.sh
```

This script performs the following steps:

* Creates a new conda environment named **`retriever`**.
* Downloads the **Wikipedia data** used to build the retrieval database.
* Downloads the **training and testing datasets**.
* Stores all data under the newly created **`data/`** directory.

The environment setup and data-processing logic are adapted from [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1).

---

## Prepare Retrieval Server

To start the retrieval server, run:

```bash
bash retrieval_launch.sh
```

This script activates the previously created **`retriever`** environment and starts a **retrieval server** at `http://127.0.0.1:8000` using the downloaded Wikipedia data. The server receives user queries and returns a ranked list of retrieved text passages.

The retrieval server implementation is based on `search_r1/search/retrieval_server.py`](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py).

> ⚠️ **Note:** Keep the retrieval server running during training (for example, in a separate `tmux` session or terminal window).

---

## Run RL Training (GRPO) with Llama-3.2-3B-Instruct

1. **Start Ray**

   ```bash
   bash ../../scripts/restart_ray.sh
   ```

   > If you plan to use WandB for experiment tracking, set the environment variable
   > `WANDB_API_KEY` before starting Ray.

2. **Start the Training Server**
   In another terminal, run:

   ```bash
   python scripts/train_search_r1_agent.py llama
   ```

   This script starts the RL training. Each agent follows the Search-R1 workflow, retrieving information from the database and generating answers accordingly.

---

## Run RL Training (GRPO) with Qwen2.5-3B-Instruct on H100 80 GB GPUs

The `qwen7b` configuration targets `Qwen/Qwen2.5-3B-Instruct` and is optimised for nodes equipped with **NVIDIA H100 80 GB HBM3** GPUs. Compared to the default configuration it:

* raises `gpu_memory_utilization` to **0.6** so vLLM can use more of the available VRAM for KV-cache,
* **disables CPU offloading** for both actor and reference model parameters and the actor optimiser states, eliminating the PCIe bottleneck on each training step.

1. **Start Ray**

   ```bash
   bash ../../scripts/restart_ray.sh
   ```

2. **Start the Training Server**

   ```bash
   python scripts/train_search_r1_agent.py qwen7b
   ```

---

## Benchmark Results

We evaluated Search-R1 across seven diverse question-answering benchmarks, covering both General QA (NQ, TriviaQA, PopQA) and complex multi-hop reasoning tasks (HotpotQA, 2WikiMultiHopQA, Musique, and Bamboogle).

The following tables compare the performance of the original Search-R1 implementation and the Agent-Lightning version across various base models.

| Model | Source | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | Musique | Bamboogle |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-3B-Instruct** | **Search-R1 (Original)** | 34.1 | 54.5 | 37.8 | 32.4 | 31.9 | 10.3 | 26.4 |
| | **Agent-Lightning** | **45.3** | **61.7** | **43.8** | **42.6** | **36.4** | **17.1** | **37.6** |
| **Qwen2.5-7B-Instruct** | **Search-R1 (Original)** | 39.3 | 61.0 | 39.7 | 37.0 | 41.4 | 14.6 | 36.8 |
| | **Agent-Lightning** | **46.5** | **65.9** | **46.8** | **43.7** | **46.2** | **20.3** | **47.2** |
| **Llama-3.2-3B** | **Search-R1 (Reproduced)** | 26.3 | 49.0 | 23.0 | 21.6 | 27.3 | 4.5 | 9.7 |
| | **Agent-Lightning** | **29.6** | **51.9** | **25.7** | **23.2** | **28.3** | **5.8** | 9.6 |
