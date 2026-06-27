# Search-R1 Example

## Overview

This example implements **Search R1** within Agent Lightning. It also serves as a demonstration of a **framework-free agent training pipeline**, showing how to run end-to-end RL training without relying on specialized frameworks. **It's tested and compatible with Agent-lightning v0.2.x**.

The example is designed to run on a single node with 8 GPUs, each having at least 40 GB of memory.

## Included Files

| Path | Description |
|------|-------------|
| `scripts/` | Python entrypoints: `train_search_r1_agent.py`, `eval_search_r1_agent.py`, `eval_gepa_prompt.py`, `monitor_best_and_eval.py`, `search_r1_agent.py`, `qa_em.py`, `retrieval_server.py`, `wandb_run.py` |
| `train/` | LSF bsub scripts for GRPO and GEPA training (`train_qwen3b*.bsub`, `train_gepa.bsub`) |
| `serve/` | LSF bsub scripts for per-variant BM25 retrieval servers (`serve_retrieval_*.bsub`) |
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
- **Retrieval:** dedicated BM25 server from `serve/serve_retrieval_gepa.bsub` (`outputs/bm25_server_addr_gepa.txt`)
- **Metric:** exact match via `qa_em.py`
- **WandB:** project `AgentLightning`, run `searchr1_qwen25_3b_gepa`

1. Start the GEPA-dedicated BM25 retrieval server (`bsub < serve/serve_retrieval_gepa.bsub`).
2. Submit the GEPA job:

```bash
bsub < train/train_gepa.bsub
```

Optional env vars: `GEPA_MAX_METRIC_CALLS` (default 1500), `GEPA_REFLECTION_LM` (default: same local vLLM endpoint).

### Retrieval server pairing (GRPO + GEPA)

Each training or eval job must use its **own** BM25 retrieval server. Do not share `serve_retrieval_*.bsub` or `bm25_server_addr_*.txt` across variants.

| Variant | LSF job name | Serve script | Addr file | Train script | Eval |
| :--- | :--- | :--- | :--- | :--- | :--- |
| baseline (`qwen7b`) | `serve_bm25_qwen25_3b_baseline` | `serve/serve_retrieval_baseline.bsub` | `outputs/bm25_server_addr_baseline.txt` | `train/train_qwen3b.bsub` | `eval/eval_checkpoint.bsub` (via `scripts/monitor_best_and_eval.py`) |
| baseline_a (`qwen3_8b`) | `serve_bm25_qwen25_3b_baseline_a` | `serve/serve_retrieval_baseline_a.bsub` | `outputs/bm25_server_addr_baseline_a.txt` | `train/train_qwen3b_a.bsub` | same |
| rewrite | `serve_bm25_qwen25_3b_rewrite` | `serve/serve_retrieval_rewrite.bsub` | `outputs/bm25_server_addr_rewrite.txt` | `train/train_qwen3b_rewrite.bsub` | same |
| rewrite_em | `serve_bm25_qwen25_3b_rewrite_em` | `serve/serve_retrieval_rewrite_em.bsub` | `outputs/bm25_server_addr_rewrite_em.txt` | `train/train_qwen3b_rewrite_em.bsub` | same |
| shaped | `serve_bm25_qwen25_3b_shaped` | `serve/serve_retrieval_shaped.bsub` | `outputs/bm25_server_addr_shaped.txt` | `train/train_qwen3b_shaped.bsub` | same |
| gepa | `serve_bm25_qwen25_3b_gepa` | `serve/serve_retrieval_gepa.bsub` | `outputs/bm25_server_addr_gepa.txt` | `train/train_gepa.bsub` | `eval/eval_gepa_prompt.bsub` |

Submit the matching `serve/serve_retrieval_<variant>.bsub` job **before** train or full-test eval for that variant. Train and eval for the **same** variant share the same addr file; eval jobs poll until it appears. Do not reuse legacy paths such as `bm25_server_addr.txt` or `bm25_server_addr_qwen3.txt`. `scripts/monitor_best_and_eval.py` fills `%ADDR_FILE%` from the table above when it generates eval bsub scripts under `eval/generated/`.

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
