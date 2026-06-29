#!/bin/bash
# Shared retrieval server launch (sourced from serve/serve_retrieval_*.bsub).
#
# Production default: dense e5 + FAISS with /search micro-batching (MultiGpuEncoder
# shards encode batches across all visible GPUs; faiss_gpu shards the index).
#
# BM25 fallback: set RETRIEVAL_MODE=bm25 and pass WIKI (uses GPU torch BM25 by default;
# bm25s/Lucene via BM25_BACKEND=bm25s|lucene). gpu_keepalive runs only for BM25 mode.
#
# Requires: PYTHON, ADDR_FILE, RECIPE. Optional: DENSE_INDEX, DENSE_CORPUS, RETRIEVER_MODEL.

RETRIEVAL_SCRIPT="${RECIPE}/scripts/retrieval_server.py"
KEEPALIVE_SCRIPT="${RECIPE}/scripts/gpu_keepalive.py"
KEEPALIVE_TARGET_UTIL=${KEEPALIVE_TARGET_UTIL:-0.05}
RETRIEVAL_MODE=${RETRIEVAL_MODE:-dense}
SEARCH_BATCH_SIZE=${SEARCH_BATCH_SIZE:-32}
SEARCH_BATCH_WAIT_MS=${SEARCH_BATCH_WAIT_MS:-10}
RECIPE_DATA="${RECIPE}/data"
DENSE_INDEX=${DENSE_INDEX:-${RECIPE_DATA}/e5_Flat.index}
DENSE_CORPUS=${DENSE_CORPUS:-${RECIPE_DATA}/wiki-18.jsonl}
RETRIEVER_MODEL=${RETRIEVER_MODEL:-intfloat/e5-base-v2}

if [[ "${RETRIEVAL_MODE}" == "bm25" ]]; then
    BM25_MAX_PROCESS_NUM=${BM25_MAX_PROCESS_NUM:-8}
    BM25_BACKEND=${BM25_BACKEND:-torch}
    TORCH_BM25_DEVICE=${TORCH_BM25_DEVICE:-cuda}
    echo "Starting retrieval server (BM25 backend=${BM25_BACKEND}, torch_bm25_device=${TORCH_BM25_DEVICE}, max_process_num=${BM25_MAX_PROCESS_NUM}, dp=all_visible_gpus tp=1) ..."
    "${PYTHON}" "${RETRIEVAL_SCRIPT}" \
        --wiki-dir "${WIKI}" \
        --retriever_name bm25 \
        --bm25-backend "${BM25_BACKEND}" \
        --torch-bm25-device "${TORCH_BM25_DEVICE}" \
        --max-process-num "${BM25_MAX_PROCESS_NUM}" \
        --addr-file "${ADDR_FILE}" &
    SRV_PID=$!
    export KEEPALIVE_TARGET_UTIL
    "${PYTHON}" "${KEEPALIVE_SCRIPT}" &
    KEEP_PID=$!
    trap "kill ${KEEP_PID} 2>/dev/null; rm -f ${ADDR_FILE}" EXIT
else
    echo "Starting retrieval server (dense e5, search_batch=${SEARCH_BATCH_SIZE}, faiss_gpu=on) ..."
    "${PYTHON}" "${RETRIEVAL_SCRIPT}" \
        --index_path "${DENSE_INDEX}" \
        --corpus_path "${DENSE_CORPUS}" \
        --retriever_name e5 \
        --retriever_model "${RETRIEVER_MODEL}" \
        --search-batch-size "${SEARCH_BATCH_SIZE}" \
        --search-batch-wait-ms "${SEARCH_BATCH_WAIT_MS}" \
        --faiss_gpu \
        --addr-file "${ADDR_FILE}" &
    SRV_PID=$!
    trap "rm -f ${ADDR_FILE}" EXIT
fi

wait ${SRV_PID}
