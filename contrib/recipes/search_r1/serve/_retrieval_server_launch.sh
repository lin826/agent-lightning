#!/bin/bash
# Shared retrieval server launch (sourced from serve/serve_retrieval_*.bsub).
#
# Default: CPU BM25 (bm25s) matching the original Search-R1 setup. Set RETRIEVAL_MODE=bm25
# and pass WIKI; choose the backend via BM25_BACKEND=bm25s|lucene (default bm25s). No GPU
# is required or used for BM25 hosting.
#
# Dense e5 + FAISS remains available for local dev (RETRIEVAL_MODE=dense) and uses GPUs.
#
# Requires: PYTHON, ADDR_FILE, RECIPE. Optional: DENSE_INDEX, DENSE_CORPUS, RETRIEVER_MODEL.

RETRIEVAL_SCRIPT="${RECIPE}/scripts/retrieval_server.py"
RETRIEVAL_MODE=${RETRIEVAL_MODE:-bm25}
SEARCH_BATCH_SIZE=${SEARCH_BATCH_SIZE:-32}
SEARCH_BATCH_WAIT_MS=${SEARCH_BATCH_WAIT_MS:-10}
RECIPE_DATA="${RECIPE}/data"
DENSE_INDEX=${DENSE_INDEX:-${RECIPE_DATA}/e5_Flat.index}
DENSE_CORPUS=${DENSE_CORPUS:-${RECIPE_DATA}/wiki-18.jsonl}
RETRIEVER_MODEL=${RETRIEVER_MODEL:-intfloat/e5-base-v2}

if [[ "${RETRIEVAL_MODE}" == "bm25" ]]; then
    BM25_MAX_PROCESS_NUM=${BM25_MAX_PROCESS_NUM:-8}
    BM25_BACKEND=${BM25_BACKEND:-bm25s}
    echo "Starting retrieval server (CPU BM25 backend=${BM25_BACKEND}, max_process_num=${BM25_MAX_PROCESS_NUM}) ..."
    "${PYTHON}" "${RETRIEVAL_SCRIPT}" \
        --wiki-dir "${WIKI}" \
        --retriever_name bm25 \
        --bm25-backend "${BM25_BACKEND}" \
        --max-process-num "${BM25_MAX_PROCESS_NUM}" \
        --addr-file "${ADDR_FILE}" &
    SRV_PID=$!
    trap "rm -f ${ADDR_FILE}" EXIT
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
