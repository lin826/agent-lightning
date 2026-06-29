# Copyright (c) Microsoft. All rights reserved.

"""GPU-accelerated BM25 via bm25_pt with multi-GPU row sharding.

Adapted from prompt-policy-rl ``torch_bm25_tools.py``. Shards ``_corpus_scores`` row-wise
across visible GPUs (dp=N, tp=1) so scoring uses all granted GPUs instead of pinning cuda:0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class _Shard:
    bm25: object
    row_offset: int
    device: str


@dataclass
class _TorchBM25Store:
    shards: List[_Shard]
    corpus: List[str]
    docs_by_title: Dict[str, str]
    devices: List[str]


def resolve_torch_bm25_devices(device: str) -> List[str]:
    """Expand ``device`` into the list of devices to shard over."""
    if device == "cuda":
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return [f"cuda:{i}" for i in range(n)] if n > 0 else ["cpu"]
    return [device]


def _load_or_build_corpus_scores(
    *,
    corpus: Sequence[str],
    cache_dir: Path,
    k1: float,
    b: float,
) -> Tuple[object, torch.Tensor]:
    from bm25_pt import BM25  # lazy: heavy import

    cache_dir.mkdir(parents=True, exist_ok=True)
    state_path = cache_dir / "torch_bm25_state.pt"

    bm25 = BM25(k1=k1, b=b, device="cpu")

    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        return bm25.tokenizer, state["_corpus_scores"].coalesce()

    logger.info("Building torch BM25 corpus scores (one-time, CPU); cache -> %s", state_path)
    bm25.index(list(corpus))
    scores_cpu = bm25._corpus_scores.cpu().coalesce()  # type: ignore[attr-defined]
    torch.save(
        {
            "_corpus": bm25._corpus.cpu(),  # type: ignore[attr-defined]
            "_corpus_scores": scores_cpu,
            "_corpus_lengths": bm25._corpus_lengths.cpu(),  # type: ignore[attr-defined]
            "_average_document_length": bm25._average_document_length,  # type: ignore[attr-defined]
            "_IDF": bm25._IDF.cpu(),  # type: ignore[attr-defined]
            "_documents_containing_word": bm25._documents_containing_word.cpu(),  # type: ignore[attr-defined]
            "_word_counts": bm25._word_counts.cpu(),  # type: ignore[attr-defined]
        },
        state_path,
    )
    return bm25.tokenizer, scores_cpu


def _build_shards(tokenizer: object, scores_cpu: torch.Tensor, devices: List[str], k1: float, b: float) -> List[_Shard]:
    from bm25_pt import BM25  # lazy

    num_docs, vocab = int(scores_cpu.size(0)), int(scores_cpu.size(1))
    ndev = max(1, min(len(devices), num_docs))
    idx = scores_cpu.indices()
    rows, cols = idx[0], idx[1]
    vals = scores_cpu.values()
    bounds = [round(j * num_docs / ndev) for j in range(ndev + 1)]
    pos = torch.searchsorted(rows, torch.tensor(bounds, dtype=rows.dtype)).tolist()

    shards: List[_Shard] = []
    for j in range(ndev):
        row_start, row_end = bounds[j], bounds[j + 1]
        p0, p1 = pos[j], pos[j + 1]
        dev = devices[j]
        shard_scores = (
            torch.sparse_coo_tensor(
                torch.stack([rows[p0:p1] - row_start, cols[p0:p1]]),
                vals[p0:p1],
                size=(row_end - row_start, vocab),
            )
            .coalesce()
            .to(dev)
        )
        shard_bm25 = BM25(tokenizer=tokenizer, k1=k1, b=b, device=dev)
        shard_bm25._corpus_scores = shard_scores  # type: ignore[attr-defined]
        shards.append(_Shard(bm25=shard_bm25, row_offset=row_start, device=dev))
    return shards


def build_torch_bm25_store(
    *,
    corpus: List[str],
    cache_dir: Path,
    device: str = "cuda",
    k1: float = 0.9,
    b: float = 0.4,
    cache_docs: bool = True,
) -> _TorchBM25Store:
    devices = resolve_torch_bm25_devices(device)
    tokenizer, scores_cpu = _load_or_build_corpus_scores(corpus=corpus, cache_dir=cache_dir, k1=k1, b=b)
    shards = _build_shards(tokenizer, scores_cpu, devices, k1, b)
    del scores_cpu

    docs_by_title: Dict[str, str] = {}
    if cache_docs:
        for passage in corpus:
            if " | " in passage:
                title, text = passage.split(" | ", 1)
                docs_by_title.setdefault(title, text)

    logger.info(
        "Torch BM25 sharded %d docs across %d device(s): %s",
        len(corpus),
        len(shards),
        ", ".join(s.device for s in shards),
    )
    return _TorchBM25Store(shards=shards, corpus=corpus, docs_by_title=docs_by_title, devices=devices)


def torch_bm25_topk(store: _TorchBM25Store, query: str, k: int) -> List[str]:
    cand_vals: List[torch.Tensor] = []
    cand_idxs: List[torch.Tensor] = []
    for shard in store.shards:
        scores: torch.Tensor = shard.bm25.score(query)  # type: ignore[no-untyped-call]
        m = min(k, scores.numel())
        top = torch.topk(scores, m)
        cand_vals.append(top.values.detach().cpu())
        cand_idxs.append((top.indices + shard.row_offset).detach().cpu())
    all_vals = torch.cat(cand_vals)
    all_idxs = torch.cat(cand_idxs)
    merged = torch.topk(all_vals, min(k, all_vals.numel()))
    return [store.corpus[int(all_idxs[i])] for i in merged.indices.tolist()]
