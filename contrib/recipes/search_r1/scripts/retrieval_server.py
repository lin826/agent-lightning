# Copyright (c) Microsoft. All rights reserved.

# Copied and adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py

"""FastAPI retrieval server for Search-R1 training and eval.

Supports dense (FAISS + multi-GPU encoder with /search micro-batching), GPU torch BM25
(bm25_pt row-sharded across GPUs), and CPU BM25 fallbacks (bm25s / pyserini Lucene).
Exposes ``GET /health``, ``POST /search``, and ``POST /lookup`` for agent rollouts plus
``POST /retrieve`` for batch Search-R1-style queries.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import socket
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import datasets
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from numpy.typing import NDArray
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from torch_bm25_backend import build_torch_bm25_store, torch_bm25_topk

logger = logging.getLogger(__name__)

# ---- Small helpers / aliases
Doc = Dict[str, Any]
Docs = List[Doc]
BatchDocs = List[Docs]
Scores = List[float]
BatchScores = List[Scores]


def _visible_cuda_devices() -> List[str]:
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def _is_bm25s_index(index_path: str) -> bool:
    return (Path(index_path) / "data.csc.index.npy").is_file()


def load_corpus(corpus_path: str) -> Any:
    corpus: Any = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)  # type: ignore
    return corpus


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_docs(corpus: Any, doc_idxs: Sequence[int]) -> Docs:
    results: Docs = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path: str, device: str, use_fp16: bool = False) -> Tuple[torch.nn.Module, Any]:
    _model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  # type: ignore
    model: torch.nn.Module = AutoModel.from_pretrained(model_path, trust_remote_code=True)  # type: ignore
    model.eval()  # type: ignore
    if device.startswith("cuda"):
        model = model.to(device)  # type: ignore
    else:
        model = model.cpu()  # type: ignore
    if use_fp16:
        model = model.half()  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)  # type: ignore
    return model, tokenizer  # type: ignore


def pooling(
    pooler_output: torch.Tensor,
    last_hidden_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    pooling_method: str = "mean",
) -> torch.Tensor:
    if pooling_method == "mean":
        assert attention_mask is not None, "attention_mask is required for mean pooling"
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    """Single-device encoder replica."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        pooling_method: str,
        max_length: int,
        use_fp16: bool,
        device: str,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.device = device

        self.model, self.tokenizer = load_model(model_path=model_path, device=device, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()  # type: ignore
    def encode(self, query_list: Union[List[str], str], is_query: bool = True) -> NDArray[np.float32]:
        if isinstance(query_list, str):
            query_list = [query_list]
        if not query_list:
            return np.zeros((0, 1), dtype=np.float32)

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [
                    f"Represent this sentence for searching relevant passages: {query}" for query in query_list
                ]

        inputs: Dict[str, torch.Tensor] = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )  # type: ignore[call-arg]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output,
                output.last_hidden_state,
                inputs["attention_mask"],
                self.pooling_method,
            )
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_np: NDArray[np.float32] = query_emb.detach().cpu().numpy().astype(np.float32, order="C")  # type: ignore

        del inputs, output

        return query_np


class MultiGpuEncoder:
    """Load one encoder replica per visible GPU and shard encode batches across them."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        pooling_method: str,
        max_length: int,
        use_fp16: bool,
    ) -> None:
        devices = _visible_cuda_devices()
        if not devices:
            devices = ["cpu"]
            logger.warning("No CUDA devices visible; encoder runs on CPU")
        self.replicas: List[Encoder] = [
            Encoder(
                model_name=model_name,
                model_path=model_path,
                pooling_method=pooling_method,
                max_length=max_length,
                use_fp16=use_fp16,
                device=device,
            )
            for device in devices
        ]
        logger.info(
            "Loaded %d encoder replica(s) on device(s): %s",
            len(self.replicas),
            ", ".join(r.device for r in self.replicas),
        )

    @torch.no_grad()  # type: ignore
    def encode(self, query_list: Union[List[str], str], is_query: bool = True) -> NDArray[np.float32]:
        if isinstance(query_list, str):
            query_list = [query_list]
        if not query_list:
            return np.zeros((0, 1), dtype=np.float32)
        if len(self.replicas) == 1:
            return self.replicas[0].encode(query_list, is_query=is_query)

        n = len(self.replicas)
        chunk_size = max(1, (len(query_list) + n - 1) // n)
        chunks: List[List[str]] = [query_list[i : i + chunk_size] for i in range(0, len(query_list), chunk_size)]
        while len(chunks) < n:
            chunks.append([])

        parts: List[Optional[NDArray[np.float32]]] = [None] * n

        def _encode_chunk(replica_idx: int, chunk: List[str]) -> Tuple[int, NDArray[np.float32]]:
            if not chunk:
                return replica_idx, np.zeros((0, 1), dtype=np.float32)
            return replica_idx, self.replicas[replica_idx].encode(chunk, is_query=is_query)

        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = [executor.submit(_encode_chunk, i, chunks[i]) for i in range(n) if chunks[i]]
            for fut in as_completed(futures):
                idx, emb = fut.result()
                parts[idx] = emb

        ordered = [p for p in parts if p is not None and p.shape[0] > 0]
        if not ordered:
            return np.zeros((0, 1), dtype=np.float32)
        return np.concatenate(ordered, axis=0)


class Config:
    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        max_process_num: int = 8,
        bm25_backend: str = "bm25s",
        torch_bm25_device: str = "cuda",
        torch_bm25_cache_dir: Optional[str] = None,
        search_batch_size: int = 32,
        search_batch_wait_ms: float = 10.0,
    ) -> None:
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.max_process_num = max_process_num
        self.bm25_backend = bm25_backend
        self.torch_bm25_device = torch_bm25_device
        self.torch_bm25_cache_dir = torch_bm25_cache_dir
        self.search_batch_size = search_batch_size
        self.search_batch_wait_ms = search_batch_wait_ms


class BaseRetriever:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.retrieval_method: str = config.retrieval_method
        self.topk: int = config.retrieval_topk
        self.max_process_num: int = config.max_process_num

        self.index_path: str = config.index_path
        self.corpus_path: str = config.corpus_path

    def _search(self, query: str, num: Optional[int], return_score: bool) -> Union[Docs, Tuple[Docs, Scores]]:
        raise NotImplementedError

    def _batch_search(
        self, query_list: List[str], num: Optional[int], return_score: bool
    ) -> Union[BatchDocs, Tuple[BatchDocs, BatchScores]]:
        raise NotImplementedError

    def search(
        self, query: str, num: Optional[int] = None, return_score: bool = False
    ) -> Union[Docs, Tuple[Docs, Scores]]:
        return self._search(query, num, return_score)

    def batch_search(
        self, query_list: List[str], num: Optional[int] = None, return_score: bool = False
    ) -> Union[BatchDocs, Tuple[BatchDocs, BatchScores]]:
        return self._batch_search(query_list, num, return_score)

    def lookup(self, title: str) -> str:
        raise NotImplementedError(f"lookup not supported for {type(self).__name__}")


def _parallel_batch_search(
    retriever: BaseRetriever,
    query_list: List[str],
    num: Optional[int],
    return_score: bool,
) -> Union[BatchDocs, Tuple[BatchDocs, BatchScores]]:
    """Run single-query searches in parallel using ``max_process_num`` worker threads."""
    if len(query_list) <= 1:
        results: BatchDocs = []
        scores: BatchScores = []
        for query in query_list:
            item_result, item_score = retriever._search(query, num, True)  # type: ignore[misc]
            results.append(item_result)  # type: ignore[arg-type]
            scores.append(item_score)  # type: ignore[arg-type]
        if return_score:
            return results, scores
        return results

    indexed_results: List[Optional[Docs]] = [None] * len(query_list)
    indexed_scores: List[Optional[Scores]] = [None] * len(query_list)

    def _one(idx: int, query: str) -> Tuple[int, Docs, Scores]:
        docs, sc = retriever._search(query, num, True)  # type: ignore[misc]
        return idx, docs, sc  # type: ignore[return-value]

    workers = min(retriever.max_process_num, len(query_list))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_one, i, q) for i, q in enumerate(query_list)]
        for fut in as_completed(futures):
            idx, docs, sc = fut.result()
            indexed_results[idx] = docs
            indexed_scores[idx] = sc

    results_out: BatchDocs = [r for r in indexed_results if r is not None]  # type: ignore[misc]
    scores_out: BatchScores = [s for s in indexed_scores if s is not None]  # type: ignore[misc]
    if return_score:
        return results_out, scores_out
    return results_out


class BM25Retriever(BaseRetriever):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        try:
            from pyserini.search.lucene import LuceneSearcher  # type: ignore
        except Exception:  # pragma: no cover - typing convenience
            LuceneSearcher = Any  # type: ignore[assignment,misc]

        self.searcher: Any = LuceneSearcher(self.index_path)  # type: ignore
        self.contain_doc: bool = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus: Any = load_corpus(self.corpus_path)
        logger.info("BM25 Lucene index ready at %s (max_process_num=%d)", self.index_path, self.max_process_num)

    def _check_contain_doc(self) -> bool:
        doc = self.searcher.doc(0)
        try:
            _ = doc.raw()
            return True
        except Exception:
            return False

    def _search(
        self, query: str, num: Optional[int] = None, return_score: bool = False
    ) -> Union[Docs, Tuple[Docs, Scores]]:
        k = self.topk if num is None else num
        hits: List[Any] = self.searcher.search(query, k)
        if len(hits) < 1:
            if return_score:
                return [], []
            return []

        scores: Scores = [float(hit.score) for hit in hits]
        if len(hits) < k:
            warnings.warn("Not enough documents retrieved!")
        else:
            hits = hits[:k]

        if self.contain_doc:
            all_contents: List[str] = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
            results: Docs = [
                {
                    "title": content.split("\n")[0].strip('"'),
                    "text": "\n".join(content.split("\n")[1:]),
                    "contents": content,
                }
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [int(hit.docid) for hit in hits])

        if return_score:
            return results, scores
        return results

    def _batch_search(
        self, query_list: List[str], num: Optional[int] = None, return_score: bool = False
    ) -> Union[BatchDocs, Tuple[BatchDocs, BatchScores]]:
        return _parallel_batch_search(self, query_list, num, return_score)

    def lookup(self, title: str) -> str:
        docs, _ = self._search(title, num=10, return_score=True)  # type: ignore[misc]
        for doc in docs:
            if doc.get("title") == title:
                return doc.get("text") or doc.get("contents", "")
        return f"No Wikipedia page found for title: {title}"


def _find_wiki_jsonl(wiki_dir: Path) -> Path:
    candidates = [
        wiki_dir / "wiki.abstracts.2017.jsonl",
        wiki_dir / "wiki_abstracts_2017.jsonl",
        wiki_dir / "wiki2017.jsonl",
        wiki_dir / "corpus.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
        gz = path.with_suffix(path.suffix + ".gz")
        if gz.exists():
            return gz
    jsonls = sorted(wiki_dir.glob("*.jsonl"))
    if jsonls:
        return jsonls[0]
    raise FileNotFoundError(f"Could not find a Wikipedia JSONL corpus under {wiki_dir}")


def _maybe_extract_jsonl(src: Path) -> Path:
    if src.suffix != ".gz":
        return src
    dest = src.with_suffix("")
    if dest.exists():
        return dest
    with gzip.open(src, "rb") as fin, open(dest, "wb") as fout:
        while True:
            chunk = fin.read(1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
    return dest


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_coerce_text(x) for x in value if x is not None)
    if isinstance(value, dict):
        for key in ("text", "sentence", "content"):
            if key in value:
                return _coerce_text(value[key])
        return " ".join(_coerce_text(v) for v in value.values())
    return str(value)


def _parse_title_and_text(record: dict) -> Tuple[str, str]:
    title = record.get("title") or record.get("id") or record.get("key") or ""
    title = str(title).strip()
    text = ""
    for field in ("text", "abstract", "contents", "sentences", "paragraphs"):
        if field in record:
            text = _coerce_text(record[field])
            if text:
                break
    if not text:
        text = _coerce_text({k: v for k, v in record.items() if k not in {"title", "id", "key"}})
    text = re.sub(r"\s+", " ", text).strip()
    return title, text


def _load_bm25s_corpus(jsonl_path: Path) -> List[str]:
    corpus: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            title, text = _parse_title_and_text(record)
            if not title:
                continue
            corpus.append(f"{title} | {text}" if text else f"{title} | ")
    if not corpus:
        raise RuntimeError(f"Loaded zero corpus entries from {jsonl_path}")
    return corpus


class Bm25sRetriever(BaseRetriever):
    """BM25 via bm25s (wiki2017 prebuilt index under ``--wiki-dir``)."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        try:
            import bm25s  # type: ignore
            import Stemmer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("bm25s and PyStemmer are required for bm25s index. pip install bm25s PyStemmer") from exc

        jsonl_path = _maybe_extract_jsonl(Path(config.corpus_path))
        self.passage_corpus: List[str] = _load_bm25s_corpus(jsonl_path)
        self.stemmer = Stemmer.Stemmer("english")
        self.bm25s = bm25s
        self.retriever = bm25s.BM25.load(config.index_path, mmap=True)
        self.docs_by_title: Dict[str, str] = {}
        for passage in self.passage_corpus:
            if " | " in passage:
                title, text = passage.split(" | ", 1)
                self.docs_by_title.setdefault(title, text)
        logger.info(
            "BM25 bm25s index ready at %s (%d docs, max_process_num=%d)",
            config.index_path,
            len(self.passage_corpus),
            self.max_process_num,
        )

    def _retrieve_passages(self, query: str, k: int) -> List[str]:
        tokens = self.bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer, show_progress=False)
        # Avoid 8 outer batch threads × n_threads oversubscription on CPU bm25s.
        n_threads = 1 if self.max_process_num > 1 else self.max_process_num
        results, _scores = self.retriever.retrieve(  # type: ignore[no-untyped-call]
            tokens, k=k, n_threads=n_threads, show_progress=False
        )
        ranked: List[str] = []
        for doc_idx in results[0]:
            idx = int(doc_idx)
            if idx < len(self.passage_corpus):
                ranked.append(self.passage_corpus[idx])
        return ranked

    def _passages_to_docs(self, passages: List[str]) -> Docs:
        docs: Docs = []
        for passage in passages:
            if " | " in passage:
                title, text = passage.split(" | ", 1)
            else:
                title, text = passage, ""
            docs.append({"title": title, "text": text, "contents": passage})
        return docs

    def _search(
        self, query: str, num: Optional[int] = None, return_score: bool = False
    ) -> Union[Docs, Tuple[Docs, Scores]]:
        k = self.topk if num is None else num
        passages = self._retrieve_passages(query, k)
        docs = self._passages_to_docs(passages)
        if return_score:
            return docs, [0.0] * len(docs)
        return docs

    def _batch_search(
        self, query_list: List[str], num: Optional[int] = None, return_score: bool = False
    ) -> Union[BatchDocs, Tuple[BatchDocs, BatchScores]]:
        return _parallel_batch_search(self, query_list, num, return_score)

    def lookup(self, title: str) -> str:
        if title in self.docs_by_title:
            return self.docs_by_title[title]
        results = [p for p in self._retrieve_passages(title, 10) if p.startswith(title + " | ")]
        if not results:
            return f"No Wikipedia page found for title: {title}"
        return results[0].split(" | ", 1)[1] if " | " in results[0] else results[0]


class TorchBM25Retriever(BaseRetriever):
    """GPU BM25 via bm25_pt with row-wise multi-GPU sharding (dp=N, tp=1)."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        jsonl_path = _maybe_extract_jsonl(Path(config.corpus_path))
        passage_corpus = _load_bm25s_corpus(jsonl_path)
        cache_dir = Path(config.torch_bm25_cache_dir) if config.torch_bm25_cache_dir else Path(config.index_path).parent / "torch_bm25_index_k1_0.9_b0.4"
        self._store = build_torch_bm25_store(
            corpus=passage_corpus,
            cache_dir=cache_dir,
            device=config.torch_bm25_device,
        )
        self.docs_by_title = self._store.docs_by_title

    def _passages_to_docs(self, passages: List[str]) -> Docs:
        docs: Docs = []
        for passage in passages:
            if " | " in passage:
                title, text = passage.split(" | ", 1)
            else:
                title, text = passage, ""
            docs.append({"title": title, "text": text, "contents": passage})
        return docs

    def _search(
        self, query: str, num: Optional[int] = None, return_score: bool = False
    ) -> Union[Docs, Tuple[Docs, Scores]]:
        k = self.topk if num is None else num
        passages = torch_bm25_topk(self._store, query, k)
        docs = self._passages_to_docs(passages)
        if return_score:
            return docs, [0.0] * len(docs)
        return docs

    def _batch_search(
        self, query_list: List[str], num: Optional[int] = None, return_score: bool = False
    ) -> Union[BatchDocs, Tuple[BatchDocs, BatchScores]]:
        k = self.topk if num is None else num
        results: BatchDocs = []
        scores: BatchScores = []
        for query in query_list:
            docs, sc = self._search(query, k, True)  # type: ignore[misc]
            results.append(docs)  # type: ignore[arg-type]
            scores.append(sc)  # type: ignore[arg-type]
        if return_score:
            return results, scores
        return results

    def lookup(self, title: str) -> str:
        if title in self.docs_by_title:
            return self.docs_by_title[title]
        results = [p for p in torch_bm25_topk(self._store, title, 10) if p.startswith(title + " | ")]
        if not results:
            return f"No Wikipedia page found for title: {title}"
        return results[0].split(" | ", 1)[1] if " | " in results[0] else results[0]


class DenseRetriever(BaseRetriever):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        try:
            import faiss  # type: ignore[reportMissingTypeStubs]
        except ImportError as exc:
            raise RuntimeError(
                "faiss is required for dense retrieval. pip install faiss-gpu (or faiss-cpu), "
                "or use --retriever_name bm25 / --bm25-backend torch."
            ) from exc
        index: Any = faiss.read_index(self.index_path)  # type: ignore[no-untyped-call]
        if config.faiss_gpu:
            co: Any = faiss.GpuMultipleClonerOptions()  # type: ignore[attr-defined]
            co.useFloat16 = True
            co.shard = True
            index = faiss.index_cpu_to_all_gpus(index, co=co)  # type: ignore[no-untyped-call]
            logger.info("FAISS index loaded on all visible GPUs (faiss_gpu=True, shard=True)")
        else:
            logger.info("FAISS index loaded on CPU (faiss_gpu=False)")

        self.index: Any = index
        self.corpus: Any = load_corpus(self.corpus_path)
        self.encoder = MultiGpuEncoder(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def search_queries(self, query_list: List[str], num: int) -> BatchDocs:
        """Encode and retrieve a batch of queries (used by /search micro-batching)."""
        if not query_list:
            return []
        k = num
        query_emb = self.encoder.encode(query_list)
        scores_np, idxs_np = self.index.search(query_emb, k=k)  # type: ignore[no-untyped-call]
        batch_idxs = idxs_np.tolist()
        flat_idxs: List[int] = sum(batch_idxs, [])  # type: ignore[arg-type]
        batch_results_flat = load_docs(self.corpus, flat_idxs)
        return [batch_results_flat[i * k : (i + 1) * k] for i in range(len(batch_idxs))]

    def _search(
        self, query: str, num: Optional[int] = None, return_score: bool = False
    ) -> Union[Docs, Tuple[Docs, Scores]]:
        k = self.topk if num is None else num
        query_emb = self.encoder.encode(query)
        scores_np, idxs_np = self.index.search(query_emb, k=k)  # type: ignore[no-untyped-call]
        idxs: Sequence[int] = list(map(int, idxs_np[0]))
        scores: Scores = [float(s) for s in scores_np[0]]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores
        return results

    def _batch_search(
        self, query_list: List[str], num: Optional[int] = None, return_score: bool = False
    ) -> Union[BatchDocs, Tuple[BatchDocs, BatchScores]]:
        if isinstance(query_list, str):
            query_list = [query_list]
        k = self.topk if num is None else num

        results: BatchDocs = []
        scores: BatchScores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc="Retrieval process: "):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores_np, batch_idxs_np = self.index.search(batch_emb, k=k)  # type: ignore[no-untyped-call]
            batch_scores = batch_scores_np.tolist()
            batch_idxs = batch_idxs_np.tolist()

            flat_idxs: List[int] = sum(batch_idxs, [])  # type: ignore
            batch_results_flat = load_docs(self.corpus, flat_idxs)
            chunked: List[Docs] = [batch_results_flat[i * k : (i + 1) * k] for i in range(len(batch_idxs))]

            results.extend(chunked)
            scores.extend(batch_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results_flat

        if return_score:
            return results, scores
        return results


def get_retriever(config: Config) -> BaseRetriever:
    if config.retrieval_method in ("bm25", "torch_bm25"):
        if config.retrieval_method == "torch_bm25" or config.bm25_backend == "torch":
            return TorchBM25Retriever(config)
        if config.bm25_backend == "lucene":
            return BM25Retriever(config)
        if _is_bm25s_index(config.index_path):
            return Bm25sRetriever(config)
        return BM25Retriever(config)
    return DenseRetriever(config)


@dataclass
class _PendingSearch:
    query: str
    num: int
    event: threading.Event
    result: Optional[Docs] = None
    error: Optional[BaseException] = None


class SearchBatcher:
    """Coalesce concurrent /search requests into encoder batches."""

    def __init__(self, retriever: DenseRetriever, max_batch_size: int, max_wait_s: float) -> None:
        self._retriever = retriever
        self._max_batch_size = max(1, max_batch_size)
        self._max_wait_s = max(0.0, max_wait_s)
        self._lock = threading.Lock()
        self._pending: List[_PendingSearch] = []
        self._flush_timer: Optional[threading.Timer] = None

    def search(self, query: str, num: int) -> Docs:
        pending = _PendingSearch(query=query, num=num, event=threading.Event())
        batch_to_flush: Optional[List[_PendingSearch]] = None
        with self._lock:
            self._pending.append(pending)
            if len(self._pending) >= self._max_batch_size:
                batch_to_flush = self._pending
                self._pending = []
                if self._flush_timer is not None:
                    self._flush_timer.cancel()
                    self._flush_timer = None
            elif self._flush_timer is None and self._max_wait_s > 0:
                self._flush_timer = threading.Timer(self._max_wait_s, self._timer_flush)
                self._flush_timer.daemon = True
                self._flush_timer.start()

        if batch_to_flush is not None:
            self._run_batch(batch_to_flush)

        pending.event.wait()
        if pending.error is not None:
            raise pending.error
        assert pending.result is not None
        return pending.result

    def _timer_flush(self) -> None:
        batch: Optional[List[_PendingSearch]] = None
        with self._lock:
            if self._pending:
                batch = self._pending
                self._pending = []
            self._flush_timer = None
        if batch:
            self._run_batch(batch)

    def _run_batch(self, batch: List[_PendingSearch]) -> None:
        try:
            queries = [item.query for item in batch]
            k = max(item.num for item in batch)
            batch_docs = self._retriever.search_queries(queries, num=k)
            for item, docs in zip(batch, batch_docs):
                item.result = docs[: item.num]
                item.event.set()
        except BaseException as exc:
            logger.exception("Search batch failed")
            for item in batch:
                item.error = exc
                item.event.set()


def docs_to_passages(docs: Docs, *, search_depth: int = 30) -> List[str]:
    """Format docs as ``search_wikipedia``-style passage strings for agent rollouts."""
    top = docs[:5]
    passages = [f"{doc.get('title', '')} | {doc.get('text', '')}" for doc in top]
    if len(docs) > 5:
        titles = [f"`{doc.get('title', '')}`" for doc in docs[5 : min(search_depth, 30)]]
        if titles:
            passages.append(f"Other retrieved pages have titles: {', '.join(titles)}.")
    return passages


def resolve_wiki_paths(wiki_dir: str, k1: float = 0.9, b: float = 0.4) -> Tuple[str, str]:
    root = Path(wiki_dir)
    index_path = str(root / f"bm25_index_k1_{k1}_b{b}")
    corpus_path = str(_find_wiki_jsonl(root))
    return index_path, corpus_path


def resolve_torch_bm25_cache(wiki_dir: str, k1: float = 0.9, b: float = 0.4) -> str:
    return str(Path(wiki_dir) / f"torch_bm25_index_k1_{k1}_b{b}")


#####################################
# FastAPI server below
#####################################


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


class SearchRequest(BaseModel):
    query: str


class LookupRequest(BaseModel):
    title: str


app: FastAPI = FastAPI()

config: Config = Config()
retriever: Optional[BaseRetriever] = None
search_batcher: Optional[SearchBatcher] = None
_server_url: Optional[str] = None


@app.get("/health")
def health_endpoint() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/search")
def search_endpoint(request: SearchRequest) -> Dict[str, Any]:
    """Agent rollout endpoint (compatible with ``torch_bm25_server``)."""
    assert retriever is not None, "retriever not initialized"
    if search_batcher is not None:
        docs = search_batcher.search(request.query, num=30)
    else:
        docs = retriever.search(request.query, num=30, return_score=False)  # type: ignore[assignment]
    return {"passages": docs_to_passages(docs, search_depth=30)}  # type: ignore[arg-type]


@app.post("/lookup")
def lookup_endpoint(request: LookupRequest) -> Dict[str, str]:
    assert retriever is not None, "retriever not initialized"
    try:
        result = retriever.lookup(request.title)
    except NotImplementedError:
        result = f"lookup not supported for retriever {type(retriever).__name__}"
    return {"result": result}


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest) -> Dict[str, Any]:
    assert retriever is not None, "retriever not initialized"
    if not request.topk:
        request.topk = config.retrieval_topk

    search_out = retriever.batch_search(
        query_list=request.queries, num=int(request.topk), return_score=request.return_scores
    )

    if request.return_scores:
        results, scores = search_out  # type: ignore[misc]
    else:
        results = search_out  # type: ignore[assignment]
        scores = None

    resp: List[Any] = []
    for i, single_result in enumerate(results):  # type: ignore[arg-type]
        if request.return_scores and scores is not None:
            combined: List[Dict[str, Any]] = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": float(score)})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


def _write_addr_file(addr_file: str, url: str) -> None:
    path = Path(addr_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(url, encoding="utf-8")
    logger.info("Wrote server URL %s to %s", url, addr_file)


def _remove_addr_file(addr_file: Optional[str]) -> None:
    if not addr_file:
        return
    try:
        os.remove(addr_file)
    except OSError:
        pass


def _resolve_port(explicit_port: Optional[int]) -> int:
    if explicit_port is not None and explicit_port > 0:
        return explicit_port
    return 19000 + (int(os.environ.get("LSB_JOBID", "0")) % 2000)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Search-R1 retrieval server.")
    parser.add_argument("--wiki-dir", type=str, default=None, help="Wiki2017 directory (BM25 corpus + index/cache).")
    parser.add_argument(
        "--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Index file or directory."
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl",
        help="Local corpus file.",
    )
    parser.add_argument("--topk", type=int, default=3, help="Default number of retrieved passages per query.")
    parser.add_argument(
        "--retriever_name",
        type=str,
        default="e5",
        help="Retriever name ('bm25', 'torch_bm25', or dense model tag such as 'e5').",
    )
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Dense retriever model path.")
    parser.add_argument(
        "--bm25-backend",
        type=str,
        choices=("torch", "bm25s", "lucene"),
        default=os.environ.get("BM25_BACKEND", "torch"),
        help="BM25 implementation when --wiki-dir or retriever_name=bm25 (default: torch GPU).",
    )
    parser.add_argument(
        "--torch-bm25-device",
        type=str,
        default=os.environ.get("TORCH_BM25_DEVICE", "cuda"),
        help="'cuda' shards torch BM25 across all visible GPUs; 'cuda:0' pins one GPU.",
    )
    parser.add_argument(
        "--torch-bm25-cache-dir",
        type=str,
        default=None,
        help="Cache dir for torch BM25 _corpus_scores (default: <wiki-dir>/torch_bm25_index_k1_0.9_b0.4).",
    )
    parser.add_argument(
        "--faiss_gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shard FAISS index across visible GPUs (default: enabled).",
    )
    parser.add_argument(
        "--max-process-num",
        type=int,
        default=int(os.environ.get("BM25_MAX_PROCESS_NUM", "8")),
        help="Parallel BM25 batch-search worker threads (CPU bm25s/Lucene only).",
    )
    parser.add_argument(
        "--search-batch-size",
        type=int,
        default=int(os.environ.get("SEARCH_BATCH_SIZE", "32")),
        help="Max concurrent /search queries to coalesce before MultiGpuEncoder.encode.",
    )
    parser.add_argument(
        "--search-batch-wait-ms",
        type=float,
        default=float(os.environ.get("SEARCH_BATCH_WAIT_MS", "10")),
        help="Max wait (ms) to fill a /search micro-batch before flushing.",
    )
    parser.add_argument("--addr-file", type=str, default=None, help="Write http://host:port here when ready.")
    parser.add_argument("--port", type=int, default=0, help="Listen port (0 -> derive from LSB_JOBID).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    index_path = args.index_path
    corpus_path = args.corpus_path
    retrieval_method = args.retriever_name
    bm25_backend = args.bm25_backend
    torch_bm25_cache_dir = args.torch_bm25_cache_dir
    if args.wiki_dir:
        index_path, corpus_path = resolve_wiki_paths(args.wiki_dir)
        if retrieval_method not in ("bm25", "torch_bm25"):
            retrieval_method = "bm25"
        if bm25_backend == "torch":
            torch_bm25_cache_dir = torch_bm25_cache_dir or resolve_torch_bm25_cache(args.wiki_dir)
        logger.info(
            "Using wiki-dir %s -> index=%s corpus=%s bm25_backend=%s",
            args.wiki_dir,
            index_path,
            corpus_path,
            bm25_backend,
        )

    global config, retriever, search_batcher, _server_url
    config = Config(
        retrieval_method=retrieval_method,
        index_path=index_path,
        corpus_path=corpus_path,
        retrieval_topk=int(args.topk),
        faiss_gpu=bool(args.faiss_gpu),
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
        max_process_num=int(args.max_process_num),
        bm25_backend=bm25_backend,
        torch_bm25_device=args.torch_bm25_device,
        torch_bm25_cache_dir=torch_bm25_cache_dir,
        search_batch_size=int(args.search_batch_size),
        search_batch_wait_ms=float(args.search_batch_wait_ms),
    )

    retriever = get_retriever(config)
    search_batcher = None
    if isinstance(retriever, DenseRetriever):
        search_batcher = SearchBatcher(
            retriever,
            max_batch_size=config.search_batch_size,
            max_wait_s=config.search_batch_wait_ms / 1000.0,
        )
        logger.info(
            "Dense /search micro-batching enabled (batch_size=%d, wait_ms=%.1f)",
            config.search_batch_size,
            config.search_batch_wait_ms,
        )

    port = _resolve_port(args.port if args.port > 0 else None)
    host = "0.0.0.0"
    _server_url = f"http://{socket.getfqdn()}:{port}"

    if args.addr_file:
        _write_addr_file(args.addr_file, _server_url)

    logger.info("Starting retrieval server at %s (retriever=%s, faiss_gpu=%s)", _server_url, retrieval_method, args.faiss_gpu)
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        _remove_addr_file(args.addr_file)


if __name__ == "__main__":
    main()
