# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

SEARCH_R1_SCRIPTS = Path(__file__).resolve().parents[4] / "contrib" / "recipes" / "search_r1" / "scripts"
sys.path.insert(0, str(SEARCH_R1_SCRIPTS))

import retrieval_server as rs  # noqa: E402


class _StubRetriever(rs.BaseRetriever):
    def __init__(self) -> None:
        super().__init__(rs.Config(retrieval_method="bm25", max_process_num=4))

    def _search(
        self, query: str, num: int | None = None, return_score: bool = False
    ) -> rs.Docs | Tuple[rs.Docs, rs.Scores]:
        k = num or self.topk
        docs = [{"title": f"T-{query}", "text": f"body-{i}"} for i in range(k)]
        if return_score:
            return docs, [float(i) for i in range(k)]
        return docs

    def lookup(self, title: str) -> str:
        return f"text-for-{title}"


@pytest.fixture
def stub_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(rs, "config", rs.Config())
    monkeypatch.setattr(rs, "retriever", _StubRetriever())
    return TestClient(rs.app)


def test_health_endpoint(stub_client: TestClient) -> None:
    resp = stub_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_search_endpoint_formats_passages(stub_client: TestClient) -> None:
    resp = stub_client.post("/search", json={"query": "python"})
    assert resp.status_code == 200
    passages = resp.json()["passages"]
    assert passages[0].startswith("T-python | body-0")
    assert any(p.startswith("Other retrieved pages have titles:") for p in passages)


def test_lookup_endpoint(stub_client: TestClient) -> None:
    resp = stub_client.post("/lookup", json={"title": "Paris"})
    assert resp.status_code == 200
    assert resp.json()["result"] == "text-for-Paris"


def test_docs_to_passages_truncates_other_titles() -> None:
    docs = [{"title": f"D{i}", "text": f"x{i}"} for i in range(10)]
    passages = rs.docs_to_passages(docs, search_depth=30)
    assert len(passages) == 6  # top 5 + summary line
    assert passages[-1].startswith("Other retrieved pages have titles:")


def test_is_bm25s_index(tmp_path: Path) -> None:
    assert not rs._is_bm25s_index(str(tmp_path))
    (tmp_path / "data.csc.index.npy").write_bytes(b"")
    assert rs._is_bm25s_index(str(tmp_path))


def test_parallel_batch_search_preserves_order() -> None:
    retriever = _StubRetriever()

    def slow_search(query: str, num: int | None, return_score: bool) -> Tuple[rs.Docs, rs.Scores]:
        idx = int(query.split("-")[-1])
        docs = [{"title": f"Q{idx}", "text": "x"}]
        return docs, [1.0]

    retriever._search = slow_search  # type: ignore[method-assign]
    queries = [f"q-{i}" for i in range(6)]
    results, scores = rs._parallel_batch_search(retriever, queries, num=1, return_score=True)  # type: ignore[misc]
    assert [d[0]["title"] for d in results] == [f"Q{i}" for i in range(6)]
    assert len(scores) == 6


def test_multi_gpu_encoder_single_device_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rs.torch.cuda, "is_available", lambda: False)

    class FakeEncoder:
        def __init__(self, **kwargs: Any) -> None:
            self.device = kwargs.get("device", "cpu")

        def encode(self, query_list: Any, is_query: bool = True) -> np.ndarray:
            n = len(query_list) if isinstance(query_list, list) else 1
            return np.ones((n, 4), dtype=np.float32)

    monkeypatch.setattr(rs, "Encoder", FakeEncoder)
    enc = rs.MultiGpuEncoder(
        model_name="e5",
        model_path="/fake",
        pooling_method="mean",
        max_length=32,
        use_fp16=False,
    )
    assert len(enc.replicas) == 1
    out = enc.encode(["a", "b", "c"])
    assert out.shape == (3, 4)


def test_multi_gpu_encoder_shards_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rs, "_visible_cuda_devices", lambda: ["cuda:0", "cuda:1"])

    class FakeEncoder:
        def __init__(self, **kwargs: Any) -> None:
            self.device = kwargs["device"]

        def encode(self, query_list: Any, is_query: bool = True) -> np.ndarray:
            items = query_list if isinstance(query_list, list) else [query_list]
            return np.full((len(items), 2), float(self.device[-1]), dtype=np.float32)

    monkeypatch.setattr(rs, "Encoder", FakeEncoder)
    enc = rs.MultiGpuEncoder(
        model_name="e5",
        model_path="/fake",
        pooling_method="mean",
        max_length=32,
        use_fp16=False,
    )
    out = enc.encode(["a", "b", "c", "d"])
    assert out.shape == (4, 2)


def test_resolve_wiki_paths(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "wiki.abstracts.2017.jsonl").write_text('{"title":"A","text":"b"}\n')
    index_dir = wiki / "bm25_index_k1_0.9_b0.4"
    index_dir.mkdir()
    (index_dir / "data.csc.index.npy").write_bytes(b"")

    index_path, corpus_path = rs.resolve_wiki_paths(str(wiki))
    assert index_path.endswith("bm25_index_k1_0.9_b0.4")
    assert corpus_path.endswith("wiki.abstracts.2017.jsonl")


def test_resolve_port_from_lsb_jobid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LSB_JOBID", "12345")
    assert rs._resolve_port(None) == 19000 + (12345 % 2000)


def test_get_retriever_selects_bm25s(tmp_path: Path) -> None:
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    (index_dir / "data.csc.index.npy").write_bytes(b"")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"title":"Foo","text":"bar"}\n')

    cfg = rs.Config(retrieval_method="bm25", index_path=str(index_dir), corpus_path=str(corpus), bm25_backend="bm25s")
    with patch.object(rs, "Bm25sRetriever", autospec=True) as mock_cls:
        mock_cls.return_value = MagicMock()
        rs.get_retriever(cfg)
        mock_cls.assert_called_once_with(cfg)


def test_get_retriever_selects_torch_bm25(tmp_path: Path) -> None:
    cfg = rs.Config(
        retrieval_method="bm25",
        index_path=str(tmp_path),
        corpus_path=str(tmp_path / "c.jsonl"),
        bm25_backend="torch",
    )
    with patch.object(rs, "TorchBM25Retriever", autospec=True) as mock_cls:
        mock_cls.return_value = MagicMock()
        rs.get_retriever(cfg)
        mock_cls.assert_called_once_with(cfg)


def test_search_batcher_coalesces_queries() -> None:
    class FakeDense(rs.DenseRetriever):
        def __init__(self) -> None:
            pass

        def search_queries(self, query_list: List[str], num: int) -> List[rs.Docs]:
            return [[{"title": q, "text": "x"}] for q in query_list]

    batcher = rs.SearchBatcher(FakeDense(), max_batch_size=8, max_wait_s=0.05)  # type: ignore[arg-type]
    results: List[rs.Docs] = []
    errors: List[BaseException] = []

    def _run(q: str) -> None:
        try:
            results.append(batcher.search(q, num=1))
        except BaseException as exc:
            errors.append(exc)

    threads = [__import__("threading").Thread(target=_run, args=(f"q{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors
    assert len(results) == 4
    titles = {r[0]["title"] for r in results}
    assert titles == {"q0", "q1", "q2", "q3"}


def test_resolve_torch_bm25_cache() -> None:
    path = rs.resolve_torch_bm25_cache("/data/wiki")
    assert path.endswith("torch_bm25_index_k1_0.9_b0.4")
