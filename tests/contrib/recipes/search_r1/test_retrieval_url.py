# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

SEARCH_R1_SCRIPTS = Path(__file__).resolve().parents[4] / "contrib" / "recipes" / "search_r1" / "scripts"
sys.path.insert(0, str(SEARCH_R1_SCRIPTS))

import search_r1_agent as agent  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_retrieval_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    agent._cached_retrieval_url = None
    agent._cached_addr_file = None
    monkeypatch.delenv("RETRIEVAL_SERVER_URL", raising=False)
    monkeypatch.delenv("RETRIEVAL_SERVER_ADDR_FILE", raising=False)
    monkeypatch.delenv("ADDR_FILE", raising=False)


def test_retrieval_refreshes_url_from_addr_file_after_connection_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    addr_file = tmp_path / "bm25_server_addr_test.txt"
    addr_file.write_text("http://new-server:9000")

    monkeypatch.setenv("RETRIEVAL_SERVER_URL", "http://old-server:8000")
    monkeypatch.setenv("RETRIEVAL_SERVER_ADDR_FILE", str(addr_file))
    monkeypatch.setattr(agent, "_RETRIEVAL_MAX_RETRIES", 2)
    monkeypatch.setattr(agent, "_RETRIEVAL_BACKOFF_BASE_S", 0.0)

    call_count = 0
    seen_urls: list[str] = []

    def mock_post(url: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        seen_urls.append(url)
        if call_count <= 2:
            raise requests.ConnectionError("connection refused")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"passages": ["Title | passage text"]}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    monkeypatch.setattr(agent.requests, "post", mock_post)

    result = agent.retrieve_doc("test query")

    assert "passage text" in result
    assert call_count == 3
    assert seen_urls[:2] == ["http://old-server:8000/search", "http://old-server:8000/search"]
    assert seen_urls[2] == "http://new-server:9000/search"
    assert agent._cached_retrieval_url == "http://new-server:9000"


def test_refresh_retrieval_url_reads_addr_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    addr_file = tmp_path / "bm25_server_addr_baseline.txt"
    addr_file.write_text("http://fresh-server:8080")

    monkeypatch.setenv("RETRIEVAL_SERVER_URL", "http://stale-server:8000")
    monkeypatch.setenv("ADDR_FILE", str(addr_file))

    assert agent._retrieval_url() == "http://stale-server:8000"
    refreshed = agent._refresh_retrieval_url()

    assert refreshed == "http://fresh-server:8080"
    assert agent._retrieval_url() == "http://fresh-server:8080"
