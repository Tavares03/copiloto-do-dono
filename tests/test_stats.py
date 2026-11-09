import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from copiloto.api.main import app
from copiloto.config import get_settings
from copiloto.observability import configure_logging, log_event, remove_handlers
from copiloto.stats import INTERACTIONS_FILE, compute_stats, load_interactions
from fakes import fake_settings


def _run(**overrides) -> dict:
    row = {
        "ts": "2026-11-08T12:00:00.000+00:00",
        "event": "agent_run",
        "request_id": "r1",
        "question": "quanto vendi ontem?",
        "status": "answered",
        "grounded": True,
        "grounding": {"ungrounded": [], "corrected": False},
        "provider": "groq",
        "model": "qwen",
        "llm_calls": 2,
        "total_tokens": 4000,
        "latency_ms": 1500.0,
        "steps": [
            {"provider": "groq", "attempts": ["groq:ok"], "switched_from": [],
             "tools": [{"name": "get_metrics", "error": None}]},
            {"provider": "groq", "attempts": ["groq:ok"], "switched_from": [], "tools": []},
        ],
    }  # fmt: skip
    return {**row, **overrides}


ROWS = [
    _run(),
    _run(
        request_id="r2",
        provider="gemini",
        total_tokens=6000,
        latency_ms=3000.0,
        grounded=False,
        grounding={"ungrounded": ["350,00"], "corrected": True},
        steps=[
            {"provider": "gemini", "attempts": ["groq:rate_limited", "gemini:ok"],
             "switched_from": ["groq"], "tools": [{"name": "run_sql", "error": "Erro do banco"}]},
        ],
    ),
    _run(request_id="r3", status="llm_unavailable", provider=None, llm_calls=0,
         total_tokens=0, latency_ms=100.0, steps=[]),
]  # fmt: skip


def test_compute_stats():
    stats = compute_stats(ROWS)

    assert stats["perguntas"] == 3
    assert stats["por_status"] == {"answered": 2, "llm_unavailable": 1}
    assert stats["provedores"] == {"groq": 1, "gemini": 1}
    assert stats["trocas_de_provedor"] == 1
    assert stats["falhas_de_provedor"] == {"groq:rate_limited": 1}
    assert stats["tools"] == {"get_metrics": 1, "run_sql": 1}
    assert stats["erros_de_tool"] == 1
    assert stats["aterramento_pct"] == pytest.approx(66.7, abs=0.1)
    assert stats["correcoes_do_verificador"] == 1
    assert stats["tokens_total"] == 10000
    assert stats["latencia_p95_s"] == pytest.approx(2.85, abs=0.01)
    assert [r["request_id"] for r in stats["ultimas"]] == ["r3", "r2", "r1"]


def test_compute_stats_without_data():
    assert compute_stats([])["perguntas"] == 0


def test_load_interactions_skips_broken_lines(tmp_path: Path):
    path = tmp_path / INTERACTIONS_FILE
    path.write_text(
        json.dumps(ROWS[0]) + "\n{linha quebrada\n" + json.dumps(ROWS[1]) + "\n", "utf-8"
    )

    assert [r["request_id"] for r in load_interactions(path)] == ["r1", "r2"]
    assert load_interactions(tmp_path / "nao_existe.jsonl") == []


def test_interactions_file_receives_only_agent_runs(tmp_path: Path):
    path = tmp_path / INTERACTIONS_FILE
    configure_logging("INFO", interactions_path=path)
    try:
        log_event(logging.getLogger("copiloto.agent"), "agent_run", question="quanto vendi?")
        log_event(logging.getLogger("copiloto.llm"), "llm_response", provider="groq")
    finally:
        remove_handlers()

    lines = path.read_text("utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["question"] == "quanto vendi?"
    assert "quanto vendi?" in lines[0]  # acentos legíveis no arquivo (UTF-8)


def test_stats_endpoint(tmp_path: Path):
    (tmp_path / INTERACTIONS_FILE).write_text(
        "".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8"
    )
    app.dependency_overrides[get_settings] = lambda: fake_settings(log_dir=tmp_path)
    try:
        with TestClient(app) as client:
            body = client.get("/stats").json()
    finally:
        app.dependency_overrides.clear()
        remove_handlers()

    assert body["perguntas"] == 3
    assert body["provedores"] == {"groq": 1, "gemini": 1}
