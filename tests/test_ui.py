"""A interface Streamlit de ponta a ponta, com a API trocada por respostas prontas."""

from pathlib import Path
from typing import Any

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "ui" / "app.py"

HEALTH = {
    "status": "ok",
    "hoje": "2026-11-01",
    "banco": {"ok": True, "dados_de": "2026-05-01", "dados_ate": "2026-10-31"},
    "llm": [
        {"name": "groq", "model": "qwen", "active": True, "circuit": "closed",
         "last_error": None, "detail": "na fila"},
        {"name": "ollama", "model": None, "active": False, "circuit": "n/a",
         "last_error": None, "detail": "Ollama não respondeu"},
    ],
    "previsao": {"modelo": "baseline_media_4_semanas"},
}  # fmt: skip
CHAT = {
    "request_id": "abc123",
    "answer": "Ontem você faturou R$ 10.689,14.\n\nFonte: get_metrics/resumo, 31/10/2026.",
    "status": "answered",
    "grounded": True,
    "grounding": {"ungrounded": [], "corrected": True, "flagged": ["350,00"]},
    "provider": "gemini",
    "model": "gemini-2.5-flash",
    "llm_calls": 2,
    "total_tokens": 3917,
    "latency_ms": 1700.0,
    "steps": [
        {"provider": "gemini", "model": "gemini-2.5-flash", "latency_ms": 900.0,
         "prompt_tokens": 1800, "completion_tokens": 20, "switched_from": ["groq"],
         "attempts": ["groq:rate_limited", "gemini:ok"], "cached": False,
         "tools": [{"name": "get_metrics", "arguments": {"metric": "resumo"},
                    "latency_ms": 120.0, "error": None,
                    "output": {"dados": {"faturamento": 10689.14}}}]},
    ],
}  # fmt: skip
SUMMARY = {
    "day": "2026-10-31",
    "text": "Resumo de sábado, 31/10/2026\nFaturamento: R$ 10.689,14 em 136 pedidos.",
    "facts": {},
    "source": "template",
    "provider": None,
    "model": None,
    "fallback_reason": "llm_indisponivel",
    "ungrounded": [],
}


@pytest.fixture(autouse=True)
def fake_api(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, Any]]:
    calls: list[tuple[str, str, Any]] = []
    routes = {"/chat": CHAT, "/daily-summary": SUMMARY}

    def request(self: httpx.Client, method: str, path: str, **kwargs: Any) -> httpx.Response:
        calls.append((method, path, kwargs.get("json")))
        return httpx.Response(200, json=routes[path], request=httpx.Request(method, path))

    def get(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json=HEALTH, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.Client, "request", request)
    monkeypatch.setattr(httpx, "get", get)
    return calls


def _app() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()
    assert not app.exception
    return app


def test_sidebar_shows_database_and_providers():
    sidebar = " ".join(m.value for m in _app().sidebar.markdown)

    assert "dados até 2026-10-31" in sidebar
    assert "groq" in sidebar
    assert "fora da fila" in sidebar  # ollama


def test_question_shows_answer_and_inside_view(fake_api: list):
    app = _app()

    app.chat_input[0].set_value("quanto vendi ontem?").run()

    assert fake_api[-1] == ("POST", "/chat", {"question": "quanto vendi ontem?", "history": []})
    assistant = app.chat_message[1]
    assert "R$ 10.689,14" in assistant.markdown[0].value
    assert assistant.expander[0].label == "🔎 Por dentro"
    assert "350,00" in app.info[0].value  # verificador de números
    assert "groq:rate_limited → gemini:ok" in app.warning[0].value  # troca de provedor


def test_second_question_sends_history(fake_api: list):
    app = _app()
    app.chat_input[0].set_value("quanto vendi?").run()

    app.chat_input[0].set_value("outubro").run()

    history = fake_api[-1][2]["history"]
    assert [turn["role"] for turn in history] == ["user", "assistant"]


def test_observability_page(monkeypatch: pytest.MonkeyPatch):
    stats = {
        "perguntas": 3, "latencia_p50_s": 1.5, "latencia_p95_s": 2.9, "tokens_medios": 3700,
        "trocas_de_provedor": 1, "aterramento_pct": 66.7, "provedores": {"groq": 2, "gemini": 1},
        "falhas_de_provedor": {"groq:rate_limited": 1}, "tools": {"get_metrics": 3},
        "erros_de_tool": 0, "correcoes_do_verificador": 1, "por_status": {"answered": 3},
        "ultimas": [{"request_id": "r1", "pergunta": "quanto vendi?", "provedor": "groq"}],
    }  # fmt: skip
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: httpx.Response(200, json=stats, request=httpx.Request("GET", url)),
    )

    page = AppTest.from_file(str(APP.parent / "pages" / "1_Observabilidade.py"), default_timeout=30)
    page.run()

    assert not page.exception
    assert page.metric[0].value == "3"
    assert page.dataframe[0].value["pergunta"].tolist() == ["quanto vendi?"]


def test_summary_button(fake_api: list):
    app = _app()

    next(b for b in app.sidebar.button if "Resumo" in b.label).click().run()

    assert ("GET", "/daily-summary", None) in fake_api
    assert "136 pedidos" in app.text[0].value
