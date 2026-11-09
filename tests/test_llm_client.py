import logging
from typing import Any

import httpx
import pytest

from copiloto.llm import AllProvidersFailed, LLMClient, ResponseCache
from copiloto.llm.providers import Provider, SkippedProvider
from fakes import (
    FakeClock,
    FakeServer,
    completion,
    fake_settings,
    http_error,
    make_provider,
    tool_call,
)

MESSAGES = [{"role": "user", "content": "Quanto vendi ontem?"}]


def make_client(
    *providers: Provider,
    cache: ResponseCache | None = None,
    skipped: list[SkippedProvider] | None = None,
    **overrides: Any,
) -> tuple[LLMClient, FakeClock]:
    clock = FakeClock()
    client = LLMClient(
        list(providers),
        settings=fake_settings(**overrides),
        cache=cache,
        skipped=skipped or [],
        sleep=clock.sleep,
        clock=clock,
    )
    return client, clock


def outcomes(response_or_error: Any) -> list[tuple[str, str]]:
    return [(a.provider, a.outcome) for a in response_or_error.attempts]


# --- caminho feliz -----------------------------------------------------------------------


def test_first_provider_answers():
    groq, gemini = FakeServer(completion("Vendeu R$ 10,00")), FakeServer(completion("x"))
    client, _ = make_client(make_provider("groq", groq), make_provider("gemini", gemini))

    response = client.chat(MESSAGES)

    assert (response.provider, response.content) == ("groq", "Vendeu R$ 10,00")
    assert response.usage.total_tokens == 15
    assert outcomes(response) == [("groq", "ok")]
    assert response.switched_from == []
    assert gemini.calls == 0


def test_tool_calls_are_sent_and_parsed():
    server = FakeServer(completion(None, [tool_call(arguments='{"metric": "faturamento"}')]))
    client, _ = make_client(make_provider("groq", server))
    tools = [{"type": "function", "function": {"name": "get_metrics", "parameters": {}}}]

    response = client.chat(MESSAGES, tools=tools)

    assert response.content == ""
    assert response.tool_calls[0].name == "get_metrics"
    assert response.tool_calls[0].parsed_arguments() == {"metric": "faturamento"}
    assert server.requests[0]["tools"] == tools
    assert server.requests[0]["temperature"] == 0
    assert server.requests[0]["max_tokens"] == 700  # evita 429 de tokens de saída no Groq


def test_empty_response_is_a_provider_failure_and_is_never_cached(tmp_path):
    empty = completion(None)  # sem texto e sem tool call (ex.: tokens gastos pensando)
    groq, gemini = FakeServer(empty), FakeServer(completion("ok"))
    cache = ResponseCache(tmp_path)
    client, _ = make_client(
        make_provider("groq", groq), make_provider("gemini", gemini), cache=cache
    )

    response = client.chat(MESSAGES)

    assert response.provider == "gemini"
    assert outcomes(response)[:2] == [("groq", "empty_response"), ("groq", "empty_response")]
    cached = list(cache.directory.rglob("*.json"))
    assert len(cached) == 1  # só a resposta boa (do gemini) foi para o cache
    assert '"provider": "gemini"' in cached[0].read_text("utf-8")


def test_provider_request_params_are_sent():
    server = FakeServer(completion("ok"))
    provider = make_provider("gemini", server)
    client, _ = make_client(
        Provider(
            provider.name, provider.model, provider.client, params={"reasoning_effort": "none"}
        )
    )

    client.chat(MESSAGES)

    assert server.requests[0]["reasoning_effort"] == "none"


def test_think_tags_are_removed_from_content():
    server = FakeServer(completion("<think>raciocínio interno</think>\nResposta final"))
    client, _ = make_client(make_provider("groq", server))

    assert client.chat(MESSAGES).content == "Resposta final"


# --- 429 e Retry-After ---------------------------------------------------------------------


def test_429_with_short_retry_after_waits_and_retries_same_provider():
    groq = FakeServer(http_error(429, retry_after=2), completion("ok"))
    gemini = FakeServer(completion("não deveria responder"))
    client, clock = make_client(make_provider("groq", groq), make_provider("gemini", gemini))

    response = client.chat(MESSAGES)

    assert response.provider == "groq"
    assert clock.sleeps == [2.0]
    assert outcomes(response) == [("groq", "rate_limited"), ("groq", "ok")]
    assert gemini.calls == 0


def test_429_with_long_retry_after_skips_to_next_provider():
    groq, gemini = FakeServer(http_error(429, retry_after=60)), FakeServer(completion("ok"))
    client, clock = make_client(make_provider("groq", groq), make_provider("gemini", gemini))

    response = client.chat(MESSAGES)

    assert response.provider == "gemini"
    assert response.switched_from == ["groq"]
    assert groq.calls == 1
    assert clock.sleeps == []


def test_429_without_retry_after_uses_exponential_backoff():
    groq, gemini = FakeServer(http_error(429)), FakeServer(completion("ok"))
    client, clock = make_client(make_provider("groq", groq), make_provider("gemini", gemini))

    response = client.chat(MESSAGES)

    assert response.provider == "gemini"
    assert groq.calls == 3
    assert clock.sleeps == [1.0, 2.0]


# --- erros que pulam direto ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        (401, "auth_error"),
        (402, "quota_or_forbidden"),
        (403, "quota_or_forbidden"),
        (404, "model_not_found"),
        (400, "bad_request"),
    ],
)
def test_non_retryable_errors_skip_immediately(status: int, outcome: str):
    groq, gemini = FakeServer(http_error(status)), FakeServer(completion("ok"))
    client, clock = make_client(make_provider("groq", groq), make_provider("gemini", gemini))

    response = client.chat(MESSAGES)

    assert response.provider == "gemini"
    assert groq.calls == 1
    assert clock.sleeps == []
    assert response.attempts[0].outcome == outcome
    assert response.attempts[0].status_code == status


def test_malformed_tool_call_is_retried_once():
    bad = http_error(400, body={"error": {"message": "Failed to call", "code": "tool_use_failed"}})
    groq = FakeServer(bad, completion("ok"))
    client, _ = make_client(make_provider("groq", groq))

    response = client.chat(MESSAGES)

    assert outcomes(response) == [("groq", "bad_tool_call"), ("groq", "ok")]


# --- rede e servidor ------------------------------------------------------------------------


def test_timeout_retries_once_then_skips():
    groq, gemini = FakeServer(httpx.ReadTimeout), FakeServer(completion("ok"))
    client, clock = make_client(make_provider("groq", groq), make_provider("gemini", gemini))

    response = client.chat(MESSAGES)

    assert outcomes(response) == [("groq", "timeout"), ("groq", "timeout"), ("gemini", "ok")]
    assert clock.sleeps == [1.0]


def test_connection_error_is_retried_with_backoff():
    groq = FakeServer(httpx.ConnectError, httpx.ConnectError, completion("ok"))
    client, clock = make_client(make_provider("groq", groq))

    response = client.chat(MESSAGES)

    assert response.provider == "groq"
    assert clock.sleeps == [1.0, 2.0]


def test_server_error_then_success_stays_on_same_provider():
    groq = FakeServer(http_error(503), completion("ok"))
    client, clock = make_client(make_provider("groq", groq))

    assert client.chat(MESSAGES).provider == "groq"
    assert clock.sleeps == [1.0]


def test_all_providers_failing_raises_with_every_attempt():
    groq, gemini = FakeServer(http_error(403)), FakeServer(http_error(500))
    client, _ = make_client(make_provider("groq", groq), make_provider("gemini", gemini))

    with pytest.raises(AllProvidersFailed) as info:
        client.chat(MESSAGES)

    assert outcomes(info.value) == [
        ("groq", "quota_or_forbidden"),
        ("gemini", "server_error"),
        ("gemini", "server_error"),
        ("gemini", "server_error"),
    ]


# --- circuit breaker ----------------------------------------------------------------------


def test_circuit_opens_after_consecutive_failures_and_retries_after_cooldown():
    groq, gemini = FakeServer(http_error(500)), FakeServer(completion("ok"))
    client, clock = make_client(
        make_provider("groq", groq),
        make_provider("gemini", gemini),
        llm_max_retries=0,
        circuit_failure_threshold=3,
        circuit_cooldown_s=300,
    )

    for _ in range(3):
        client.chat(MESSAGES)
    assert groq.calls == 3

    response = client.chat(MESSAGES)
    assert groq.calls == 3  # em espera: nem tentou
    assert outcomes(response) == [("groq", "circuit_open"), ("gemini", "ok")]

    clock.now += 300
    client.chat(MESSAGES)
    assert groq.calls == 4  # passou o tempo de espera: tenta de novo


def test_hard_failure_opens_circuit_immediately():
    groq, gemini = FakeServer(http_error(402)), FakeServer(completion("ok"))
    client, _ = make_client(make_provider("groq", groq), make_provider("gemini", gemini))

    client.chat(MESSAGES)
    client.chat(MESSAGES)

    assert groq.calls == 1


def test_success_resets_failure_count():
    groq = FakeServer(
        http_error(500), http_error(500), completion("ok"), http_error(500), http_error(500),
        completion("ok"),
    )  # fmt: skip
    gemini = FakeServer(completion("ok"))
    client, _ = make_client(
        make_provider("groq", groq),
        make_provider("gemini", gemini),
        llm_max_retries=0,
        circuit_failure_threshold=3,
    )

    for _ in range(6):
        client.chat(MESSAGES)

    assert groq.calls == 6


# --- histórico entre provedores -------------------------------------------------------------


def _history_with_gemini_signature() -> list[dict[str, Any]]:
    signature = {"google": {"thought_signature": "abc"}}
    return [
        *MESSAGES,
        {"role": "assistant", "content": "", "tool_calls": [tool_call(extra_content=signature)]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"faturamento": 10.0}'},
    ]


def test_gemini_extras_are_sent_to_gemini():
    gemini = FakeServer(completion("ok"))
    client, _ = make_client(make_provider("gemini", gemini))

    client.chat(_history_with_gemini_signature())

    sent = gemini.requests[0]["messages"][1]["tool_calls"][0]
    assert sent["extra_content"] == {"google": {"thought_signature": "abc"}}


def test_gemini_extras_are_stripped_for_other_providers():
    groq = FakeServer(completion("ok"))
    client, _ = make_client(make_provider("groq", groq))

    client.chat(_history_with_gemini_signature())

    sent = groq.requests[0]["messages"][1]["tool_calls"][0]
    assert "extra_content" not in sent
    assert sent["function"]["name"] == "get_metrics"


def test_as_message_keeps_tool_call_extras():
    signature = {"google": {"thought_signature": "abc"}}
    gemini = FakeServer(completion(None, [tool_call(extra_content=signature)]))
    client, _ = make_client(make_provider("gemini", gemini))

    message = client.chat(MESSAGES).as_message()

    assert message["role"] == "assistant"
    assert message["tool_calls"][0]["function"]["name"] == "get_metrics"
    assert message["tool_calls"][0]["extra_content"] == signature


# --- observabilidade ----------------------------------------------------------------------


def test_failover_is_logged_with_reason(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="copiloto.llm")
    groq, gemini = FakeServer(http_error(403)), FakeServer(completion("ok"))
    client, _ = make_client(make_provider("groq", groq), make_provider("gemini", gemini))

    client.chat(MESSAGES)

    events = {r.getMessage(): r.fields for r in caplog.records if hasattr(r, "fields")}
    assert events["llm_attempt_failed"]["provider"] == "groq"
    assert events["llm_attempt_failed"]["outcome"] == "quota_or_forbidden"
    assert events["llm_response"]["provider"] == "gemini"
    assert events["llm_response"]["switched_from"] == ["groq"]


def test_status_reports_circuit_and_skipped_providers():
    groq = FakeServer(http_error(402))
    client, _ = make_client(
        make_provider("groq", groq),
        skipped=[SkippedProvider("ollama", "Ollama não respondeu")],
    )
    with pytest.raises(AllProvidersFailed):
        client.chat(MESSAGES)

    groq_status, ollama_status = client.status()

    assert groq_status.name == "groq" and groq_status.active
    assert groq_status.circuit == "open"
    assert "quota_or_forbidden" in (groq_status.last_error or "")
    assert ollama_status.name == "ollama" and not ollama_status.active
    assert ollama_status.detail == "Ollama não respondeu"
