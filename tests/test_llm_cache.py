from pathlib import Path

import pytest

from copiloto.llm import AllProvidersFailed, LLMClient, ResponseCache
from fakes import FakeClock, FakeServer, completion, fake_settings, http_error, make_provider

MESSAGES = [{"role": "user", "content": "Quanto vendi ontem?"}]


def _client(server: FakeServer, cache: ResponseCache) -> LLMClient:
    clock = FakeClock()
    return LLMClient(
        [make_provider("groq", server)],
        settings=fake_settings(),
        cache=cache,
        sleep=clock.sleep,
        clock=clock,
    )


def test_cache_hit_avoids_the_network(tmp_path: Path):
    server = FakeServer(completion("R$ 10,00"))
    client = _client(server, ResponseCache(tmp_path))

    first = client.chat(MESSAGES)
    second = client.chat(MESSAGES)

    assert server.calls == 1
    assert second.cached and not first.cached
    assert second.content == "R$ 10,00"
    assert second.latency_ms == first.latency_ms  # latência original, não a do disco
    assert second.attempts[0].outcome == "cache_hit"


def test_cache_persists_across_instances(tmp_path: Path):
    _client(FakeServer(completion("ok")), ResponseCache(tmp_path)).chat(MESSAGES)
    server = FakeServer(completion("outra"))

    response = _client(server, ResponseCache(tmp_path)).chat(MESSAGES)

    assert server.calls == 0
    assert response.content == "ok"


def test_cache_key_depends_on_provider_model_and_messages():
    base = {"provider": "groq", "model": "m", "messages": MESSAGES, "tools": None}
    key = ResponseCache.make_key(**base, temperature=0.0)

    assert key == ResponseCache.make_key(**base, temperature=0.0)
    assert key != ResponseCache.make_key(**{**base, "provider": "gemini"}, temperature=0.0)
    assert key != ResponseCache.make_key(**{**base, "model": "outro"}, temperature=0.0)
    other_messages = [{"role": "user", "content": "E hoje?"}]
    assert key != ResponseCache.make_key(**{**base, "messages": other_messages}, temperature=0.0)
    with_params = ResponseCache.make_key(
        **base, temperature=0.0, params={"reasoning_effort": "none"}
    )
    assert key != with_params


def test_failures_are_not_cached(tmp_path: Path):
    client = _client(FakeServer(http_error(403)), ResponseCache(tmp_path))

    with pytest.raises(AllProvidersFailed):
        client.chat(MESSAGES)

    assert not any(tmp_path.rglob("*.json"))
