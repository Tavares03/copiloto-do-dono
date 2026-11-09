import httpx

from copiloto.llm.providers import ProbeResult, build_providers, probe_ollama
from fakes import fake_settings


def _ollama_up(base_url: str, model: str) -> ProbeResult:
    return ProbeResult(True, "ok")


def _ollama_down(base_url: str, model: str) -> ProbeResult:
    return ProbeResult(False, "Ollama não respondeu")


def test_provider_order_is_parsed_from_env_string():
    settings = fake_settings(llm_providers=" Groq , gemini ,, ")

    assert settings.provider_order == ["groq", "gemini"]


def test_providers_follow_configured_order():
    settings = fake_settings(
        llm_providers="gemini,groq,ollama", groq_api_key="g", gemini_api_key="m"
    )

    providers, skipped = build_providers(settings, probe=_ollama_up)

    assert [p.name for p in providers] == ["gemini", "groq", "ollama"]
    assert skipped == []
    assert providers[0].model == settings.gemini_model


def test_reasoning_effort_is_only_sent_when_configured():
    settings = fake_settings(
        llm_providers="groq,gemini",
        groq_api_key="g",
        gemini_api_key="m",
        groq_reasoning_effort="",
        gemini_reasoning_effort="none",
    )

    groq, gemini = build_providers(settings, probe=_ollama_up)[0]

    assert groq.params == {}
    assert gemini.params == {"reasoning_effort": "none"}


def test_provider_without_key_is_skipped():
    settings = fake_settings(llm_providers="groq,gemini", groq_api_key="", gemini_api_key="m")

    providers, skipped = build_providers(settings, probe=_ollama_up)

    assert [p.name for p in providers] == ["gemini"]
    assert [(s.name, s.reason) for s in skipped] == [("groq", "sem GROQ_API_KEY no .env")]


def test_ollama_is_skipped_when_not_running():
    settings = fake_settings(llm_providers="ollama")

    providers, skipped = build_providers(settings, probe=_ollama_down)

    assert providers == []
    assert skipped[0].reason == "Ollama não respondeu"


def test_unknown_and_repeated_providers():
    settings = fake_settings(llm_providers="groq,openai,groq", groq_api_key="g")

    providers, skipped = build_providers(settings, probe=_ollama_up)

    assert [p.name for p in providers] == ["groq"]
    assert [s.name for s in skipped] == ["openai"]


def _tags_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_probe_ollama_finds_pulled_model():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b"}]})

    result = probe_ollama("http://localhost:11434/v1", "qwen2.5:7b", http=_tags_client(handler))

    assert result.ok


def test_probe_ollama_reports_missing_model():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})

    result = probe_ollama("http://localhost:11434/v1", "qwen2.5:7b", http=_tags_client(handler))

    assert not result.ok
    assert "ollama pull qwen2.5:7b" in result.detail


def test_probe_ollama_reports_server_down():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("recusado", request=request)

    result = probe_ollama("http://localhost:11434/v1", "qwen2.5:7b", http=_tags_client(handler))

    assert not result.ok
    assert "não respondeu" in result.detail
