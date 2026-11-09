"""Provedores de LLM falsos: um servidor HTTP em memória por trás do SDK real da OpenAI.

Assim os testes exercitam o mesmo caminho de exceções da produção (RateLimitError,
APITimeoutError...) sem nenhuma chamada de rede.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
from openai import OpenAI

from copiloto.config import Settings
from copiloto.llm import Attempt, LLMResponse, ToolCall, Usage
from copiloto.llm.providers import Provider


class ScriptedLLM:
    """LLM de mentira para testar quem usa o LLMClient: devolve respostas prontas, em ordem
    (a última se repete), e guarda as mensagens que recebeu."""

    def __init__(self, *responses: LLMResponse | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: Sequence[dict], tools: Sequence[dict] | None = None) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


def reply(content: str = "", *calls: ToolCall, provider: str = "groq", **kw: Any) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=list(calls),
        provider=provider,
        model="modelo",
        usage=Usage(100, 20),
        latency_ms=10.0,
        attempts=kw.get("attempts", [Attempt(provider, "modelo", "ok")]),
    )


def completion(
    content: str | None = "ok",
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-teste",
        "object": "chat.completion",
        "created": 0,
        "model": "modelo-teste",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": message,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def tool_call(
    call_id: str = "call_1",
    name: str = "get_metrics",
    arguments: str = '{"metric": "faturamento"}',
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
        **extra,
    }


def http_error(
    status: int, *, retry_after: float | None = None, body: Any = None
) -> httpx.Response:
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    payload = body if body is not None else {"error": {"message": f"erro {status}"}}
    return httpx.Response(status, headers=headers, json=payload)


class FakeServer:
    """Responde a cada requisição com o próximo item do roteiro (o último se repete).

    Itens: dict -> 200 com esse JSON; httpx.Response -> devolvida como está;
    classe de exceção do httpx (ex.: httpx.ReadTimeout) -> lançada.
    """

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, type) and issubclass(item, Exception):
            raise item("falha simulada", request=request)
        if isinstance(item, httpx.Response):
            return item
        return httpx.Response(200, json=item)


def make_provider(name: str, server: FakeServer, model: str = "modelo-teste") -> Provider:
    http = httpx.Client(transport=httpx.MockTransport(server))
    client = OpenAI(
        base_url=f"https://{name}.example/v1",
        api_key="chave-de-teste",
        max_retries=0,
        http_client=http,
    )
    return Provider(name=name, model=model, client=client)


class FakeClock:
    """Relógio controlável: sleep() só avança o tempo e registra quanto 'dormiu'."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def fake_settings(**overrides: Any) -> Settings:
    """Settings isolado do .env (que tem chaves reais)."""
    defaults: dict[str, Any] = {
        "groq_api_key": "",
        "gemini_api_key": "",
        "llm_max_retries": 2,
        "llm_backoff_base_s": 1.0,
        "llm_retry_after_max_s": 10.0,
        "circuit_failure_threshold": 3,
        "circuit_cooldown_s": 300.0,
        "llm_temperature": 0.0,
        "llm_cache_enabled": False,
    }
    return Settings(_env_file=None, **{**defaults, **overrides})
