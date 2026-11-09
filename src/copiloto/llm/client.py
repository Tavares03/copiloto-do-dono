"""LLMClient: interface única sobre vários provedores gratuitos, com failover.

O resto do código só chama `client.chat(messages, tools)` e nunca sabe qual provedor
respondeu — essa informação vem na resposta, para log e para a UI.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import openai

from copiloto.config import Settings, get_settings
from copiloto.llm.cache import ResponseCache
from copiloto.llm.providers import Provider, SkippedProvider, build_providers
from copiloto.llm.resilience import CircuitBreaker, EmptyResponseError, classify, retry_delay
from copiloto.llm.types import (
    AllProvidersFailed,
    Attempt,
    LLMResponse,
    Message,
    ToolCall,
    Usage,
)
from copiloto.observability import log_event

logger = logging.getLogger("copiloto.llm")

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_TOOL_CALL_KEYS = {"id", "type", "function"}
# Campos extras de tool_call que cada provedor aceita de volta no histórico.
_PASSTHROUGH_EXTRAS: dict[str, set[str]] = {"gemini": {"extra_content"}}


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    model: str | None
    active: bool  # está na fila de failover
    circuit: str  # closed | open | half_open | n/a
    consecutive_failures: int
    retry_in_s: float | None
    last_error: str | None
    last_latency_ms: float | None
    detail: str


class LLMClient:
    def __init__(
        self,
        providers: Sequence[Provider],
        *,
        settings: Settings | None = None,
        cache: ResponseCache | None = None,
        skipped: Sequence[SkippedProvider] = (),
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings or get_settings()
        self.providers = list(providers)
        self.skipped = list(skipped)
        self.cache = cache
        self._sleep = sleep
        self._clock = clock
        self._breakers = {
            p.name: CircuitBreaker(
                self.settings.circuit_failure_threshold, self.settings.circuit_cooldown_s, clock
            )
            for p in self.providers
        }
        self._last_error: dict[str, str] = {}
        self._last_latency: dict[str, float] = {}

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> LLMClient:
        settings = settings or get_settings()
        providers, skipped = build_providers(settings)
        cache = ResponseCache(settings.llm_cache_dir) if settings.llm_cache_enabled else None
        for item in skipped:
            log_event(logger, "llm_provider_unavailable", provider=item.name, reason=item.reason)
        return cls(providers, settings=settings, cache=cache, skipped=skipped)

    # --- API pública ----------------------------------------------------------------------

    def chat(
        self, messages: Sequence[Message], tools: Sequence[dict[str, Any]] | None = None
    ) -> LLMResponse:
        """Envia a conversa ao primeiro provedor disponível, trocando de provedor se falhar."""
        tool_list = list(tools) if tools else None
        attempts: list[Attempt] = []

        for provider in self.providers:
            prepared = _prepare_messages(provider.name, messages)
            cache_key = None
            if self.cache is not None:
                cache_key = ResponseCache.make_key(
                    provider=provider.name,
                    model=provider.model,
                    messages=prepared,
                    tools=tool_list,
                    temperature=self.settings.llm_temperature,
                    params={"max_tokens": self.settings.llm_max_tokens, **provider.params},
                )
                cached = self.cache.get(cache_key)
                if cached is not None:
                    response = _with_attempts(cached, [*attempts, *cached.attempts])
                    self._log_response(response)
                    return response

            breaker = self._breakers[provider.name]
            if not breaker.allow():
                retry_in = breaker.retry_in_s()
                attempts.append(
                    Attempt(provider.name, provider.model, "circuit_open",
                            detail=f"em espera por mais {retry_in:.0f}s")
                )  # fmt: skip
                log_event(logger, "llm_provider_skipped", provider=provider.name,
                          reason="circuit_open", retry_in_s=retry_in)  # fmt: skip
                continue

            response = self._call_with_retries(provider, prepared, tool_list, attempts)
            if response is not None:
                if self.cache is not None and cache_key is not None:
                    self.cache.put(cache_key, response)
                self._log_response(response)
                return response

        log_event(
            logger,
            "llm_all_failed",
            logging.ERROR,
            attempts=[f"{a.provider}:{a.outcome}" for a in attempts],
        )
        raise AllProvidersFailed(attempts)

    def status(self) -> list[ProviderStatus]:
        """Estado de cada provedor, para o /health. Não faz nenhuma chamada de rede."""
        active = [
            ProviderStatus(
                name=p.name,
                model=p.model,
                active=True,
                circuit=self._breakers[p.name].state,
                consecutive_failures=self._breakers[p.name].consecutive_failures,
                retry_in_s=self._breakers[p.name].retry_in_s(),
                last_error=self._last_error.get(p.name),
                last_latency_ms=self._last_latency.get(p.name),
                detail="na fila",
            )
            for p in self.providers
        ]
        inactive = [
            ProviderStatus(s.name, None, False, "n/a", 0, None, None, None, s.reason)
            for s in self.skipped
        ]
        return active + inactive

    # --- internos ---------------------------------------------------------------------------

    def _call_with_retries(
        self,
        provider: Provider,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        attempts: list[Attempt],
    ) -> LLMResponse | None:
        breaker = self._breakers[provider.name]
        kwargs: dict[str, Any] = {
            "model": provider.model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            # Respostas são curtas. Sem teto, o Groq "reserva" ~900 tokens de saída por chamada
            # e estoura o limite gratuito de tokens de saída por minuto.
            "max_tokens": self.settings.llm_max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        kwargs.update(provider.params)  # ajustes por provedor (ex.: reasoning_effort)

        retry = 0
        while True:
            started = self._clock()
            try:
                completion = provider.client.chat.completions.create(**kwargs)
                latency_ms = (self._clock() - started) * 1000
                response = _parse(completion, provider, latency_ms, attempts)
            except openai.OpenAIError as exc:
                latency_ms = (self._clock() - started) * 1000
                failure = classify(exc, self.settings.llm_max_retries)
                wait = retry_delay(
                    failure,
                    exc,
                    retry,
                    backoff_base_s=self.settings.llm_backoff_base_s,
                    retry_after_max_s=self.settings.llm_retry_after_max_s,
                )
                attempts.append(
                    Attempt(provider.name, provider.model, failure.kind, failure.status_code,
                            failure.detail, latency_ms, wait or 0.0)
                )  # fmt: skip
                log_event(
                    logger,
                    "llm_attempt_failed",
                    logging.WARNING,
                    provider=provider.name,
                    model=provider.model,
                    outcome=failure.kind,
                    status_code=failure.status_code,
                    detail=failure.detail,
                    retry=retry,
                    will_retry=wait is not None,
                    wait_s=wait,
                    latency_ms=round(latency_ms, 1),
                )
                if wait is None:
                    breaker.record_failure(hard=failure.hard)
                    self._last_error[provider.name] = f"{failure.kind}: {failure.detail[:120]}"
                    return None
                if wait > 0:
                    self._sleep(wait)
                retry += 1
                continue

            breaker.record_success()
            self._last_latency[provider.name] = latency_ms
            return response

    def _log_response(self, response: LLMResponse) -> None:
        log_event(
            logger,
            "llm_response",
            provider=response.provider,
            model=response.model,
            cached=response.cached,
            latency_ms=round(response.latency_ms, 1),
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            tool_calls=[call.name for call in response.tool_calls],
            switched_from=response.switched_from,
            attempts=[f"{a.provider}:{a.outcome}" for a in response.attempts],
        )


def _prepare_messages(provider_name: str, messages: Sequence[Message]) -> list[Message]:
    """Copia o histórico removendo campos de tool_call que este provedor não aceita."""
    allowed = _TOOL_CALL_KEYS | _PASSTHROUGH_EXTRAS.get(provider_name, set())
    prepared: list[Message] = []
    for message in messages:
        if message.get("tool_calls"):
            message = {
                **message,
                "tool_calls": [
                    {k: v for k, v in call.items() if k in allowed}
                    for call in message["tool_calls"]
                ],
            }
        prepared.append(dict(message))
    return prepared


def _parse(
    completion: Any, provider: Provider, latency_ms: float, attempts: list[Attempt]
) -> LLMResponse:
    if not completion.choices:
        raise EmptyResponseError("resposta sem choices")
    choice = completion.choices[0]
    message = choice.message
    content = _THINK_BLOCK.sub("", message.content or "").strip()

    tool_calls: list[ToolCall] = []
    for raw in message.tool_calls or []:
        function = getattr(raw, "function", None)
        if function is None:
            continue
        dumped = raw.model_dump(exclude_none=True)
        extra = {k: v for k, v in dumped.items() if k not in _TOOL_CALL_KEYS}
        tool_calls.append(ToolCall(raw.id, function.name, function.arguments or "{}", extra))

    if not content and not tool_calls:
        # 200 OK sem nada útil (ex.: o modelo gastou o max_tokens "pensando"). Tratar como falha
        # garante nova tentativa/troca de provedor e que isso nunca vá para o cache.
        raise EmptyResponseError(f"resposta vazia (finish_reason={choice.finish_reason})")

    usage = completion.usage
    attempts.append(Attempt(provider.name, provider.model, "ok", latency_ms=latency_ms))
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        provider=provider.name,
        model=provider.model,
        usage=Usage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        ),
        latency_ms=latency_ms,
        attempts=list(attempts),
    )


def _with_attempts(response: LLMResponse, attempts: list[Attempt]) -> LLMResponse:
    return LLMResponse(
        content=response.content,
        tool_calls=response.tool_calls,
        provider=response.provider,
        model=response.model,
        usage=response.usage,
        latency_ms=response.latency_ms,
        attempts=attempts,
        cached=response.cached,
    )
