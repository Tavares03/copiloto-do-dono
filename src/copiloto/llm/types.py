"""Tipos da camada de LLM, independentes de provedor."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON cru, exatamente como o modelo gerou
    # Campos específicos do provedor que precisam voltar no histórico
    # (ex.: extra_content.google.thought_signature dos modelos Gemini 3).
    extra: dict[str, Any] = field(default_factory=dict)

    def parsed_arguments(self) -> dict[str, Any]:
        """Argumentos como dict. Lança ValueError se o modelo gerou JSON inválido."""
        parsed = json.loads(self.arguments or "{}")
        if not isinstance(parsed, dict):
            raise ValueError("os argumentos da tool precisam ser um objeto JSON")
        return parsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
            **self.extra,
        }


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Attempt:
    """Uma tentativa de chamada (ou um provedor pulado) dentro de um chat()."""

    provider: str
    model: str
    # ok | cache_hit | circuit_open | rate_limited | quota_or_forbidden | auth_error |
    # model_not_found | bad_tool_call | bad_request | server_error | timeout |
    # connection_error | invalid_response
    outcome: str
    status_code: int | None = None
    detail: str = ""
    latency_ms: float = 0.0
    wait_s: float = 0.0  # quanto esperou antes de tentar de novo


@dataclass(frozen=True)
class LLMResponse:
    content: str
    tool_calls: list[ToolCall]
    provider: str
    model: str
    usage: Usage
    latency_ms: float  # latência da chamada que respondeu (no cache: a original)
    attempts: list[Attempt]  # tudo o que aconteceu até a resposta, em ordem
    cached: bool = False

    @property
    def switched_from(self) -> list[str]:
        """Provedores que falharam (ou foram pulados) antes do que respondeu."""
        seen: list[str] = []
        for attempt in self.attempts:
            if attempt.provider != self.provider and attempt.provider not in seen:
                seen.append(attempt.provider)
        return seen

    def as_message(self) -> Message:
        """Mensagem do assistente pronta para voltar ao histórico da conversa."""
        message: Message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        return message


class AllProvidersFailed(RuntimeError):
    def __init__(self, attempts: list[Attempt]) -> None:
        self.attempts = attempts
        summary = ", ".join(f"{a.provider}={a.outcome}" for a in attempts) or "nenhum provedor"
        super().__init__(f"Nenhum provedor de LLM respondeu ({summary}).")
