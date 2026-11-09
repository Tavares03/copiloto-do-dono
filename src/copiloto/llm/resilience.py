"""Classificação de erros, Retry-After e circuit breaker."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Literal

import openai

CircuitState = Literal["closed", "open", "half_open"]


@dataclass(frozen=True)
class Failure:
    kind: str  # rótulo que vai para Attempt.outcome e para os logs
    retries: int  # novas tentativas permitidas no MESMO provedor
    hard: bool  # abre o circuito na hora (chave inválida, cota acabou, modelo sumiu)
    status_code: int | None
    detail: str


class EmptyResponseError(openai.OpenAIError):
    """O provedor respondeu 200, mas sem texto e sem tool call."""


def error_text(exc: openai.OpenAIError) -> str:
    """Mensagem + corpo do erro: o SDK guarda campos como `code` só no corpo."""
    body = getattr(exc, "body", None)
    extra = json.dumps(body, ensure_ascii=False, default=str) if body is not None else ""
    return f"{exc} {extra}".strip()


def classify(exc: openai.OpenAIError, max_retries: int) -> Failure:
    """Decide o que fazer com cada erro do SDK."""
    detail = error_text(exc)[:300]
    if isinstance(exc, EmptyResponseError):
        return Failure("empty_response", min(1, max_retries), False, None, detail)
    if isinstance(exc, openai.APITimeoutError):  # antes de APIConnectionError (é subclasse)
        return Failure("timeout", min(1, max_retries), False, None, detail)
    if isinstance(exc, openai.APIConnectionError):
        return Failure("connection_error", max_retries, False, None, detail)
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        if status == 429:
            return Failure("rate_limited", max_retries, False, status, detail)
        if status == 401:
            return Failure("auth_error", 0, True, status, detail)
        if status in (402, 403):
            return Failure("quota_or_forbidden", 0, True, status, detail)
        if status == 404:
            return Failure("model_not_found", 0, True, status, detail)
        if status == 400 and "tool_use_failed" in detail:
            return Failure("bad_tool_call", 1, False, status, detail)
        if status >= 500:
            return Failure("server_error", max_retries, False, status, detail)
        return Failure("bad_request", 0, False, status, detail)
    return Failure("invalid_response", 0, False, None, detail)


_RETRY_DELAY_IN_BODY = re.compile(r'"?retryDelay"?\s*:\s*"(\d+(?:\.\d+)?)s"')


def retry_after_seconds(
    headers: Mapping[str, str], text: str = "", *, now: datetime | None = None
) -> float | None:
    """Quanto o provedor pediu para esperar, se pediu.

    Fontes, em ordem: header retry-after-ms, header retry-after (segundos ou data HTTP)
    e o campo retryDelay que o Gemini coloca no corpo do erro.
    """
    if (millis := headers.get("retry-after-ms")) is not None:
        try:
            return max(0.0, float(millis) / 1000)
        except ValueError:
            pass
    if (value := headers.get("retry-after")) is not None:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                when = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                when = None
            if when is not None:
                return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())
    if match := _RETRY_DELAY_IN_BODY.search(text):
        return float(match.group(1))
    return None


def retry_delay(
    failure: Failure,
    exc: openai.OpenAIError,
    retry: int,
    *,
    backoff_base_s: float,
    retry_after_max_s: float,
) -> float | None:
    """Segundos até a próxima tentativa no mesmo provedor, ou None para desistir dele."""
    if retry >= failure.retries:
        return None
    response = getattr(exc, "response", None)
    hint = retry_after_seconds(response.headers if response is not None else {}, error_text(exc))
    if hint is not None:
        return hint if hint <= retry_after_max_s else None
    if failure.kind == "bad_tool_call":
        return 0.0
    return backoff_base_s * 2**retry


class CircuitBreaker:
    """Depois de `threshold` falhas seguidas, o provedor fica em espera por `cooldown_s`.

    Passada a espera, entra em "half_open": a próxima chamada é um teste. Se der certo,
    o circuito fecha; se falhar, volta a esperar.
    """

    def __init__(
        self, threshold: int, cooldown_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self.consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self.cooldown_s:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def retry_in_s(self) -> float | None:
        if self.state != "open" or self._opened_at is None:
            return None
        return self.cooldown_s - (self._clock() - self._opened_at)

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self._opened_at = None

    def record_failure(self, *, hard: bool = False) -> None:
        self.consecutive_failures += 1
        if hard or self.consecutive_failures >= self.threshold:
            self._opened_at = self._clock()
