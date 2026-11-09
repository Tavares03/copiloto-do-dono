"""Quais provedores entram na fila, em que ordem e com qual cliente."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import OpenAI

from copiloto.config import Settings


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    client: OpenAI
    base_url: str = ""
    params: dict[str, Any] = field(default_factory=dict)  # parâmetros extras por provedor


@dataclass(frozen=True)
class SkippedProvider:
    name: str
    reason: str


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    detail: str


def probe_ollama(
    base_url: str, model: str, *, timeout_s: float = 0.5, http: httpx.Client | None = None
) -> ProbeResult:
    """Ollama está rodando e com o modelo baixado? Timeout curto para não atrasar a subida."""
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        response = (http or httpx.Client()).get(f"{root}/api/tags", timeout=timeout_s)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return ProbeResult(False, f"Ollama não respondeu em {root} ({type(exc).__name__})")
    names = {m.get("name") for m in response.json().get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        return ProbeResult(False, f"modelo {model} não baixado (rode: ollama pull {model})")
    return ProbeResult(True, "ok")


Probe = Callable[[str, str], ProbeResult]


def build_providers(
    settings: Settings, *, probe: Probe = probe_ollama
) -> tuple[list[Provider], list[SkippedProvider]]:
    """Monta a fila a partir de LLM_PROVIDERS, pulando quem não está utilizável."""
    catalog = {
        "groq": (settings.groq_base_url, settings.groq_model, settings.groq_api_key),
        "gemini": (settings.gemini_base_url, settings.gemini_model, settings.gemini_api_key),
        "ollama": (settings.ollama_base_url, settings.ollama_model, "ollama"),
    }
    reasoning = {
        "groq": settings.groq_reasoning_effort,
        "gemini": settings.gemini_reasoning_effort,
    }
    providers: list[Provider] = []
    skipped: list[SkippedProvider] = []
    seen: set[str] = set()

    for name in settings.provider_order:
        if name in seen:
            continue
        seen.add(name)
        if name not in catalog:
            skipped.append(SkippedProvider(name, "provedor desconhecido"))
            continue
        base_url, model, api_key = catalog[name]
        if not api_key:
            skipped.append(SkippedProvider(name, f"sem {name.upper()}_API_KEY no .env"))
            continue
        if name == "ollama":
            result = probe(base_url, model)
            if not result.ok:
                skipped.append(SkippedProvider(name, result.detail))
                continue
        client = OpenAI(
            base_url=base_url, api_key=api_key, timeout=settings.llm_timeout_s, max_retries=0
        )
        effort = reasoning.get(name)
        params = {"reasoning_effort": effort} if effort else {}
        providers.append(Provider(name, model, client, base_url=base_url, params=params))

    return providers, skipped
