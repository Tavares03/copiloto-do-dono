"""Cache em disco das respostas do LLM, para economizar cota gratuita (principalmente nas evals)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from copiloto.llm.types import Attempt, LLMResponse, Message, ToolCall, Usage


class ResponseCache:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    @staticmethod
    def make_key(
        *,
        provider: str,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        params: dict[str, Any] | None = None,  # max_tokens, reasoning_effort...: mudam a resposta
    ) -> str:
        payload = json.dumps(
            {
                "provider": provider,
                "model": model,
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
                "params": params or {},
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        # 128 bits do hash bastam e mantêm o caminho curto (o Windows limita a 260 caracteres)
        return self.directory / key[:2] / f"{key[2:32]}.json"

    def get(self, key: str) -> LLMResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return LLMResponse(
            content=data["content"],
            tool_calls=[ToolCall(**call) for call in data["tool_calls"]],
            provider=data["provider"],
            model=data["model"],
            usage=Usage(**data["usage"]),
            latency_ms=data["latency_ms"],
            attempts=[
                Attempt(data["provider"], data["model"], "cache_hit", latency_ms=data["latency_ms"])
            ],
            cached=True,
        )

    def put(self, key: str, response: LLMResponse) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "content": response.content,
            "tool_calls": [asdict(call) for call in response.tool_calls],
            "provider": response.provider,
            "model": response.model,
            "usage": asdict(response.usage),
            "latency_ms": response.latency_ms,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # escrita atômica: nunca deixa um JSON pela metade
