"""Log estruturado em JSON: cada evento é uma linha com campos consultáveis.

Dois destinos:
- stderr: todos os eventos (o que aparece no `docker compose logs`);
- logs/interactions.jsonl: só o evento `agent_run`, um por pergunta, com id da requisição,
  pergunta, provedor/modelo, trocas de provedor, tools com argumentos, tokens e latência de
  cada etapa. É a base das estatísticas (python -m copiloto.stats, GET /stats e a página de
  observabilidade da interface).
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# id da requisição atual, propagado automaticamente para todos os logs do mesmo fluxo
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

INTERACTION_EVENTS = frozenset({"agent_run"})


class JsonFormatter(logging.Formatter):
    def __init__(self, *, ensure_ascii: bool = True) -> None:
        # ASCII no stderr: acentos viram ç, que qualquer terminal (inclusive o do
        # Windows) mostra sem corromper e continua sendo JSON válido.
        super().__init__()
        self.ensure_ascii = ensure_ascii

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=self.ensure_ascii, default=str)


class _InteractionsOnly(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() in INTERACTION_EVENTS


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Registra um evento com campos estruturados (ficam em record.fields)."""
    logger.log(level, event, extra={"fields": fields})


def _ours(handler: logging.Handler) -> bool:
    return getattr(handler, "_copiloto", False)


def configure_logging(level: str = "INFO", *, interactions_path: Path | None = None) -> None:
    """Liga a saída JSON. Chamado pelos pontos de entrada (CLI, API), nunca no import."""
    root = logging.getLogger()
    if any(_ours(h) for h in root.handlers):
        return

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(JsonFormatter())
    handlers: list[logging.Handler] = [console]

    if interactions_path is not None:
        interactions_path.parent.mkdir(parents=True, exist_ok=True)
        interactions = logging.FileHandler(interactions_path, encoding="utf-8")
        interactions.setFormatter(JsonFormatter(ensure_ascii=False))
        interactions.addFilter(_InteractionsOnly())
        interactions.setLevel(logging.INFO)  # mesmo se o console estiver em WARNING
        handlers.append(interactions)

    for handler in handlers:
        handler._copiloto = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    console.setLevel(level)
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def remove_handlers() -> None:
    """Desfaz configure_logging (usado nos testes)."""
    root = logging.getLogger()
    for handler in [h for h in root.handlers if _ours(h)]:
        root.removeHandler(handler)
        handler.close()
