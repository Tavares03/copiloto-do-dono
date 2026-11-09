"""Registro de tools: esquemas para o LLM e execução que nunca derruba o loop do agente."""

from __future__ import annotations

import json
import logging
import time
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from pydantic import ValidationError

from copiloto.agent.tools.common import Category, Channel, TimeSlot
from copiloto.agent.tools.forecast import Forecaster, forecast_demand
from copiloto.agent.tools.metrics import MetricName, get_metrics
from copiloto.agent.tools.run_sql import run_sql
from copiloto.config import Settings
from copiloto.data.repository import AnalyticsRepository
from copiloto.observability import log_event

logger = logging.getLogger("copiloto.agent.tools")

Handler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema dos argumentos
    handler: Handler

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    arguments: dict[str, Any] | str  # str quando o modelo mandou JSON inválido
    output: dict[str, Any]
    latency_ms: float

    @property
    def error(self) -> str | None:
        return self.output.get("erro")


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors()[:5]:
        where = ".".join(str(p) for p in error["loc"]) or "argumentos"
        parts.append(f"{where}: {error['msg']}")
    return "; ".join(parts)


class ToolRegistry:
    def __init__(
        self, tools: Sequence[Tool], *, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._clock = clock

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, raw_arguments: str, call_id: str = "") -> ToolResult:
        """Roda a tool. Qualquer problema vira {"erro": ...} para o LLM poder corrigir."""
        started = self._clock()
        arguments: dict[str, Any] | str = raw_arguments
        tool = self._tools.get(name)
        if tool is None:
            output = {"erro": f"Tool '{name}' não existe. Disponíveis: {', '.join(self.names)}."}
        else:
            try:
                parsed = json.loads(raw_arguments or "{}")
                if not isinstance(parsed, dict):
                    raise ValueError("os argumentos precisam ser um objeto JSON")
                arguments = parsed
            except ValueError as exc:
                output = {"erro": f"Argumentos não são um JSON válido: {exc}"}
            else:
                output = self._run(tool, parsed)
        return ToolResult(call_id, name, arguments, output, (self._clock() - started) * 1000)

    @staticmethod
    def _run(tool: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            return tool.handler(arguments)
        except ValidationError as exc:
            return {"erro": f"Argumentos inválidos: {_validation_message(exc)}"}
        except ValueError as exc:  # inclui ToolInputError
            return {"erro": str(exc)}
        except Exception as exc:
            log_event(logger, "tool_failed", logging.ERROR, tool=tool.name, error=repr(exc))
            return {"erro": "Falha interna ao executar a tool; tente outra abordagem."}


# --- tools reais -------------------------------------------------------------------------------

_DATE = {"type": "string", "description": "Data no formato YYYY-MM-DD."}


def _enum(literal: Any, description: str) -> dict[str, Any]:
    return {"type": "string", "enum": list(typing.get_args(literal)), "description": description}


GET_METRICS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "metric": _enum(
            MetricName,
            "resumo = faturamento, pedidos, ticket médio, cancelados e margem; "
            "top_itens = mais vendidos; variacao_itens = itens que mais caíram/subiram vs. "
            "período de comparação; margem_por_canal; horarios_pico; cancelamentos; "
            "tempo_preparo; vendas_por_dia (máx. 92 dias).",
        ),
        "start_date": {**_DATE, "description": "Início do período (inclusive), YYYY-MM-DD."},
        "end_date": {**_DATE, "description": "Fim do período (inclusive), YYYY-MM-DD."},
        "compare_start_date": {
            **_DATE,
            "description": "Início do período de comparação (opcional).",
        },
        "compare_end_date": {**_DATE, "description": "Fim do período de comparação (opcional)."},
        "channel": _enum(Channel, "Filtra um canal (opcional)."),
        "category": _enum(Category, "Filtra uma categoria do cardápio (opcional)."),
        "weekday": {
            "type": "integer",
            "description": "Filtra um dia da semana: 1=segunda ... 7=domingo (opcional).",
        },
        "limit": {"type": "integer", "description": "Máximo de linhas, de 1 a 20 (padrão 5)."},
    },
    "required": ["metric", "start_date", "end_date"],
}

RUN_SQL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Um único SELECT (PostgreSQL) sobre as views orders, sold_items, "
            "menu, channels e holidays. LIMIT de 200 linhas é aplicado automaticamente.",
        }
    },
    "required": ["query"],
}

FORECAST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_date": {
            **_DATE,
            "description": "Dia a prever, de hoje até 6 dias à frente (YYYY-MM-DD).",
        },
        "time_slot": _enum(TimeSlot, "Uma faixa só (opcional): almoco, tarde ou jantar."),
        "top_items": {"type": "integer", "description": "Quantos itens listar, 1 a 15 (padrão 5)."},
    },
    "required": ["target_date"],
}


def build_registry(
    *,
    settings: Settings,
    repo: AnalyticsRepository,
    forecaster: Forecaster,
    today: date,
) -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                "get_metrics",
                "Métricas de vendas prontas e testadas. Prefira esta tool a run_sql.",
                GET_METRICS_SCHEMA,
                lambda args: get_metrics(args, repo=repo, today=today),
            ),
            Tool(
                "run_sql",
                "Consulta SQL somente leitura, para perguntas que nenhuma métrica responde.",
                RUN_SQL_SCHEMA,
                lambda args: run_sql(args, settings=settings),
            ),
            Tool(
                "forecast_demand",
                "Previsão de pedidos e itens para um dia futuro, por faixa de horário.",
                FORECAST_SCHEMA,
                lambda args: forecast_demand(args, forecaster=forecaster, repo=repo, today=today),
            ),
        ]
    )
