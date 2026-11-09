"""Tool forecast_demand: quantos pedidos (e itens) esperar num dia futuro, por faixa."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from copiloto.agent.tools.common import (
    WEEKDAYS_PT,
    FlexibleDate,
    TimeSlot,
    ToolInputError,
    br_date,
)
from copiloto.data.repository import AnalyticsRepository
from copiloto.ml.baseline import SLOTS

# O modelo usa a demanda de 7 dias antes do dia previsto: ela precisa já ter acontecido.
MAX_HORIZON_DAYS = 6


class Forecaster(Protocol):
    name: str
    description: str
    mae: float | None

    def reference_days(self, target: date) -> list[date]: ...

    def predict(self, target: date) -> dict[str, float]: ...


class ForecastArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_date: FlexibleDate
    time_slot: TimeSlot | None = None
    top_items: int = Field(default=5, ge=1, le=15)


def forecast_demand(
    raw_args: Mapping[str, Any],
    *,
    forecaster: Forecaster,
    repo: AnalyticsRepository,
    today: date,
) -> dict[str, Any]:
    args = ForecastArgs.model_validate(dict(raw_args))
    target = args.target_date
    if target < today:
        raise ToolInputError(
            f"forecast_demand é só para hoje ou datas futuras (hoje é {br_date(today)}). "
            "Para o passado use get_metrics."
        )
    if target > today + timedelta(days=MAX_HORIZON_DAYS):
        last = today + timedelta(days=MAX_HORIZON_DAYS)
        raise ToolInputError(
            f"A previsão vai no máximo {MAX_HORIZON_DAYS} dias à frente (até {br_date(last)})."
        )

    predicted = forecaster.predict(target)
    reference = forecaster.reference_days(target)
    items, orders_per_slot = repo.item_mix(reference)

    # itens por pedido em cada faixa nos dias de referência × pedidos previstos
    per_order: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in items:
        orders = orders_per_slot.get(row["faixa"], 0)
        if orders:
            per_order[row["faixa"]].append({**row, "por_pedido": row["quantidade"] / orders})

    slots = [args.time_slot] if args.time_slot else list(SLOTS)
    forecast = []
    for slot in slots:
        expected_orders = predicted.get(slot, 0.0)
        estimates = [
            {
                "item": row["item"],
                "categoria": row["categoria"],
                "quantidade": round(expected_orders * row["por_pedido"], 1),
            }
            for row in per_order.get(slot, [])
        ]
        by_category: dict[str, float] = defaultdict(float)
        for row in per_order.get(slot, []):
            by_category[row["categoria"]] += expected_orders * row["por_pedido"]
        estimates.sort(key=lambda r: (-r["quantidade"], r["item"]))
        forecast.append(
            {
                "faixa": slot,
                "pedidos_previstos": round(expected_orders, 2),
                "itens_principais": estimates[: args.top_items],
                "por_categoria": {cat: round(qty, 1) for cat, qty in sorted(by_category.items())},
            }
        )

    total = round(sum(slot["pedidos_previstos"] for slot in forecast), 2)
    result: dict[str, Any] = {
        "data": target.isoformat(),
        "dia_semana": WEEKDAYS_PT[target.weekday()],
        "previsao": forecast,
        "total_pedidos_previstos": total,
        "modelo": forecaster.name,
        "como_funciona": forecaster.description,
        "dias_de_referencia": [day.isoformat() for day in reference],
        "erro_medio_tipico_pedidos": forecaster.mae,
        "fonte": f"forecast_demand ({forecaster.name})",
    }
    if total == 0:
        result["aviso"] = (
            f"Nenhum pedido nas últimas {len(reference)} {WEEKDAYS_PT[target.weekday()]}s: "
            "o restaurante provavelmente não abre nesse dia (fecha às segundas)."
        )
    return result
