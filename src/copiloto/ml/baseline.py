"""Baseline de previsão: média da mesma faixa, no mesmo dia da semana, nas últimas semanas.

É o "chute educado" que qualquer dono faria. O modelo de ML só vale a pena se bater isso.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
from typing import Protocol

SLOTS = ("almoco", "tarde", "jantar")


def same_weekday_history(target: date, origin: date, weeks: int = 4) -> list[date]:
    """As `weeks` datas mais recentes com o mesmo dia da semana de `target`, antes de `origin`.

    `origin` é o "hoje" da previsão: só existem dados de dias anteriores a ele.
    """
    day = target
    while day >= origin:
        day -= timedelta(days=7)
    return [day - timedelta(days=7 * i) for i in range(weeks)]


def baseline_forecast(
    history: Mapping[tuple[date, str], float], target: date, origin: date, weeks: int = 4
) -> dict[str, float]:
    """Média de pedidos por faixa nos dias de referência; dia sem registro conta como zero."""
    days = same_weekday_history(target, origin, weeks)
    return {slot: sum(history.get((day, slot), 0.0) for day in days) / len(days) for slot in SLOTS}


class OrdersSource(Protocol):
    def orders_by_day_slot(self, start: date, end: date) -> list[dict]: ...


class BaselineForecaster:
    name = "baseline_media_4_semanas"
    description = "média de pedidos da mesma faixa no mesmo dia da semana nas últimas 4 semanas"
    mae: float | None = None  # erro médio medido no backtest (preenchido no treino)

    def __init__(self, source: OrdersSource, today: date, weeks: int = 4) -> None:
        self.source = source
        self.today = today
        self.weeks = weeks

    def reference_days(self, target: date) -> list[date]:
        return same_weekday_history(target, self.today, self.weeks)

    def predict(self, target: date) -> dict[str, float]:
        days = self.reference_days(target)
        rows = self.source.orders_by_day_slot(min(days), max(days))
        history = {(row["order_date"], row["time_slot"]): float(row["pedidos"]) for row in rows}
        return baseline_forecast(history, target, self.today, self.weeks)
