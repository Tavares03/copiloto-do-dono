"""Features da previsão de demanda (pedidos concluídos por dia e faixa de horário).

Regra de ouro contra vazamento: para prever o dia D, só se usa demanda de D-7 para trás.
Por isso a previsão vai no máximo 6 dias à frente de "hoje".
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from typing import Any

import pandas as pd

from copiloto.calendar_br import SpecialDay
from copiloto.ml.baseline import SLOTS

History = dict[tuple[date, str], float]  # (dia, faixa) -> pedidos concluídos

LAG_WEEKS = 4
WARMUP_DAYS = 7 * LAG_WEEKS
CALENDAR_FLAGS = ["is_month_end", "is_payday", "is_holiday", "is_holiday_eve", "is_commemorative"]
# Árvores: dia da semana e faixa como números + lags (o baseline vira uma feature).
GBR_FEATURES = ["weekday", "slot", "day_of_month", *CALENDAR_FLAGS, "lag_7", "lag_mean_4"]
# GLM de Poisson: um nível por (dia da semana × faixa), tendência e flags de calendário.
GLM_FEATURES = ["weekday_slot", "trend", *CALENDAR_FLAGS]


def history_from_rows(rows: Iterable[Mapping[str, Any]]) -> History:
    return {(row["order_date"], row["time_slot"]): float(row["pedidos"]) for row in rows}


def closed_weekdays(history: History, start: date, end: date) -> set[int]:
    """Dias da semana sem nenhum pedido no período (ex.: a loja fecha às segundas)."""
    totals = dict.fromkeys(range(7), 0.0)
    for (day, _), orders in history.items():
        if start <= day <= end:
            totals[day.weekday()] += orders
    return {weekday for weekday, total in totals.items() if total == 0}


def fifth_business_day(year: int, month: int, holidays: set[date]) -> date:
    """5º dia útil do mês: quando a maioria recebe o salário."""
    day, count = date(year, month, 1), 0
    while True:
        if day.weekday() < 5 and day not in holidays:
            count += 1
            if count == 5:
                return day
        day += timedelta(days=1)


def feature_row(
    day: date,
    slot: str,
    history: History,
    specials: Mapping[date, SpecialDay],
    trend_origin: date,
) -> dict[str, float]:
    lags = [history.get((day - timedelta(days=7 * k), slot), 0.0) for k in range(1, 5)]
    special = specials.get(day)
    next_special = specials.get(day + timedelta(days=1))
    holidays = {d for d, s in specials.items() if s.kind == "feriado"}
    payday = fifth_business_day(day.year, day.month, holidays)
    return {
        "weekday": day.weekday(),
        "slot": SLOTS.index(slot),
        "weekday_slot": day.weekday() * len(SLOTS) + SLOTS.index(slot),
        "trend": (day - trend_origin).days / 30,  # meses desde o início do histórico
        "day_of_month": day.day,
        "is_month_end": int(day.day >= 26),
        "is_payday": int(payday <= day <= payday + timedelta(days=2) or day.day in (20, 21)),
        "is_holiday": int(special is not None and special.kind == "feriado"),
        "is_holiday_eve": int(next_special is not None and next_special.kind == "feriado"),
        "is_commemorative": int(special is not None and special.kind == "data_comemorativa"),
        "lag_7": lags[0],
        "lag_mean_4": sum(lags) / len(lags),
    }


def build_dataset(
    history: History,
    start: date,
    end: date,
    specials: Mapping[date, SpecialDay],
    closed: set[int],
) -> pd.DataFrame:
    """Uma linha por (dia aberto, faixa), depois das 4 semanas de aquecimento dos lags."""
    rows = []
    day = start + timedelta(days=WARMUP_DAYS)
    while day <= end:
        if day.weekday() not in closed:
            for slot in SLOTS:
                rows.append(
                    {
                        "day": day,
                        "slot_name": slot,
                        **feature_row(day, slot, history, specials, trend_origin=start),
                        "target": history.get((day, slot), 0.0),
                    }
                )
        day += timedelta(days=1)
    return pd.DataFrame(rows)
