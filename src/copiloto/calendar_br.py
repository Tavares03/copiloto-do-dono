"""Feriados nacionais e datas comemorativas que mexem com o movimento de restaurantes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import holidays as pyholidays

Kind = Literal["feriado", "data_comemorativa"]


@dataclass(frozen=True)
class SpecialDay:
    day: date
    name: str
    kind: Kind


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-ésimo dia da semana do mês (weekday: 0 = segunda ... 6 = domingo)."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _commemorative(year: int) -> list[tuple[date, str]]:
    return [
        (_nth_weekday(year, 5, 6, 2), "Dia das Mães"),
        (date(year, 6, 12), "Dia dos Namorados"),
        (_nth_weekday(year, 8, 6, 2), "Dia dos Pais"),
        (date(year, 10, 12), "Dia das Crianças"),
        (_nth_weekday(year, 11, 3, 4) + timedelta(days=1), "Black Friday"),
    ]


def special_days(start: date, end: date) -> list[SpecialDay]:
    """Feriados nacionais + datas comemorativas entre start e end (inclusive), ordenados.

    Quando uma data comemorativa cai num feriado, o dia continua como "feriado" e os nomes
    são combinados (ex.: "Nossa Senhora Aparecida / Dia das Crianças").
    """
    years = range(start.year, end.year + 1)
    national = pyholidays.country_holidays("BR", years=years, language="pt_BR")

    result: dict[date, SpecialDay] = {
        day: SpecialDay(day, name, "feriado")
        for day, name in national.items()
        if start <= day <= end
    }
    for year in years:
        for day, name in _commemorative(year):
            if not start <= day <= end:
                continue
            if day in result:
                current = result[day]
                result[day] = SpecialDay(day, f"{current.name} / {name}", current.kind)
            else:
                result[day] = SpecialDay(day, name, "data_comemorativa")

    return sorted(result.values(), key=lambda s: s.day)
