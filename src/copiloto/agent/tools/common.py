"""Utilidades compartilhadas pelas tools."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator


def _accept_br_date(value: Any) -> Any:
    """Aceita também dd/mm/aaaa: modelos às vezes copiam o formato brasileiro do prompt."""
    if isinstance(value, str) and len(value) == 10 and value[2] == "/" and value[5] == "/":
        return datetime.strptime(value, "%d/%m/%Y").date()
    return value


FlexibleDate = Annotated[date, BeforeValidator(_accept_br_date)]
Channel = Literal["ifood", "99food", "balcao", "whatsapp"]
Category = Literal["burger", "acompanhamento", "bebida", "sobremesa"]
TimeSlot = Literal["almoco", "tarde", "jantar"]
WEEKDAYS_PT = ("segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo")


class ToolInputError(ValueError):
    """Argumento inválido vindo do LLM; a mensagem volta para ele corrigir."""


def money(value: Any) -> float:
    return round(float(value or 0), 2)


def one_decimal(value: Any) -> float | None:
    return None if value is None else round(float(value), 1)


def br_date(day: date) -> str:
    return day.strftime("%d/%m/%Y")


def jsonable(value: Any) -> Any:
    """Converte valores do banco para algo que vira JSON sem perder sentido."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value
