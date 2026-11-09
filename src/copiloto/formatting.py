"""Números no formato brasileiro, para textos que o dono vai ler."""

from __future__ import annotations

from datetime import date

CHANNEL_NAMES = {"ifood": "iFood", "99food": "99Food", "balcao": "Balcão", "whatsapp": "WhatsApp"}


def _swap_separators(text: str) -> str:
    return text.replace(",", "§").replace(".", ",").replace("§", ".")


def brl(value: float) -> str:
    """10689.14 -> 'R$ 10.689,14'."""
    return "R$ " + _swap_separators(f"{value:,.2f}")


def number(value: float, decimals: int = 0) -> str:
    """1234.5 -> '1.234' (decimals=0) ou '1.234,5' (decimals=1)."""
    return _swap_separators(f"{value:,.{decimals}f}")


def pct(value: float, *, signed: bool = True) -> str:
    """-57.14 -> '-57,1%'; com signed=True positivos ganham '+'."""
    return _swap_separators(f"{value:{'+' if signed else ''}.1f}") + "%"


def short_date(day: date) -> str:
    return day.strftime("%d/%m")


def channel_name(code: str) -> str:
    return CHANNEL_NAMES.get(code, code)
