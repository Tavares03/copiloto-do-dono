"""Verificador de números: todo número da resposta precisa ter vindo de algum lugar.

"Algum lugar" = saída de uma tool desta conversa, a pergunta ou o contexto do system prompt.
Aceita arredondamentos (137,6 → 138; 29,5% → 30%). Datas, horários de calendário e nomes como
"99food" não contam como números.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_MONTHS = (
    "janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro"
)
# Removidos antes de procurar números: datas (31/10/2026, 31/10, 2026-10-31), início de
# intervalos ("19 a 25/10"), datas por extenso ("1º de outubro"), "dia 12" e o nome 99Food.
_NOT_NUMBERS = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}(?:/\d{2,4})?"
    r"|\b\d{1,2}º?(?=\s*(?:a|e|até|-|–)\s*\d{1,2}(?:º)?(?:/\d{1,2}|\s+de\s+(?:" + _MONTHS + r")))"
    r"|\b\d{1,2}º?\s+de\s+(?:" + _MONTHS + r")"
    r"|\bdia\s+\d{1,2}\b(?!/)"  # "dia 12"; "dia 06/11" fica para o padrão de data
    r"|\b99\s*food\b",
    re.IGNORECASE,
)
# Número em PT-BR (1.234,56 / 1234,56 / 1234) ou com ponto decimal (29.5).
_ANSWER_NUMBER = re.compile(r"(?<![\w.,])-?(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)")
_SOURCE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _to_float(token: str) -> float:
    if "," in token:  # formato brasileiro: ponto é milhar, vírgula é decimal
        return float(token.replace(".", "").replace(",", "."))
    if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", token):  # 1.234 = mil duzentos e trinta e quatro
        return float(token.replace(".", ""))
    return float(token)


def parse_answer_numbers(text: str) -> list[tuple[str, float]]:
    """Números de um texto em português, como (trecho original, valor)."""
    cleaned = _NOT_NUMBERS.sub(" ", text)
    return [(match, _to_float(match)) for match in _ANSWER_NUMBER.findall(cleaned)]


def _source_numbers(sources: Iterable[str]) -> set[float]:
    numbers: set[float] = set()
    for source in sources:
        numbers.update(abs(float(n)) for n in _SOURCE_NUMBER.findall(_NOT_NUMBERS.sub(" ", source)))
        numbers.update(abs(value) for _, value in parse_answer_numbers(source))
    return numbers


def _matches(value: float, known: float) -> bool:
    if abs(value - known) <= max(0.011, 0.005 * known):
        return True
    return value in (round(known), round(known, 1))


def ungrounded_numbers(answer: str, sources: Iterable[str]) -> list[str]:
    """Trechos numéricos da resposta que não aparecem em nenhuma fonte."""
    known = _source_numbers(sources)
    return [
        token
        for token, value in parse_answer_numbers(answer)
        if not any(_matches(abs(value), number) for number in known)
    ]
