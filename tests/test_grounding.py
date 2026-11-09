import json

import pytest

from copiloto.agent.grounding import parse_answer_numbers, ungrounded_numbers

TOOL_OUTPUT = json.dumps(
    {
        "periodo": {"inicio": "2026-10-01", "fim": "2026-10-31"},
        "dados": {"faturamento": 10689.14, "pedidos": 136, "variacao_pct": -29.5, "hora": 19},
        "canal": "99food",
        "previsao": 137.6,
    }
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("faturou R$ 10.689,14 em 136 pedidos", [10689.14, 136.0]),
        ("queda de 29,5% na média", [29.5]),
        ("cerca de 29.5 por cento", [29.5]),
        ("R$ 1.234 no total", [1234.0]),
        ("pico às 19h", [19.0]),
    ],
)
def test_parses_brazilian_number_formats(text: str, expected: list[float]):
    assert [value for _, value in parse_answer_numbers(text)] == expected


def test_dates_and_channel_names_are_not_numbers():
    text = "Fonte: get_metrics/resumo, 01/10/2026-31/10/2026 e 2026-10-31, canal 99Food, dia 06/11."

    assert parse_answer_numbers(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "Na semana passada (19 a 25/10) você faturou bem.",
        "De 1º a 31 de outubro o movimento cresceu.",
        "No dia 12 houve feriado.",
        "Entre 23 e 24/10 a loja abriu normalmente.",
    ],
)
def test_date_ranges_and_written_dates_are_not_numbers(text: str):
    assert parse_answer_numbers(text) == []


def test_grounded_answer_passes():
    answer = (
        "Ontem você faturou R$ 10.689,14 em 136 pedidos; o Smash caiu 29,5%. "
        "Pico às 19h. Fonte: get_metrics/resumo, 31/10/2026."
    )

    assert ungrounded_numbers(answer, [TOOL_OUTPUT]) == []


def test_rounded_values_are_accepted():
    assert ungrounded_numbers("Prepare uns 138 burgers.", [TOOL_OUTPUT]) == []  # 137,6
    assert ungrounded_numbers("Caiu 30%.", [TOOL_OUTPUT]) == []  # 29,5 arredondado


def test_invented_number_is_flagged():
    answer = "O pedido mais caro foi de R$ 350,00 e veio pelo iFood."

    assert ungrounded_numbers(answer, [TOOL_OUTPUT]) == ["350,00"]


def test_numbers_from_question_or_context_are_allowed():
    answer = "Quer ver os últimos 30 dias ou só outubro?"

    assert ungrounded_numbers(answer, ["sugira os últimos 30 dias"]) == []
    assert ungrounded_numbers(answer, []) == ["30"]
