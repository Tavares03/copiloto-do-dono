"""get_metrics contra os 6 pedidos de tiny_data.py (contas no docstring de lá)."""

from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError

from copiloto.agent.tools.metrics import get_metrics
from copiloto.config import Settings
from copiloto.data.repository import AnalyticsRepository

pytestmark = pytest.mark.db

TODAY = date(2026, 11, 1)
PERIOD = {"start_date": "2026-10-30", "end_date": "2026-10-31"}


@pytest.fixture(scope="module")
def metrics(tiny_db: Settings):
    repo = AnalyticsRepository(tiny_db)

    def run(metric: str, **args: Any) -> dict[str, Any]:
        return get_metrics({"metric": metric, **PERIOD, **args}, repo=repo, today=TODAY)

    return run


def test_resumo(metrics):
    result = metrics("resumo")

    assert result["dados"] == {
        "faturamento": 130.0,  # (60 - 5) + 45 + 30
        "pedidos": 3,
        "ticket_medio": 43.33,
        "pedidos_cancelados": 1,
        "margem_contribuicao": 70.2,  # 21,20 + 29 + 20
        "dias_com_venda": 2,
    }
    assert result["periodo"] == {"inicio": "2026-10-30", "fim": "2026-10-31"}
    assert result["fonte"].startswith("get_metrics/resumo")


def test_resumo_filtered_by_channel(metrics):
    dados = metrics("resumo", channel="ifood")["dados"]

    assert (dados["faturamento"], dados["pedidos"], dados["pedidos_cancelados"]) == (55.0, 1, 1)


def test_resumo_filtered_by_weekday(metrics):
    dados = metrics("resumo", start_date="2026-10-23", weekday=6)["dados"]  # sábados: 24 e 31

    assert (dados["faturamento"], dados["pedidos"]) == (100.0, 2)


def test_resumo_with_comparison(metrics):
    result = metrics("resumo", compare_start_date="2026-10-23", compare_end_date="2026-10-24")

    comparison = result["comparacao"]
    assert comparison["dados"]["faturamento"] == 100.0
    assert comparison["variacao_pct"] == {
        "faturamento": 30.0,
        "pedidos": 50.0,
        "ticket_medio": -13.3,
    }


def test_top_itens(metrics):
    dados = metrics("top_itens")["dados"]

    assert [(i["item"], i["quantidade"], i["receita"]) for i in dados] == [
        ("Clássico", 3, 90.0),
        ("Cheddar Bacon", 1, 35.0),  # cancelado (pedido 4) não conta
        ("Refrigerante Lata", 1, 10.0),
    ]


def test_top_itens_by_category_and_limit(metrics):
    dados = metrics("top_itens", category="burger", limit=1)["dados"]

    assert [i["item"] for i in dados] == ["Clássico"]


def test_variacao_itens(metrics):
    result = metrics(
        "variacao_itens", compare_start_date="2026-10-23", compare_end_date="2026-10-24"
    )

    rows = {i["item"]: i for i in result["dados"]}
    assert [i["item"] for i in result["dados"]] == [
        "Cheddar Bacon",
        "Clássico",
        "Refrigerante Lata",
    ]
    assert rows["Cheddar Bacon"]["variacao_pct"] == -50.0
    assert rows["Clássico"]["variacao_pct"] == 200.0
    assert rows["Refrigerante Lata"]["variacao_pct"] is None  # não vendia antes
    assert result["comparacao"]["periodo"] == {"inicio": "2026-10-23", "fim": "2026-10-24"}


def test_variacao_itens_defaults_to_previous_period_of_same_length(metrics):
    result = metrics("variacao_itens")

    assert result["comparacao"]["periodo"] == {"inicio": "2026-10-28", "fim": "2026-10-29"}


def test_margem_por_canal(metrics):
    dados = metrics("margem_por_canal")["dados"]

    assert [(c["canal"], c["margem_contribuicao"], c["margem_pct"]) for c in dados] == [
        ("whatsapp", 29.0, 64.4),
        ("ifood", 21.2, 38.5),  # 55 - 20 de custo - 13,80 de comissão
        ("balcao", 20.0, 66.7),
    ]
    ifood = dados[1]
    assert (ifood["faturamento"], ifood["comissao"], ifood["custo_itens"]) == (55.0, 13.8, 20.0)


def test_horarios_pico(metrics):
    dados = metrics("horarios_pico")["dados"]

    assert dados["dias_considerados"] == 2
    assert [(h["hora"], h["pedidos"], h["media_por_dia"]) for h in dados["por_hora"]] == [
        (12, 1, 0.5),
        (19, 1, 0.5),
        (20, 1, 0.5),  # o pedido cancelado das 20h15 não conta
    ]


def test_cancelamentos(metrics):
    dados = metrics("cancelamentos")["dados"]

    assert (dados["pedidos_total"], dados["cancelados"], dados["taxa_pct"]) == (4, 1, 25.0)
    assert dados["por_motivo"] == [{"motivo": "demora na entrega", "pedidos": 1}]
    assert dados["por_canal"][0] == {
        "canal": "ifood",
        "pedidos_total": 2,
        "cancelados": 1,
        "taxa_pct": 50.0,
    }


def test_tempo_preparo(metrics):
    dados = metrics("tempo_preparo")["dados"]

    assert dados == [
        {"faixa": "almoco", "pedidos": 1, "media_min": 15.0, "mediana_min": 15.0, "p90_min": 15.0},
        {"faixa": "jantar", "pedidos": 2, "media_min": 22.5, "mediana_min": 22.5, "p90_min": 24.5},
    ]


def test_vendas_por_dia(metrics):
    dados = metrics("vendas_por_dia")["dados"]

    assert dados == [
        {"data": "2026-10-30", "dia_semana": "sexta", "pedidos": 2, "faturamento": 100.0},
        {"data": "2026-10-31", "dia_semana": "sábado", "pedidos": 1, "faturamento": 30.0},
    ]


def test_period_after_last_data_has_warning(metrics):
    result = metrics("resumo", start_date="2026-11-01", end_date="2026-11-01")

    assert result["dados"]["faturamento"] == 0.0
    assert "31/10/2026" in result["aviso"]


@pytest.mark.parametrize(
    "bad_args",
    [
        {"start_date": "2026-10-31", "end_date": "2026-10-01"},  # fim antes do início
        {"compare_start_date": "2026-10-01"},  # comparação incompleta
        {"metric": "lucro_magico"},
        {"channel": "rappi"},
        {"start_date": "ontem"},
    ],
)
def test_invalid_arguments_are_rejected(metrics, bad_args: dict[str, Any]):
    with pytest.raises((ValidationError, ValueError)):
        metric = bad_args.pop("metric", "resumo")
        metrics(metric, **bad_args)
