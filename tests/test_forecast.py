from datetime import date

import pytest

from copiloto.agent.tools.forecast import forecast_demand
from copiloto.config import Settings
from copiloto.data.repository import AnalyticsRepository
from copiloto.ml.baseline import BaselineForecaster, baseline_forecast, same_weekday_history

TODAY = date(2026, 11, 1)  # domingo; dados até 31/10
FRIDAY = date(2026, 11, 6)


# --- baseline (puro, sem banco) ---------------------------------------------------------


def test_same_weekday_history_uses_the_4_previous_weeks():
    assert same_weekday_history(FRIDAY, origin=TODAY) == [
        date(2026, 10, 30), date(2026, 10, 23), date(2026, 10, 16), date(2026, 10, 9),
    ]  # fmt: skip


def test_same_weekday_history_never_includes_the_origin_day():
    assert same_weekday_history(TODAY, origin=TODAY, weeks=1) == [date(2026, 10, 25)]


def test_baseline_averages_and_treats_missing_days_as_zero():
    history = {
        (date(2026, 10, 30), "jantar"): 100.0,
        (date(2026, 10, 23), "jantar"): 80.0,
        (date(2026, 10, 16), "jantar"): 60.0,
        (date(2026, 10, 30), "almoco"): 40.0,
    }

    result = baseline_forecast(history, FRIDAY, origin=TODAY, weeks=4)

    assert result == {"almoco": 10.0, "tarde": 0.0, "jantar": 60.0}  # (100+80+60+0)/4


# --- tool forecast_demand (banco com tiny_data) --------------------------------------------


@pytest.fixture(scope="module")
def forecast(tiny_db: Settings):
    repo = AnalyticsRepository(tiny_db)
    forecaster = BaselineForecaster(repo, today=TODAY)

    def run(**args):
        return forecast_demand(args, forecaster=forecaster, repo=repo, today=TODAY)

    return run


@pytest.mark.db
def test_forecast_for_friday(forecast):
    result = forecast(target_date="2026-11-06")

    slots = {s["faixa"]: s for s in result["previsao"]}
    # sextas de referência: 30, 23, 16 e 09/10 → almoço (1+1+0+0)/4, jantar (1+0+0+0)/4
    assert slots["almoco"]["pedidos_previstos"] == 0.5
    assert slots["jantar"]["pedidos_previstos"] == 0.25
    # no almoço dessas sextas saíram 3 Clássicos em 2 pedidos → 1,5 por pedido × 0,5 pedido
    assert slots["almoco"]["itens_principais"][0] == {
        "item": "Clássico", "categoria": "burger", "quantidade": 0.8,
    }  # fmt: skip
    assert result["dia_semana"] == "sexta"
    assert result["dias_de_referencia"][0] == "2026-10-30"
    assert result["modelo"] == "baseline_media_4_semanas"


@pytest.mark.db
def test_forecast_single_slot(forecast):
    result = forecast(target_date="2026-11-06", time_slot="jantar")

    assert [s["faixa"] for s in result["previsao"]] == ["jantar"]
    assert result["previsao"][0]["por_categoria"] == {"burger": 0.2, "bebida": 0.2}


@pytest.mark.db
def test_forecast_for_monday_warns_closed(forecast):
    result = forecast(target_date="2026-11-02")

    assert result["total_pedidos_previstos"] == 0
    assert "segunda" in result["aviso"]


@pytest.mark.db
@pytest.mark.parametrize("target", ["2026-10-31", "2026-11-08", "2026-12-31"])
def test_forecast_only_from_today_to_6_days_ahead(forecast, target: str):
    # 08/11 é 7 dias à frente: a demanda de 7 dias antes (hoje) ainda não existe
    with pytest.raises(ValueError):
        forecast(target_date=target)


@pytest.mark.db
def test_last_day_of_the_horizon_is_accepted(forecast):
    assert forecast(target_date="2026-11-07")["dia_semana"] == "sábado"
