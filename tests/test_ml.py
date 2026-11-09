"""Previsão de demanda: features sem vazamento, backtest temporal e modelo versionado.

Usa o gerador direto (sem banco): mesmos dados da aplicação, seed 42.
"""

import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
import pytest

from copiloto.calendar_br import special_days
from copiloto.data.generator import generate
from copiloto.ml.baseline import SLOTS, baseline_forecast
from copiloto.ml.features import (
    History,
    build_dataset,
    closed_weekdays,
    feature_row,
    fifth_business_day,
    history_from_rows,
)
from copiloto.ml.model import ModelForecaster, latest_model_path
from copiloto.ml.train import TrainReport, train

START, END = date(2026, 5, 1), date(2026, 10, 31)
TODAY = date(2026, 11, 1)
FRIDAY = date(2026, 10, 30)


def _rows() -> list[dict]:
    data = generate(seed=42)
    orders = data.orders[data.orders["status"] == "concluido"]
    local = orders["created_at"].dt.tz_convert("America/Sao_Paulo")
    slot = pd.cut(local.dt.hour, [0, 15, 18, 24], right=False, labels=list(SLOTS))
    counts = (
        pd.DataFrame({"order_date": local.dt.date, "time_slot": slot.astype(str)})
        .value_counts()
        .reset_index(name="pedidos")
    )
    return counts.to_dict("records")


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return _rows()


@pytest.fixture(scope="module")
def history(rows: list[dict]) -> History:
    return history_from_rows(rows)


# --- features --------------------------------------------------------------------------------


def test_lag_features_are_exactly_the_baseline(history: History):
    for slot in SLOTS:
        row = feature_row(FRIDAY, slot, history, {}, trend_origin=START)
        expected = baseline_forecast(history, FRIDAY, origin=FRIDAY - timedelta(days=6))[slot]
        assert row["lag_mean_4"] == pytest.approx(expected)
        assert row["lag_7"] == history.get((FRIDAY - timedelta(days=7), slot), 0.0)


def test_features_never_see_the_6_days_before_the_target(history: History):
    tampered = dict(history)
    for back in range(7):
        for slot in SLOTS:
            tampered[(FRIDAY - timedelta(days=back), slot)] = 9999.0

    for slot in SLOTS:
        assert feature_row(FRIDAY, slot, tampered, {}, START) == feature_row(
            FRIDAY, slot, history, {}, START
        )


def test_fifth_business_day():
    assert fifth_business_day(2026, 10, set()) == date(2026, 10, 7)
    assert fifth_business_day(2026, 11, {date(2026, 11, 2)}) == date(2026, 11, 9)  # Finados


def test_calendar_features():
    specials = {s.day: s for s in special_days(date(2026, 10, 1), date(2026, 10, 31))}

    def row(day: date) -> dict:
        return feature_row(day, "jantar", {}, specials, trend_origin=START)

    assert row(date(2026, 10, 12))["is_holiday"] == 1
    assert row(date(2026, 5, 31))["trend"] == 1.0  # meses desde o início dos dados
    assert row(date(2026, 10, 30))["weekday_slot"] == 4 * 3 + 2  # sexta × jantar
    assert row(date(2026, 10, 11))["is_holiday_eve"] == 1
    assert row(date(2026, 10, 7))["is_payday"] == 1  # 5º dia útil
    assert row(date(2026, 10, 20))["is_payday"] == 1  # vale
    assert row(date(2026, 10, 14))["is_payday"] == 0
    assert row(date(2026, 10, 27))["is_month_end"] == 1


def test_closed_weekdays_are_learned_from_history(history: History):
    assert closed_weekdays(history, START, END) == {0}  # segunda


def test_dataset_skips_warmup_and_closed_days(history: History):
    specials = {s.day: s for s in special_days(START, END)}

    dataset = build_dataset(history, START, END, specials, closed={0})

    assert dataset["day"].min() == START + timedelta(days=28)
    assert not any(day.weekday() == 0 for day in dataset["day"])
    assert set(dataset["slot_name"]) == set(SLOTS)
    assert not dataset.isna().any().any()


# --- treino e backtest -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(rows: list[dict], tmp_path_factory: pytest.TempPathFactory) -> tuple[TrainReport, Path]:
    models_dir = tmp_path_factory.mktemp("models")
    report = train(
        rows, data_start=START, data_end=END, models_dir=models_dir, trained_on=date(2026, 11, 8)
    )
    return report, models_dir


def test_training_saves_versioned_model_and_metrics(trained: tuple[TrainReport, Path]):
    report, models_dir = trained

    assert report.model_path.name == "demand_v1_2026-11-08.joblib"
    metadata = json.loads((models_dir / "demand_v1_2026-11-08.json").read_text("utf-8"))
    assert [fold["fold"] for fold in metadata["backtest"]] == ["2026-08", "2026-09", "2026-10"]
    assert set(metadata["backtest"][0]["mae"]) == {"baseline", "gbr", "poisson_glm"}
    assert latest_model_path(models_dir) == report.model_path


def test_selection_gate_picks_a_model_only_if_it_beats_the_baseline_last_month(
    trained: tuple[TrainReport, Path],
):
    report, _ = trained
    last = report.folds[-1].mae

    assert report.selected == min(last, key=last.get)
    assert report.selected != "baseline"
    assert report.mae_model == last[report.selected] < report.mae_baseline == last["baseline"]


class _Source:
    def __init__(self, history: History) -> None:
        self.history = history

    def orders_by_day_slot(self, start: date, end: date) -> list[dict]:
        return [
            {"order_date": day, "time_slot": slot, "pedidos": value}
            for (day, slot), value in self.history.items()
            if start <= day <= end
        ]


def test_model_forecaster(trained: tuple[TrainReport, Path], history: History):
    report, _ = trained
    forecaster = ModelForecaster(joblib.load(report.model_path), _Source(history), today=TODAY)

    friday = forecaster.predict(date(2026, 11, 6))
    assert friday["jantar"] > friday["almoco"] > friday["tarde"] > 0
    assert forecaster.predict(date(2026, 11, 2)) == {slot: 0.0 for slot in SLOTS}  # segunda
    assert forecaster.mae == report.mae_model
    assert forecaster.name == "demand_v1_2026-11-08"
    with pytest.raises(ValueError):
        forecaster.predict(date(2026, 11, 8))  # 7 dias à frente: o lag de 7 dias ainda não existe
