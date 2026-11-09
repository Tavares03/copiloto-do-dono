"""Treina a previsão de demanda com backtest temporal, escolhe o modelo e salva uma versão datada.

    python -m copiloto.ml.train               # treina e salva models/demand_v1_AAAA-MM-DD.joblib
    python -m copiloto.ml.train --if-missing  # só treina se ainda não houver modelo (docker)

Validação: janela expansível por mês. Para avaliar agosto, treina só com o que veio antes de
agosto; idem setembro e outubro. Nada do futuro entra no treino de cada dobra.

Seleção: vence quem tiver o menor MAE no mês mais recente, que é o que melhor representa o
modelo final (treinado com todo o histórico). Se ninguém bater o baseline, fica o baseline.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from copiloto.calendar_br import special_days
from copiloto.ml.features import (
    CALENDAR_FLAGS,
    GBR_FEATURES,
    GLM_FEATURES,
    build_dataset,
    closed_weekdays,
    history_from_rows,
)

VERSION = "v1"
BACKTEST_MONTHS = 3
BASELINE = "baseline"


def make_gbr() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="absolute_error",  # otimiza a mesma métrica que reportamos (MAE)
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )


def make_poisson_glm() -> Any:
    """Contagem com efeitos multiplicativos: log(pedidos) = dia×faixa + tendência + calendário."""
    return make_pipeline(
        ColumnTransformer(
            [
                ("dia_faixa", OneHotEncoder(handle_unknown="ignore"), ["weekday_slot"]),
                ("numericas", StandardScaler(), ["trend", *CALENDAR_FLAGS]),
            ]
        ),
        PoissonRegressor(alpha=1e-3, max_iter=1000),
    )


@dataclass(frozen=True)
class Candidate:
    make: Callable[[], RegressorMixin]
    features: list[str]
    description: str


CANDIDATES: dict[str, Candidate] = {
    "gbr": Candidate(
        make_gbr,
        GBR_FEATURES,
        "GradientBoosting com calendário (dia da semana, faixa, feriados, dia de pagamento, "
        "fim de mês) e as médias das últimas semanas",
    ),
    "poisson_glm": Candidate(
        make_poisson_glm,
        GLM_FEATURES,
        "regressão de Poisson com um nível por dia da semana × faixa, tendência e calendário "
        "(feriados, véspera, datas comemorativas, dia de pagamento, fim de mês)",
    ),
}


@dataclass(frozen=True)
class BacktestFold:
    fold: str  # "2026-10"
    start: str
    end: str
    rows: int
    mae: dict[str, float]  # baseline e cada candidato
    mae_special_days: dict[str, float | None]  # feriados, vésperas e datas comemorativas


@dataclass(frozen=True)
class TrainReport:
    model_path: Path
    metadata_path: Path
    folds: list[BacktestFold]
    selected: str
    mae_baseline: float  # no mês mais recente
    mae_model: float  # do selecionado, no mês mais recente


def month_folds(end: date, months: int = BACKTEST_MONTHS) -> list[tuple[str, date, date]]:
    """Os últimos `months` meses até `end`, do mais antigo ao mais novo."""
    folds = []
    month_end = end
    for _ in range(months):
        month_start = month_end.replace(day=1)
        folds.append((month_start.strftime("%Y-%m"), month_start, month_end))
        month_end = month_start - timedelta(days=1)
    return list(reversed(folds))


def _mae(actual: Any, predicted: Any) -> float:
    return round(float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted)))), 3)


def backtest(dataset: pd.DataFrame, folds: list[tuple[str, date, date]]) -> list[BacktestFold]:
    results = []
    for name, start, end in folds:
        train_set = dataset[dataset["day"] < start]
        test_set = dataset[(dataset["day"] >= start) & (dataset["day"] <= end)]
        special = (
            (test_set["is_holiday"] == 1)
            | (test_set["is_holiday_eve"] == 1)
            | (test_set["is_commemorative"] == 1)
        ).to_numpy()
        predictions = {BASELINE: test_set["lag_mean_4"].to_numpy()}
        for key, candidate in CANDIDATES.items():
            model = candidate.make().fit(train_set[candidate.features], train_set["target"])
            predictions[key] = np.clip(model.predict(test_set[candidate.features]), 0, None)
        actual = test_set["target"].to_numpy()
        results.append(
            BacktestFold(
                fold=name,
                start=start.isoformat(),
                end=end.isoformat(),
                rows=len(test_set),
                mae={key: _mae(actual, pred) for key, pred in predictions.items()},
                mae_special_days={
                    key: _mae(actual[special], pred[special]) if special.any() else None
                    for key, pred in predictions.items()
                },
            )
        )
    return results


def train(
    rows: Iterable[Mapping[str, Any]],
    *,
    data_start: date,
    data_end: date,
    models_dir: Path,
    trained_on: date,
) -> TrainReport:
    history = history_from_rows(rows)
    specials = {s.day: s for s in special_days(data_start, data_end + timedelta(days=60))}
    closed = closed_weekdays(history, data_start, data_end)
    dataset = build_dataset(history, data_start, data_end, specials, closed)

    folds = backtest(dataset, month_folds(data_end))
    latest = folds[-1].mae
    selected = min(latest, key=latest.get)  # baseline incluído: só troca se ganhar dele

    estimator = None
    if selected != BASELINE:
        candidate = CANDIDATES[selected]
        estimator = candidate.make().fit(dataset[candidate.features], dataset["target"])

    weights = [fold.rows for fold in folds]
    metadata = {
        "name": f"demand_{VERSION}_{trained_on.isoformat()}",
        "version": VERSION,
        "trained_on": trained_on.isoformat(),
        "data_start": data_start.isoformat(),
        "data_end": data_end.isoformat(),
        "target": "pedidos concluídos por dia e faixa (almoco, tarde, jantar)",
        "selected": selected,
        "description": (
            CANDIDATES[selected].description
            if selected != BASELINE
            else "média da mesma faixa no mesmo dia da semana nas últimas 4 semanas"
        ),
        "features": CANDIDATES[selected].features if selected != BASELINE else [],
        "trend_origin": data_start.isoformat(),
        "closed_weekdays": sorted(closed),
        "train_rows": len(dataset),
        "mae": {"baseline": latest[BASELINE], "modelo": latest[selected]},
        "mae_backtest_medio": {
            key: round(float(np.average([f.mae[key] for f in folds], weights=weights)), 3)
            for key in latest
        },
        "backtest": [asdict(fold) for fold in folds],
        "sklearn": sklearn.__version__,
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{metadata['name']}.joblib"
    metadata_path = model_path.with_suffix(".json")
    joblib.dump({"model": estimator, **metadata}, model_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    return TrainReport(
        model_path, metadata_path, folds, selected, latest[BASELINE], latest[selected]
    )


def main(argv: list[str] | None = None) -> int:
    from copiloto.config import get_settings
    from copiloto.data.repository import AnalyticsRepository
    from copiloto.ml.model import latest_model_path

    parser = argparse.ArgumentParser(description="Treina o modelo de previsão de demanda.")
    parser.add_argument("--if-missing", action="store_true", help="pula se já houver modelo")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.if_missing and (existing := latest_model_path(settings.model_dir)):
        print(f"Modelo já existe: {existing.name}. Nada a fazer.")
        return 0

    repo = AnalyticsRepository(settings)
    first, last = repo.data_range()
    if first is None or last is None:
        print("Sem dados no banco. Rode antes: python -m copiloto.data.setup")
        return 1
    report = train(
        repo.orders_by_day_slot(first, last),
        data_start=first,
        data_end=last,
        models_dir=settings.model_dir,
        trained_on=date.today(),
    )
    names = [BASELINE, *CANDIDATES]
    print(f"Modelo salvo em {report.model_path}\n")
    print("MAE (pedidos por faixa) no backtest temporal:")
    print(f"{'mês':<9}{'linhas':>7}" + "".join(f"{n:>14}" for n in names))
    for fold in report.folds:
        print(f"{fold.fold:<9}{fold.rows:>7}" + "".join(f"{fold.mae[n]:>14.2f}" for n in names))
    gain = 1 - report.mae_model / report.mae_baseline
    print(
        f"\nSelecionado: {report.selected} (MAE {report.mae_model:.2f} vs. baseline "
        f"{report.mae_baseline:.2f} no último mês, {gain:.1%} menos erro)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
