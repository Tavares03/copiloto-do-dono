"""Uso do modelo treinado, com a mesma interface do baseline (Forecaster)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from copiloto.calendar_br import special_days
from copiloto.ml.baseline import SLOTS, BaselineForecaster, OrdersSource, same_weekday_history
from copiloto.ml.features import LAG_WEEKS, feature_row, history_from_rows

MAX_HORIZON_DAYS = 6  # o lag de 7 dias precisa ser um dia que já aconteceu


class ModelForecaster:
    def __init__(self, bundle: dict[str, Any], source: OrdersSource, today: date) -> None:
        self._model = bundle["model"]
        self._features: list[str] = bundle["features"]
        self._closed = set(bundle["closed_weekdays"])
        self._trend_origin = date.fromisoformat(bundle["trend_origin"])
        self.name: str = bundle["name"]
        self.description: str = bundle["description"] + "; validado contra o baseline em backtest"
        self.mae: float | None = bundle["mae"]["modelo"]
        self.source = source
        self.today = today

    def reference_days(self, target: date) -> list[date]:
        return same_weekday_history(target, self.today, LAG_WEEKS)

    def predict(self, target: date) -> dict[str, float]:
        if target - timedelta(days=7) >= self.today:
            raise ValueError(f"o modelo prevê no máximo {MAX_HORIZON_DAYS} dias à frente")
        if target.weekday() in self._closed:
            return dict.fromkeys(SLOTS, 0.0)

        first = target - timedelta(days=7 * LAG_WEEKS)
        history = history_from_rows(self.source.orders_by_day_slot(first, self.today))
        specials = {s.day: s for s in special_days(first, target + timedelta(days=1))}
        rows = pd.DataFrame(
            [feature_row(target, slot, history, specials, self._trend_origin) for slot in SLOTS]
        )
        predicted = np.clip(self._model.predict(rows[self._features]), 0, None)
        return {slot: float(value) for slot, value in zip(SLOTS, predicted, strict=True)}


def latest_model_path(models_dir: Path) -> Path | None:
    """O artefato mais recente: maior versão e, dentro dela, a data mais nova."""
    candidates = sorted(Path(models_dir).glob("demand_v*_*.joblib"))
    return candidates[-1] if candidates else None


def load_forecaster(source: OrdersSource, models_dir: Path, today: date) -> Any:
    """Modelo treinado se ele venceu o baseline; senão o baseline (nunca fica sem previsão)."""
    path = latest_model_path(models_dir)
    if path is None:
        return BaselineForecaster(source, today=today)
    bundle = joblib.load(path)
    if bundle["model"] is None:  # o treino concluiu que o baseline é melhor
        baseline = BaselineForecaster(source, today=today)
        baseline.mae = bundle["mae"]["baseline"]
        return baseline
    return ModelForecaster(bundle, source, today=today)
