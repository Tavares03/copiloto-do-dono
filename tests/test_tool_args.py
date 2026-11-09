from datetime import date

import pytest
from pydantic import ValidationError

from copiloto.agent.tools.forecast import ForecastArgs
from copiloto.agent.tools.metrics import MetricArgs


def test_dates_accept_iso_and_brazilian_format():
    args = MetricArgs.model_validate(
        {"metric": "resumo", "start_date": "30/10/2026", "end_date": "2026-10-31"}
    )

    assert (args.start_date, args.end_date) == (date(2026, 10, 30), date(2026, 10, 31))
    assert ForecastArgs.model_validate({"target_date": "06/11/2026"}).target_date == date(
        2026, 11, 6
    )


@pytest.mark.parametrize("bad", ["31/02/2026", "ontem", "2026/10/31"])
def test_invalid_dates_are_still_rejected(bad: str):
    with pytest.raises(ValidationError):
        MetricArgs.model_validate({"metric": "resumo", "start_date": bad, "end_date": bad})


def test_unknown_arguments_are_rejected():
    with pytest.raises(ValidationError):
        MetricArgs.model_validate(
            {"metric": "resumo", "start_date": "2026-10-01", "end_date": "2026-10-31", "x": 1}
        )
