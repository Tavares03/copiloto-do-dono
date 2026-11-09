from datetime import date

import psycopg
import pytest

from copiloto.config import Settings
from copiloto.data import setup
from copiloto.data.db import connect
from copiloto.data.generator import SyntheticData, generate

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def data() -> SyntheticData:
    return generate(seed=3, start=date(2026, 6, 1), end=date(2026, 6, 21))


@pytest.fixture(scope="module")
def seeded(db_settings: Settings, data: SyntheticData) -> setup.SetupReport:
    return setup.run(db_settings, reset=True, data_factory=lambda: data)


def test_loads_every_row(seeded: setup.SetupReport, db_settings: Settings, data: SyntheticData):
    tables = ("channels", "menu_items", "holidays", "orders", "order_items")
    with connect(db_settings) as conn:
        counts = {t: conn.execute(f"SELECT count(*) FROM core.{t}").fetchone()[0] for t in tables}

    assert seeded.seeded is True
    assert counts == {
        "channels": 4,
        "menu_items": 18,
        "holidays": len(data.holidays),
        "orders": len(data.orders),
        "order_items": len(data.order_items),
    }


def test_running_again_keeps_existing_data(seeded: setup.SetupReport, db_settings: Settings):
    again = setup.run(db_settings)

    assert again.seeded is False
    assert again.orders == seeded.orders


def test_net_revenue_ignores_cancelled_orders(
    seeded: setup.SetupReport, db_settings: Settings, data: SyntheticData
):
    completed = data.orders[data.orders["status"] == "concluido"]
    expected_revenue = float((completed["subtotal"] - completed["discount"]).sum())

    with connect(db_settings, readonly=True) as conn:
        total, count = conn.execute(
            "SELECT sum(net_revenue), count(net_revenue) FROM orders"
        ).fetchone()

    assert float(total) == pytest.approx(expected_revenue, abs=0.01)
    assert count == len(completed)


def test_orders_view_uses_local_time(seeded: setup.SetupReport, db_settings: Settings):
    with connect(db_settings, readonly=True) as conn:
        first_day, last_day, min_hour, max_hour, mondays = conn.execute(
            """SELECT min(order_date), max(order_date), min(order_hour), max(order_hour),
                      count(*) FILTER (WHERE weekday = 1)
               FROM orders"""
        ).fetchone()
        slots = {row[0] for row in conn.execute("SELECT DISTINCT time_slot FROM orders")}

    assert (first_day, last_day) == (date(2026, 6, 2), date(2026, 6, 21))  # 01/06 é segunda
    assert (min_hour, max_hour) == (11, 22)
    assert mondays == 0
    assert slots == {"almoco", "tarde", "jantar"}


def test_sold_items_only_has_completed_orders(
    seeded: setup.SetupReport, db_settings: Settings, data: SyntheticData
):
    completed_ids = set(data.orders.loc[data.orders["status"] == "concluido", "id"])
    expected = data.order_items["order_id"].isin(completed_ids).sum()

    with connect(db_settings, readonly=True) as conn:
        (lines,) = conn.execute("SELECT count(*) FROM sold_items").fetchone()

    assert lines == expected


def test_readonly_user_cannot_see_core_tables(seeded: setup.SetupReport, db_settings: Settings):
    with (
        connect(db_settings, readonly=True, autocommit=True) as conn,
        pytest.raises(psycopg.errors.InsufficientPrivilege),
    ):
        conn.execute("SELECT * FROM core.orders LIMIT 1")


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM analytics.orders",
        "CREATE TABLE public.hack (id int)",
        "CREATE TEMP TABLE hack (id int)",
        "DROP VIEW analytics.orders",
    ],
)
def test_readonly_user_cannot_write_even_after_disabling_read_only_mode(
    seeded: setup.SetupReport, db_settings: Settings, statement: str
):
    with connect(db_settings, readonly=True, autocommit=True) as conn:
        conn.execute("SET default_transaction_read_only = off")  # o papel não deve depender disso
        with pytest.raises(psycopg.Error):
            conn.execute(statement)


def test_readonly_user_has_timeout_and_search_path(
    seeded: setup.SetupReport, db_settings: Settings
):
    with connect(db_settings, readonly=True) as conn:
        timeout = conn.execute("SHOW statement_timeout").fetchone()[0]
        search_path = conn.execute("SHOW search_path").fetchone()[0]

    assert timeout == "5s"
    assert search_path == "analytics"
