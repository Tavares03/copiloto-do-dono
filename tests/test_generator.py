from datetime import date

import pandas as pd
import pytest

from copiloto.data.generator import SyntheticData, generate


@pytest.fixture(scope="module")
def data() -> SyntheticData:
    return generate(seed=42)


@pytest.fixture(scope="module")
def local_ts(data: SyntheticData) -> pd.Series:
    return data.orders["created_at"].dt.tz_convert("America/Sao_Paulo")


@pytest.fixture(scope="module")
def lines(data: SyntheticData) -> pd.DataFrame:
    """Itens com nome, data local e status do pedido."""
    orders = data.orders.assign(
        day=data.orders["created_at"].dt.tz_convert("America/Sao_Paulo").dt.date
    )
    return data.order_items.merge(
        orders[["id", "day", "status"]], left_on="order_id", right_on="id", suffixes=("", "_o")
    ).merge(
        data.menu_items[["id", "name"]], left_on="menu_item_id", right_on="id", suffixes=("", "_m")
    )


def test_same_seed_gives_identical_data():
    kwargs = {"seed": 7, "start": date(2026, 6, 1), "end": date(2026, 6, 14)}
    a, b = generate(**kwargs), generate(**kwargs)

    pd.testing.assert_frame_equal(a.orders, b.orders)
    pd.testing.assert_frame_equal(a.order_items, b.order_items)


def test_different_seed_gives_different_data():
    a = generate(seed=1, start=date(2026, 6, 1), end=date(2026, 6, 14))
    b = generate(seed=2, start=date(2026, 6, 1), end=date(2026, 6, 14))

    assert not a.orders["subtotal"].equals(b.orders["subtotal"])


def test_volume_is_realistic(data: SyntheticData):
    assert 15_000 <= len(data.orders) <= 25_000
    assert 2.0 <= len(data.order_items) / len(data.orders) <= 4.5


def test_closed_on_mondays_and_open_from_11_to_23(local_ts: pd.Series):
    assert (local_ts.dt.day_of_week != 0).all()
    assert local_ts.dt.hour.between(11, 22).all()
    assert local_ts.dt.date.min() >= date(2026, 5, 1)
    assert local_ts.dt.date.max() <= date(2026, 10, 31)


def test_weekend_is_busier_than_midweek(local_ts: pd.Series):
    per_day = local_ts.dt.date.value_counts()
    weekday = pd.Series([d.weekday() for d in per_day.index], index=per_day.index)

    weekend_avg = per_day[weekday.isin([4, 5, 6])].mean()
    midweek_avg = per_day[weekday.isin([1, 2, 3])].mean()
    assert weekend_avg > 1.3 * midweek_avg


def test_lunch_and_dinner_peaks(local_ts: pd.Series):
    by_hour = local_ts.dt.hour.value_counts()

    assert by_hour[12] > 2 * by_hour[16]
    assert by_hour[20] > 2 * by_hour[16]
    assert by_hour[20] > by_hour[12]


def test_channel_mix(data: SyntheticData, local_ts: pd.Series):
    codes = data.orders["channel_id"].map(data.channels.set_index("id")["code"])
    share = codes.value_counts(normalize=True)

    assert set(share.index) == {"ifood", "99food", "balcao", "whatsapp"}
    assert 0.38 <= share["ifood"] <= 0.52
    may = codes[local_ts.dt.month == 5].value_counts(normalize=True)["99food"]
    october = codes[local_ts.dt.month == 10].value_counts(normalize=True)["99food"]
    assert october > may + 0.04


def test_cancellations(data: SyntheticData):
    orders = data.orders
    cancelled = orders["status"] == "cancelado"

    assert 0.02 <= cancelled.mean() <= 0.08
    assert orders.loc[cancelled, "cancel_reason"].notna().all()
    assert orders.loc[~cancelled, "cancel_reason"].isna().all()
    assert (orders.loc[cancelled, "channel_fee"] == 0).all()
    assert orders.loc[~cancelled, "prep_time_min"].notna().all()


def test_order_money_is_consistent(data: SyntheticData):
    orders = data.orders.set_index("id")
    items = data.order_items
    line_sum = (items["quantity"] * items["unit_price"]).groupby(items["order_id"]).sum()
    commission = data.orders["channel_id"].map(data.channels.set_index("id")["commission_pct"])
    completed = data.orders["status"] == "concluido"

    assert (line_sum - orders["subtotal"]).abs().max() < 0.01
    expected_total = orders["subtotal"] - orders["discount"] + orders["delivery_fee"]
    assert (expected_total - orders["total"]).abs().max() < 0.01
    expected_fee = data.orders["subtotal"] * commission / 100
    fee_diff = (expected_fee[completed] - data.orders.loc[completed, "channel_fee"]).abs()
    assert fee_diff.max() <= 0.005 + 1e-9  # no máximo meio centavo de arredondamento
    assert (data.orders["discount"] <= data.orders["subtotal"]).all()


def test_prep_time_is_longer_at_peak(data: SyntheticData, local_ts: pd.Series):
    prep = data.orders["prep_time_min"].astype(float)
    saturday_dinner = (local_ts.dt.day_of_week == 5) & local_ts.dt.hour.between(19, 21)
    afternoon = local_ts.dt.hour.between(15, 17)

    assert prep[saturday_dinner].median() > prep[afternoon].median() + 4


def test_smash_bacon_price_increase_reduces_sales(lines: pd.DataFrame):
    smash = lines[(lines["name"] == "Smash Bacon") & (lines["status"] == "concluido")]
    before = smash[smash["day"] < date(2026, 10, 1)]
    after = smash[smash["day"] >= date(2026, 10, 1)]

    assert set(before["unit_price"]) == {33.90}
    assert set(after["unit_price"]) == {37.90}
    september = before[before["day"] >= date(2026, 9, 1)]["quantity"].sum() / 30
    october = after["quantity"].sum() / 31
    assert october < 0.8 * september


def test_trufado_only_sells_after_launch(lines: pd.DataFrame):
    trufado = lines[lines["name"] == "Burger Trufado"]

    assert trufado["day"].min() >= date(2026, 8, 15)
    assert trufado["quantity"].sum() > 300


def test_burger_cost_rises_in_august(lines: pd.DataFrame):
    classico = lines[lines["name"] == "Clássico"]
    july = classico[classico["day"] < date(2026, 8, 1)]["unit_cost"].unique()
    september = classico[classico["day"] >= date(2026, 9, 1)]["unit_cost"].unique()

    assert len(july) == 1 and len(september) == 1
    assert september[0] == pytest.approx(july[0] * 1.06, abs=0.01)


def test_menu_shows_current_prices(data: SyntheticData):
    menu = data.menu_items.set_index("name")

    assert menu.loc["Smash Bacon", "price"] == 37.90
    assert len(menu) == 18
    assert set(menu["category"]) == {"burger", "acompanhamento", "bebida", "sobremesa"}


def test_holidays_cover_the_period(data: SyntheticData):
    days = dict(zip(data.holidays["day"], data.holidays["kind"], strict=True))

    assert days[date(2026, 9, 7)] == "feriado"
    assert days[date(2026, 8, 9)] == "data_comemorativa"
