from datetime import date

from copiloto.calendar_br import special_days


def _by_day(start: date, end: date):
    return {s.day: s for s in special_days(start, end)}


def test_commemorative_dates_2026():
    days = _by_day(date(2026, 1, 1), date(2026, 12, 31))

    assert days[date(2026, 5, 10)].name == "Dia das Mães"
    assert days[date(2026, 5, 10)].kind == "data_comemorativa"
    assert days[date(2026, 6, 12)].name == "Dia dos Namorados"
    assert days[date(2026, 8, 9)].name == "Dia dos Pais"
    assert days[date(2026, 11, 27)].name == "Black Friday"


def test_national_holidays_are_feriado():
    days = _by_day(date(2026, 1, 1), date(2026, 12, 31))

    assert days[date(2026, 9, 7)].kind == "feriado"
    assert days[date(2026, 11, 2)].kind == "feriado"


def test_commemorative_date_on_a_holiday_keeps_feriado_and_both_names():
    oct_12 = _by_day(date(2026, 10, 1), date(2026, 10, 31))[date(2026, 10, 12)]

    assert oct_12.kind == "feriado"
    assert "Aparecida" in oct_12.name
    assert "Dia das Crianças" in oct_12.name


def test_range_is_inclusive_and_sorted():
    result = special_days(date(2026, 9, 7), date(2026, 10, 12))

    assert [s.day for s in result] == [date(2026, 9, 7), date(2026, 10, 12)]
