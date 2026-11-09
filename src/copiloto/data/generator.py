"""Gerador de dados sintéticos de uma hamburgueria, reproduzível com seed fixa.

Todos os efeitos (picos de horário, fim de semana, canais, cancelamentos e as "histórias"
plantadas para a demo) estão declarados como constantes no topo, para ficarem fáceis de auditar.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from copiloto.calendar_br import SpecialDay, special_days

TZ = ZoneInfo("America/Sao_Paulo")
DEFAULT_START = date(2026, 5, 1)
DEFAULT_END = date(2026, 10, 31)
HOLIDAYS_AHEAD_DAYS = 120  # feriados futuros, para a previsão de demanda


@dataclass(frozen=True)
class Channel:
    id: int
    code: str
    name: str
    commission_pct: float


CHANNELS: tuple[Channel, ...] = (
    Channel(1, "ifood", "iFood", 23.0),
    Channel(2, "99food", "99Food", 12.0),
    Channel(3, "balcao", "Balcão", 0.0),
    Channel(4, "whatsapp", "WhatsApp", 0.0),
)
IFOOD, NINETY_NINE, BALCAO, WHATSAPP = range(len(CHANNELS))  # posições em CHANNELS


@dataclass(frozen=True)
class MenuItem:
    id: int
    name: str
    category: str
    price: float  # preço de lançamento
    cost: float  # custo de lançamento
    weight: float  # popularidade relativa dentro da categoria
    launched_on: date = date(2024, 3, 1)


MENU: tuple[MenuItem, ...] = (
    MenuItem(1, "Clássico", "burger", 29.90, 10.40, 0.17),
    MenuItem(2, "Cheddar Bacon", "burger", 34.90, 13.10, 0.17),
    MenuItem(3, "Smash Duplo", "burger", 32.90, 11.70, 0.16),
    MenuItem(4, "Smash Bacon", "burger", 33.90, 12.40, 0.16),
    MenuItem(5, "Frango Crispy", "burger", 28.90, 9.30, 0.11),
    MenuItem(6, "Veggie de Grão-de-Bico", "burger", 31.90, 10.10, 0.06),
    MenuItem(7, "Costela BBQ", "burger", 38.90, 15.60, 0.10),
    MenuItem(8, "Burger Trufado", "burger", 41.90, 17.20, 0.12, date(2026, 8, 15)),
    MenuItem(9, "Batata Frita", "acompanhamento", 14.90, 3.70, 0.50),
    MenuItem(10, "Batata Cheddar e Bacon", "acompanhamento", 22.90, 7.00, 0.20),
    MenuItem(11, "Onion Rings", "acompanhamento", 18.90, 5.10, 0.18),
    MenuItem(12, "Nuggets (6 un.)", "acompanhamento", 16.90, 5.50, 0.12),
    MenuItem(13, "Refrigerante Lata", "bebida", 6.90, 2.80, 0.50),
    MenuItem(14, "Suco Natural", "bebida", 9.90, 3.00, 0.15),
    MenuItem(15, "Milkshake", "bebida", 19.90, 6.10, 0.20),
    MenuItem(16, "Água Mineral", "bebida", 4.50, 1.10, 0.15),
    MenuItem(17, "Brownie", "sobremesa", 14.90, 4.20, 0.60),
    MenuItem(18, "Petit Gâteau", "sobremesa", 18.90, 5.80, 0.40),
)
MENU_BY_ID = {item.id: item for item in MENU}

# --- Histórias plantadas -----------------------------------------------------------------
PRICE_CHANGES: dict[int, tuple[date, float]] = {4: (date(2026, 10, 1), 37.90)}  # Smash Bacon
DEMAND_SHOCKS: dict[int, tuple[date, float]] = {4: (date(2026, 10, 1), 0.62)}  # reação ao aumento
BURGER_COST_INCREASE = (date(2026, 8, 1), 1.06)  # carne mais cara a partir de agosto
LAUNCH_RAMP_DAYS = 21  # lançamento leva 3 semanas para atingir a popularidade normal

# --- Movimento -----------------------------------------------------------------------------
# Pedidos por hora num dia "base", antes dos multiplicadores. Abre 11h, último pedido 22h59.
HOURLY_BASE = {
    11: 4.0, 12: 12.5, 13: 11.5, 14: 5.5,  # almoço
    15: 2.3, 16: 1.7, 17: 2.9,  # tarde
    18: 7.0, 19: 13.5, 20: 16.0, 21: 11.5, 22: 5.0,  # jantar
}  # fmt: skip
DINNER_FROM_HOUR = 18
# Multiplicadores por dia da semana (0 = segunda, fechado): (almoço e tarde, jantar)
WEEKDAY_FACTORS = {
    1: (0.95, 0.85),
    2: (1.00, 0.90),
    3: (1.00, 1.05),
    4: (1.10, 1.55),
    5: (1.35, 1.85),
    6: (1.60, 1.50),
}
HOLIDAY_FACTORS = (1.25, 1.20)
COMMEMORATIVE_FACTORS = {
    "Dia das Mães": (1.25, 1.00),
    "Dia dos Namorados": (1.00, 1.60),
    "Dia dos Pais": (1.50, 1.20),
    "Black Friday": (1.10, 1.30),
}
TREND_GROWTH = 0.10  # +10% ao longo do período
PAYDAY_DAYS, PAYDAY_FACTOR = frozenset({5, 6, 7, 20, 21}), 1.12
MONTH_END_FROM, MONTH_END_FACTOR = 26, 0.92
RAIN_PROBABILITY, RAIN_FACTOR = 0.12, 1.10
DAILY_NOISE_SIGMA = 0.08

# --- Canais ------------------------------------------------------------------------------------
APPS_SHARE = 0.55  # iFood + 99Food
NINETY_NINE_SHARE = (0.06, 0.14)  # 99Food cresce do início ao fim do período
BALCAO_SHARE, WHATSAPP_SHARE = 0.20, 0.25

# --- Cesta -------------------------------------------------------------------------------------
BURGERS_PER_ORDER = {"delivery": (0.58, 0.30, 0.08, 0.04), "balcao": (0.70, 0.24, 0.05, 0.01)}
SIDE_PER_BURGER = 0.55
DRINK_PER_BURGER = 0.60
DESSERT_PER_ORDER = {"weekday": 0.12, "weekend": 0.18}

# --- Dinheiro ----------------------------------------------------------------------------------
DELIVERY_FEES = {
    IFOOD: (5.99, 7.99, 9.99),
    NINETY_NINE: (0.0, 3.99, 5.99),
    WHATSAPP: (5.0, 7.0, 9.0),
    BALCAO: (0.0,),
}
IFOOD_COUPON = (0.15, 0.10, 15.0)  # probabilidade, % do subtotal, teto em R$
NINETY_NINE_COUPON = (0.30, (5.0, 8.0, 10.0))  # probabilidade, valores possíveis

# --- Operação ----------------------------------------------------------------------------------
CANCEL_BASE = 0.025
CANCEL_EXTRA = {IFOOD: 0.02, NINETY_NINE: 0.02, WHATSAPP: 0.01, BALCAO: 0.0}
CANCEL_PEAK_EXTRA, CANCEL_RAIN_EXTRA = 0.02, 0.01
CANCEL_REASONS = {
    "app": (
        ("cliente desistiu", "demora na entrega", "item em falta",
         "endereço não encontrado", "pagamento recusado"),
        (0.30, 0.30, 0.15, 0.10, 0.15),
    ),
    "whatsapp": (
        ("cliente desistiu", "demora na entrega", "item em falta", "endereço não encontrado"),
        (0.35, 0.35, 0.15, 0.15),
    ),
    "balcao": (("cliente desistiu", "item em falta"), (0.60, 0.40)),
}  # fmt: skip
CANCELLED_BEFORE_PREP = 0.6
PREP_BASE_MIN, PREP_PER_BURGER, PREP_PER_ORDER_IN_HOUR, PREP_NOISE = 9.0, 2.2, 0.30, 2.5


@dataclass
class SyntheticData:
    channels: pd.DataFrame
    menu_items: pd.DataFrame
    holidays: pd.DataFrame
    orders: pd.DataFrame
    order_items: pd.DataFrame


def price_on(item: MenuItem, day: date) -> float:
    change = PRICE_CHANGES.get(item.id)
    return change[1] if change and day >= change[0] else item.price


def cost_on(item: MenuItem, day: date) -> float:
    start, factor = BURGER_COST_INCREASE
    if item.category == "burger" and day >= start:
        return round(item.cost * factor, 2)
    return item.cost


def channels_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [(c.id, c.code, c.name, c.commission_pct) for c in CHANNELS],
        columns=["id", "code", "name", "commission_pct"],
    )


def menu_frame(as_of: date) -> pd.DataFrame:
    """Cardápio com preço e custo vigentes na data."""
    return pd.DataFrame(
        [
            (i.id, i.name, i.category, price_on(i, as_of), cost_on(i, as_of), True, i.launched_on)
            for i in MENU
        ],
        columns=["id", "name", "category", "price", "cost", "active", "launched_on"],
    )


@dataclass(frozen=True)
class _Choices:
    ids: np.ndarray
    probs: np.ndarray


def _category_choices(category: str) -> _Choices:
    items = [item for item in MENU if item.category == category]
    weights = np.array([item.weight for item in items])
    return _Choices(np.array([item.id for item in items]), weights / weights.sum())


SIDES, DRINKS, DESSERTS = (_category_choices(c) for c in ("acompanhamento", "bebida", "sobremesa"))


def _burger_choices(day: date) -> _Choices:
    ids: list[int] = []
    weights: list[float] = []
    for item in MENU:
        if item.category != "burger" or day < item.launched_on:
            continue
        ramp = min(1.0, ((day - item.launched_on).days + 1) / LAUNCH_RAMP_DAYS)
        weight = item.weight * ramp
        shock = DEMAND_SHOCKS.get(item.id)
        if shock and day >= shock[0]:
            weight *= shock[1]
        ids.append(item.id)
        weights.append(weight)
    w = np.array(weights)
    return _Choices(np.array(ids), w / w.sum())


@dataclass(frozen=True)
class _Day:
    day: date
    progress: float  # 0 no início do período, 1 no fim
    lunch_factor: float
    dinner_factor: float
    rainy: bool
    weekend: bool
    burgers: _Choices


def _day_context(
    day: date, progress: float, specials: dict[date, SpecialDay], rng: np.random.Generator
) -> _Day:
    lunch, dinner = WEEKDAY_FACTORS[day.weekday()]

    level = 1.0 + TREND_GROWTH * progress
    if day.day in PAYDAY_DAYS:
        level *= PAYDAY_FACTOR
    elif day.day >= MONTH_END_FROM:
        level *= MONTH_END_FACTOR

    special = specials.get(day)
    if special is not None:
        if special.kind == "feriado":
            special_lunch, special_dinner = HOLIDAY_FACTORS
        else:
            special_lunch, special_dinner = COMMEMORATIVE_FACTORS.get(special.name, (1.0, 1.0))
        lunch *= special_lunch
        dinner *= special_dinner

    rainy = bool(rng.random() < RAIN_PROBABILITY)
    if rainy:
        level *= RAIN_FACTOR
    level *= float(rng.lognormal(0.0, DAILY_NOISE_SIGMA))

    return _Day(
        day=day,
        progress=progress,
        lunch_factor=lunch * level,
        dinner_factor=dinner * level,
        rainy=rainy,
        weekend=day.weekday() >= 4,
        burgers=_burger_choices(day),
    )


def _channel_probs(ctx: _Day, hour: int) -> np.ndarray:
    lo, hi = NINETY_NINE_SHARE
    ninety_nine = lo + (hi - lo) * ctx.progress
    probs = np.zeros(len(CHANNELS))
    probs[IFOOD] = APPS_SHARE - ninety_nine
    probs[NINETY_NINE] = ninety_nine
    probs[BALCAO] = BALCAO_SHARE
    probs[WHATSAPP] = WHATSAPP_SHARE
    if hour < 15:
        probs[BALCAO] *= 1.5  # almoço: mais gente comendo no local
    elif hour >= DINNER_FROM_HOUR:
        probs[BALCAO] *= 0.8
    if ctx.rainy:
        probs[BALCAO] *= 0.5
        probs[[IFOOD, NINETY_NINE, WHATSAPP]] *= 1.1
    return probs / probs.sum()


def _basket(rng: np.random.Generator, ctx: _Day, channel: int) -> list[tuple[int, int]]:
    """Itens do pedido como [(menu_item_id, quantidade)], ordenados por id."""
    sizes = BURGERS_PER_ORDER["balcao" if channel == BALCAO else "delivery"]
    n_burgers = int(rng.choice(len(sizes), p=sizes)) + 1

    picked: list[int] = list(rng.choice(ctx.burgers.ids, size=n_burgers, p=ctx.burgers.probs))
    n_sides = int((rng.random(n_burgers) < SIDE_PER_BURGER).sum())
    picked += list(rng.choice(SIDES.ids, size=n_sides, p=SIDES.probs))
    n_drinks = int((rng.random(n_burgers) < DRINK_PER_BURGER).sum())
    picked += list(rng.choice(DRINKS.ids, size=n_drinks, p=DRINKS.probs))
    if rng.random() < DESSERT_PER_ORDER["weekend" if ctx.weekend else "weekday"]:
        picked.append(int(rng.choice(DESSERTS.ids, p=DESSERTS.probs)))

    return sorted(Counter(int(i) for i in picked).items())


def _discount(rng: np.random.Generator, channel: int, subtotal: float) -> float:
    if channel == IFOOD:
        prob, pct, cap = IFOOD_COUPON
        if rng.random() < prob:
            return round(min(subtotal * pct, cap), 2)
    elif channel == NINETY_NINE:
        prob, values = NINETY_NINE_COUPON
        if rng.random() < prob:
            return round(min(float(rng.choice(values)), subtotal * 0.3), 2)
    return 0.0


def _cancel_reason(rng: np.random.Generator, channel: int) -> str:
    group = {IFOOD: "app", NINETY_NINE: "app", WHATSAPP: "whatsapp", BALCAO: "balcao"}[channel]
    reasons, probs = CANCEL_REASONS[group]
    return str(rng.choice(reasons, p=probs))


def generate(
    seed: int = 42,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
    holidays_until: date | None = None,
) -> SyntheticData:
    """Gera o histórico completo da hamburgueria entre start e end (inclusive)."""
    rng = np.random.default_rng(seed)
    specials = {s.day: s for s in special_days(start, end)}
    total_days = max((end - start).days, 1)

    orders: list[dict[str, object]] = []
    lines: list[tuple[int, int, int, int, float, float]] = []

    for offset in range((end - start).days + 1):
        day = start + timedelta(days=offset)
        if day.weekday() == 0:  # fechado às segundas
            continue
        ctx = _day_context(day, offset / total_days, specials, rng)
        prices = {item.id: (price_on(item, day), cost_on(item, day)) for item in MENU}

        for hour, base_rate in HOURLY_BASE.items():
            factor = ctx.dinner_factor if hour >= DINNER_FROM_HOUR else ctx.lunch_factor
            n_orders = int(rng.poisson(base_rate * factor))
            seconds = np.sort(rng.integers(0, 3600, size=n_orders))
            peak = day.weekday() in (4, 5) and 19 <= hour <= 21

            for second in seconds:
                order_id = len(orders) + 1
                channel = int(rng.choice(len(CHANNELS), p=_channel_probs(ctx, hour)))
                basket = _basket(rng, ctx, channel)
                subtotal = round(sum(qty * prices[item_id][0] for item_id, qty in basket), 2)
                discount = _discount(rng, channel, subtotal)
                delivery_fee = float(rng.choice(DELIVERY_FEES[channel]))

                cancel_prob = CANCEL_BASE + CANCEL_EXTRA[channel]
                cancel_prob += CANCEL_PEAK_EXTRA if peak else 0.0
                cancel_prob += CANCEL_RAIN_EXTRA if ctx.rainy else 0.0
                cancelled = bool(rng.random() < cancel_prob)

                n_burgers = sum(
                    qty for item_id, qty in basket if MENU_BY_ID[item_id].category == "burger"
                )
                prep = (
                    PREP_BASE_MIN
                    + PREP_PER_BURGER * n_burgers
                    + PREP_PER_ORDER_IN_HOUR * n_orders
                    + rng.normal(0.0, PREP_NOISE)
                )
                prep_time: int | None = round(min(max(prep, 6.0), 60.0))
                cancel_reason: str | None = None
                if cancelled:
                    cancel_reason = _cancel_reason(rng, channel)
                    if rng.random() < CANCELLED_BEFORE_PREP:
                        prep_time = None

                commission = CHANNELS[channel].commission_pct
                orders.append(
                    {
                        "id": order_id,
                        "channel_id": CHANNELS[channel].id,
                        "created_at": datetime(
                            day.year, day.month, day.day, hour, int(second) // 60,
                            int(second) % 60, tzinfo=TZ,
                        ),
                        "status": "cancelado" if cancelled else "concluido",
                        "cancel_reason": cancel_reason,
                        "subtotal": subtotal,
                        "discount": discount,
                        "delivery_fee": delivery_fee,
                        "channel_fee": 0.0 if cancelled else round(subtotal * commission / 100, 2),
                        "total": round(subtotal - discount + delivery_fee, 2),
                        "prep_time_min": prep_time,
                    }
                )  # fmt: skip
                for item_id, qty in basket:
                    price, cost = prices[item_id]
                    lines.append((len(lines) + 1, order_id, item_id, qty, price, cost))

    orders_df = pd.DataFrame(orders)
    orders_df["created_at"] = pd.to_datetime(orders_df["created_at"])
    orders_df["prep_time_min"] = orders_df["prep_time_min"].astype("Int64")

    holidays = special_days(start, holidays_until or end + timedelta(days=HOLIDAYS_AHEAD_DAYS))

    return SyntheticData(
        channels=channels_frame(),
        menu_items=menu_frame(end),
        holidays=pd.DataFrame(
            [(h.day, h.name, h.kind) for h in holidays], columns=["day", "name", "kind"]
        ),
        orders=orders_df,
        order_items=pd.DataFrame(
            lines,
            columns=["id", "order_id", "menu_item_id", "quantity", "unit_price", "unit_cost"],
        ),
    )
