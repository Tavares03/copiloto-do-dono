"""Seis pedidos escritos à mão: dá para conferir cada número das métricas de cabeça.

Período principal: 30 e 31/10/2026 (sexta e sábado). Comparação: 23 e 24/10.
Preço/custo: Clássico 30/10, Cheddar Bacon 35/13, Refrigerante 10/3.

| id | dia   | hora  | canal    | status    | itens         | subtotal | desc | comissão | custo |
|----|-------|-------|----------|-----------|---------------|----------|------|----------|-------|
| 1  | 30/10 | 12:10 | ifood    | concluido | 2 Clássico    | 60       | 5    | 13,80    | 20    |
| 2  | 30/10 | 20:05 | whatsapp | concluido | Cheddar+Refri | 45       | 0    | 0        | 16    |
| 3  | 31/10 | 19:30 | balcao   | concluido | Clássico      | 30       | 0    | 0        | 10    |
| 4  | 31/10 | 20:15 | ifood    | cancelado | Cheddar       | 35       | 0    | 0        | 13    |
| 5  | 24/10 | 20:00 | ifood    | concluido | 2 Cheddar     | 70       | 0    | 16,10    | 26    |
| 6  | 23/10 | 13:00 | whatsapp | concluido | Clássico      | 30       | 0    | 0        | 10    |
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from copiloto.data.generator import SyntheticData, channels_frame, menu_frame

TZ = ZoneInfo("America/Sao_Paulo")
CLASSICO, CHEDDAR, REFRI = 1, 2, 13
IFOOD, BALCAO, WHATSAPP = 1, 3, 4

# id, canal, horário, status, motivo, subtotal, desconto, entrega, comissão, preparo, itens
ORDERS = [
    (1, IFOOD, datetime(2026, 10, 30, 12, 10, tzinfo=TZ), "concluido", None,
     60.0, 5.0, 7.99, 13.80, 15, [(CLASSICO, 2, 30.0, 10.0)]),
    (2, WHATSAPP, datetime(2026, 10, 30, 20, 5, tzinfo=TZ), "concluido", None,
     45.0, 0.0, 5.0, 0.0, 20, [(CHEDDAR, 1, 35.0, 13.0), (REFRI, 1, 10.0, 3.0)]),
    (3, BALCAO, datetime(2026, 10, 31, 19, 30, tzinfo=TZ), "concluido", None,
     30.0, 0.0, 0.0, 0.0, 25, [(CLASSICO, 1, 30.0, 10.0)]),
    (4, IFOOD, datetime(2026, 10, 31, 20, 15, tzinfo=TZ), "cancelado", "demora na entrega",
     35.0, 0.0, 7.99, 0.0, None, [(CHEDDAR, 1, 35.0, 13.0)]),
    (5, IFOOD, datetime(2026, 10, 24, 20, 0, tzinfo=TZ), "concluido", None,
     70.0, 0.0, 7.99, 16.10, 18, [(CHEDDAR, 2, 35.0, 13.0)]),
    (6, WHATSAPP, datetime(2026, 10, 23, 13, 0, tzinfo=TZ), "concluido", None,
     30.0, 0.0, 5.0, 0.0, 12, [(CLASSICO, 1, 30.0, 10.0)]),
]  # fmt: skip


def tiny_data() -> SyntheticData:
    orders, lines = [], []
    for (oid, channel, created, status, reason, subtotal, discount, delivery, fee, prep,
         items) in ORDERS:  # fmt: skip
        orders.append(
            {
                "id": oid,
                "channel_id": channel,
                "created_at": created,
                "status": status,
                "cancel_reason": reason,
                "subtotal": subtotal,
                "discount": discount,
                "delivery_fee": delivery,
                "channel_fee": fee,
                "total": round(subtotal - discount + delivery, 2),
                "prep_time_min": prep,
            }
        )
        for item_id, qty, price, cost in items:
            lines.append((len(lines) + 1, oid, item_id, qty, price, cost))

    orders_df = pd.DataFrame(orders)
    orders_df["created_at"] = pd.to_datetime(orders_df["created_at"])
    orders_df["prep_time_min"] = orders_df["prep_time_min"].astype("Int64")
    return SyntheticData(
        channels=channels_frame(),
        menu_items=menu_frame(date(2026, 10, 31)),
        holidays=pd.DataFrame(columns=["day", "name", "kind"]),
        orders=orders_df,
        order_items=pd.DataFrame(
            lines,
            columns=["id", "order_id", "menu_item_id", "quantity", "unit_price", "unit_cost"],
        ),
    )
