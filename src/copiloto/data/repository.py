"""Acesso aos dados de vendas: todo o SQL das métricas mora aqui, sempre sobre o schema analytics.

É a fronteira com a fonte de dados. Para ligar no ComandaHub basta que as views de analytics
existam lá com as mesmas colunas (ou escrever outra classe com os mesmos métodos).
Sempre usa o usuário somente leitura.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal
from typing import Any

from psycopg.rows import dict_row

from copiloto.config import Settings, get_settings
from copiloto.data.db import connect

Row = dict[str, Any]

# Filtros opcionais: NULL desliga o filtro. Sempre parametrizados, nunca concatenados.
_ORDER_FILTERS = """
    order_date BETWEEN %(start)s AND %(end)s
    AND (%(channel)s::text IS NULL OR channel = %(channel)s)
    AND (%(weekday)s::int IS NULL OR weekday = %(weekday)s)
"""
_ITEM_FILTERS = _ORDER_FILTERS + " AND (%(category)s::text IS NULL OR category = %(category)s)"


def _plain(value: Any) -> Any:
    return float(value) if isinstance(value, Decimal) else value


class AnalyticsRepository:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def fetch(self, sql: str, params: Mapping[str, Any] | None = None) -> list[Row]:
        with (
            connect(self.settings, readonly=True) as conn,
            conn.cursor(row_factory=dict_row) as cur,
        ):
            cur.execute(sql, params or {})
            return [{key: _plain(value) for key, value in row.items()} for row in cur.fetchall()]

    @staticmethod
    def _params(start: date, end: date, **filters: Any) -> dict[str, Any]:
        defaults = {"channel": None, "weekday": None, "category": None}
        return {"start": start, "end": end, **defaults, **filters}

    # --- período e resumo ----------------------------------------------------------------

    def data_range(self) -> tuple[date | None, date | None]:
        row = self.fetch("SELECT min(order_date) AS first, max(order_date) AS last FROM orders")[0]
        return row["first"], row["last"]

    def summary(
        self, start: date, end: date, *, channel: str | None = None, weekday: int | None = None
    ) -> Row:
        return self.fetch(
            f"""SELECT coalesce(sum(net_revenue), 0)                      AS faturamento,
                       count(net_revenue)                                  AS pedidos,
                       avg(net_revenue)                                    AS ticket_medio,
                       count(*) FILTER (WHERE status = 'cancelado')        AS pedidos_cancelados,
                       coalesce(sum(contribution_margin), 0)               AS margem_contribuicao,
                       count(DISTINCT order_date) FILTER (WHERE status = 'concluido')
                                                                           AS dias_com_venda
                FROM orders WHERE {_ORDER_FILTERS}""",
            self._params(start, end, channel=channel, weekday=weekday),
        )[0]

    def daily(self, start: date, end: date, *, channel: str | None = None) -> list[Row]:
        return self.fetch(
            f"""SELECT order_date, min(weekday_name) AS dia_semana,
                       count(net_revenue) AS pedidos, coalesce(sum(net_revenue), 0) AS faturamento
                FROM orders WHERE {_ORDER_FILTERS}
                GROUP BY order_date ORDER BY order_date""",
            self._params(start, end, channel=channel),
        )

    # --- itens ------------------------------------------------------------------------------

    def top_items(
        self,
        start: date,
        end: date,
        *,
        channel: str | None = None,
        category: str | None = None,
        weekday: int | None = None,
        limit: int = 5,
    ) -> list[Row]:
        return self.fetch(
            f"""SELECT item_name AS item, category AS categoria, sum(quantity) AS quantidade,
                       sum(line_revenue) AS receita, sum(line_margin) AS margem
                FROM sold_items WHERE {_ITEM_FILTERS}
                GROUP BY item_name, category
                ORDER BY quantidade DESC, receita DESC, item_name
                LIMIT %(limit)s""",
            self._params(
                start, end, channel=channel, category=category, weekday=weekday, limit=limit
            ),
        )

    def item_quantities(
        self, start: date, end: date, *, channel: str | None = None, category: str | None = None
    ) -> dict[str, Row]:
        rows = self.fetch(
            f"""SELECT item_name AS item, category AS categoria, sum(quantity) AS quantidade
                FROM sold_items WHERE {_ITEM_FILTERS}
                GROUP BY item_name, category""",
            self._params(start, end, channel=channel, category=category),
        )
        return {row["item"]: row for row in rows}

    # --- canais, horários, operação -----------------------------------------------------------

    def channel_margins(self, start: date, end: date, *, weekday: int | None = None) -> list[Row]:
        return self.fetch(
            f"""SELECT channel AS canal,
                       count(net_revenue)                       AS pedidos,
                       coalesce(sum(net_revenue), 0)            AS faturamento,
                       coalesce(sum(items_cost), 0)             AS custo_itens,
                       coalesce(sum(channel_fee) FILTER (WHERE status = 'concluido'), 0)
                                                                AS comissao,
                       coalesce(sum(contribution_margin), 0)    AS margem_contribuicao,
                       100 * sum(contribution_margin) / nullif(sum(net_revenue), 0)
                                                                AS margem_pct,
                       avg(contribution_margin)                 AS margem_por_pedido
                FROM orders WHERE {_ORDER_FILTERS}
                GROUP BY channel
                HAVING count(net_revenue) > 0
                ORDER BY margem_contribuicao DESC""",
            self._params(start, end, weekday=weekday),
        )

    def hourly(
        self, start: date, end: date, *, channel: str | None = None, weekday: int | None = None
    ) -> tuple[int, list[Row]]:
        params = self._params(start, end, channel=channel, weekday=weekday)
        days = self.fetch(
            f"SELECT count(DISTINCT order_date) AS n FROM orders WHERE {_ORDER_FILTERS}", params
        )[0]["n"]
        rows = self.fetch(
            f"""SELECT order_hour AS hora, count(net_revenue) AS pedidos,
                       coalesce(sum(net_revenue), 0) AS faturamento
                FROM orders WHERE {_ORDER_FILTERS}
                GROUP BY order_hour HAVING count(net_revenue) > 0 ORDER BY order_hour""",
            params,
        )
        return days, rows

    def cancellations(self, start: date, end: date, *, channel: str | None = None) -> Row:
        params = self._params(start, end, channel=channel)
        totals = self.fetch(
            f"""SELECT count(*) AS pedidos_total,
                       count(*) FILTER (WHERE status = 'cancelado') AS cancelados
                FROM orders WHERE {_ORDER_FILTERS}""",
            params,
        )[0]
        reasons = self.fetch(
            f"""SELECT cancel_reason AS motivo, count(*) AS pedidos
                FROM orders WHERE status = 'cancelado' AND {_ORDER_FILTERS}
                GROUP BY cancel_reason ORDER BY pedidos DESC, motivo""",
            params,
        )
        channels = self.fetch(
            f"""SELECT channel AS canal, count(*) AS pedidos_total,
                       count(*) FILTER (WHERE status = 'cancelado') AS cancelados
                FROM orders WHERE {_ORDER_FILTERS}
                GROUP BY channel""",
            params,
        )
        return {**totals, "por_motivo": reasons, "por_canal": channels}

    def prep_times(self, start: date, end: date, *, channel: str | None = None) -> list[Row]:
        return self.fetch(
            f"""SELECT time_slot AS faixa, count(prep_time_min) AS pedidos,
                       avg(prep_time_min) AS media_min,
                       percentile_cont(0.5) WITHIN GROUP (ORDER BY prep_time_min) AS mediana_min,
                       percentile_cont(0.9) WITHIN GROUP (ORDER BY prep_time_min) AS p90_min
                FROM orders WHERE status = 'concluido' AND {_ORDER_FILTERS}
                GROUP BY time_slot
                ORDER BY array_position(ARRAY['almoco', 'tarde', 'jantar'], time_slot)""",
            self._params(start, end, channel=channel),
        )

    # --- previsão de demanda ------------------------------------------------------------------

    def orders_by_day_slot(self, start: date, end: date) -> list[Row]:
        """Pedidos concluídos por dia e faixa (dias sem pedido simplesmente não aparecem)."""
        return self.fetch(
            """SELECT order_date, time_slot, count(net_revenue) AS pedidos
               FROM orders WHERE order_date BETWEEN %(start)s AND %(end)s
               GROUP BY order_date, time_slot""",
            {"start": start, "end": end},
        )

    def item_mix(self, days: Sequence[date]) -> tuple[list[Row], dict[str, int]]:
        """Quantidade de cada item por faixa nos dias dados + pedidos concluídos por faixa."""
        params = {"days": list(days)}
        items = self.fetch(
            """SELECT time_slot AS faixa, item_name AS item, category AS categoria,
                      sum(quantity) AS quantidade
               FROM sold_items WHERE order_date = ANY(%(days)s)
               GROUP BY time_slot, item_name, category""",
            params,
        )
        orders = self.fetch(
            """SELECT time_slot AS faixa, count(net_revenue) AS pedidos
               FROM orders WHERE order_date = ANY(%(days)s) GROUP BY time_slot""",
            params,
        )
        return items, {row["faixa"]: row["pedidos"] for row in orders}
