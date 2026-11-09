"""Tool get_metrics: métricas prontas e testadas, preferidas ao SQL livre."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from copiloto.agent.tools.common import (
    Category,
    Channel,
    FlexibleDate,
    ToolInputError,
    br_date,
    money,
    one_decimal,
)
from copiloto.data.repository import AnalyticsRepository

MetricName = Literal[
    "resumo",
    "top_itens",
    "variacao_itens",
    "margem_por_canal",
    "horarios_pico",
    "cancelamentos",
    "tempo_preparo",
    "vendas_por_dia",
]

DEFINITIONS: dict[str, str] = {
    "resumo": "faturamento = soma de (subtotal - desconto) dos pedidos concluídos, sem taxa de "
    "entrega; ticket médio = faturamento / pedidos concluídos; margem de contribuição = "
    "faturamento - custo dos itens - comissão do canal",
    "top_itens": "itens de pedidos concluídos, por quantidade; receita bruta antes de descontos",
    "variacao_itens": "quantidade média por dia de cada item no período vs. no período de "
    "comparação (variação em %; None = item não vendia antes)",
    "margem_por_canal": "margem de contribuição = faturamento - custo dos itens - comissão",
    "horarios_pico": "pedidos concluídos por hora; média por dia em que o restaurante vendeu",
    "cancelamentos": "taxa = pedidos cancelados / todos os pedidos do período",
    "tempo_preparo": "minutos de preparo dos pedidos concluídos, por faixa de horário",
    "vendas_por_dia": "faturamento e pedidos concluídos de cada dia",
}
VIEWS = {"top_itens": "sold_items", "variacao_itens": "sold_items"}
MAX_DAILY_DAYS = 92


class MetricArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: MetricName
    start_date: FlexibleDate
    end_date: FlexibleDate
    compare_start_date: FlexibleDate | None = None
    compare_end_date: FlexibleDate | None = None
    channel: Channel | None = None
    category: Category | None = None
    weekday: int | None = Field(default=None, ge=1, le=7)
    limit: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def _check_periods(self) -> MetricArgs:
        if self.end_date < self.start_date:
            raise ValueError("end_date não pode ser antes de start_date")
        if (self.compare_start_date is None) != (self.compare_end_date is None):
            raise ValueError("informe compare_start_date e compare_end_date juntos")
        if (
            self.compare_start_date
            and self.compare_end_date
            and self.compare_end_date < self.compare_start_date
        ):
            raise ValueError("compare_end_date não pode ser antes de compare_start_date")
        if (
            self.metric == "vendas_por_dia"
            and (self.end_date - self.start_date).days >= MAX_DAILY_DAYS
        ):
            raise ValueError(f"vendas_por_dia aceita no máximo {MAX_DAILY_DAYS} dias")
        return self


def _period(start: date, end: date) -> dict[str, str]:
    return {"inicio": start.isoformat(), "fim": end.isoformat()}


def _pct_change(now: float | None, before: float | None) -> float | None:
    if now is None or not before:
        return None
    return round((now - before) / before * 100, 1)


def _summary(repo: AnalyticsRepository, start: date, end: date, args: MetricArgs) -> dict:
    row = repo.summary(start, end, channel=args.channel, weekday=args.weekday)
    return {
        "faturamento": money(row["faturamento"]),
        "pedidos": row["pedidos"],
        "ticket_medio": money(row["ticket_medio"]) if row["ticket_medio"] is not None else None,
        "pedidos_cancelados": row["pedidos_cancelados"],
        "margem_contribuicao": money(row["margem_contribuicao"]),
        "dias_com_venda": row["dias_com_venda"],
    }


def _item_variation(
    repo: AnalyticsRepository, args: MetricArgs, compare: tuple[date, date]
) -> list[dict]:
    days_now = (args.end_date - args.start_date).days + 1
    days_before = (compare[1] - compare[0]).days + 1
    now = repo.item_quantities(
        args.start_date, args.end_date, channel=args.channel, category=args.category
    )
    before = repo.item_quantities(*compare, channel=args.channel, category=args.category)

    rows = []
    for item in sorted(set(now) | set(before)):
        qty_now = (now.get(item) or {}).get("quantidade", 0)
        qty_before = (before.get(item) or {}).get("quantidade", 0)
        avg_now, avg_before = qty_now / days_now, qty_before / days_before
        rows.append(
            {
                "item": item,
                "categoria": (now.get(item) or before.get(item) or {}).get("categoria"),
                "quantidade_periodo": qty_now,
                "quantidade_comparacao": qty_before,
                "diferenca_quantidade": qty_now - qty_before,  # contas prontas: o LLM não calcula
                "media_dia_periodo": round(avg_now, 2),
                "media_dia_comparacao": round(avg_before, 2),
                "variacao_pct": _pct_change(avg_now, avg_before),
            }
        )
    # maiores quedas primeiro; itens novos (sem base de comparação) por último
    rows.sort(key=lambda r: (r["variacao_pct"] is None, r["variacao_pct"] or 0, r["item"]))
    return rows[: args.limit]


def get_metrics(
    raw_args: Mapping[str, Any], *, repo: AnalyticsRepository, today: date
) -> dict[str, Any]:
    args = MetricArgs.model_validate(dict(raw_args))
    start, end = args.start_date, args.end_date
    result: dict[str, Any] = {"metrica": args.metric, "periodo": _period(start, end)}
    filters = {
        key: value
        for key, value in {
            "canal": args.channel,
            "categoria": args.category,
            "dia_semana_iso": args.weekday,
        }.items()
        if value is not None
    }
    if filters:
        result["filtros"] = filters

    match args.metric:
        case "resumo":
            result["dados"] = _summary(repo, start, end, args)
            if args.compare_start_date and args.compare_end_date:
                before = _summary(repo, args.compare_start_date, args.compare_end_date, args)
                result["comparacao"] = {
                    "periodo": _period(args.compare_start_date, args.compare_end_date),
                    "dados": before,
                    "variacao_pct": {
                        key: _pct_change(result["dados"][key], before[key])
                        for key in ("faturamento", "pedidos", "ticket_medio")
                    },
                    "diferenca": {
                        "faturamento": money(
                            result["dados"]["faturamento"] - before["faturamento"]
                        ),
                        "pedidos": result["dados"]["pedidos"] - before["pedidos"],
                    },
                }
        case "top_itens":
            result["dados"] = [
                {**row, "receita": money(row["receita"]), "margem": money(row["margem"])}
                for row in repo.top_items(
                    start, end, channel=args.channel, category=args.category,
                    weekday=args.weekday, limit=args.limit,
                )
            ]  # fmt: skip
        case "variacao_itens":
            if args.compare_start_date and args.compare_end_date:
                compare = (args.compare_start_date, args.compare_end_date)
            else:
                length = end - start
                compare_end = start - timedelta(days=1)
                compare = (compare_end - length, compare_end)
            result["comparacao"] = {"periodo": _period(*compare)}
            result["dados"] = _item_variation(repo, args, compare)
        case "margem_por_canal":
            result["dados"] = [
                {
                    "canal": row["canal"],
                    "pedidos": row["pedidos"],
                    "faturamento": money(row["faturamento"]),
                    "custo_itens": money(row["custo_itens"]),
                    "comissao": money(row["comissao"]),
                    "margem_contribuicao": money(row["margem_contribuicao"]),
                    "margem_pct": one_decimal(row["margem_pct"]),
                    "margem_por_pedido": money(row["margem_por_pedido"]),
                }
                for row in repo.channel_margins(start, end, weekday=args.weekday)
            ]
        case "horarios_pico":
            days, rows = repo.hourly(start, end, channel=args.channel, weekday=args.weekday)
            result["dados"] = {
                "dias_considerados": days,
                "por_hora": [
                    {
                        "hora": row["hora"],
                        "pedidos": row["pedidos"],
                        "media_por_dia": round(row["pedidos"] / days, 1) if days else 0.0,
                        "faturamento": money(row["faturamento"]),
                    }
                    for row in rows
                ],
            }
        case "cancelamentos":
            raw = repo.cancellations(start, end, channel=args.channel)
            by_channel = [
                {
                    "canal": row["canal"],
                    "pedidos_total": row["pedidos_total"],
                    "cancelados": row["cancelados"],
                    "taxa_pct": one_decimal(100 * row["cancelados"] / row["pedidos_total"]),
                }
                for row in raw["por_canal"]
            ]
            by_channel.sort(key=lambda r: (-(r["taxa_pct"] or 0), r["canal"]))
            total = raw["pedidos_total"]
            result["dados"] = {
                "pedidos_total": total,
                "cancelados": raw["cancelados"],
                "taxa_pct": one_decimal(100 * raw["cancelados"] / total) if total else 0.0,
                "por_motivo": raw["por_motivo"],
                "por_canal": by_channel,
            }
        case "tempo_preparo":
            result["dados"] = [
                {
                    "faixa": row["faixa"],
                    "pedidos": row["pedidos"],
                    "media_min": one_decimal(row["media_min"]),
                    "mediana_min": one_decimal(row["mediana_min"]),
                    "p90_min": one_decimal(row["p90_min"]),
                }
                for row in repo.prep_times(start, end, channel=args.channel)
            ]
        case "vendas_por_dia":
            result["dados"] = [
                {
                    "data": row["order_date"].isoformat(),
                    "dia_semana": row["dia_semana"],
                    "pedidos": row["pedidos"],
                    "faturamento": money(row["faturamento"]),
                }
                for row in repo.daily(start, end, channel=args.channel)
            ]
        case _:  # pragma: no cover - o Literal já barra
            raise ToolInputError(f"métrica desconhecida: {args.metric}")

    _, last_day = repo.data_range()
    if last_day is not None and end > last_day:
        result["aviso"] = (
            f"Os dados vão até {br_date(last_day)} (hoje é {br_date(today)}); "
            "dias depois disso ainda não têm vendas registradas."
        )
    result["definicao"] = DEFINITIONS[args.metric]
    view = VIEWS.get(args.metric, "orders")
    result["fonte"] = f"get_metrics/{args.metric} (analytics.{view})"
    return result
