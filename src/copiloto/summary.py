"""Resumo diário automático.

Os números são calculados por código; o LLM só redige. Depois o verificador de números confere
o texto: se aparecer qualquer número que não está nos fatos (ou se nenhum LLM responder), vale
o template determinístico. O dono nunca recebe um número inventado.

    python -m copiloto.summary                 # resumo de ontem
    python -m copiloto.summary --date 2026-10-31 --template
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

from copiloto.agent.grounding import ungrounded_numbers
from copiloto.agent.loop import ChatModel
from copiloto.agent.tools.common import WEEKDAYS_PT, money, one_decimal
from copiloto.agent.tools.forecast import Forecaster
from copiloto.data.repository import AnalyticsRepository
from copiloto.formatting import brl, channel_name, number, pct, short_date
from copiloto.llm import AllProvidersFailed
from copiloto.observability import log_event

logger = logging.getLogger("copiloto.summary")

Mode = Literal["llm", "template"]

SUMMARY_PROMPT = """\
Você escreve o resumo diário de uma hamburgueria para o dono, que vai ler no celular.
Use SOMENTE os números do JSON que o usuário enviar, no formato brasileiro (R$ 1.234,56; \
12,5%). Não calcule números novos e não invente causas.
No máximo 6 linhas curtas, sem títulos nem tabelas, tom direto e amigável. Comece pelo \
faturamento e pela comparação com o mesmo dia da semana anterior; depois destaque o lanche mais \
vendido, o canal mais lucrativo, os cancelamentos e, se houver, a previsão do próximo dia."""


@dataclass
class DailySummary:
    day: date
    text: str
    facts: dict[str, Any]
    source: Literal["llm", "template"]
    provider: str | None = None
    model: str | None = None
    fallback_reason: str | None = None  # por que o texto do LLM não foi usado
    ungrounded: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["day"] = self.day.isoformat()
        return data


def _pct_change(now: float, before: float) -> float | None:
    return round((now - before) / before * 100, 1) if before else None


def collect_facts(
    repo: AnalyticsRepository, day: date, *, today: date, forecaster: Forecaster | None
) -> dict[str, Any]:
    """Todos os números do resumo, já arredondados, com nomes em português."""
    current = repo.summary(day, day)
    previous_day = day - timedelta(days=7)
    previous = repo.summary(previous_day, previous_day)
    cancels = repo.cancellations(day, day)

    facts: dict[str, Any] = {
        "dia": day.isoformat(),
        "dia_semana": WEEKDAYS_PT[day.weekday()],
        "faturamento": money(current["faturamento"]),
        "pedidos": current["pedidos"],
        "ticket_medio": money(current["ticket_medio"]) if current["ticket_medio"] else None,
        "pedidos_cancelados": current["pedidos_cancelados"],
        "taxa_cancelamento_pct": (
            one_decimal(100 * cancels["cancelados"] / cancels["pedidos_total"])
            if cancels["pedidos_total"]
            else 0.0
        ),
        "margem_contribuicao": money(current["margem_contribuicao"]),
        "mesmo_dia_semana_anterior": {
            "dia": previous_day.isoformat(),
            "faturamento": money(previous["faturamento"]),
            "pedidos": previous["pedidos"],
        },
        "variacao_faturamento_pct": _pct_change(
            money(current["faturamento"]), money(previous["faturamento"])
        ),
        "top_lanches": [
            {"item": row["item"], "quantidade": row["quantidade"]}
            for row in repo.top_items(day, day, category="burger", limit=3)
        ],
        "canal_mais_lucrativo": None,
        "principal_motivo_cancelamento": (
            cancels["por_motivo"][0]["motivo"] if cancels["por_motivo"] else None
        ),
    }
    channels = repo.channel_margins(day, day)
    if channels:
        best = channels[0]
        facts["canal_mais_lucrativo"] = {
            "canal": channel_name(best["canal"]),
            "margem_contribuicao": money(best["margem_contribuicao"]),
            "margem_pct": one_decimal(best["margem_pct"]),
        }

    next_day = day + timedelta(days=1)
    if forecaster is not None and next_day >= today:
        predicted = forecaster.predict(next_day)
        total = round(sum(predicted.values()))
        if total > 0:
            facts["previsao_proximo_dia"] = {
                "dia": next_day.isoformat(),
                "dia_semana": WEEKDAYS_PT[next_day.weekday()],
                "pedidos_previstos": total,
                "pedidos_jantar": round(predicted.get("jantar", 0.0)),
            }
    return facts


def template_text(facts: dict[str, Any]) -> str:
    """Texto determinístico: sempre disponível, sem LLM."""
    day = date.fromisoformat(facts["dia"])
    lines = [f"Resumo de {facts['dia_semana']}, {day.strftime('%d/%m/%Y')}"]
    if not facts["pedidos"]:
        lines.append("Sem vendas registradas nesse dia (a loja fecha às segundas).")
        return "\n".join(lines)

    line = (
        f"Faturamento: {brl(facts['faturamento'])} em {facts['pedidos']} pedidos "
        f"(ticket médio {brl(facts['ticket_medio'] or 0)})"
    )
    previous = facts["mesmo_dia_semana_anterior"]
    if facts["variacao_faturamento_pct"] is not None:
        line += (
            f", {pct(facts['variacao_faturamento_pct'])} vs. {facts['dia_semana']} anterior "
            f"({short_date(date.fromisoformat(previous['dia']))})"
        )
    lines.append(line + ".")
    if facts["top_lanches"]:
        tops = ", ".join(f"{i['item']} ({i['quantidade']})" for i in facts["top_lanches"])
        lines.append(f"Lanches mais vendidos: {tops}.")
    if best := facts["canal_mais_lucrativo"]:
        lines.append(
            f"Canal mais lucrativo: {best['canal']}, {brl(best['margem_contribuicao'])} de margem "
            f"({pct(best['margem_pct'], signed=False)})."
        )
    if facts["pedidos_cancelados"]:
        reason = facts["principal_motivo_cancelamento"]
        lines.append(
            f"Cancelamentos: {facts['pedidos_cancelados']} "
            f"({pct(facts['taxa_cancelamento_pct'], signed=False)})"
            + (f", principal motivo: {reason}." if reason else ".")
        )
    if forecast := facts.get("previsao_proximo_dia"):
        lines.append(
            f"Previsão para {forecast['dia_semana']} "
            f"({short_date(date.fromisoformat(forecast['dia']))}): cerca de "
            f"{number(forecast['pedidos_previstos'])} pedidos, "
            f"{number(forecast['pedidos_jantar'])} no jantar."
        )
    return "\n".join(lines)


def build_daily_summary(
    day: date,
    *,
    repo: AnalyticsRepository,
    llm: ChatModel,
    forecaster: Forecaster | None,
    today: date,
    mode: Mode = "llm",
) -> DailySummary:
    facts = collect_facts(repo, day, today=today, forecaster=forecaster)
    template = template_text(facts)
    summary = DailySummary(day=day, text=template, facts=facts, source="template")

    if mode == "llm" and facts["pedidos"]:
        facts_json = json.dumps(facts, ensure_ascii=False)
        try:
            response = llm.chat(
                [
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": facts_json},
                ]
            )
        except AllProvidersFailed:
            summary.fallback_reason = "llm_indisponivel"
        else:
            text = response.content.strip()
            missing = ungrounded_numbers(text, [facts_json, template])
            if text and not missing:
                summary.text, summary.source = text, "llm"
                summary.provider, summary.model = response.provider, response.model
            else:
                summary.fallback_reason = "numeros_sem_fonte" if text else "resposta_vazia"
                summary.ungrounded = missing

    log_event(
        logger,
        "daily_summary",
        day=day.isoformat(),
        source=summary.source,
        provider=summary.provider,
        fallback_reason=summary.fallback_reason,
        ungrounded=summary.ungrounded,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    from copiloto.config import get_settings
    from copiloto.llm import LLMClient
    from copiloto.ml.model import load_forecaster

    parser = argparse.ArgumentParser(description="Resumo diário do Copiloto do Dono.")
    parser.add_argument("--date", type=date.fromisoformat, help="dia (padrão: ontem)")
    parser.add_argument("--template", action="store_true", help="não usa LLM")
    args = parser.parse_args(argv)

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8", errors="replace")
    settings = get_settings()
    today = settings.copiloto_today
    repo = AnalyticsRepository(settings)
    summary = build_daily_summary(
        args.date or today - timedelta(days=1),
        repo=repo,
        llm=LLMClient.from_settings(settings),
        forecaster=load_forecaster(repo, settings.model_dir, today),
        today=today,
        mode="template" if args.template else "llm",
    )
    print(summary.text)
    origin = f"{summary.provider}/{summary.model}" if summary.source == "llm" else "template"
    print(
        f"\n[gerado por: {origin}"
        + (f"; motivo: {summary.fallback_reason}]" if summary.fallback_reason else "]")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
