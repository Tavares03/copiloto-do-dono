"""Roda as evals com UM provedor de LLM e salva detalhes e resumo.

    python -m copiloto.evals.run --provider groq               # todas as perguntas
    python -m copiloto.evals.run --provider gemini --limit 5   # só as 5 primeiras
    python -m copiloto.evals.run --provider groq --ids F01,M11 # perguntas escolhidas
    python -m copiloto.evals.report                            # tabela comparativa

Para caber nos planos gratuitos: cache em disco das respostas do LLM (uma rodada interrompida
continua de onde parou sem gastar cota), pausa entre perguntas e espera longa por Retry-After.
O gabarito é calculado na hora: SQL de referência (usuário read-only) ou a tool de previsão.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from copiloto.agent.factory import build_agent
from copiloto.agent.loop import Agent, AgentResult
from copiloto.agent.tools.forecast import Forecaster, forecast_demand
from copiloto.config import Settings, get_settings
from copiloto.data.repository import AnalyticsRepository
from copiloto.evals.dataset import EvalItem, load_dataset
from copiloto.evals.scoring import is_refusal, score_answer, summarize
from copiloto.llm import LLMClient
from copiloto.ml.model import load_forecaster
from copiloto.observability import configure_logging

DATASET = Path("evals/dataset.yaml")
RESULTS_DIR = Path("evals/results")
DEFAULT_PAUSE_S = {"groq": 30.0, "gemini": 15.0, "ollama": 0.0}


def eval_settings(base: Settings, provider: str, *, use_cache: bool) -> Settings:
    """Um provedor só, com cache e paciência: nas evals é melhor esperar do que pular."""
    return base.model_copy(
        update={
            "llm_providers": provider,
            "llm_cache_enabled": use_cache,
            "llm_retry_after_max_s": 65.0,
            "llm_max_retries": 5,
            "circuit_failure_threshold": 1000,
        }
    )


def expected_values(
    item: EvalItem, *, repo: AnalyticsRepository, forecaster: Forecaster, today: Any
) -> list[Any]:
    spec = item.esperado
    if spec.sql:
        row = repo.fetch(spec.sql)[0]
        values = list(row.values())
        return values if spec.qualquer_coluna else values[:1]
    if spec.tipo == "previsao":
        args: dict[str, Any] = {"target_date": spec.data.isoformat()}  # type: ignore[union-attr]
        if spec.faixa:
            args["time_slot"] = spec.faixa
        output = forecast_demand(args, forecaster=forecaster, repo=repo, today=today)
        if spec.campo == "total":
            return [output["total_pedidos_previstos"]]
        return [sum(slot["por_categoria"].get(spec.campo, 0.0) for slot in output["previsao"])]
    return []


def to_record(item: EvalItem, result: AgentResult, expected: list[Any]) -> dict[str, Any]:
    score = score_answer(item, result.answer, expected)
    steps = result.steps
    model_latency_ms = sum(s.latency_ms for s in steps) + sum(
        t.latency_ms for s in steps for t in s.tool_results
    )
    return {
        "id": item.id,
        "categoria": item.categoria,
        "tipo": item.esperado.tipo,
        "pergunta": item.pergunta,
        "esperado": expected,
        "resposta": result.answer,
        "correto": score.correct,
        "motivo": score.reason,
        "recusou": is_refusal(result.answer) and not score.correct,
        "grounded": result.grounded,
        "grounding": result.grounding,
        "status": result.status,
        "falha_llm": result.llm_failure,
        "provedores": sorted({s.provider for s in steps}),
        "trocou_provedor": any(s.switched_from for s in steps),
        "tools": [
            {"name": t.name, "arguments": t.arguments, "error": t.error}
            for s in steps
            for t in s.tool_results
        ],
        "tokens": result.total_tokens,
        "chamadas_llm": result.llm_calls,
        # latência das chamadas (LLM + tools) sem as esperas de rate limit;
        # quando vem do cache, vale a latência original
        "latencia_s": round(model_latency_ms / 1000, 2),
        "tempo_total_s": round(result.latency_ms / 1000, 2),
        "cache": bool(steps) and all(s.cached for s in steps),
        "request_id": result.request_id,
    }


_TRANSIENT = (":rate_limited", ":circuit_open", ":timeout", ":connection_error", ":server_error")


def should_pause_run(record: dict[str, Any]) -> bool:
    """Cota esgotada ou rede fora: pausar a rodada. Falha do modelo (ex.: resposta vazia): erro."""
    return record["status"] == "llm_unavailable" and any(
        failure.endswith(_TRANSIENT) for failure in record["falha_llm"]
    )


def run(
    provider: str,
    *,
    limit: int | None = None,
    ids: list[str] | None = None,
    pause_s: float | None = None,
    use_cache: bool = True,
) -> dict[str, Any] | None:
    settings = eval_settings(get_settings(), provider, use_cache=use_cache)
    llm = LLMClient.from_settings(settings)
    if not llm.providers:
        reason = llm.skipped[0].reason if llm.skipped else "não configurado"
        print(f"Provedor {provider} indisponível: {reason}")
        return None

    repo = AnalyticsRepository(settings)
    forecaster = load_forecaster(repo, settings.model_dir, settings.copiloto_today)
    agent: Agent = build_agent(settings, llm=llm, repo=repo, forecaster=forecaster)
    items = load_dataset(DATASET)
    if ids:
        items = [i for i in items if i.id in ids]
    if limit:
        items = items[:limit]
    partial = bool(ids or limit)
    pause = DEFAULT_PAUSE_S.get(provider, 10.0) if pause_s is None else pause_s

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_parcial" if partial else ""
    details_path = RESULTS_DIR / f"detalhes_{provider}{suffix}.jsonl"
    model = llm.providers[0].model
    print(f"Avaliando {provider} ({model}) em {len(items)} perguntas; pausa de {pause:.0f}s.")

    records = []
    with details_path.open("w", encoding="utf-8", newline="\n") as details:
        for number, item in enumerate(items, 1):
            expected = expected_values(
                item, repo=repo, forecaster=forecaster, today=settings.copiloto_today
            )
            record = to_record(item, agent.run(item.pergunta), expected)
            if should_pause_run(record):
                # Cota diária esgotada (ou provedor fora do ar): parar em vez de registrar
                # erros falsos. Rodar de novo depois continua daqui; o que já foi respondido
                # vem do cache sem gastar cota.
                print(
                    f"\nInterrompido em {item.id}: {provider} indisponível (provável cota "
                    f"esgotada). {len(records)} perguntas concluídas e guardadas no cache; "
                    "rode o mesmo comando mais tarde para continuar.",
                    flush=True,
                )
                return None
            records.append(record)
            details.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            details.flush()
            mark = "ok " if record["correto"] else "ERRO"
            print(
                f"[{number:>2}/{len(items)}] {item.id} {mark} {record['latencia_s']:>5.1f}s "
                f"{record['tokens']:>6} tok {'(cache) ' if record['cache'] else ''}"
                f"{record['motivo']}",
                flush=True,
            )
            if pause and not record["cache"] and number < len(items):
                time.sleep(pause)

    summary = {
        "provider": provider,
        "model": model,
        "executado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "parcial": partial,
        "respostas_do_cache": sum(r["cache"] for r in records),
        **summarize(records),
    }
    summary_path = RESULTS_DIR / f"resumo_{provider}{suffix}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    print(
        f"\n{provider}: {summary['acertos']}/{summary['perguntas']} corretas · aterramento "
        f"{summary['aterramento_pct']}% · latência média {summary['latencia_media_s']}s "
        f"(p95 {summary['latencia_p95_s']}s) · {summary['tokens_medios']} tokens/pergunta"
    )
    print(f"Detalhes em {details_path}; resumo em {summary_path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evals do Copiloto do Dono, por provedor.")
    parser.add_argument("--provider", required=True, choices=["groq", "gemini", "ollama"])
    parser.add_argument("--limit", type=int, help="só as N primeiras perguntas")
    parser.add_argument("--ids", help="ids separados por vírgula (ex.: F01,M11)")
    parser.add_argument("--pause", type=float, help="segundos entre perguntas")
    parser.add_argument("--no-cache", action="store_true", help="ignora o cache do LLM")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")
    configure_logging("ERROR")
    summary = run(
        args.provider,
        limit=args.limit,
        ids=args.ids.split(",") if args.ids else None,
        pause_s=args.pause,
        use_cache=not args.no_cache,
    )
    return 0 if summary else 2


if __name__ == "__main__":
    raise SystemExit(main())
