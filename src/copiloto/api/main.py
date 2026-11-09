"""API HTTP do Copiloto do Dono.

    uvicorn copiloto.api.main:app --reload

Rotas: POST /chat, GET /daily-summary, GET /health, GET /stats, POST /webhook/whatsapp
(documentação interativa em /docs).
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date, timedelta
from typing import Annotated, Any

import openai
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response

from copiloto import __version__
from copiloto.agent.loop import Agent
from copiloto.api import deps
from copiloto.api.conversations import ConversationStore
from copiloto.api.schemas import ChatRequest, WebhookMessage, WebhookReply
from copiloto.config import Settings, get_settings
from copiloto.observability import configure_logging, log_event, request_id_var
from copiloto.stats import INTERACTIONS_FILE, compute_stats, load_interactions
from copiloto.summary import Mode

logger = logging.getLogger("copiloto.api")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, interactions_path=settings.log_dir / INTERACTIONS_FILE)
    yield


app = FastAPI(
    title="Copiloto do Dono",
    version=__version__,
    description="Agente de IA que responde perguntas do dono de restaurante sobre o negócio.",
    lifespan=lifespan,
)

SettingsDep = Annotated[Settings, Depends(get_settings)]


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Any) -> Response:
    """Cada requisição ganha um id (ou usa o do chamador) que aparece em todos os logs."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    token = request_id_var.set(request_id)
    try:
        response: Response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"nome": "Copiloto do Dono", "versao": __version__, "docs": "/docs"}


@app.post("/chat")
def chat(body: ChatRequest, agent: Annotated[Agent, Depends(deps.get_agent)]) -> dict[str, Any]:
    """Pergunta em linguagem natural → resposta + o "por dentro" (passos, tools, provedor)."""
    result = agent.run(body.question, history=[turn.model_dump() for turn in body.history])
    return result.to_dict()


@app.get("/daily-summary")
def daily_summary(
    settings: SettingsDep,
    builder: Annotated[deps.SummaryBuilder, Depends(deps.get_summary_builder)],
    day: Annotated[date | None, Query(alias="date", description="padrão: ontem")] = None,
    mode: Mode = "llm",
) -> dict[str, Any]:
    """Resumo do dia. Números calculados por código; o LLM só redige (com verificação)."""
    target = day or settings.copiloto_today - timedelta(days=1)
    return builder(target, mode).to_dict()


@app.get("/health")
def health(
    settings: SettingsDep,
    repo: Annotated[Any, Depends(deps.get_repo)],
    llm: Annotated[Any, Depends(deps.get_llm)],
    forecaster: Annotated[Any, Depends(deps.get_forecaster)],
    deep: bool = False,
) -> dict[str, Any]:
    """Estado do banco e de cada provedor de LLM. Não gasta tokens; deep=true valida as chaves."""
    try:
        first, last = repo.data_range()
        database: dict[str, Any] = {
            "ok": True,
            "dados_de": first.isoformat() if first else None,
            "dados_ate": last.isoformat() if last else None,
        }
    except Exception as exc:
        database = {"ok": False, "detalhe": f"{type(exc).__name__}: {str(exc)[:120]}"}

    providers = [asdict(status) for status in llm.status()]
    if deep:
        valid = {}
        for provider in llm.providers:
            try:
                provider.client.with_options(timeout=5).models.list()
                valid[provider.name] = True
            except openai.OpenAIError:
                valid[provider.name] = False
        for status in providers:
            if status["name"] in valid:
                status["chave_valida"] = valid[status["name"]]

    usable = any(p["active"] and p["circuit"] != "open" for p in providers)
    return {
        "status": "ok" if database["ok"] and usable else "degraded",
        "versao": __version__,
        "hoje": settings.copiloto_today.isoformat(),
        "banco": database,
        "llm": providers,
        "previsao": {"modelo": forecaster.name},
    }


@app.get("/stats")
def stats(settings: SettingsDep) -> dict[str, Any]:
    """Estatísticas das perguntas respondidas (a partir de logs/interactions.jsonl)."""
    return compute_stats(load_interactions(settings.log_dir / INTERACTIONS_FILE))


def _digits(phone: str) -> str:
    return "".join(ch for ch in phone if ch.isdigit())


@app.post("/webhook/whatsapp")
def whatsapp_webhook(
    body: WebhookMessage,
    settings: SettingsDep,
    agent: Annotated[Agent, Depends(deps.get_agent)],
    conversations: Annotated[ConversationStore, Depends(deps.get_conversations)],
    x_webhook_secret: Annotated[str | None, Header()] = None,
) -> WebhookReply:
    """Ponto de entrada para o N8N (ou outro orquestrador) ligado a uma API de WhatsApp."""
    if not settings.webhook_secret:
        raise HTTPException(503, "Webhook desativado: defina WEBHOOK_SECRET no .env.")
    if not secrets.compare_digest(x_webhook_secret or "", settings.webhook_secret):
        raise HTTPException(401, "Segredo do webhook inválido.")
    sender = _digits(body.sender)
    if settings.allowed_senders and sender not in settings.allowed_senders:
        raise HTTPException(403, "Número não autorizado.")

    key = hashlib.sha256(sender.encode()).hexdigest()[:12]  # telefone nunca vai para o log
    result = agent.run(body.message, history=conversations.history(key))
    conversations.append(key, body.message, result.answer)
    log_event(
        logger, "whatsapp_message", sender_hash=key, status=result.status,
        request_id=result.request_id,
    )  # fmt: skip
    return WebhookReply(reply=result.answer, request_id=result.request_id, status=result.status)
