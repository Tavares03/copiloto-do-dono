"""Monta o agente com as peças reais: LLM multi-provedor, banco read-only e previsão."""

from __future__ import annotations

from copiloto.agent.loop import Agent, ChatModel
from copiloto.agent.tools.forecast import Forecaster
from copiloto.agent.tools.registry import build_registry
from copiloto.config import Settings, get_settings
from copiloto.data.repository import AnalyticsRepository
from copiloto.llm import LLMClient
from copiloto.ml.model import load_forecaster


def build_agent(
    settings: Settings | None = None,
    *,
    llm: ChatModel | None = None,
    repo: AnalyticsRepository | None = None,
    forecaster: Forecaster | None = None,
) -> Agent:
    settings = settings or get_settings()
    today = settings.copiloto_today
    repo = repo or AnalyticsRepository(settings)
    forecaster = forecaster or load_forecaster(repo, settings.model_dir, today)
    first_day, _ = repo.data_range()
    tools = build_registry(settings=settings, repo=repo, forecaster=forecaster, today=today)
    return Agent(
        llm or LLMClient.from_settings(settings),
        tools,
        today=today,
        data_start=first_day or settings.data_start,
        max_steps=settings.agent_max_steps,
    )
