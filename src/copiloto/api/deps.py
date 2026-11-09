"""Dependências da API: uma instância de cada peça por processo (e substituíveis nos testes).

O mesmo LLMClient atende /chat, /daily-summary e /health, então o estado do circuit breaker
mostrado no /health é o mesmo que o agente está usando.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from functools import lru_cache

from copiloto.agent.factory import build_agent
from copiloto.agent.loop import Agent
from copiloto.agent.tools.forecast import Forecaster
from copiloto.api.conversations import ConversationStore
from copiloto.config import get_settings
from copiloto.data.repository import AnalyticsRepository
from copiloto.llm import LLMClient
from copiloto.ml.model import load_forecaster
from copiloto.summary import DailySummary, Mode, build_daily_summary

SummaryBuilder = Callable[[date, Mode], DailySummary]


@lru_cache
def get_repo() -> AnalyticsRepository:
    return AnalyticsRepository(get_settings())


@lru_cache
def get_llm() -> LLMClient:
    return LLMClient.from_settings(get_settings())


@lru_cache
def get_forecaster() -> Forecaster:
    settings = get_settings()
    return load_forecaster(get_repo(), settings.model_dir, settings.copiloto_today)


@lru_cache
def get_agent() -> Agent:
    return build_agent(get_settings(), llm=get_llm(), repo=get_repo(), forecaster=get_forecaster())


def get_summary_builder() -> SummaryBuilder:
    settings = get_settings()

    def build(day: date, mode: Mode) -> DailySummary:
        return build_daily_summary(
            day,
            repo=get_repo(),
            llm=get_llm(),
            forecaster=get_forecaster(),
            today=settings.copiloto_today,
            mode=mode,
        )

    return build


_conversations = ConversationStore()


def get_conversations() -> ConversationStore:
    return _conversations
