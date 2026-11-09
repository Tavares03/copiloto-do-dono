"""Resumo diário contra tiny_data.py: sábado 31/10 vs. sábado 24/10."""

from datetime import date

import pytest

from copiloto.config import Settings
from copiloto.data.repository import AnalyticsRepository
from copiloto.formatting import brl, pct
from copiloto.llm import AllProvidersFailed, Attempt
from copiloto.summary import build_daily_summary, collect_facts, template_text
from fakes import ScriptedLLM, reply

pytestmark = pytest.mark.db

TODAY = date(2026, 11, 1)
SATURDAY = date(2026, 10, 31)


def test_brazilian_formatting():
    assert brl(10689.14) == "R$ 10.689,14"
    assert brl(30) == "R$ 30,00"
    assert pct(-57.14) == "-57,1%"
    assert pct(12.0) == "+12,0%"


@pytest.fixture(scope="module")
def repo(tiny_db: Settings) -> AnalyticsRepository:
    return AnalyticsRepository(tiny_db)


def test_collect_facts(repo: AnalyticsRepository):
    facts = collect_facts(repo, SATURDAY, today=TODAY, forecaster=None)

    assert facts["faturamento"] == 30.0
    assert facts["pedidos"] == 1
    assert facts["pedidos_cancelados"] == 1
    assert facts["taxa_cancelamento_pct"] == 50.0
    assert facts["mesmo_dia_semana_anterior"] == {
        "dia": "2026-10-24", "faturamento": 70.0, "pedidos": 1,
    }  # fmt: skip
    assert facts["variacao_faturamento_pct"] == -57.1
    assert facts["top_lanches"] == [{"item": "Clássico", "quantidade": 1}]
    assert facts["canal_mais_lucrativo"] == {
        "canal": "Balcão", "margem_contribuicao": 20.0, "margem_pct": 66.7,
    }  # fmt: skip
    assert facts["principal_motivo_cancelamento"] == "demora na entrega"


def test_template_text_uses_only_the_facts(repo: AnalyticsRepository):
    text = template_text(collect_facts(repo, SATURDAY, today=TODAY, forecaster=None))

    assert text.startswith("Resumo de sábado, 31/10/2026")
    for expected in ("R$ 30,00", "-57,1%", "Clássico", "Balcão", "demora na entrega"):
        assert expected in text


def test_llm_text_is_used_when_every_number_checks_out(repo: AnalyticsRepository):
    llm = ScriptedLLM(reply("Sábado você faturou R$ 30,00, 57,1% abaixo do sábado anterior."))

    summary = build_daily_summary(SATURDAY, repo=repo, llm=llm, forecaster=None, today=TODAY)

    assert summary.source == "llm"
    assert summary.provider == "groq"
    assert summary.text.startswith("Sábado você faturou")
    assert "R$ 30,00" in llm.calls[0][-1]["content"] or "30.0" in llm.calls[0][-1]["content"]


def test_invented_number_falls_back_to_template(repo: AnalyticsRepository):
    llm = ScriptedLLM(reply("Sábado você faturou R$ 99,00."))

    summary = build_daily_summary(SATURDAY, repo=repo, llm=llm, forecaster=None, today=TODAY)

    assert summary.source == "template"
    assert summary.fallback_reason == "numeros_sem_fonte"
    assert summary.ungrounded == ["99,00"]
    assert "R$ 30,00" in summary.text


def test_llm_unavailable_falls_back_to_template(repo: AnalyticsRepository):
    llm = ScriptedLLM(AllProvidersFailed([Attempt("groq", "m", "rate_limited")]))

    summary = build_daily_summary(SATURDAY, repo=repo, llm=llm, forecaster=None, today=TODAY)

    assert (summary.source, summary.fallback_reason) == ("template", "llm_indisponivel")


def test_template_mode_never_calls_the_llm(repo: AnalyticsRepository):
    llm = ScriptedLLM(reply("não deveria ser chamado"))

    summary = build_daily_summary(
        SATURDAY, repo=repo, llm=llm, forecaster=None, today=TODAY, mode="template"
    )

    assert summary.source == "template"
    assert llm.calls == []


def test_day_without_sales(repo: AnalyticsRepository):
    llm = ScriptedLLM(reply("não deveria ser chamado"))

    summary = build_daily_summary(
        date(2026, 10, 26), repo=repo, llm=llm, forecaster=None, today=TODAY
    )

    assert "Sem vendas registradas" in summary.text
    assert llm.calls == []
