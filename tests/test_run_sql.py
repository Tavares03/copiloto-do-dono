import json

import pytest

from copiloto.agent.tools.run_sql import run_sql
from copiloto.config import Settings

pytestmark = pytest.mark.db


def test_runs_a_select_with_the_readonly_user(tiny_db: Settings):
    result = run_sql(
        {"query": "SELECT sum(net_revenue) AS fat FROM orders "
                  "WHERE order_date BETWEEN '2026-10-30' AND '2026-10-31'"},
        settings=tiny_db,
    )  # fmt: skip

    assert result["colunas"] == ["fat"]
    assert result["linhas"] == [[130.0]]
    assert result["sql_executado"].endswith("LIMIT 200")
    assert result["fonte"] == "run_sql sobre analytics.orders"
    json.dumps(result)  # tudo serializável para voltar ao LLM


def test_dates_come_back_as_iso_strings(tiny_db: Settings):
    result = run_sql({"query": "SELECT min(order_date) FROM orders"}, settings=tiny_db)

    assert result["linhas"] == [["2026-10-23"]]


@pytest.mark.parametrize(
    "query",
    ["DELETE FROM orders", "SELECT * FROM core.orders", "SELECT 1; DROP TABLE orders"],
)
def test_unsafe_queries_return_an_error_for_the_agent(tiny_db: Settings, query: str):
    result = run_sql({"query": query}, settings=tiny_db)

    assert result["erro"].startswith("Consulta recusada")


def test_generate_series_runs_and_is_capped(tiny_db: Settings):
    result = run_sql({"query": "SELECT g FROM generate_series(1, 500) AS g"}, settings=tiny_db)

    assert result["total_linhas"] == 200
    assert result["atingiu_limite"] is True


def test_slow_query_hits_the_timeout(tiny_db: Settings):
    fast_timeout = tiny_db.model_copy(update={"sql_timeout_ms": 50})

    result = run_sql(
        {"query": "SELECT count(*) FROM generate_series(1, 300000000) AS g"},
        settings=fast_timeout,
    )

    assert "tempo limite" in result["erro"]


def test_database_errors_are_returned_not_raised(tiny_db: Settings):
    result = run_sql({"query": "SELECT coluna_que_nao_existe FROM orders"}, settings=tiny_db)

    assert result["erro"].startswith("Erro do banco")
