"""Tool run_sql: SQL somente leitura, validado com sqlglot e executado com o usuário read-only."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import psycopg
from psycopg import sql
from pydantic import BaseModel, ConfigDict, Field

from copiloto.agent.tools.common import jsonable
from copiloto.agent.tools.sql_guard import UnsafeQueryError, validate_sql
from copiloto.config import Settings
from copiloto.data.db import connect


class RunSqlArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)


def run_sql(raw_args: Mapping[str, Any], *, settings: Settings) -> dict[str, Any]:
    """Executa a consulta. Erros voltam como {"erro": ...} para o agente poder corrigir."""
    args = RunSqlArgs.model_validate(dict(raw_args))
    try:
        query = validate_sql(args.query, max_rows=settings.sql_row_limit)
    except UnsafeQueryError as exc:
        return {"erro": f"Consulta recusada: {exc}"}

    timeout = sql.SQL("SET statement_timeout = {}").format(
        sql.Literal(f"{settings.sql_timeout_ms}ms")
    )
    try:
        with connect(settings, readonly=True) as conn:
            conn.read_only = True  # terceira camada: transação READ ONLY
            conn.execute(timeout)
            cursor = conn.execute(query.sql)  # type: ignore[arg-type]  # SQL já validado
            columns = [column.name for column in cursor.description or []]
            rows = cursor.fetchall()
    except psycopg.errors.QueryCanceled:
        return {
            "erro": f"A consulta passou do tempo limite de {settings.sql_timeout_ms} ms. "
            "Filtre por período ou agregue mais."
        }
    except psycopg.Error as exc:
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return {"erro": f"Erro do banco: {first_line}"}

    return {
        "colunas": columns,
        "linhas": [[jsonable(value) for value in row] for row in rows],
        "total_linhas": len(rows),
        "atingiu_limite": len(rows) >= query.limit,
        "sql_executado": query.sql,
        "fonte": "run_sql sobre " + ", ".join(f"analytics.{t}" for t in query.tables),
    }
