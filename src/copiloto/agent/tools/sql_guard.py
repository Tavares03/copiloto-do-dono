"""Validação de SQL gerado pelo LLM antes de chegar ao banco.

Primeira linha de defesa (a segunda é o usuário read-only do Postgres, que só enxerga
as views de analytics). Regras:

1. exatamente uma consulta, e ela é SELECT / UNION / WITH ... SELECT;
2. nenhum nó de escrita, DDL, SELECT INTO ou FOR UPDATE em lugar nenhum da árvore
   (pega DELETE escondido dentro de CTE);
3. só tabelas da whitelist do schema analytics (CTEs declaradas na própria consulta valem);
4. nenhuma função com efeito colateral ou que leia o servidor (pg_*, dblink, set_config...);
5. LIMIT automático, nunca maior que max_rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

ALLOWED_SCHEMA = "analytics"
ALLOWED_TABLES = frozenset({"orders", "sold_items", "menu", "channels", "holidays"})
ALLOWED_TABLE_FUNCTIONS = frozenset({"generate_series", "exploding_generate_series", "unnest"})

BLOCKED_FUNCTIONS = frozenset(
    {"set_config", "current_setting", "dblink", "dblink_exec", "nextval", "setval", "currval",
     "query_to_xml", "query_to_xml_and_xmlschema", "cursor_to_xml", "table_to_xml",
     "schema_to_xml", "database_to_xml", "txid_current", "inet_server_addr"}
)  # fmt: skip
BLOCKED_PREFIXES = ("pg_", "lo_", "dblink")

_FORBIDDEN_NODE_NAMES = (
    "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "TruncateTable",
    "Command", "Copy", "Set", "Grant", "Revoke", "Into", "Lock", "Transaction", "Commit",
    "Rollback", "Use", "Pragma", "LoadData", "Cache", "Uncache", "Describe", "Kill",
)  # fmt: skip
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name) for name in _FORBIDDEN_NODE_NAMES if hasattr(exp, name)
)

ONLY_SELECT = "Só consultas de leitura (SELECT) são permitidas."


class UnsafeQueryError(ValueError):
    """A consulta foi recusada antes de chegar ao banco."""


@dataclass(frozen=True)
class ValidatedQuery:
    sql: str  # SQL normalizado, já com LIMIT
    tables: list[str]  # tabelas da whitelist usadas, em ordem alfabética
    limit: int


def _function_name(node: exp.Func) -> str:
    return (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()


def _is_blocked_function(name: str) -> bool:
    return name in BLOCKED_FUNCTIONS or name.startswith(BLOCKED_PREFIXES)


def _parse_single(sql: str) -> exp.Expression:
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except SqlglotError as exc:
        raise UnsafeQueryError(f"SQL inválido: {str(exc).splitlines()[0][:150]}") from exc
    if not statements:
        raise UnsafeQueryError("Consulta vazia.")
    if len(statements) > 1:
        raise UnsafeQueryError("Envie apenas uma consulta por vez (sem ';' separando comandos).")
    return statements[0]


def _check_tables(root: exp.Expression) -> list[str]:
    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    used: set[str] = set()
    for table in root.find_all(exp.Table):
        if isinstance(table.this, exp.Func):
            name = _function_name(table.this)
            if name not in ALLOWED_TABLE_FUNCTIONS:
                raise UnsafeQueryError(f"Função '{name}' não permitida no FROM.")
            continue
        name, schema = table.name.lower(), table.db.lower()
        if table.catalog:
            raise UnsafeQueryError(f"Tabela '{table.sql()}' não permitida.")
        if not schema and name in cte_names:
            continue
        if schema not in ("", ALLOWED_SCHEMA) or name not in ALLOWED_TABLES:
            allowed = ", ".join(sorted(ALLOWED_TABLES))
            raise UnsafeQueryError(f"Tabela '{table.sql()}' não permitida. Use apenas: {allowed}.")
        used.add(name)
    return sorted(used)


def _apply_limit(root: exp.Expression, max_rows: int) -> int:
    current: int | None = None
    node = root.args.get("limit")
    if isinstance(node, exp.Limit):
        value = node.args.get("expression")
        if isinstance(value, exp.Literal) and value.is_int:
            current = int(value.this)
    limit = max_rows if current is None else min(current, max_rows)
    root.set("limit", exp.Limit(expression=exp.Literal.number(limit)))
    return limit


def validate_sql(sql: str, max_rows: int) -> ValidatedQuery:
    """Valida e normaliza a consulta. Lança UnsafeQueryError com o motivo, em PT-BR."""
    root = _parse_single(sql or "")
    while isinstance(root, exp.Subquery):  # "(SELECT ...)" sozinho
        root = root.this

    if not isinstance(root, exp.Select | exp.SetOperation):
        raise UnsafeQueryError(ONLY_SELECT)
    if any(isinstance(node, FORBIDDEN_NODES) for node in root.walk()):
        raise UnsafeQueryError(ONLY_SELECT + " (escrita, SELECT INTO e FOR UPDATE são bloqueados)")

    for func in root.find_all(exp.Func):
        name = _function_name(func)
        if _is_blocked_function(name):
            raise UnsafeQueryError(f"Função '{name}' não permitida.")

    tables = _check_tables(root)
    limit = _apply_limit(root, max_rows)
    normalized = root.sql(dialect="postgres", comments=False)  # comentários não vão ao banco
    return ValidatedQuery(sql=normalized, tables=tables, limit=limit)
