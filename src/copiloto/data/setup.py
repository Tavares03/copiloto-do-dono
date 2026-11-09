"""Prepara o banco: aplica o schema, cria o usuário somente leitura e carrega os dados.

Idempotente: rodar de novo não duplica nada. Uso:

    python -m copiloto.data.setup            # cria o que faltar e popula se estiver vazio
    python -m copiloto.data.setup --reset    # apaga e recria tudo
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql

from copiloto.config import Settings, get_settings
from copiloto.data.db import connect
from copiloto.data.generator import SyntheticData, generate

SQL_DIR = Path(__file__).parent / "sql"
SERIAL_TABLES = ("menu_items", "orders", "order_items")


@dataclass(frozen=True)
class SetupReport:
    seeded: bool  # True se os dados foram carregados nesta execução
    orders: int
    order_items: int


def drop_schemas(conn: psycopg.Connection) -> None:
    conn.execute("DROP SCHEMA IF EXISTS analytics CASCADE")
    conn.execute("DROP SCHEMA IF EXISTS core CASCADE")


def apply_schema(conn: psycopg.Connection) -> None:
    for name in ("schema.sql", "analytics.sql"):
        conn.execute((SQL_DIR / name).read_text(encoding="utf-8"))


def grant_readonly(conn: psycopg.Connection, settings: Settings) -> None:
    """Cria/atualiza o usuário do agente: só SELECT nas views de analytics, nada mais."""
    role = sql.Identifier(settings.db_ro_user)
    database = sql.Identifier(settings.db_name)
    exists = conn.execute(
        "SELECT 1 FROM pg_roles WHERE rolname = %s", (settings.db_ro_user,)
    ).fetchone()
    verb = "ALTER" if exists else "CREATE"
    conn.execute(
        sql.SQL(
            verb + " ROLE {} WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD {}"
        ).format(role, sql.Literal(settings.db_ro_password))
    )

    statements = [
        "ALTER ROLE {role} SET default_transaction_read_only = on",
        "ALTER ROLE {role} SET statement_timeout = {timeout}",
        "ALTER ROLE {role} SET search_path = analytics",
        "REVOKE TEMPORARY ON DATABASE {database} FROM PUBLIC",
        "REVOKE CREATE ON SCHEMA public FROM PUBLIC",
        "GRANT CONNECT ON DATABASE {database} TO {role}",
        "GRANT USAGE ON SCHEMA analytics TO {role}",
        "GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO {role}",
    ]
    timeout = sql.Literal(f"{settings.sql_timeout_ms}ms")
    for statement in statements:
        conn.execute(sql.SQL(statement).format(role=role, database=database, timeout=timeout))


def has_data(conn: psycopg.Connection) -> bool:
    return conn.execute("SELECT EXISTS (SELECT 1 FROM core.orders)").fetchone()[0]


def _copy(conn: psycopg.Connection, table: str, frame: pd.DataFrame) -> None:
    columns = sql.SQL(", ").join(sql.Identifier(c) for c in frame.columns)
    statement = sql.SQL("COPY {} ({}) FROM STDIN").format(sql.Identifier("core", table), columns)
    rows = frame.astype(object).where(frame.notna(), None).itertuples(index=False, name=None)
    with conn.cursor() as cur, cur.copy(statement) as copy:
        for row in rows:
            copy.write_row(row)


def load(conn: psycopg.Connection, data: SyntheticData) -> None:
    _copy(conn, "channels", data.channels)
    _copy(conn, "menu_items", data.menu_items)
    _copy(conn, "holidays", data.holidays)
    _copy(conn, "orders", data.orders)
    _copy(conn, "order_items", data.order_items)
    for table in SERIAL_TABLES:
        conn.execute(
            sql.SQL(
                "SELECT setval(pg_get_serial_sequence({qualified}, 'id'), max(id)) FROM {t}"
            ).format(qualified=sql.Literal(f"core.{table}"), t=sql.Identifier("core", table))
        )


def run(
    settings: Settings,
    *,
    reset: bool = False,
    data_factory: Callable[[], SyntheticData] | None = None,
) -> SetupReport:
    def default_data() -> SyntheticData:
        return generate(seed=settings.data_seed, start=settings.data_start, end=settings.data_end)

    with connect(settings) as conn:
        if reset:
            drop_schemas(conn)
        apply_schema(conn)
        grant_readonly(conn, settings)
        seeded = not has_data(conn)
        if seeded:
            load(conn, (data_factory or default_data)())
        orders, items = conn.execute(
            "SELECT (SELECT count(*) FROM core.orders), (SELECT count(*) FROM core.order_items)"
        ).fetchone()
        conn.commit()
    return SetupReport(seeded=seeded, orders=orders, order_items=items)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepara o banco do Copiloto do Dono.")
    parser.add_argument("--reset", action="store_true", help="apaga e recria os schemas")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    report = run(get_settings(), reset=args.reset)
    status = "dados carregados" if report.seeded else "dados já existiam"
    elapsed = time.perf_counter() - started
    print(
        f"Banco pronto ({status}): {report.orders} pedidos, "
        f"{report.order_items} itens, {elapsed:.1f}s."
    )


if __name__ == "__main__":
    main()
