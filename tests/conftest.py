import os

import psycopg
import pytest
from psycopg import sql

from copiloto.config import Settings
from copiloto.data import setup
from copiloto.observability import remove_handlers
from tiny_data import tiny_data

TEST_DB = "copiloto_test"


@pytest.fixture(autouse=True)
def _isolated_logging():
    """Cada teste começa sem os handlers do app (o lifespan da API configura logging)."""
    yield
    remove_handlers()


@pytest.fixture(scope="session")
def db_settings() -> Settings:
    """Banco de testes recriado do zero a cada sessão.

    Sem Postgres no ar, os testes de banco são pulados — a não ser que REQUIRE_DB=1
    (como no CI), quando passam a falhar.
    """
    settings = Settings(db_name=TEST_DB)
    try:
        with psycopg.connect(settings.conninfo(dbname="postgres"), autocommit=True) as conn:
            name = sql.Identifier(TEST_DB)
            conn.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(name))
            conn.execute(sql.SQL("CREATE DATABASE {}").format(name))
    except psycopg.OperationalError as exc:
        message = (
            f"Postgres indisponível ({exc}). Suba com: docker compose up -d db "
            "(ou o Postgres compartilhado, se o .env apontar para ele)"
        )
        if os.environ.get("REQUIRE_DB") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return settings


@pytest.fixture(scope="module")
def tiny_db(db_settings: Settings) -> Settings:
    """Banco de testes com os 6 pedidos de tests/tiny_data.py."""
    setup.run(db_settings, reset=True, data_factory=tiny_data)
    return db_settings
