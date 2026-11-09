"""Conexões com o PostgreSQL: admin (setup) e somente leitura (agente)."""

from __future__ import annotations

import psycopg

from copiloto.config import Settings, get_settings


def connect(
    settings: Settings | None = None, *, readonly: bool = False, autocommit: bool = False
) -> psycopg.Connection:
    """Abre uma conexão. Com readonly=True usa o usuário restrito ao schema analytics."""
    settings = settings or get_settings()
    return psycopg.connect(settings.conninfo(readonly=readonly), autocommit=autocommit)
