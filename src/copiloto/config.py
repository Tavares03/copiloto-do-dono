"""Configuração central, lida de variáveis de ambiente (e do arquivo .env, se existir)."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from psycopg.conninfo import make_conninfo
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Banco de dados
    db_host: str = "localhost"
    db_port: int = 5433
    db_name: str = "copiloto"
    db_admin_user: str = "copiloto"
    db_admin_password: str = "copiloto"
    db_ro_user: str = "copiloto_ro"
    db_ro_password: str = "copiloto_ro"
    sql_timeout_ms: int = 5000
    sql_row_limit: int = 200

    # Agente e previsão
    agent_max_steps: int = 6
    model_dir: Path = Path("models")

    # API, UI e webhook
    log_level: str = "INFO"
    log_dir: Path = Path("logs")  # interactions.jsonl: uma linha por pergunta respondida
    api_url: str = "http://localhost:8000"  # usado pela UI Streamlit
    api_timeout_s: float = 120.0
    webhook_secret: str = ""  # vazio = webhook desligado
    webhook_allowed_senders: str = ""  # telefones separados por vírgula; vazio = qualquer um

    @property
    def allowed_senders(self) -> set[str]:
        return {
            digits
            for raw in self.webhook_allowed_senders.split(",")
            if (digits := "".join(ch for ch in raw if ch.isdigit()))
        }

    # Dados sintéticos
    data_seed: int = 42
    data_start: date = date(2026, 5, 1)
    data_end: date = date(2026, 10, 31)
    copiloto_today: date = date(2026, 11, 1)

    # LLM
    llm_providers: str = "groq,gemini,ollama"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "qwen/qwen3.8-27b"
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-2.5-flash"
    # "none" desliga o raciocínio interno do Gemini 2.5 Flash: sem isso ele gasta o max_tokens
    # "pensando" e devolve a resposta vazia ou cortada. Vazio = não envia o parâmetro.
    gemini_reasoning_effort: str = "none"
    groq_reasoning_effort: str = ""
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b"

    # Resiliência
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    llm_backoff_base_s: float = 1.0
    llm_retry_after_max_s: float = 10.0
    circuit_failure_threshold: int = 3
    circuit_cooldown_s: float = 300.0
    llm_temperature: float = 0.0
    llm_max_tokens: int = 700

    # Cache
    llm_cache_enabled: bool = False
    llm_cache_dir: Path = Path(".cache/llm")

    @property
    def provider_order(self) -> list[str]:
        """LLM_PROVIDERS="groq, gemini" -> ["groq", "gemini"]."""
        return [p.strip().lower() for p in self.llm_providers.split(",") if p.strip()]

    def conninfo(self, *, readonly: bool = False, dbname: str | None = None) -> str:
        """String de conexão do usuário admin ou do usuário somente leitura."""
        user, password = (
            (self.db_ro_user, self.db_ro_password)
            if readonly
            else (self.db_admin_user, self.db_admin_password)
        )
        return make_conninfo(
            host=self.db_host,
            port=self.db_port,
            dbname=dbname or self.db_name,
            user=user,
            password=password,
            connect_timeout=5,
            application_name="copiloto",
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
