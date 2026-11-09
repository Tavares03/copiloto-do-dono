"""Contratos de entrada da API (a saída é o to_dict() de cada resultado)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]  # system/tool nunca vêm de fora
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=500)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class WebhookMessage(BaseModel):
    """Formato simples para o N8N montar a partir de qualquer API de WhatsApp."""

    model_config = ConfigDict(populate_by_name=True)

    sender: str = Field(alias="from", min_length=3, max_length=40)
    message: str = Field(min_length=1, max_length=500)


class WebhookReply(BaseModel):
    reply: str
    request_id: str
    status: str
