"""Memória curta de conversa por remetente do WhatsApp (em memória, por processo).

Permite o fluxo "quanto vendi?" → "qual período?" → "outubro". Expira depois de `ttl_s` sem
mensagens. Para várias réplicas da API, trocaria por Redis; aqui uma réplica basta.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from copiloto.llm import Message


@dataclass
class _Conversation:
    turns: list[Message] = field(default_factory=list)
    updated_at: float = 0.0


class ConversationStore:
    def __init__(
        self,
        max_messages: int = 6,
        ttl_s: float = 1800,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_messages = max_messages
        self.ttl_s = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._conversations: dict[str, _Conversation] = {}

    def history(self, key: str) -> list[Message]:
        with self._lock:
            conversation = self._conversations.get(key)
            if conversation is None:
                return []
            if self._clock() - conversation.updated_at > self.ttl_s:
                del self._conversations[key]
                return []
            return list(conversation.turns)

    def append(self, key: str, question: str, answer: str) -> None:
        with self._lock:
            conversation = self._conversations.setdefault(key, _Conversation())
            conversation.turns += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            conversation.turns = conversation.turns[-self.max_messages :]
            conversation.updated_at = self._clock()
