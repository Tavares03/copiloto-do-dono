"""Loop de tool calling feito à mão: pergunta → LLM → tools → LLM → ... → resposta.

Sem framework: são poucas linhas, cada passo fica visível nos logs e o comportamento não depende
de como uma biblioteca trata cada provedor.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal, Protocol

from copiloto.agent.grounding import ungrounded_numbers
from copiloto.agent.prompts import build_system_prompt
from copiloto.agent.tools.registry import ToolRegistry, ToolResult
from copiloto.llm import AllProvidersFailed, LLMResponse, Message
from copiloto.observability import log_event, request_id_var

logger = logging.getLogger("copiloto.agent")

Status = Literal["answered", "max_steps", "llm_unavailable", "empty_answer"]

MAX_TOOL_OUTPUT_CHARS = 6000
MAX_HISTORY_MESSAGES = 10
ANSWERS: dict[Status, str] = {
    "max_steps": "Não consegui chegar a uma resposta confiável dentro do limite de passos. "
    'Tente perguntar de um jeito mais direto, com o período (ex.: "faturamento de outubro").',
    "llm_unavailable": "No momento não consegui acessar nenhum modelo de IA (os provedores "
    "gratuitos falharam ou atingiram o limite). Tente de novo em alguns minutos.",
    "empty_answer": "Não consegui formular uma resposta. Pode reformular a pergunta?",
}
GROUNDING_CHECK = (
    "[verificação automática] Estes números da sua resposta não aparecem em nenhum resultado "
    "de tool desta conversa: {numbers}. Refaça a resposta usando só números que vieram das "
    "tools; se faltar dado, chame a tool adequada (run_sql responde qualquer pergunta sobre os "
    "dados). Cite na linha Fonte apenas tools que você chamou. Não mencione esta verificação."
)


class ChatModel(Protocol):
    def chat(
        self, messages: Sequence[Message], tools: Sequence[dict[str, Any]] | None = None
    ) -> LLMResponse: ...


@dataclass(frozen=True)
class AgentStep:
    """Uma chamada ao LLM e as tools que ela pediu."""

    provider: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    switched_from: list[str]
    cached: bool
    tool_results: list[ToolResult] = field(default_factory=list)
    attempts: list[str] = field(default_factory=list)  # ex.: ["groq:rate_limited", "gemini:ok"]


@dataclass
class AgentResult:
    question: str
    answer: str
    status: Status
    steps: list[AgentStep]
    latency_ms: float
    request_id: str
    # números da resposta sem origem nas tools, e se houve rodada de correção
    grounding: dict[str, Any] = field(
        default_factory=lambda: {"ungrounded": [], "corrected": False}
    )
    # quando nenhum provedor respondeu: o que aconteceu com cada um (ex.: "gemini:rate_limited")
    llm_failure: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return not self.grounding["ungrounded"]

    @property
    def tool_results(self) -> list[ToolResult]:
        return [result for step in self.steps for result in step.tool_results]

    @property
    def provider(self) -> str | None:
        return self.steps[-1].provider if self.steps else None

    @property
    def model(self) -> str | None:
        return self.steps[-1].model if self.steps else None

    @property
    def llm_calls(self) -> int:
        return len(self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.prompt_tokens + s.completion_tokens for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "question": self.question,
            "answer": self.answer,
            "status": self.status,
            "grounded": self.grounded,
            "grounding": self.grounding,
            "llm_failure": self.llm_failure,
            "provider": self.provider,
            "model": self.model,
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "steps": [_step_dict(step) for step in self.steps],
        }


def _step_dict(step: AgentStep) -> dict[str, Any]:
    return {
        "provider": step.provider,
        "model": step.model,
        "latency_ms": round(step.latency_ms, 1),
        "prompt_tokens": step.prompt_tokens,
        "completion_tokens": step.completion_tokens,
        "switched_from": step.switched_from,
        "attempts": step.attempts,
        "cached": step.cached,
        "tools": [
            {
                "name": result.name,
                "arguments": result.arguments,
                "latency_ms": round(result.latency_ms, 1),
                "error": result.error,
                "output": result.output,
            }
            for result in step.tool_results
        ],
    }


def compact_for_llm(output: dict[str, Any], max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """JSON da saída da tool, cortando linhas se ficar grande demais para o contexto."""
    text = json.dumps(output, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    for key in ("linhas", "dados"):
        rows = output.get(key)
        if not isinstance(rows, list):
            continue
        keep = len(rows)
        while keep > 1:
            keep //= 2
            trimmed = {**output, key: rows[:keep], "linhas_omitidas": len(rows) - keep}
            text = json.dumps(trimmed, ensure_ascii=False, default=str)
            if len(text) <= max_chars:
                return text
    notice = {"aviso": "Resultado grande demais. Refine a consulta (agregue ou filtre)."}
    return json.dumps(notice, ensure_ascii=False)


def _clean_history(history: Sequence[Message]) -> list[Message]:
    """Só turnos de texto do usuário e do assistente: a API não pode injetar system/tool."""
    kept = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ]
    return kept[-MAX_HISTORY_MESSAGES:]


class Agent:
    def __init__(
        self,
        llm: ChatModel,
        tools: ToolRegistry,
        *,
        today: date,
        data_start: date,
        max_steps: int = 6,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.today = today
        self.data_start = data_start
        self.max_steps = max_steps
        self._clock = clock

    def run(self, question: str, history: Sequence[Message] = ()) -> AgentResult:
        request_id = request_id_var.get() or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        started = self._clock()
        system_prompt = build_system_prompt(self.today, self.data_start)
        past_turns = _clean_history(history)
        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            *past_turns,
            {"role": "user", "content": question},
        ]
        # De onde um número pode legitimamente vir (a mensagem de verificação NÃO entra aqui).
        sources = [system_prompt, question, *(turn["content"] for turn in past_turns)]
        steps: list[AgentStep] = []
        status: Status = "max_steps"
        answer = ANSWERS["max_steps"]
        grounding: dict[str, Any] = {"ungrounded": [], "corrected": False}
        llm_failure: list[str] = []
        try:
            for step_number in range(self.max_steps):
                try:
                    response = self.llm.chat(messages, tools=self.tools.schemas())
                except AllProvidersFailed as exc:
                    status, answer = "llm_unavailable", ANSWERS["llm_unavailable"]
                    llm_failure = [f"{a.provider}:{a.outcome}" for a in exc.attempts]
                    break
                messages.append(response.as_message())
                results = [
                    self.tools.execute(call.name, call.arguments, call.id)
                    for call in response.tool_calls
                ]
                for result in results:
                    content = compact_for_llm(result.output)
                    sources.append(content)
                    messages.append(
                        {"role": "tool", "tool_call_id": result.call_id, "content": content}
                    )
                steps.append(
                    AgentStep(
                        provider=response.provider,
                        model=response.model,
                        latency_ms=response.latency_ms,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        switched_from=response.switched_from,
                        cached=response.cached,
                        tool_results=results,
                        attempts=[f"{a.provider}:{a.outcome}" for a in response.attempts],
                    )
                )
                if response.tool_calls:
                    continue
                answer = response.content.strip()
                if not answer:
                    status, answer = "empty_answer", ANSWERS["empty_answer"]
                    break
                missing = ungrounded_numbers(answer, sources)
                can_retry = step_number < self.max_steps - 1
                if missing and not grounding["corrected"] and can_retry:
                    grounding["corrected"] = True
                    grounding["flagged"] = missing  # o que foi barrado na 1ª tentativa
                    messages.append(
                        {
                            "role": "user",
                            "content": GROUNDING_CHECK.format(numbers=", ".join(missing)),
                        }
                    )
                    continue
                grounding["ungrounded"] = missing
                status = "answered"
                break

            result = AgentResult(
                question=question,
                answer=answer,
                status=status,
                steps=steps,
                latency_ms=(self._clock() - started) * 1000,
                request_id=request_id,
                grounding=grounding,
                llm_failure=llm_failure,
            )
            self._log(result)
            return result
        finally:
            request_id_var.reset(token)

    @staticmethod
    def _log(result: AgentResult) -> None:
        data = result.to_dict()
        for step in data["steps"]:  # o log guarda argumentos e erros, não a saída inteira
            for tool in step["tools"]:
                tool.pop("output", None)
        log_event(logger, "agent_run", **data)
