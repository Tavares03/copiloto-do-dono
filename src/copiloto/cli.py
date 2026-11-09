"""Pergunte ao Copiloto pelo terminal.

python -m copiloto.cli "quanto vendi ontem?"
python -m copiloto.cli -v "qual canal deu mais lucro em outubro?"   # mostra o "por dentro"
python -m copiloto.cli --json "quanto vendi ontem?"                 # resultado completo
python -m copiloto.cli                                              # modo conversa
"""

from __future__ import annotations

import argparse
import json
import sys

from copiloto.agent.factory import build_agent
from copiloto.agent.loop import AgentResult
from copiloto.config import get_settings
from copiloto.llm import Message
from copiloto.observability import configure_logging
from copiloto.stats import INTERACTIONS_FILE


def format_details(result: AgentResult) -> str:
    lines = [f"\n-- por dentro · request {result.request_id} · status {result.status} --"]
    for number, step in enumerate(result.steps, 1):
        switched = f" (trocou de {', '.join(step.switched_from)})" if step.switched_from else ""
        cached = " [cache]" if step.cached else ""
        tokens = step.prompt_tokens + step.completion_tokens
        lines.append(
            f"passo {number}: {step.provider}/{step.model}{switched}{cached} · "
            f"{step.latency_ms:.0f} ms · {tokens} tokens"
        )
        for tool in step.tool_results:
            arguments = (
                json.dumps(tool.arguments, ensure_ascii=False)
                if isinstance(tool.arguments, dict)
                else tool.arguments
            )
            outcome = f"ERRO: {tool.error}" if tool.error else "ok"
            lines.append(f"   -> {tool.name} {arguments} · {tool.latency_ms:.0f} ms · {outcome}")
    lines.append(
        f"total: {result.llm_calls} chamada(s) ao LLM · {result.total_tokens} tokens · "
        f"{result.latency_ms / 1000:.1f} s"
    )
    if result.grounding["corrected"]:
        lines.append("verificador de números: pediu uma correção ao modelo")
    if not result.grounded:
        lines.append(f"ATENÇÃO: números sem fonte nas tools: {result.grounding['ungrounded']}")
    return "\n".join(lines)


def _print_result(result: AgentResult, *, verbose: bool, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))
        return
    print(result.answer)
    if verbose:
        print(format_details(result))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copiloto do Dono pelo terminal.")
    parser.add_argument("question", nargs="?", help="pergunta; sem ela, abre o modo conversa")
    parser.add_argument("-v", "--verbose", action="store_true", help="mostra passos e tools")
    parser.add_argument("--json", action="store_true", help="imprime o resultado em JSON")
    parser.add_argument("--log-level", default="WARNING", help="nível dos logs JSON no stderr")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):  # console do Windows não é UTF-8 por padrão
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")
    settings = get_settings()
    configure_logging(args.log_level, interactions_path=settings.log_dir / INTERACTIONS_FILE)
    agent = build_agent(settings)

    if args.question:
        result = agent.run(args.question)
        _print_result(result, verbose=args.verbose, as_json=args.json)
        return 1 if result.status == "llm_unavailable" else 0

    print("Copiloto do Dono — pergunte sobre suas vendas (Ctrl+C para sair).")
    history: list[Message] = []
    while True:
        try:
            question = input("\nvocê> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        result = agent.run(question, history=history)
        print()
        _print_result(result, verbose=args.verbose, as_json=args.json)
        history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": result.answer},
        ]


if __name__ == "__main__":
    raise SystemExit(main())
