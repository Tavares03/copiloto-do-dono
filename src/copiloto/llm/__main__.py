"""Confere os provedores configurados no .env.

python -m copiloto.llm            # uma pergunta curtinha em cada provedor da fila
python -m copiloto.llm --models   # modelos disponíveis para as suas chaves
"""

from __future__ import annotations

import argparse

import openai

from copiloto.config import get_settings
from copiloto.llm.client import LLMClient
from copiloto.llm.providers import build_providers
from copiloto.llm.types import AllProvidersFailed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnóstico dos provedores de LLM.")
    parser.add_argument("--models", action="store_true", help="lista os modelos disponíveis")
    args = parser.parse_args(argv)

    settings = get_settings()
    providers, skipped = build_providers(settings)
    for item in skipped:
        print(f"- {item.name}: fora da fila ({item.reason})")
    if not providers:
        print("Nenhum provedor utilizável. Configure GROQ_API_KEY ou GEMINI_API_KEY no .env.")
        return 1

    for provider in providers:
        if args.models:
            try:
                ids = sorted(model.id for model in provider.client.models.list())
            except openai.OpenAIError as exc:
                print(f"x {provider.name}: não consegui listar ({exc})")
                continue
            print(f"\n{provider.name} ({len(ids)} modelos):")
            print("\n".join(f"  {model_id}" for model_id in ids))
            continue

        client = LLMClient([provider], settings=settings)
        try:
            response = client.chat([{"role": "user", "content": "Responda só com a palavra: ok"}])
        except AllProvidersFailed as exc:
            last = exc.attempts[-1]
            print(f"x {provider.name} / {provider.model}: {last.outcome} ({last.detail[:150]})")
            continue
        print(
            f"+ {provider.name} / {provider.model}: {response.content!r} "
            f"em {response.latency_ms:.0f} ms, {response.usage.total_tokens} tokens"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
