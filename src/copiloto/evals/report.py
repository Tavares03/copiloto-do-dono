"""Tabela comparativa por provedor, em markdown, a partir dos resumos de cada rodada.

python -m copiloto.evals.report   # escreve evals/results/comparativo.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from copiloto.formatting import number

RESULTS_DIR = Path("evals/results")
PROVIDERS = ["groq", "gemini", "ollama"]
CATEGORY_LABELS = {
    "facil": "Fáceis",
    "media": "Médias",
    "ambigua": "Ambíguas (pedir período)",
    "fora_do_escopo": "Fora do escopo",
    "ataque": "Tentativas de ataque",
}


def _ratio(value: dict[str, int]) -> str:
    if not value["total"]:
        return "-"
    return f"{value['acertos']}/{value['total']} ({100 * value['acertos'] / value['total']:.0f}%)"


ROWS: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
    ("**Acurácia geral**", lambda s: f"**{_ratio(_ratio_of(s))}**"),
    ("Dados corretos (números, nomes, previsão)", lambda s: _ratio(s["dados_corretos"])),
    ("Pediu o período quando faltava", lambda s: _ratio(s["esclarecimentos_corretos"])),
    ("Recusas corretas (escopo + ataques)", lambda s: _ratio(s["recusas_corretas"])),
    ("Ataques bloqueados", lambda s: _ratio(s["ataques_bloqueados"])),
    ("Recusas indevidas", lambda s: str(s["recusas_indevidas"])),
    ("Respostas aterradas (verificador de números)", lambda s: f"{s['aterramento_pct']:.0f}%"),
    ("Erros (sem resposta)", lambda s: str(s["erros"])),
    (
        "Latência média / p95",
        lambda s: f"{number(s['latencia_media_s'], 1)} s / {number(s['latencia_p95_s'], 1)} s",
    ),
    ("Tokens por pergunta", lambda s: number(s["tokens_medios"])),
    ("Chamadas ao LLM por pergunta", lambda s: number(s["chamadas_llm_medias"], 1)),
]


def _ratio_of(summary: dict[str, Any]) -> dict[str, int]:
    return {"acertos": summary["acertos"], "total": summary["perguntas"]}


def render_report(summaries: list[dict[str, Any]], expected: list[str] = PROVIDERS) -> str:
    by_provider = {s["provider"]: s for s in summaries}
    columns = [p for p in expected if p in by_provider] + [
        p for p in expected if p not in by_provider
    ]
    header = ["Métrica"] + [
        f"{p} (`{by_provider[p]['model']}`)" if p in by_provider else p for p in columns
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    def row(label: str, render: Callable[[dict[str, Any]], str]) -> str:
        cells = [render(by_provider[p]) if p in by_provider else "não executado" for p in columns]
        return f"| {label} | " + " | ".join(cells) + " |"

    lines += [row(label, render) for label, render in ROWS]
    for key, label in CATEGORY_LABELS.items():
        lines.append(
            row(
                f"↳ {label}",
                lambda s, k=key: _ratio(s["por_categoria"].get(k, {"acertos": 0, "total": 0})),
            )
        )
    dates = sorted({s.get("executado_em", "")[:10] for s in summaries if s.get("executado_em")})
    footer = f"\n_{summaries[0]['perguntas'] if summaries else 0} perguntas por provedor"
    footer += f"; rodadas em {', '.join(dates)}._" if dates else "._"
    return "\n".join(lines) + "\n" + footer + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tabela comparativa das evals por provedor.")
    parser.add_argument("--dir", type=Path, default=RESULTS_DIR, help="pasta com resumo_*.json")
    args = parser.parse_args(argv)

    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.dir.glob("resumo_*.json"))
        if not path.stem.endswith("_parcial")
    ]
    markdown = render_report(summaries)
    output = args.dir / "comparativo.md"
    output.write_text(markdown, encoding="utf-8", newline="\n")
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8", errors="replace")
    print(markdown)
    print(f"Escrito em {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
