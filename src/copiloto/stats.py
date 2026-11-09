"""Estatísticas das interações registradas em logs/interactions.jsonl.

python -m copiloto.stats
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

INTERACTIONS_FILE = "interactions.jsonl"
RECENT = 15


def load_interactions(path: Path) -> list[dict[str, Any]]:
    """Linhas do arquivo de interações; ignora linhas quebradas (ex.: processo morto no meio)."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def compute_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"perguntas": 0}

    latencies = [row.get("latency_ms", 0.0) / 1000 for row in rows]
    steps = [step for row in rows for step in row.get("steps", [])]
    tools = [tool for step in steps for tool in step.get("tools", [])]
    failures = Counter(
        attempt
        for step in steps
        for attempt in step.get("attempts", [])
        if not attempt.endswith((":ok", ":cache_hit"))
    )
    return {
        "perguntas": len(rows),
        "por_status": dict(Counter(row.get("status") for row in rows)),
        "provedores": dict(Counter(row["provider"] for row in rows if row.get("provider"))),
        "trocas_de_provedor": sum(
            1 for row in rows if any(s.get("switched_from") for s in row.get("steps", []))
        ),
        "falhas_de_provedor": dict(failures),
        "tools": dict(Counter(tool["name"] for tool in tools)),
        "erros_de_tool": sum(1 for tool in tools if tool.get("error")),
        "aterramento_pct": round(100 * sum(bool(r.get("grounded")) for r in rows) / len(rows), 1),
        "correcoes_do_verificador": sum(
            1 for row in rows if (row.get("grounding") or {}).get("corrected")
        ),
        "latencia_media_s": round(float(np.mean(latencies)), 2),
        "latencia_p50_s": round(float(np.percentile(latencies, 50)), 2),
        "latencia_p95_s": round(float(np.percentile(latencies, 95)), 2),
        "tokens_total": sum(row.get("total_tokens", 0) for row in rows),
        "tokens_medios": round(float(np.mean([row.get("total_tokens", 0) for row in rows]))),
        "chamadas_llm_medias": round(float(np.mean([row.get("llm_calls", 0) for row in rows])), 2),
        "ultimas": [
            {
                "ts": row.get("ts"),
                "request_id": row.get("request_id"),
                "pergunta": row.get("question"),
                "status": row.get("status"),
                "provedor": row.get("provider"),
                "latencia_s": round(row.get("latency_ms", 0.0) / 1000, 2),
                "tokens": row.get("total_tokens", 0),
                "aterrada": row.get("grounded"),
            }
            for row in reversed(rows[-RECENT:])
        ],
    }


def main() -> int:
    from copiloto.config import get_settings

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8", errors="replace")
    path = get_settings().log_dir / INTERACTIONS_FILE
    stats = compute_stats(load_interactions(path))
    if not stats["perguntas"]:
        print(f"Nenhuma interação registrada ainda em {path}.")
        return 0
    recent = stats.pop("ultimas")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\nÚltimas perguntas:")
    for row in recent:
        when, provider = row["ts"][:19], row["provedor"] or "-"
        print(f"  {when}  {provider:<7} {row['latencia_s']:>5}s  {row['pergunta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
