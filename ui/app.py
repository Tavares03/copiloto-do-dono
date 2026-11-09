"""Interface de chat do Copiloto do Dono. Fala com a API por HTTP (não importa o agente).

streamlit run ui/app.py
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import streamlit as st

from copiloto.config import get_settings
from copiloto.formatting import number

settings = get_settings()

st.set_page_config(page_title="Copiloto do Dono", page_icon="🍔", layout="wide")

EXAMPLES = [
    "Quanto vendi ontem?",
    "Qual lanche caiu mais em outubro em relação a setembro?",
    "Qual canal deu mais lucro em outubro?",
    "Quantos burgers preparar sexta à noite?",
    "Qual o horário de pico aos sábados em outubro?",
    "Qual canal dá mais lucro?",
]
CIRCUIT_LABELS = {
    "closed": "🟢 disponível",
    "half_open": "🟡 testando de novo",
    "open": "🔴 em espera",
}


@st.cache_resource
def http() -> httpx.Client:
    return httpx.Client(base_url=settings.api_url, timeout=settings.api_timeout_s)


def call_api(method: str, path: str, **kwargs: Any) -> dict[str, Any] | None:
    try:
        response = http().request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        st.error(f"Não consegui falar com a API em {settings.api_url}: {exc}")
        return None


@st.cache_data(ttl=15, show_spinner=False)
def fetch_health() -> dict[str, Any] | None:
    try:
        return httpx.get(f"{settings.api_url}/health", timeout=10).json()
    except httpx.HTTPError:
        return None


# --- barra lateral ---------------------------------------------------------------------------


def render_sidebar() -> None:
    with st.sidebar:
        st.title("🍔 Copiloto do Dono")
        st.caption("Hamburgueria fictícia · dados sintéticos de mai a out/2026")

        health = fetch_health()
        st.subheader("Status")
        if health is None:
            st.error(f"API fora do ar ({settings.api_url})")
        else:
            database = health["banco"]
            if database["ok"]:
                st.markdown(f"🟢 **Banco** · dados até {database['dados_ate']}")
            else:
                st.markdown(f"🔴 **Banco** · {database.get('detalhe', 'indisponível')}")
            for provider in health["llm"]:
                if not provider["active"]:
                    st.markdown(f"⚪ **{provider['name']}** · fora da fila")
                    st.caption(provider["detail"])
                    continue
                label = CIRCUIT_LABELS.get(provider["circuit"], provider["circuit"])
                st.markdown(f"{label} · **{provider['name']}** `{provider['model']}`")
                if provider.get("last_error"):
                    st.caption(f"último erro: {provider['last_error'][:90]}")
            st.caption(f"Previsão: `{health['previsao']['modelo']}` · hoje = {health['hoje']}")

        st.divider()
        if st.button("📋 Resumo de ontem", use_container_width=True):
            with st.spinner("Montando o resumo..."):
                st.session_state.summary = call_api("GET", "/daily-summary")
        if st.button("🧹 Nova conversa", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("summary", None)

        st.subheader("Experimente")
        for example in EXAMPLES:
            st.button(
                example,
                key=f"ex-{example}",
                use_container_width=True,
                on_click=lambda q=example: st.session_state.update(pending=q),
            )


# --- "por dentro" ------------------------------------------------------------------------------


def render_details(result: dict[str, Any]) -> None:
    with st.expander("🔎 Por dentro", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Respondeu", result["provider"] or "-")
        cols[1].metric("Chamadas ao LLM", result["llm_calls"])
        cols[2].metric("Tokens", number(result["total_tokens"]))
        cols[3].metric("Tempo", f"{result['latency_ms'] / 1000:.1f} s")

        grounding = result["grounding"]
        if not result["grounded"]:
            st.error(f"Números sem fonte nas tools: {', '.join(grounding['ungrounded'])}")
        elif grounding["corrected"]:
            flagged = ", ".join(grounding.get("flagged", []))
            st.info(
                f"O verificador de números barrou {flagged} na primeira tentativa (número sem "
                "fonte nas tools) e o modelo corrigiu a resposta."
            )
        else:
            st.success("Todos os números da resposta vieram das tools.")

        for index, step in enumerate(result["steps"], 1):
            tokens = step["prompt_tokens"] + step["completion_tokens"]
            line = (
                f"**Passo {index}** · `{step['provider']}/{step['model']}` · "
                f"{step['latency_ms']:.0f} ms · {tokens} tokens"
            )
            if step["cached"]:
                line += " · do cache"
            st.markdown(line)
            if step["switched_from"]:
                st.warning("Troca de provedor: " + " → ".join(step["attempts"]))
            for tool in step["tools"]:
                arguments = tool["arguments"]
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                st.markdown(f"&nbsp;&nbsp;↳ `{tool['name']}` · {tool['latency_ms']:.0f} ms")
                st.code(arguments, language="json")
                if tool["error"]:
                    st.error(tool["error"])
                st.json(tool["output"], expanded=False)
        st.caption(f"request_id `{result['request_id']}` · status `{result['status']}`")


# --- página principal --------------------------------------------------------------------------


def render_summary() -> None:
    summary = st.session_state.get("summary")
    if not summary:
        return
    origin = (
        f"redigido por {summary['provider']}/{summary['model']}"
        if summary["source"] == "llm"
        else f"template ({summary['fallback_reason'] or 'modo sem LLM'})"
    )
    with st.container(border=True):
        st.markdown("#### 📋 Resumo do dia")
        st.text(summary["text"])
        st.caption(f"Números calculados pelo sistema · texto {origin}")


def ask(question: str) -> None:
    history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Consultando os dados..."):
            result = call_api("POST", "/chat", json={"question": question, "history": history})
        if result is None:
            return
        st.markdown(result["answer"])
        render_details(result)
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "result": result}
    )


def main() -> None:
    st.session_state.setdefault("messages", [])
    render_sidebar()

    st.title("Pergunte sobre o seu negócio")
    st.caption(
        "Faturamento, itens, canais, horários, cancelamentos e previsão de demanda. "
        "Todo número vem de uma consulta ao banco: abra o **Por dentro** para conferir."
    )
    render_summary()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("result"):
                render_details(message["result"])

    question = st.chat_input("Ex.: quanto vendi ontem?")
    pending = st.session_state.pop("pending", None)
    if question or pending:
        ask(question or pending)


main()
