"""Observabilidade: o que o Copiloto respondeu, com quem, quanto custou e quanto demorou."""

from __future__ import annotations

import httpx
import pandas as pd
import streamlit as st

from copiloto.config import get_settings
from copiloto.formatting import number

settings = get_settings()
st.set_page_config(page_title="Observabilidade · Copiloto do Dono", page_icon="📈", layout="wide")
st.title("📈 Observabilidade")
st.caption(
    "Cada pergunta respondida vira uma linha em `logs/interactions.jsonl` "
    "(id da requisição, provedor, trocas, tools com argumentos, tokens e latência por etapa)."
)

try:
    stats = httpx.get(f"{settings.api_url}/stats", timeout=15).json()
except httpx.HTTPError as exc:
    st.error(f"Não consegui falar com a API em {settings.api_url}: {exc}")
    st.stop()

if not stats.get("perguntas"):
    st.info("Nenhuma pergunta registrada ainda. Faça algumas no chat e volte aqui.")
    st.stop()

cols = st.columns(5)
cols[0].metric("Perguntas", number(stats["perguntas"]))
cols[1].metric(
    "Latência p50 / p95", f"{stats['latencia_p50_s']:.1f} / {stats['latencia_p95_s']:.1f} s"
)
cols[2].metric("Tokens por pergunta", number(stats["tokens_medios"]))
cols[3].metric("Trocas de provedor", stats["trocas_de_provedor"])
cols[4].metric("Respostas aterradas", f"{stats['aterramento_pct']:.0f}%")

left, right = st.columns(2)
with left:
    st.subheader("Quem respondeu")
    if stats["provedores"]:
        st.bar_chart(pd.Series(stats["provedores"], name="respostas"))
    st.subheader("Falhas de provedor (antes da troca)")
    if stats["falhas_de_provedor"]:
        st.bar_chart(pd.Series(stats["falhas_de_provedor"], name="ocorrências"))
    else:
        st.caption("Nenhuma falha registrada.")
with right:
    st.subheader("Tools mais usadas")
    if stats["tools"]:
        st.bar_chart(pd.Series(stats["tools"], name="chamadas"))
    st.metric("Erros de tool (corrigidos pelo agente ou não)", stats["erros_de_tool"])
    st.metric("Correções pedidas pelo verificador de números", stats["correcoes_do_verificador"])
    st.write("**Status das respostas**", stats["por_status"])

st.subheader("Últimas perguntas")
st.dataframe(pd.DataFrame(stats["ultimas"]), use_container_width=True, hide_index=True)
