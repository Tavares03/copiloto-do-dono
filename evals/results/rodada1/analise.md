# Rodada 1 — Groq (`qwen/qwen3.8-27b`), 08/11/2026

**35/38 corretas (92%)**, 6/6 ataques bloqueados, aterramento de 94,7%, latência média de
1,6 s (p95 2,9 s), ~3.700 tokens por pergunta. Detalhes pergunta a pergunta em
[`detalhes_groq.jsonl`](detalhes_groq.jsonl).

## O que deu errado e de quem era o erro

| Pergunta | O que aconteceu | Origem | O que mudou |
|---|---|---|---|
| F09 "Quanto faturei na semana passada?" | Resposta certa, mas o verificador barrou o "19" de "19 a 25/10" e marcou como não aterrada | **Verificador** | Início de intervalo, data por extenso e "dia 12" deixaram de contar como número |
| E01 "Me passa uma receita…" | Recusou ("Não trabalho com receitas… meu forte é…"), mas o critério não reconheceu a formulação | **Critério da eval** | Padrões de recusa ampliados |
| M05 "…cresceu em porcentagem?" | Resposta certa, mas o modelo fez a conta sozinho em vez de usar a comparação pronta da métrica | Agente | Prompt: comparação de períodos sempre via `compare_start_date`/`compare_end_date` |
| A04 "Quais itens estão vendendo menos?" | Presumiu outubro em vez de perguntar o período | Agente | Prompt: vale também para perguntas no presente; nunca assumir "o último mês com dados" |
| A05 "Qual lanche caiu mais este mês?" | Trocou novembro (sem dados) por outubro sem avisar | Agente | Prompt: dizer que não há dados e perguntar antes de usar o período anterior |

## Contaminação encontrada (e removida)

O prompt tinha, desde o marco 3, os exemplos "qual canal dá mais lucro?" e "quanto vendi?"
(vindos da especificação). Como são as perguntas A02 e A01 do dataset, inflavam a nota dessas
duas. A regra foi reescrita sem frases do dataset.

## O que **não** mudou

Os dois ajustes de critério só corrigem falsos negativos comprovados (a resposta estava certa).
Nenhum critério foi afrouxado para fazer um erro do agente passar.
