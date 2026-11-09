# Copiloto do Dono — plano de design e implementação

> Documento de design e plano de implementação. O README é a porta de entrada.

## Objetivo

Agente de IA que responde, em português e com números rastreáveis, perguntas do dono de uma
hamburgueria pequena sobre o próprio negócio, e gera um resumo diário. Tudo roda em planos
gratuitos (Groq, Gemini, Ollama), sem cartão de crédito.

## Arquitetura

```
pergunta (CLI · Streamlit · POST /chat · webhook N8N/WhatsApp)
  └─► Agente (loop de tool calling feito à mão, máx. 6 passos)
        ├─► LLMClient.chat(messages, tools)
        │     fila do .env: groq → gemini → ollama
        │     retry com backoff + Retry-After · circuit breaker · cache em disco
        └─► tools
              ├─ run_sql          → sqlglot + usuário read-only → Postgres (schema analytics)
              ├─ get_metrics      → SQL pronto e testado      → Postgres (schema analytics)
              └─ forecast_demand  → models/demand_v1_AAAA-MM-DD.joblib
  └─► resposta PT-BR + fonte de cada número + log JSON da interação
```

## Restrições globais

- Somente planos gratuitos; nenhuma API paga como padrão, exemplo ou fallback.
- SDK `openai` apontando para endpoints compatíveis; nada de SDK específico de provedor.
- Python 3.12, dependências via `uv` (lockfile commitado).
- Código e banco em inglês; textos para o dono em PT-BR; commits em português.
- Testes nunca chamam LLM real (mocks); testes de banco rodam contra Postgres real
  (Docker local / service container no CI) e falham — em vez de pular — quando `REQUIRE_DB=1`.
- Dados reproduzíveis: seed 42, período 01/05/2026–31/10/2026, “hoje” = `COPILOTO_TODAY` (padrão 2026-11-01).

## Banco de dados

Duas camadas:

- `core` — dados brutos (hoje: gerador sintético; amanhã: ComandaHub).
  `channels`, `menu_items`, `orders`, `order_items` (preço/custo congelados na venda), `holidays`.
- `analytics` — somente views; a **única** coisa visível para o usuário `copiloto_ro`.
  - `orders`: pedido + canal + data/hora local, dia da semana, faixa (`almoco` < 15h ≤ `tarde` < 18h ≤ `jantar`),
    `net_revenue`, `items_cost`, `contribution_margin` (NULL para cancelados → SUM/AVG corretos sem filtro).
  - `sold_items`: itens de pedidos **concluídos** com receita bruta, custo e margem da linha.
  - `menu`, `channels`, `holidays`.

Usuário read-only: só `SELECT` em `analytics`, `default_transaction_read_only=on`,
`statement_timeout=5s`, `search_path=analytics`. Criado por `python -m copiloto.data.setup`
(idempotente: aplica SQL, cria/atualiza papel, popula se vazio). Trocar a fonte = reescrever as views.

### Glossário de métricas

- **Faturamento** = Σ (subtotal − desconto) dos pedidos concluídos (taxa de entrega é repasse).
- **Ticket médio** = faturamento ÷ pedidos concluídos.
- **Margem de contribuição** = faturamento − CMV − comissão do canal.

## Dados sintéticos (hamburgueria)

- Fechada às segundas; abre 11h–23h; picos 12h–14h e 19h–22h; fim de semana 1,5–1,8×.
- Canais: iFood ~45% (comissão ilustrativa 23%), WhatsApp ~25%, balcão ~20%, 99Food ~10%
  (comissão 12%, crescendo de 6% → 14% no período).
- Efeitos: tendência +10% no semestre, dia de pagamento (dias 5–7 e 20–21) +12%, fim de mês −8%,
  feriados e datas comemorativas, dias de chuva aleatórios (mais delivery, menos balcão), ruído diário.
- Cancelamentos 3–6% (maiores nos apps e nos picos), com motivo; tempo de preparo cresce com a lotação.
- Histórias plantadas: Smash Bacon sobe de R$ 33,90 para R$ 37,90 em 01/10 e cai ~35%;
  Burger Trufado lançado em 15/08 ganha tração; custo dos burgers +6% a partir de 01/08.

## Camada de LLM

`LLMClient.chat(messages, tools) -> LLMResponse` com fila configurável (`LLM_PROVIDERS`).
Provedor sem chave é ignorado; Ollama só entra se `/api/tags` responder em < 0,5 s.

| Erro | Ação |
|---|---|
| 429 | espera `Retry-After` se ≤ `LLM_RETRY_AFTER_MAX_S`, senão pula |
| 5xx, timeout, conexão | backoff exponencial (2 tentativas), depois pula |
| 402, 403, 404 (modelo inexistente) | pula imediatamente |
| 400 `tool_use_failed` | 1 nova tentativa, depois pula |

Circuit breaker: `CIRCUIT_FAILURE_THRESHOLD` falhas seguidas → provedor em espera por
`CIRCUIT_COOLDOWN_S`. Cache em disco (hash de mensagens + tools + modelo + provedor) guarda a
latência original. Cada troca de provedor é logada com o motivo.

## Agente e tools

- Loop manual, máximo `AGENT_MAX_STEPS` (6). System prompt: nunca inventar números, citar a fonte,
  recusar fora do escopo, pedir esclarecimento quando faltar período, PT-BR curto.
- `run_sql`: sqlglot (postgres) → 1 statement SELECT/UNION/WITH, nenhum nó DML/DDL, tabelas na
  whitelist (CTEs permitidas), funções perigosas bloqueadas, `LIMIT 200` automático; conexão read-only.
- `get_metrics`: faturamento, pedidos, ticket_medio, top_itens, itens_em_queda, margem_por_canal,
  horarios_pico, cancelamentos, tempo_preparo; período + comparação + filtro de canal.
- `forecast_demand`: pedidos previstos por faixa + itens principais + versão do modelo + MAE.

## ML

Alvo: pedidos por dia × faixa. Baseline: média da mesma faixa/dia da semana nas últimas 4 semanas.
Modelo: `GradientBoostingRegressor` com calendário (dia da semana, faixa, feriado, véspera,
fim/início de mês, dia de pagamento) + a média da baseline (só dados ≥ 7 dias antes → sem vazamento).
Split temporal: treino até 30/09, teste em outubro. Itens = pedidos previstos × mix recente da faixa.
Artefato: `models/demand_v1_AAAA-MM-DD.joblib` + `models/demand_v1_AAAA-MM-DD.json` (métricas).

## Evals

`evals/dataset.yaml` com ~36 perguntas (fáceis, médias, ambíguas, fora do escopo, ataques), gabarito
por `reference_sql` executado na hora. Métricas: acurácia numérica com tolerância, comportamento
correto (responder / esclarecer / recusar), aterramento (todo número citado aparece nas saídas das
tools), latência média e p95, tokens e chamadas por pergunta. Execução por provedor com `--provider`,
`--limit`, pausa entre chamadas e cache; gera tabela markdown comparativa.

## Marcos

- [x] **1. Estrutura + dados + schema** — pyproject/uv, config, `src/copiloto/data/sql/{schema,analytics}.sql`,
  `copiloto.data.{generator,db,setup}`, docker-compose (db), testes do gerador e do setup.
- [x] **2. LLM multi-provedor** — `copiloto.llm.{providers,resilience,cache,client}`, testes com
  provedores mockados (429/402/403/404/timeout/5xx, Retry-After, circuit breaker, cache).
- [x] **3. Tools + agente CLI** — `sql_guard`, `run_sql`, `repository` + `get_metrics`, loop do agente,
  prompts, `python -m copiloto.cli "pergunta"`; testes de SQL guard, métricas (fixture conhecida) e loop.
- [x] **4. API + UI** — FastAPI (`/chat`, `/daily-summary`, `/health`, `/webhook/whatsapp`),
  resumo diário com verificador de números, Streamlit com painel “por dentro”.
- [x] **5. ML + evals** (rodada 1 no Groq feita; rodada 2 no Groq e Gemini em lotes diários,
  por causa da cota gratuita de 20 requisições/dia do Gemini 2.5 Flash) — features, baseline, treino versionado, `forecast_demand`, dataset de evals,
  runner, relatório por provedor.
- [x] **6. Infra e acabamento** — Dockerfile + compose completo (db, init, api, ui), CI (ruff + pytest
  com Postgres), log JSON + página de observabilidade, README completo, exemplo de fluxo N8N.

## Riscos

- Cota diária de tokens do Groq vs. ~200 mil tokens por rodada completa de evals → prompt enxuto,
  lotes com `--limit`, cache, começar cedo.
- Modelos locais (7–8B em CPU) erram mais tool calls → aparece na tabela, como resultado legítimo.
