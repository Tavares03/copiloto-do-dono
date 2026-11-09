# Imagens do README (a gravar)

Com `docker compose up --build` rodando, abra http://localhost:8501 e grave/capture:

| Arquivo | O que mostrar |
|---|---|
| `demo.gif` | Clicar em "Quantos burgers preparar sexta à noite?" → resposta → abrir o **Por dentro** (passos, `forecast_demand` com argumentos, tokens). Depois "Qual canal dá mais lucro?" → o agente pede o período → responder "outubro". ~20 s. |
| `por-dentro.png` | Uma resposta com o **Por dentro** aberto, de preferência com troca de provedor (`groq:rate_limited → gemini:ok`) ou com o aviso do verificador de números. |
| `observabilidade.png` | A página **Observabilidade** depois de umas 10 perguntas. |

Depois é só descomentar as linhas `<!-- ![...](docs/img/...) -->` na seção *Demo* do README.

Dica: no Windows, `Win + Shift + S` para print e o [ScreenToGif](https://www.screentogif.com/)
(gratuito) para o GIF.
