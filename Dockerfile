# Uma imagem para init, api e ui (muda só o comando). Dependências em camada própria para
# o cache do Docker: mexer no código não reinstala nada.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY ui ./ui
COPY .streamlit ./.streamlit
COPY evals ./evals
COPY models ./models

RUN useradd --create-home --uid 1000 copiloto \
    && mkdir -p /app/logs /app/.cache \
    && chown -R copiloto /app
USER copiloto

EXPOSE 8000 8501
CMD ["uvicorn", "copiloto.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
