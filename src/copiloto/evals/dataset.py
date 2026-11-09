"""Carrega e valida evals/dataset.yaml."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

Category = Literal["facil", "media", "ambigua", "fora_do_escopo", "ataque"]
ExpectedKind = Literal["numero", "texto", "previsao", "esclarecimento", "recusa"]


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tipo: ExpectedKind
    sql: str | None = None
    qualquer_coluna: bool = False
    tolerancia_abs: float | None = None
    tolerancia_rel: float | None = None
    # previsão
    data: date | None = None
    faixa: Literal["almoco", "tarde", "jantar"] | None = None
    campo: str | None = None  # "total" ou uma categoria (burger, bebida...)

    @model_validator(mode="after")
    def _check(self) -> Expected:
        if self.tipo in ("numero", "texto") and not self.sql:
            raise ValueError(f"tipo {self.tipo} precisa de sql")
        if self.tipo == "previsao" and not (self.data and self.campo):
            raise ValueError("tipo previsao precisa de data e campo")
        return self


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    categoria: Category
    pergunta: str
    esperado: Expected
    nao_deve_conter: list[str] = []


def load_dataset(path: Path) -> list[EvalItem]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = [EvalItem.model_validate(entry) for entry in raw]
    ids = [i.id for i in items]
    duplicated = {i for i in ids if ids.count(i) > 1}
    if duplicated:
        raise ValueError(f"ids repetidos no dataset: {sorted(duplicated)}")
    return items
