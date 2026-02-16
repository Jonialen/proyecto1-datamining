"""API de alto nivel para notebooks: carga datos desde DuckDB."""

from __future__ import annotations

import pandas as pd
import polars as pl

from src.config import settings
from src.db import duckdb as duck
from src.schemas import get_column_descriptions


def _ensure_views():
    duck.register_views()


def load(
    dataset_type: str,
    sample_n: int = 0,
    where: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Carga datos desde DuckDB como pandas DataFrame.

    Args:
        dataset_type: Tipo de dataset (nacimientos, defunciones, etc.)
        sample_n: Si > 0, toma una muestra aleatoria (USING SAMPLE)
        where: Cláusula WHERE SQL (ej: "_year = 2020")
        columns: Lista de columnas a seleccionar (None = todas)
    """
    _ensure_views()

    cols = ", ".join(columns) if columns else "*"
    sql = f"SELECT {cols} FROM {dataset_type}"

    if where:
        sql += f" WHERE {where}"
    if sample_n > 0:
        sql += f" USING SAMPLE {sample_n}"

    return duck.query(sql)


def agg(sql: str) -> pd.DataFrame:
    """Ejecuta SQL directo contra DuckDB. Para queries agregadas en notebooks."""
    _ensure_views()
    return duck.query(sql)


def yearly_counts(dataset_type: str) -> pd.DataFrame:
    """Conteo por año, push-down a DuckDB."""
    _ensure_views()
    return duck.query(
        f"SELECT _year, COUNT(*) as n FROM {dataset_type} "
        f"GROUP BY _year ORDER BY _year"
    )


def list_datasets() -> list[str]:
    """Lista los datasets disponibles con datos."""
    _ensure_views()
    return list(duck.table_counts().keys())


def describe(dataset_type: str) -> pd.DataFrame:
    """Describe columnas de un dataset (nombre, tipo, nulls, uniques)."""
    _ensure_views()
    return duck.query(f"SUMMARIZE SELECT * FROM {dataset_type}")


def get_column_labels(dataset_type: str) -> dict[str, str]:
    """Retorna {col: descripción legible} para un dataset."""
    return get_column_descriptions(dataset_type)
