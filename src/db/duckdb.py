"""Conexión y utilidades DuckDB."""

from __future__ import annotations

import duckdb
import pandas as pd
import polars as pl

from src.config import settings


_conn: duckdb.DuckDBPyConnection | None = None


def get_conn() -> duckdb.DuckDBPyConnection:
    """Retorna conexión singleton a DuckDB."""
    global _conn
    if _conn is None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        _conn = duckdb.connect(str(settings.duckdb_path))
    return _conn


def close():
    """Cierra la conexión."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


def register_views():
    """Crea views sobre los directorios parquet para cada tipo de dataset."""
    conn = get_conn()
    for dtype in settings.dataset_types:
        parquet_dir = settings.parquet_dir / dtype
        if parquet_dir.exists() and any(parquet_dir.glob("*.parquet")):
            glob_path = str(parquet_dir / "*.parquet")
            conn.execute(
                f"CREATE OR REPLACE VIEW {dtype} AS "
                f"SELECT * FROM read_parquet('{glob_path}', union_by_name=true)"
            )


def query(sql: str) -> pd.DataFrame:
    """Ejecuta SQL y retorna pandas DataFrame."""
    conn = get_conn()
    return conn.execute(sql).fetchdf()


def query_polars(sql: str) -> pl.DataFrame:
    """Ejecuta SQL y retorna Polars DataFrame."""
    conn = get_conn()
    return conn.execute(sql).pl()


def sample(dataset_type: str, n: int = 1000) -> pd.DataFrame:
    """Muestra aleatoria eficiente usando USING SAMPLE de DuckDB."""
    return query(f"SELECT * FROM {dataset_type} USING SAMPLE {n}")


def table_counts() -> dict[str, int]:
    """Retorna conteos de cada view/tabla registrada."""
    conn = get_conn()
    counts = {}
    for dtype in settings.dataset_types:
        try:
            result = conn.execute(f"SELECT COUNT(*) FROM {dtype}").fetchone()
            counts[dtype] = result[0] if result else 0
        except duckdb.CatalogException:
            pass
    return counts
