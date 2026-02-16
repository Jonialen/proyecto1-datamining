"""Escritura de DataFrames armonizados a Parquet, DuckDB y MongoDB."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.config import settings
from src.db import duckdb as duck
from src.db import mongo


def write_parquet(df: pl.DataFrame, dataset_type: str, year: int) -> Path:
    """Escribe un DataFrame a parquet con compresión zstd."""
    out_dir = settings.parquet_dir / dataset_type
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{year}.parquet"
    df.write_parquet(path, compression=settings.parquet_compression)
    return path


def load_to_duckdb():
    """Registra views en DuckDB sobre los parquet files."""
    duck.register_views()


def load_to_mongo(
    df: pl.DataFrame,
    dataset_type: str,
    column_labels: dict[str, str] | None = None,
    verbose: bool = True,
) -> int:
    """Inserta DataFrame en MongoDB y guarda metadata."""
    n = mongo.bulk_insert(df, dataset_type, verbose=verbose)
    if column_labels:
        mongo.save_metadata(dataset_type, column_labels)
    return n


def write_all(
    df: pl.DataFrame,
    dataset_type: str,
    year: int,
    column_labels: dict[str, str] | None = None,
    skip_mongo: bool = False,
    verbose: bool = True,
) -> Path:
    """Ejecuta escritura completa: Parquet + DuckDB view + MongoDB."""
    # 1. Parquet
    path = write_parquet(df, dataset_type, year)
    if verbose:
        print(f"    -> Parquet: {path} ({df.height:,} rows)")

    # 2. DuckDB view (registrar después de escribir parquet)
    load_to_duckdb()

    # 3. MongoDB
    if not skip_mongo:
        load_to_mongo(df, dataset_type, column_labels, verbose=verbose)

    return path
