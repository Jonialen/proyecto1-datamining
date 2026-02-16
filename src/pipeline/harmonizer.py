"""Armonización de esquemas entre años: renombra aliases, agrega columnas faltantes, normaliza valores."""

from __future__ import annotations

import polars as pl

from src.schemas import (
    CANONICAL_SCHEMAS,
    VALUE_NORMALIZATION,
    build_alias_map,
    get_canonical_columns,
)


def harmonize(
    df: pl.DataFrame,
    dataset_type: str,
    year: int | None,
    source_file: str,
) -> pl.DataFrame:
    """
    Armoniza un DataFrame Polars al esquema canónico.

    1. Lowercase columnas
    2. Renombra aliases → nombre canónico
    3. Agrega columnas faltantes como null
    4. Reordena columnas
    5. Normaliza valores de texto
    6. Agrega _year y _source_file
    """
    # 1. Lowercase column names
    rename_lower = {col: col.lower() for col in df.columns}
    df = df.rename(rename_lower)

    # Drop caudef.descrip if present (only in 2013 defunciones, not useful)
    if "caudef.descrip" in df.columns:
        df = df.drop("caudef.descrip")

    # 2. Rename aliases → canonical
    alias_map = build_alias_map(dataset_type)
    rename_map = {}
    for col in df.columns:
        canonical = alias_map.get(col)
        if canonical and canonical != col:
            rename_map[col] = canonical
    if rename_map:
        df = df.rename(rename_map)

    # 3. Add missing canonical columns as null
    canonical_cols = get_canonical_columns(dataset_type)
    for col in canonical_cols:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))

    # 4. Reorder to canonical order + keep only canonical columns
    df = df.select(canonical_cols)

    # 5. Text normalization and cleaning
    # 5a. Generic cleaning: strip and Title Case to ensure consistency
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col(col).str.strip_chars().str.to_titlecase().alias(col)
            )

    # 5b. Specific value replacement based on VALUE_NORMALIZATION
    for col, norm_map in VALUE_NORMALIZATION.items():
        if col in df.columns:
            # TitleCase keys in the map to match the previous step
            tc_map = {k.strip().title(): v for k, v in norm_map.items()}
            df = df.with_columns(
                pl.col(col).cast(pl.Utf8, strict=False).replace(tc_map).alias(col)
            )

    # 6. Add metadata columns
    df = df.with_columns(
        pl.lit(year).cast(pl.Int16).alias("_year"),
        pl.lit(source_file).alias("_source_file"),
    )

    return df
