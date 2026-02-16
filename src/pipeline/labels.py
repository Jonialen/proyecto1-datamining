"""Aplicación inteligente de labels SPSS usando Polars."""

from __future__ import annotations

import polars as pl


def apply_labels(df: pl.DataFrame, variable_value_labels: dict) -> pl.DataFrame:
    """
    Aplica labels SPSS de forma inteligente:
    - Si >50% de valores tienen label → columna categórica (aplica labels)
    - Si <50% tienen label → columna numérica (reemplaza sentinelas con null)

    Args:
        df: DataFrame Polars
        variable_value_labels: dict {col_original: {valor: label}} de pyreadstat meta

    Returns:
        DataFrame con labels aplicados
    """
    # Build case-insensitive lookup: lowercase col → (original col name, labels dict)
    labels_lower: dict[str, dict] = {}
    for col, labels in variable_value_labels.items():
        if labels:
            labels_lower[col.lower()] = labels

    for col in df.columns:
        col_key = col.lower()
        if col_key not in labels_lower:
            continue

        labels = labels_lower[col_key]
        if not labels:
            continue

        series = df[col]
        n_valid = series.drop_nulls().len()
        if n_valid == 0:
            continue

        # Count how many non-null values have a label
        label_keys = list(labels.keys())
        n_labeled = series.is_in(label_keys).sum()
        labeled_ratio = n_labeled / n_valid

        if labeled_ratio > 0.5:
            # Categorical: replace values with their labels
            df = df.with_columns(
                pl.col(col)
                .cast(pl.Utf8, strict=False)
                .replace(
                    {str(k): str(v) for k, v in labels.items()}
                )
                .alias(col)
            )
        else:
            # Numeric: replace sentinel values with null
            df = df.with_columns(
                pl.when(pl.col(col).is_in(label_keys))
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )

    return df
