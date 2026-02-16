"""Conexión y utilidades MongoDB."""

from __future__ import annotations

import polars as pl
from pymongo import MongoClient

from src.config import settings


def get_db():
    """Retorna conexión a la base de datos MongoDB."""
    client = MongoClient(settings.mongo_uri)
    return client[settings.mongo_db]


def bulk_insert(
    df: pl.DataFrame,
    collection_name: str,
    batch_size: int | None = None,
    verbose: bool = True,
) -> int:
    """Inserta un DataFrame Polars en MongoDB en batches."""
    if batch_size is None:
        batch_size = settings.batch_size

    db = get_db()
    collection = db[collection_name]

    records = df.to_pandas().to_dict(orient="records")

    # Replace NaN with None for MongoDB
    for record in records:
        for key, val in record.items():
            if isinstance(val, float) and val != val:  # NaN check
                record[key] = None

    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        collection.insert_many(batch)
        total += len(batch)

    if verbose:
        print(f"    -> {total:,} docs insertados en '{collection_name}'")

    return total


def save_metadata(dataset_type: str, column_labels: dict[str, str]):
    """Guarda metadata de columnas en colección _metadata."""
    db = get_db()
    db["_metadata"].update_one(
        {"dataset_type": dataset_type},
        {"$set": {"column_labels": column_labels}},
        upsert=True,
    )


def collection_stats() -> dict[str, dict]:
    """Retorna estadísticas de cada colección."""
    db = get_db()
    stats = {}
    for name in sorted(db.list_collection_names()):
        col = db[name]
        count = col.estimated_document_count()
        sample_doc = col.find_one({}, {"_id": 0})
        stats[name] = {
            "count": count,
            "fields": list(sample_doc.keys()) if sample_doc else [],
        }
    return stats


def drop_all():
    """Elimina todas las colecciones."""
    db = get_db()
    for name in db.list_collection_names():
        db[name].drop()
