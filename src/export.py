"""
Paso 3: Exportar datos desde MongoDB a DataFrames o CSV.
"""
import pandas as pd
from pymongo import MongoClient
from src.config import MONGO_URI, DB_NAME


def get_db():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]


def list_collections() -> list[str]:
    """Lista todas las colecciones disponibles."""
    db = get_db()
    return sorted(db.list_collection_names())


def to_dataframe(collection_name: str, query: dict = None,
                 projection: dict = None, limit: int = 0) -> pd.DataFrame:
    """
    Obtiene datos de una colección MongoDB como DataFrame.

    Args:
        collection_name: Nombre de la colección
        query: Filtro MongoDB (ej: {"_year": 2020})
        projection: Campos a incluir/excluir (ej: {"_id": 0})
        limit: Máximo de documentos (0 = todos)

    Returns:
        DataFrame con los datos
    """
    db = get_db()
    collection = db[collection_name]

    if query is None:
        query = {}
    if projection is None:
        projection = {"_id": 0}

    cursor = collection.find(query, projection)
    if limit > 0:
        cursor = cursor.limit(limit)

    df = pd.DataFrame(list(cursor))
    return df


def to_csv(collection_name: str, output_path: str, query: dict = None) -> str:
    """Exporta una colección a CSV."""
    df = to_dataframe(collection_name, query)
    df.to_csv(output_path, index=False)
    print(f"Exportado {len(df):,} filas a {output_path}")
    return output_path


def get_column_labels(dataset_type: str) -> dict:
    """Retorna el mapeo {col_normalizada: descripción legible} para un dataset."""
    db = get_db()
    doc = db["_metadata"].find_one({"dataset_type": dataset_type}, {"_id": 0})
    if doc and "column_labels" in doc:
        return doc["column_labels"]
    return {}


def collection_stats() -> dict:
    """Retorna estadísticas de cada colección."""
    db = get_db()
    stats = {}
    for name in sorted(db.list_collection_names()):
        col = db[name]
        count = col.count_documents({})
        sample = col.find_one({}, {"_id": 0})
        stats[name] = {
            "count": count,
            "fields": list(sample.keys()) if sample else [],
        }
    return stats


if __name__ == "__main__":
    print("=== Colecciones disponibles ===\n")
    stats = collection_stats()
    for name, data in stats.items():
        print(f"  {name:25s}: {data['count']:>9,} docs | {len(data['fields']):>3} campos")
        print(f"    Campos: {', '.join(data['fields'][:10])}{'...' if len(data['fields']) > 10 else ''}")
