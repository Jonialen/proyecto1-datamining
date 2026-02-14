"""
Paso 2: Cargar datos desde .sav a MongoDB.
Una colección por tipo de dataset. Normaliza columnas y aplica labels SPSS.
"""
import re
import pyreadstat
import pandas as pd
from pymongo import MongoClient
from src.config import MONGO_URI, DB_NAME
from src.identify import scan_all_files, get_unique_files


def _normalize_column_name(col: str) -> str:
    """Normaliza nombre de columna: lowercase, sin espacios, sin caracteres raros."""
    col = col.lower().strip()
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[^a-z0-9_]", "", col)
    return col


def _apply_value_labels(df: pd.DataFrame, meta) -> pd.DataFrame:
    """
    Aplica labels SPSS de forma inteligente:
    - Si la mayoría de valores tienen label → columna categórica, aplicar labels
    - Si solo unos pocos tienen label (ej: 999=Ignorado) → columna numérica, reemplazar
      esos valores centinela con NaN y mantener los números
    """
    for col in df.columns:
        if col not in meta.variable_value_labels:
            continue
        labels = meta.variable_value_labels[col]
        if not labels:
            continue

        data = df[col].dropna()
        if len(data) == 0:
            continue

        # Contar qué proporción de valores tienen label
        labeled_mask = data.isin(labels.keys())
        labeled_ratio = labeled_mask.sum() / len(data) if len(data) > 0 else 0

        if labeled_ratio > 0.5:
            # Mayoría tiene label → es categórica, aplicar labels
            df[col] = df[col].map(lambda x, lb=labels: lb.get(x, x))
        else:
            # Pocos valores tienen label (sentinelas como 999=Ignorado)
            # → mantener numérica, reemplazar sentinelas con NaN
            sentinel_keys = set(labels.keys())
            df[col] = df[col].apply(lambda x, sk=sentinel_keys: None if x in sk else x)
    return df


def get_db():
    """Retorna conexión a la base de datos."""
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]


def load_file_to_mongo(filepath: str, dataset_type: str, year: int | None,
                       db=None, apply_labels: bool = True, verbose: bool = True) -> int:
    """
    Carga un archivo .sav a MongoDB.

    Args:
        filepath: Ruta al archivo .sav
        dataset_type: Tipo de dataset (nacimientos, defunciones, etc.)
        year: Año de los datos
        db: Conexión a MongoDB (si None, crea una nueva)
        apply_labels: Si True, reemplaza códigos por labels SPSS
        verbose: Imprimir progreso

    Returns:
        Número de documentos insertados
    """
    if db is None:
        db = get_db()

    df, meta = pyreadstat.read_sav(filepath)

    if apply_labels:
        df = _apply_value_labels(df, meta)

    # Guardar mapeo de labels descriptivos antes de normalizar
    col_labels = {_normalize_column_name(col): meta.column_names_to_labels.get(col, col)
                  for col in df.columns}

    # Normalizar nombres de columnas
    rename_map = {col: _normalize_column_name(col) for col in df.columns}
    df = df.rename(columns=rename_map)

    # Agregar metadatos
    df["_year"] = year
    df["_source_file"] = filepath.split("/")[-1]

    # Convertir NaN a None para MongoDB
    df = df.where(pd.notnull(df), None)

    # Insertar en colección
    collection = db[dataset_type]
    records = df.to_dict(orient="records")

    if records:
        collection.insert_many(records)

    # Guardar metadatos de columnas (labels descriptivos) en colección _metadata
    metadata_col = db["_metadata"]
    metadata_col.update_one(
        {"dataset_type": dataset_type},
        {"$set": {"column_labels": col_labels}},
        upsert=True,
    )

    if verbose:
        print(f"    -> {len(records):,} docs insertados en '{dataset_type}'")

    return len(records)


def load_all(verbose: bool = True) -> dict:
    """
    Carga todos los archivos .sav únicos a MongoDB.

    Returns:
        Dict con estadísticas de carga por colección.
    """
    print("Escaneando archivos...")
    all_files = scan_all_files(verbose=False)
    unique_files = get_unique_files(all_files)

    # Filtrar archivos desconocidos y con error
    loadable = [f for f in unique_files if f["dataset_type"] not in ("desconocido", "error")]

    db = get_db()
    stats = {}

    print(f"\nCargando {len(loadable)} archivos a MongoDB...")
    for file_info in loadable:
        dtype = file_info["dataset_type"]
        if verbose:
            print(f"\n  [{dtype}] {file_info['filename'][:50]}... (año {file_info['year'] or '?'})")

        try:
            n = load_file_to_mongo(
                filepath=file_info["filepath"],
                dataset_type=dtype,
                year=file_info["year"],
                db=db,
                verbose=verbose,
            )
            if dtype not in stats:
                stats[dtype] = {"files": 0, "docs": 0}
            stats[dtype]["files"] += 1
            stats[dtype]["docs"] += n
        except Exception as e:
            print(f"    ERROR: {e}")

    # También cargar los desconocidos en una colección separada
    unknown = [f for f in unique_files if f["dataset_type"] == "desconocido"]
    if unknown:
        print(f"\n  {len(unknown)} archivos no clasificados, cargando en 'otros'...")
        for file_info in unknown:
            try:
                n = load_file_to_mongo(
                    filepath=file_info["filepath"],
                    dataset_type="otros",
                    year=file_info["year"],
                    db=db,
                    verbose=verbose,
                )
                if "otros" not in stats:
                    stats["otros"] = {"files": 0, "docs": 0}
                stats["otros"]["files"] += 1
                stats["otros"]["docs"] += n
            except Exception as e:
                print(f"    ERROR: {e}")

    return stats


def drop_all_collections(verbose: bool = True):
    """Elimina todas las colecciones de la DB (útil para re-cargar)."""
    db = get_db()
    for name in db.list_collection_names():
        db[name].drop()
        if verbose:
            print(f"  Colección '{name}' eliminada")


if __name__ == "__main__":
    print("=== Cargando datos a MongoDB ===\n")
    stats = load_all()
    print("\n=== Resumen de carga ===")
    total_docs = 0
    for dtype, data in sorted(stats.items()):
        print(f"  {dtype:25s}: {data['files']:2d} archivos, {data['docs']:>9,} documentos")
        total_docs += data["docs"]
    print(f"\n  Total: {total_docs:,} documentos")
