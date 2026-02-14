"""
Paso 1: Identificar y clasificar cada archivo .sav del INE.
Lee metadatos, clasifica por tipo de dataset, detecta duplicados.
"""
import os
import re
import pyreadstat
from pathlib import Path
from src.config import RAWDATA_DIR, DATASET_SIGNATURES


def _extract_year_from_filename(filename: str) -> int | None:
    """Intenta extraer el año del prefijo del nombre de archivo (ej: 20171204...)."""
    match = re.match(r"^(\d{4})", filename)
    if match:
        year = int(match.group(1))
        if 2009 <= year <= 2030:
            return year
    return None


def _classify_dataset(columns: list[str]) -> str:
    """Clasifica un dataset según sus columnas usando las firmas definidas en config."""
    cols_lower = {c.lower() for c in columns}

    for dataset_type, signature_sets in DATASET_SIGNATURES.items():
        for signature in signature_sets:
            if all(kw in cols_lower for kw in signature):
                # Divorcios y matrimonios comparten columnas;
                # matrimonios tiene CLAUNI, divorcios no
                if dataset_type == "matrimonios" and "clauni" not in cols_lower:
                    continue
                return dataset_type

    return "desconocido"


def _extract_year_from_data(filepath: str) -> int | None:
    """Intenta extraer el año leyendo la primera fila del archivo."""
    try:
        df, meta = pyreadstat.read_sav(str(filepath), row_limit=5)
        cols_lower = {c.lower(): c for c in df.columns}

        # Buscar columnas que contengan 'anio' o 'año'
        for col_lower, col_orig in cols_lower.items():
            if "anio" in col_lower or "año" in col_lower:
                vals = df[col_orig].dropna()
                if len(vals) > 0:
                    year = int(vals.iloc[0])
                    if 2009 <= year <= 2030:
                        return year
    except Exception:
        pass
    return None


def scan_all_files(verbose: bool = True) -> list[dict]:
    """
    Escanea todos los .sav en rowdata/ y retorna info de cada uno.

    Returns:
        Lista de dicts con: filename, filepath, dataset_type, year, nrows, ncols, columns, is_duplicate
    """
    sav_files = sorted(RAWDATA_DIR.glob("*.sav"))
    results = []
    seen_hashes = {}  # Para detectar duplicados por nombre base sin "(1)"

    for filepath in sav_files:
        filename = filepath.name
        is_duplicate = "(1)" in filename

        # Hash base: nombre sin "(1)" para detectar duplicados
        base_name = filename.replace("(1)", "")

        try:
            df, meta = pyreadstat.read_sav(str(filepath), metadataonly=True)
            columns = list(meta.column_names)
            nrows = meta.number_rows
            ncols = meta.number_columns
            labels = meta.column_names_to_labels
            dataset_type = _classify_dataset(columns)

            # Extraer año
            year = _extract_year_from_filename(filename)
            if year is None:
                year = _extract_year_from_data(filepath)

            info = {
                "filename": filename,
                "filepath": str(filepath),
                "dataset_type": dataset_type,
                "year": year,
                "nrows": nrows,
                "ncols": ncols,
                "columns": columns,
                "column_labels": labels,
                "is_duplicate": is_duplicate,
            }

            if base_name in seen_hashes and is_duplicate:
                info["is_duplicate"] = True
            seen_hashes[base_name] = info

            results.append(info)

            if verbose:
                dup_tag = " [DUPLICADO]" if info["is_duplicate"] else ""
                print(f"  {dataset_type:25s} | {year or '????':4} | {nrows:>7,} rows | {ncols:>3} cols | {filename[:40]}...{dup_tag}")

        except Exception as e:
            if verbose:
                print(f"  ERROR leyendo {filename}: {e}")
            results.append({
                "filename": filename,
                "filepath": str(filepath),
                "dataset_type": "error",
                "year": None,
                "nrows": 0,
                "ncols": 0,
                "columns": [],
                "column_labels": {},
                "is_duplicate": is_duplicate,
                "error": str(e),
            })

    return results


def get_unique_files(file_info: list[dict]) -> list[dict]:
    """Filtra archivos duplicados, dejando solo los originales (sin '(1)')."""
    return [f for f in file_info if not f["is_duplicate"]]


def get_summary(file_info: list[dict]) -> dict:
    """Genera un resumen de la clasificación."""
    unique = get_unique_files(file_info)
    summary = {}
    for f in unique:
        dtype = f["dataset_type"]
        if dtype not in summary:
            summary[dtype] = {"count": 0, "years": [], "total_rows": 0}
        summary[dtype]["count"] += 1
        if f["year"]:
            summary[dtype]["years"].append(f["year"])
        summary[dtype]["total_rows"] += f["nrows"]

    return summary


if __name__ == "__main__":
    print("=== Escaneando archivos .sav ===\n")
    info = scan_all_files()
    print(f"\nTotal archivos: {len(info)}")

    unique = get_unique_files(info)
    print(f"Archivos únicos (sin duplicados): {len(unique)}")

    print("\n=== Resumen por tipo ===")
    summary = get_summary(info)
    for dtype, data in sorted(summary.items()):
        years_str = f"{min(data['years'])}-{max(data['years'])}" if data["years"] else "N/A"
        print(f"  {dtype:25s}: {data['count']:2d} archivos, {data['total_rows']:>9,} filas, años {years_str}")
