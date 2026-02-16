"""CLI del pipeline INE Guatemala."""

from __future__ import annotations

import re
import time
from typing import Optional

import polars as pl
import pyreadstat
import typer

from src.config import settings

app = typer.Typer(name="ine", help="Pipeline de datos INE Guatemala")


def _normalize_column_name(col: str) -> str:
    col = col.lower().strip()
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[^a-z0-9_]", "", col)
    return col


@app.command()
def scan():
    """Escanea y clasifica los archivos .sav en rowdata/."""
    from src.pipeline.scanner import get_summary, get_unique, scan_all

    print("=== Escaneando archivos .sav ===\n")
    files = scan_all(verbose=True)
    unique = get_unique(files)

    print(f"\nTotal: {len(files)} archivos, {len(unique)} únicos")

    summary = get_summary(files)
    print("\nResumen por tipo:")
    for dtype, data in sorted(summary.items()):
        years = sorted(data["years"])
        yr_str = f"{min(years)}-{max(years)}" if years else "N/A"
        print(f"  {dtype:25s}: {data['count']:2d} archivos, {data['total_rows']:>9,} filas, años {yr_str}")


@app.command()
def etl(
    dataset_type: Optional[str] = typer.Argument(None, help="Tipo de dataset (o todos si se omite)"),
    skip_mongo: bool = typer.Option(False, "--skip-mongo", help="No cargar a MongoDB"),
    reload: bool = typer.Option(False, "--reload", help="Eliminar datos existentes antes de cargar"),
):
    """Ejecuta el pipeline ETL: .sav → Parquet + DuckDB + MongoDB."""
    from src.db import mongo
    from src.pipeline.harmonizer import harmonize
    from src.pipeline.labels import apply_labels
    from src.pipeline.scanner import get_unique, scan_all
    from src.pipeline.writer import write_all

    # Scan
    print("Escaneando archivos...")
    files = scan_all(verbose=False)
    unique = get_unique(files)
    loadable = [f for f in unique if f.dataset_type not in ("desconocido", "error")]

    if dataset_type:
        loadable = [f for f in loadable if f.dataset_type == dataset_type]
        if not loadable:
            typer.echo(f"No se encontraron archivos para '{dataset_type}'")
            raise typer.Exit(1)

    # Reload
    if reload and not skip_mongo:
        print("Eliminando colecciones MongoDB existentes...")
        mongo.drop_all()

    # Process
    print(f"\nProcesando {len(loadable)} archivos...\n")
    stats: dict[str, dict] = {}
    t0 = time.time()

    for fi in loadable:
        dtype = fi.dataset_type
        print(f"  [{dtype}] {fi.filename[:50]}... (año {fi.year or '?'})")

        try:
            # Read .sav
            df_pd, meta = pyreadstat.read_sav(str(fi.filepath))

            # Convert to Polars
            df = pl.from_pandas(df_pd)

            # Apply labels
            df = apply_labels(df, meta.variable_value_labels)

            # Build column labels mapping (before harmonization renames)
            col_labels = {}
            for col in df.columns:
                norm = _normalize_column_name(col)
                col_labels[norm] = meta.column_names_to_labels.get(col, col)

            # Harmonize
            df = harmonize(df, dtype, fi.year, fi.filename)

            # Write
            write_all(df, dtype, fi.year, col_labels, skip_mongo=skip_mongo)

            if dtype not in stats:
                stats[dtype] = {"files": 0, "rows": 0}
            stats[dtype]["files"] += 1
            stats[dtype]["rows"] += df.height

        except Exception as e:
            print(f"    ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Pipeline completado en {elapsed:.1f}s\n")
    for dtype, data in sorted(stats.items()):
        print(f"  {dtype:25s}: {data['files']:2d} archivos, {data['rows']:>9,} filas")
    total_rows = sum(d["rows"] for d in stats.values())
    print(f"\n  Total: {total_rows:,} filas")


@app.command(name="query")
def run_query(sql: str = typer.Argument(..., help="Query SQL contra DuckDB")):
    """Ejecuta una query SQL contra DuckDB."""
    from src.db import duckdb as duck

    duck.register_views()
    result = duck.query(sql)
    print(result.to_string())


@app.command()
def info():
    """Muestra estadísticas de los datos cargados."""
    from src.db import duckdb as duck

    duck.register_views()
    counts = duck.table_counts()

    if not counts:
        print("No hay datos cargados. Ejecuta 'ine etl' primero.")
        raise typer.Exit(1)

    print("=== Datos en DuckDB ===\n")
    total = 0
    for dtype, count in sorted(counts.items()):
        print(f"  {dtype:25s}: {count:>9,} filas")
        total += count
    print(f"\n  Total: {total:,} filas")

    # Parquet file sizes
    print("\n=== Archivos Parquet ===\n")
    for dtype in settings.dataset_types:
        parquet_dir = settings.parquet_dir / dtype
        if parquet_dir.exists():
            files = list(parquet_dir.glob("*.parquet"))
            total_size = sum(f.stat().st_size for f in files)
            print(f"  {dtype:25s}: {len(files):2d} archivos, {total_size / 1024**2:.1f} MB")


if __name__ == "__main__":
    app()
