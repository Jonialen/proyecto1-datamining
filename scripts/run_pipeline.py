#!/usr/bin/env python3
"""
Script principal del pipeline: Identificar -> Cargar a MongoDB.
Uso: python scripts/run_pipeline.py [--reload]
"""
import sys
import os

# Agregar raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.identify import scan_all_files, get_unique_files, get_summary
from src.loader import load_all, drop_all_collections
from src.export import collection_stats


def main():
    reload = "--reload" in sys.argv

    # === Fase 1: Identificación ===
    print("=" * 60)
    print("FASE 1: IDENTIFICACIÓN DE ARCHIVOS")
    print("=" * 60)

    all_files = scan_all_files()
    unique = get_unique_files(all_files)

    print(f"\nTotal archivos encontrados: {len(all_files)}")
    print(f"Archivos únicos: {len(unique)}")
    print(f"Duplicados excluidos: {len(all_files) - len(unique)}")

    summary = get_summary(all_files)
    print("\nResumen por tipo de dataset:")
    for dtype, data in sorted(summary.items()):
        years_str = f"{min(data['years'])}-{max(data['years'])}" if data["years"] else "N/A"
        print(f"  {dtype:25s}: {data['count']:2d} archivos, {data['total_rows']:>9,} filas, años {years_str}")

    # === Fase 2: Carga a MongoDB ===
    print("\n" + "=" * 60)
    print("FASE 2: CARGA A MONGODB")
    print("=" * 60)

    if reload:
        print("\n--reload flag: eliminando colecciones existentes...")
        drop_all_collections()

    stats = load_all()

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)

    total_docs = 0
    for dtype, data in sorted(stats.items()):
        print(f"  {dtype:25s}: {data['files']:2d} archivos -> {data['docs']:>9,} documentos")
        total_docs += data["docs"]
    print(f"\n  Total documentos cargados: {total_docs:,}")

    # Verificar colecciones
    print("\n=== Verificación de colecciones en MongoDB ===")
    try:
        cs = collection_stats()
        for name, data in cs.items():
            print(f"  {name:25s}: {data['count']:>9,} docs")
    except Exception as e:
        print(f"  Error verificando: {e}")

    print("\nPipeline completado. Puedes verificar en Mongo Express: http://localhost:8081")


if __name__ == "__main__":
    main()
