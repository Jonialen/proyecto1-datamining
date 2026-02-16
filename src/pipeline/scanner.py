"""Escaneo y clasificación de archivos .sav del INE Guatemala."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pyreadstat

from src.config import settings
from src.schemas import DATASET_SIGNATURES


@dataclass
class FileInfo:
    filename: str
    filepath: Path
    dataset_type: str
    year: int | None
    nrows: int
    ncols: int
    columns: list[str]
    column_labels: dict[str, str] = field(default_factory=dict)
    is_duplicate: bool = False
    error: str | None = None


def _extract_year_from_filename(filename: str) -> int | None:
    match = re.match(r"^(\d{4})", filename)
    if match:
        year = int(match.group(1))
        if 2009 <= year <= 2030:
            return year
    return None


def _classify_dataset(columns: list[str]) -> str:
    cols_lower = {c.lower() for c in columns}
    for dataset_type, signature_sets in DATASET_SIGNATURES.items():
        for signature in signature_sets:
            if all(kw in cols_lower for kw in signature):
                if dataset_type == "matrimonios" and "clauni" not in cols_lower:
                    continue
                return dataset_type
    return "desconocido"


def _extract_year_from_data(filepath: Path) -> int | None:
    try:
        df, _ = pyreadstat.read_sav(str(filepath), row_limit=5)
        cols_lower = {c.lower(): c for c in df.columns}
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


def scan_file(filepath: Path) -> FileInfo:
    """Escanea un archivo .sav y retorna su metadata."""
    filename = filepath.name
    is_duplicate = "(1)" in filename

    try:
        _, meta = pyreadstat.read_sav(str(filepath), metadataonly=True)
        columns = list(meta.column_names)
        dataset_type = _classify_dataset(columns)

        year = _extract_year_from_filename(filename)
        if year is None:
            year = _extract_year_from_data(filepath)

        return FileInfo(
            filename=filename,
            filepath=filepath,
            dataset_type=dataset_type,
            year=year,
            nrows=meta.number_rows,
            ncols=meta.number_columns,
            columns=columns,
            column_labels=meta.column_names_to_labels,
            is_duplicate=is_duplicate,
        )
    except Exception as e:
        return FileInfo(
            filename=filename,
            filepath=filepath,
            dataset_type="error",
            year=None,
            nrows=0,
            ncols=0,
            columns=[],
            is_duplicate=is_duplicate,
            error=str(e),
        )


def scan_all(verbose: bool = True) -> list[FileInfo]:
    """Escanea todos los .sav en rowdata/ y retorna info de cada uno."""
    sav_files = sorted(settings.rawdata_dir.glob("*.sav"))
    results = []

    for filepath in sav_files:
        info = scan_file(filepath)
        results.append(info)
        if verbose:
            dup_tag = " [DUP]" if info.is_duplicate else ""
            print(
                f"  {info.dataset_type:25s} | {info.year or '????':>4} | "
                f"{info.nrows:>7,} rows | {info.ncols:>3} cols | "
                f"{info.filename[:40]}{dup_tag}"
            )

    return results


def get_unique(file_info: list[FileInfo]) -> list[FileInfo]:
    """Filtra archivos duplicados (sin '(1)')."""
    return [f for f in file_info if not f.is_duplicate]


def get_summary(file_info: list[FileInfo]) -> dict[str, dict]:
    """Genera resumen por tipo de dataset."""
    unique = get_unique(file_info)
    summary: dict[str, dict] = {}
    for f in unique:
        dtype = f.dataset_type
        if dtype not in summary:
            summary[dtype] = {"count": 0, "years": [], "total_rows": 0}
        summary[dtype]["count"] += 1
        if f.year:
            summary[dtype]["years"].append(f.year)
        summary[dtype]["total_rows"] += f.nrows
    return summary
