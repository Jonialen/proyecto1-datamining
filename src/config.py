"""Configuración centralizada con pydantic-settings. Override via env vars con prefijo INE_."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "INE_"}

    # Paths
    project_root: Path = Path(__file__).parent.parent
    rawdata_dir: Path = Path(__file__).parent.parent / "rowdata"
    data_dir: Path = Path(__file__).parent.parent / "data"
    parquet_dir: Path = Path(__file__).parent.parent / "data" / "parquet"
    duckdb_path: Path = Path(__file__).parent.parent / "data" / "ine.duckdb"

    # MongoDB
    mongo_uri: str = "mongodb://admin:admin123@localhost:27017/"
    mongo_db: str = "ine_guatemala"

    # Pipeline
    batch_size: int = 5_000
    parquet_compression: str = "zstd"

    # Años disponibles (sin 2016 en ningún tipo, sin 2014 en def. fetales)
    available_years: list[int] = [2012, 2013, 2014, 2015, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

    # Dataset types válidos
    dataset_types: list[str] = ["nacimientos", "defunciones", "defunciones_fetales", "matrimonios", "divorcios"]


settings = Settings()
