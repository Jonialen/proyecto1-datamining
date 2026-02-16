# Proyecto 1: Análisis Exploratorio de Datos INE Guatemala

Sistema de procesamiento y análisis de estadísticas vitales (Defunciones, Nacimientos, Matrimonios, etc.) del Instituto Nacional de Estadística de Guatemala para el periodo 2012-2023.

## 🚀 Inicio Rápido

### Requisitos
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (Recomendado como gestor de paquetes)

### Instalación
```bash
# Instalar dependencias
uv sync
```

### Ejecución del Pipeline
El sistema utiliza un motor híbrido: **DuckDB** para consultas analíticas rápidas y **Polars** para transformación de datos.

```bash
# 1. Escanear datos crudos (.sav) en la carpeta rowdata/
uv run ine scan

# 2. Ejecutar ETL (Limpia, armoniza y carga a DuckDB/Parquet)
uv run ine etl

# 3. Ver estadísticas de carga
uv run ine info
```

## 📊 Análisis Exploratorio (EDA)
El análisis principal se encuentra en `notebooks/01_eda.ipynb`. Cubre:
- Limpieza y armonización de esquemas entre años.
- Análisis descriptivo de variables numéricas y categóricas.
- **5 Hipótesis de investigación** validadas con datos.
- Clustering de perfiles de mortalidad (K-Means + Silueta).

Para abrir el notebook:
```bash
uv run jupyter lab
```

## 📁 Estructura del Proyecto
- `src/`: Código fuente del pipeline y API de consulta.
  - `pipeline/`: Lógica de armonización, labels SPSS y escritura.
  - `db/`: Conectores a DuckDB y MongoDB.
  - `query/`: API simplificada para notebooks.
- `data/`: Almacenamiento en formato Parquet (eficiente).
- `docs/`: Documentación formal, roadmap e informes de entrega.
- `notebooks/`: Análisis detallados e investigación.

## 📝 Informe de Entrega
El marco teórico, situación problemática y objetivos se detallan en: [docs/informe_fase1.md](docs/informe_fase1.md)

## 🛠️ Tecnologías
- **Pandas**: Manipulacion de datos y carga utlizado en su mayoria para compatividad con ipynb
- **Polars**: Manipulación de datos de alto rendimiento.
- **DuckDB**: Base de datos analítica OLAP.
- **Seaborn/Matplotlib**: Visualización de datos.
- **Scikit-Learn**: Clustering y reducción de dimensionalidad.
