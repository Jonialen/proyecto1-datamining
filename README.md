# Proyecto 1: Analisis Exploratorio de Datos - INE Guatemala

Sistema de procesamiento y analisis de estadisticas vitales del Instituto Nacional de Estadistica (INE) de Guatemala para el periodo 2012-2023. Incluye un pipeline ETL completo, una base de datos analitica y un analisis exploratorio formal con pruebas de hipotesis y clustering.

## Inicio Rapido

### Requisitos

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (gestor de paquetes)
- Docker (opcional, para MongoDB)

### Instalacion

```bash
# Clonar repositorio
git clone git@github.com:Jonialen/proyecto1-datamining.git
cd proyecto1-datamining

# Instalar dependencias
uv sync

# (Opcional) Levantar MongoDB
docker compose up -d
```

### Datos Crudos

Descargar los archivos `.sav` desde el [portal del INE](https://www.ine.gob.gt/ine/vitales/) y colocarlos en la carpeta `rowdata/`. El pipeline detecta automaticamente el tipo de dataset y el ano de cada archivo.

### Ejecucion del Pipeline

```bash
# 1. Escanear y clasificar archivos .sav
uv run ine scan

# 2. Ejecutar ETL completo (Parquet + DuckDB + MongoDB)
uv run ine etl

# 2b. Solo Parquet + DuckDB (sin MongoDB)
uv run ine etl --skip-mongo

# 3. Procesar un solo tipo de dataset
uv run ine etl defunciones

# 4. Ver estadisticas de carga
uv run ine info

# 5. Ejecutar consultas SQL ad-hoc
uv run ine query "SELECT _year, COUNT(*) as n FROM defunciones GROUP BY _year ORDER BY _year"
```

### Notebook

```bash
uv run jupyter lab
# Abrir notebooks/01_eda.ipynb
```

## Estructura del Proyecto

```
proyecto1/
├── pyproject.toml              # Dependencias y metadata (uv/hatchling)
├── README.md
├── docker-compose.yml          # MongoDB + Mongo Express
├── rowdata/                    # Archivos .sav crudos del INE (gitignored)
├── data/                       # Artefactos generados (gitignored)
│   ├── parquet/                # Formato columnar por tipo/ano
│   │   ├── defunciones/
│   │   ├── nacimientos/
│   │   ├── matrimonios/
│   │   ├── divorcios/
│   │   └── defunciones_fetales/
│   └── ine.duckdb             # Base de datos analitica
├── src/                        # Codigo fuente del pipeline
│   ├── config.py              # Configuracion centralizada (pydantic-settings)
│   ├── schemas.py             # Esquemas canonicos, alias y normalizacion
│   ├── cli.py                 # CLI con Typer (scan, etl, query, info)
│   ├── pipeline/
│   │   ├── scanner.py         # Descubrimiento y clasificacion de .sav
│   │   ├── harmonizer.py      # Armonizacion de esquemas entre anos
│   │   ├── labels.py          # Aplicacion inteligente de etiquetas SPSS
│   │   └── writer.py          # Escritura a Parquet + DuckDB + MongoDB
│   ├── db/
│   │   ├── duckdb.py          # Conexion singleton, vistas, queries SQL
│   │   └── mongo.py           # Bulk insert, metadata, estadisticas
│   └── query/
│       └── api.py             # API de alto nivel para notebooks
├── notebooks/
│   └── 01_eda.ipynb           # Analisis exploratorio principal (66 celdas)
├── scripts/
│   └── analisis_eda.py        # Script Python standalone (extraido del notebook)
└── docs/
    ├── informe_eda.md         # Informe formal completo (markdown)
    ├── informe_eda.pdf        # Informe formal (PDF, sin codigo)
    ├── 01_eda.html            # Notebook exportado (HTML, sin codigo)
    ├── guia_proyecto.md       # Guia detallada del proyecto
    ├── instructions.md        # Instrucciones del curso
    └── roadmap.md             # Roadmap de desarrollo
```

## Flujo de Datos

```
rowdata/*.sav
      |
  [scanner.py]         pyreadstat: metadata-only scan, clasifica tipo + ano
      |
  [pyreadstat]         Lee archivo completo -> pd.DataFrame + meta
      |
  [labels.py]          >50% con label -> categorica; <50% -> sentinel -> null
      |
  [harmonizer.py]      Rename aliases -> canonico, agregar cols faltantes,
      |                normalizar valores, agregar _year y _source_file
      |
  Polars DataFrame armonizado
      |
      +---> [Parquet]    data/parquet/{tipo}/{ano}.parquet (zstd)
      |
      +---> [DuckDB]     CREATE VIEW {tipo} FROM read_parquet(glob)
      |
      +---> [MongoDB]    ine_guatemala.{tipo} (bulk insert, batches de 5K)
                |
          [query/api.py]
                |
          notebooks/01_eda.ipynb
          (pandas + seaborn + scipy + sklearn)
```

## Analisis Exploratorio

El notebook `01_eda.ipynb` analiza 950,793 registros de defunciones y cubre:

### Contenido
1. **Descripcion de datos**: 28 variables (4 numericas, 24 categoricas), limpieza documentada
2. **Variables numericas**: Descriptivas, histogramas, boxplots, test de normalidad (Shapiro-Wilk)
3. **Variables categoricas**: Tablas de frecuencia, graficos de barras, proporciones
4. **Correlaciones y cruces**: Matriz de correlacion, scatter plots, tablas cruzadas
5. **Analisis temporal**: Tendencias anuales y estacionalidad mensual
6. **Outliers**: Deteccion con metodo IQR (1.5x)
7. **Clustering**: K-Means con K=5 (silueta=0.1499), interpretacion de perfiles
8. **5 hipotesis** con H0/H1, prueba estadistica (alpha=0.05) y discusion

### Hipotesis Validadas
| # | Hipotesis | Prueba | Resultado |
|---|-----------|--------|-----------|
| H1 | COVID-19 afecto mas a departamentos sin asistencia | Chi-cuadrado | Confirmada (p<0.001) |
| H2 | Mayor escolaridad = mayor edad de fallecimiento | Kruskal-Wallis | Parcial (paradoja educativa) |
| H3 | Mortalidad infantil disminuyo 2012-2021 | Chi-cuadrado | Confirmada (9.8% -> 5.3%) |
| H4 | Muertes en via publica = hombres jovenes | Test z | Confirmada (59.5% > 50%) |
| H5 | Brecha rural-urbana en asistencia medica | Chi-cuadrado | Confirmada (3.7x) |

### Clusters Identificados (K=5)
| Cluster | Nombre | Edad Media | % Total |
|---------|--------|-----------|---------|
| 0 | Mortalidad Infantil/Neonatal | 6.1 | 8.2% |
| 1 | Adultos Con Asistencia Medica | 60.6 | 39.7% |
| 2 | Adultos Mayores Rurales (Sin Asistencia) | 63.7 | 40.3% |
| 3 | Jovenes - Muerte Violenta | 38.5 | 8.6% |
| 4 | Adultos No Indigenas (Mixto) | 58.5 | 3.3% |

## Tecnologias

| Componente | Tecnologia | Proposito |
|------------|-----------|-----------|
| ETL | Polars | Transformacion de datos de alto rendimiento |
| Storage | Parquet (zstd) | Almacenamiento columnar eficiente |
| Analitica | DuckDB | Base de datos OLAP in-process |
| Documentos | MongoDB | Almacenamiento flexible de registros |
| Notebooks | Pandas + Seaborn | Analisis y visualizacion |
| Estadistica | SciPy | Pruebas de hipotesis (chi2, Kruskal, z-test) |
| Clustering | Scikit-Learn | K-Means + Silueta + PCA |
| CLI | Typer | Interfaz de linea de comandos |
| Config | Pydantic Settings | Configuracion con variables de entorno |
| Lectura SPSS | pyreadstat | Lectura de archivos .sav con metadatos |

## Entregables

- **Informe formal (PDF)**: `docs/informe_eda.pdf`
- **Informe (Markdown)**: `docs/informe_eda.md`
- **Notebook (HTML)**: `docs/01_eda.html`
- **Script Python**: `scripts/analisis_eda.py`
- **Guia del proyecto**: `docs/guia_proyecto.md`

## Autor

Jonathan Diaz - Universidad del Valle de Guatemala
Mineria de Datos - Semestre 7, Febrero 2026
