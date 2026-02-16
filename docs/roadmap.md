# Roadmap - Proyecto 1: Análisis Exploratorio INE Guatemala

## Lo que YA tienes (infraestructura + EDA base)

### Pipeline profesional (listo, funcional)
- [x] `uv` como package manager con `pyproject.toml`
- [x] Pipeline ETL: `.sav` → Polars → Parquet → DuckDB views
- [x] Armonización de esquemas entre años (aliases, columnas faltantes)
- [x] Labels SPSS aplicados inteligentemente (categóricas vs sentinelas)
- [x] CLI: `uv run ine scan`, `uv run ine etl`, `uv run ine query`, `uv run ine info`
- [x] 55 archivos procesados: 6M+ registros en ~73 MB de parquet
- [x] DuckDB para queries analíticas, MongoDB opcional para documentos
- [x] API de alto nivel para notebooks: `load()`, `agg()`, `yearly_counts()`

### Notebook `01_eda.ipynb` (EDA genérico listo)
- [x] Sección 1: Descripción general (variables, tipos, observaciones, conteos)
- [x] Sección 2: Variables numéricas (estadísticas descriptivas, histogramas, boxplots, normalidad Shapiro-Wilk)
- [x] Sección 3: Variables categóricas (tablas de frecuencia, gráficos de barras)
- [x] Sección 4: Relaciones (correlación, scatter plots, crosstabs)
- [x] Sección 5: Análisis temporal (por año, por mes)
- [x] Sección 6: Outliers (IQR)
- [x] Sección 7: Clustering (KMeans + codo + silueta + PCA + perfiles)

---

## Lo que FALTA (contenido de entrega)

### Fase 1: Completar análisis en el notebook (~5 hrs)

#### 1.1 Documentar limpieza de datos (~30 min)
Agregar celda después de cargar datos explicando:
- [ ] Categorías normalizadas: Casado(a)→Casado, Garífuna→Garifuna, etc.
- [ ] Columnas con alias entre años: `getdif`→`puedif`, `ocuhom`→`ciuohom`, etc.
- [ ] Valores sentinela (999→NaN) vs labels categóricos (decisión >50%)
- [ ] Columnas faltantes por año (rellenadas con null)

#### 1.2 Cinco hipótesis con análisis (lo más pesado, ~3-4 hrs) — 35 pts
Cada una necesita: **supuesto → código/tablas/gráficos → discusión confirmando/refutando**

- [ ] **H1: "El COVID aumentó desproporcionadamente las defunciones en departamentos con menor acceso a salud"**
  - Cruzar: `caudef` × `depocu` × `_year` × `asist`
  - Gráfico: heatmap tasa COVID por departamento, barras asistencia pre/post COVID

- [ ] **H2: "Las personas sin escolaridad mueren más jóvenes que las con educación"**
  - Cruzar: `escodif` × `edadif` × `sexo`
  - Gráfico: boxplot edad por escolaridad, separado por sexo

- [ ] **H3: "La mortalidad infantil disminuyó uniformemente en todos los departamentos"**
  - Cruzar: `perdif` × `depocu` × `_year`
  - Gráfico: líneas de mortalidad infantil por departamento en el tiempo

- [ ] **H4: "Las muertes en vía pública afectan principalmente a hombres jóvenes"**
  - Cruzar: `ocur` × `sexo` × `edadif`
  - Gráfico: histograma edad en muertes vía pública por sexo

- [ ] **H5: "La proporción de defunciones sin asistencia médica es mayor en zonas rurales"**
  - Cruzar: `asist` × `areag` × `depocu`
  - Gráfico: barras apiladas asistencia por área geográfica

> **Nota:** Las hipótesis usan `defunciones`, no `nacimientos`. Cambiar `DATASET = 'defunciones'` o cargar ambos.

#### 1.3 Interpretar y nombrar clusters (~1 hr)
- [ ] Analizar perfiles categóricos y numéricos de cada cluster
- [ ] Asignar nombre descriptivo (ej: "Adultos mayores fallecidos en domicilio sin asistencia")
- [ ] Tabla resumen: ID, nombre, %, características principales
- [ ] Discutir score de silueta

---

### Fase 2: Marco del proyecto (~1 hr) — 30 pts

#### 2.1 Situación problemática (10 pts)
- [ ] Describir el contexto: mortalidad, acceso a salud, brecha educativa en Guatemala
- [ ] Basarse en datos encontrados en el EDA (ej: "~50% mueren sin asistencia médica")

#### 2.2 Problema científico (10 pts)
- [ ] Formular como pregunta científica
  - Ej: "¿En qué medida el acceso a salud y nivel educativo determinan las condiciones de mortalidad en Guatemala entre 2012-2023?"

#### 2.3 Objetivos (10 pts)
- [ ] 1 objetivo general
- [ ] 2+ objetivos específicos medibles y alcanzables

---

### Fase 3: Hallazgos y conclusiones (~1 hr) — 20 pts

- [ ] Resumen de hallazgos del EDA
- [ ] Qué hipótesis se confirmaron/refutaron
- [ ] Nombres de clusters con justificación
- [ ] Plan de siguientes pasos (modelos predictivos, datos adicionales)
- [ ] Limitaciones identificadas

---

### Fase 4: Entregables

- [ ] **Google Doc**: informe formal (sin código), historial de cambios visible
- [ ] **PDF**: exportar Google Doc
- [ ] **Script/Notebook**: `01_eda.ipynb` limpio y ejecutable
- [ ] **GitHub**: push con `.gitignore`, link en el informe

---

## Cómo correr Jupyter

```bash
# Desde la raíz del proyecto:
uv run jupyter lab
```

Esto lanza Jupyter usando el venv de uv con todas las dependencias instaladas.

## Cómo correr el pipeline

```bash
uv run ine scan              # Ver archivos clasificados
uv run ine etl               # Pipeline completo (parquet + duckdb)
uv run ine etl --skip-mongo  # Sin MongoDB
uv run ine info              # Ver conteos
uv run ine query "SELECT _year, COUNT(*) FROM defunciones GROUP BY _year ORDER BY _year"
```

---

## Orden sugerido de ejecución

```
1. Fase 1.1  → Documentar limpieza           (~30 min)
2. Fase 1.2  → 5 hipótesis con análisis      (~3-4 hrs) ← lo más pesado
3. Fase 1.3  → Interpretar clusters          (~1 hr)
4. Fase 2    → Problema/objetivos            (~1 hr)
5. Fase 3    → Hallazgos y conclusiones      (~1 hr)
6. Fase 4    → Google Doc + PDF + GitHub     (~2-3 hrs)
```
