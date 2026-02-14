# Roadmap - Proyecto 1: Análisis Exploratorio INE Guatemala

## Estado actual

Lo que YA está hecho:
- Pipeline de datos (identify, loader, export) funcionando
- MongoDB con los 5 datasets cargados (defunciones, nacimientos, matrimonios, divorcios, def. fetales)
- Notebook `01_eda.ipynb` con:
  - Descripción general del dataset (variables, tipos, observaciones)
  - Estadísticas descriptivas de variables numéricas
  - Test de normalidad (Shapiro-Wilk)
  - Tablas de frecuencia y gráficos de barras para categóricas
  - Histogramas, boxplots
  - Matriz de correlación, scatter plots, crosstabs
  - Análisis temporal (por año, por mes)
  - Detección de outliers (IQR)
  - Clustering (KMeans + codo + silueta + PCA)
  - Perfiles de clusters (categóricos y numéricos)

---

## Fase 1: Completar el análisis en el notebook

### 1.1 Limpieza de datos documentada
**Archivo:** `notebooks/01_eda.ipynb` (agregar celda después de cargar datos)
- [ ] Documentar las categorías duplicadas encontradas y cómo se normalizaron (Casado/Casado(a), Garifuna/Garífuna, etc.)
- [ ] Documentar los valores nulos: qué columnas tienen >50% nulos (`areag` 50%, `getdif` 92%, `caudefdescrip` 92%) y decisión tomada (excluirlas o no)
- [ ] Documentar la conversión de `edadif` de string a numérica (valores centinela 999 → NaN)
- [ ] Documentar la unión de archivos de distintos años (columnas varían entre formatos)

### 1.2 Cinco preguntas/hipótesis (35 pts del análisis)
**Archivo:** `notebooks/01_eda.ipynb` (nueva sección 8)

Cada hipótesis necesita: supuesto inicial → análisis con código/tablas/gráficos → discusión si se confirma o refuta.

- [ ] **H1: "El COVID aumentó desproporcionadamente las defunciones en departamentos con menor acceso a salud"**
  - Cruzar: `caudef` (U071) × `depreg` × `_year` × `asist`
  - Gráfico: heatmap de tasa COVID por departamento, barras de asistencia médica pre/post COVID

- [ ] **H2: "Las personas sin escolaridad mueren más jóvenes que las personas con educación"**
  - Cruzar: `escodif` × `edadif` × `sexo`
  - Gráfico: boxplot de edad por escolaridad, separado por sexo

- [ ] **H3: "La mortalidad infantil ha disminuido uniformemente en todos los departamentos"**
  - Cruzar: `perdif` × `depreg` × `_year`
  - Gráfico: líneas de mortalidad infantil por departamento a lo largo del tiempo

- [ ] **H4: "Las muertes en vía pública afectan principalmente a hombres jóvenes"**
  - Cruzar: `ocur` × `sexo` × `edadif`
  - Gráfico: histograma de edad en muertes vía pública por sexo, proporción temporal

- [ ] **H5: "La proporción de defunciones sin asistencia médica es mayor en zonas rurales"**
  - Cruzar: `asist` × `areag` × `depreg`
  - Gráfico: barras apiladas de asistencia por área geográfica y departamento

### 1.3 Clustering: interpretar y nombrar grupos
**Archivo:** `notebooks/01_eda.ipynb` (sección 7, después del clustering)
- [ ] Verificar que los clusters ya no agrupan por mes (después del fix)
- [ ] Analizar el perfil de cada cluster con las variables categóricas y numéricas
- [ ] Asignar un nombre descriptivo a cada cluster basado en sus características (ej: "Adultos mayores sin escolaridad fallecidos en domicilio", "Jóvenes con muerte violenta en vía pública")
- [ ] Verificar calidad con el score de silueta y discutirlo

---

## Fase 2: Redacción del marco del proyecto

### 2.1 Situación problemática (10 pts)
**Archivo:** `notebooks/01_eda.ipynb` (nueva sección al inicio, después de la intro) o Google Docs

Tema sugerido basado en los datos:
> "Desigualdad en acceso a servicios de salud y su impacto en la mortalidad en Guatemala (2012-2022)"

Puntos a cubrir:
- [ ] ~50% de defunciones ocurren sin asistencia médica
- [ ] 62-65% mueren en domicilio (no en hospital)
- [ ] Impacto diferencial del COVID en 2021-2022 (+43% defunciones)
- [ ] Brecha educativa: personas sin escolaridad son 52% de defunciones
- [ ] Mortalidad infantil bajando pero posiblemente desigual por región

### 2.2 Problema científico (10 pts)
- [ ] Enunciar el problema como pregunta científica
  - Ejemplo: "¿En qué medida el acceso a servicios de salud y el nivel educativo determinan las condiciones de mortalidad en Guatemala, y cómo estas desigualdades se distribuyeron geográfica y temporalmente entre 2012 y 2022?"

### 2.3 Objetivos (10 pts)
- [ ] 1 objetivo general
  - Ejemplo: "Analizar los factores sociodemográficos y geográficos asociados a las condiciones de mortalidad en Guatemala durante el período 2012-2022"
- [ ] 2+ objetivos específicos medibles
  - Ejemplo 1: "Cuantificar la relación entre nivel educativo y edad de defunción, segmentado por sexo y departamento"
  - Ejemplo 2: "Evaluar el impacto del COVID-19 en la mortalidad por departamento, comparando tasas de asistencia médica pre y post pandemia"

---

## Fase 3: Hallazgos y conclusiones (20 pts)

### 3.1 Resumen de hallazgos
**Archivo:** `notebooks/01_eda.ipynb` (sección final)
- [ ] Resumir los hallazgos principales del EDA
- [ ] Resumir qué hipótesis se confirmaron y cuáles se refutaron
- [ ] Destacar hallazgos inesperados

### 3.2 Nombres de clusters
- [ ] Tabla con: ID cluster, nombre, % del total, características principales

### 3.3 Plan de siguientes pasos
- [ ] Qué análisis adicionales harían (modelos predictivos, series temporales, etc.)
- [ ] Qué datos adicionales serían útiles (censo, datos de hospitales, etc.)
- [ ] Limitaciones identificadas en los datos

---

## Fase 4: Entregables

### 4.1 Informe en Google Docs
- [ ] Crear Google Doc con el informe completo
- [ ] Verificar que el historial de cambios esté visible
- [ ] Formato formal: sin código, solo texto + tablas + gráficos exportados

### 4.2 PDF
- [ ] Exportar el Google Doc a PDF
- [ ] Verificar que no tenga código visible

### 4.3 Script Python
- [ ] Verificar que `notebooks/01_eda.ipynb` + `src/` estén limpios y ejecutables
- [ ] Agregar comentarios explicativos donde falten

### 4.4 Repositorio GitHub
- [ ] Inicializar git en el proyecto
- [ ] Crear `.gitignore` (excluir rowdata/, .venv/, mongo_data/, __pycache__/)
- [ ] Commit inicial con todo el código
- [ ] Push a GitHub
- [ ] Incluir link en el informe

---

## Orden de ejecución sugerido

```
1. Fase 1.1  → Documentar limpieza (rápido, ~30 min)
2. Fase 1.2  → 5 hipótesis con análisis (lo más pesado, ~3-4 hrs)
3. Fase 1.3  → Interpretar clusters (~1 hr)
4. Fase 2    → Redactar problema/objetivos (~1 hr)
5. Fase 3    → Hallazgos y conclusiones (~1 hr)
6. Fase 4.4  → Subir a GitHub (~15 min)
7. Fase 4.1  → Pasar al Google Doc (~2 hrs)
8. Fase 4.2  → Exportar PDF (~10 min)
```
