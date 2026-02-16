# Guia Detallada del Proyecto: INE Guatemala

Esta guia explica cada parte del proyecto a profundidad: por que se tomo cada decision, como funciona cada componente y la logica detras de cada parte del analisis. Esta pensada para que puedas responder cualquier pregunta sobre el proyecto con total confianza.

---

## Parte 1: El Pipeline ETL (Extraccion, Transformacion, Carga)

### 1.1 El Problema de los Datos Crudos

Los archivos del INE vienen en formato SPSS (`.sav`), un formato propietario de IBM que almacena datos tabulares junto con metadatos (etiquetas de variables, etiquetas de valores, tipos de datos). Cada ano tiene su propio archivo por tipo de dataset (nacimientos, defunciones, etc.), y entre anos hay inconsistencias:

- **Columnas cambian de nombre**: `getdif` en 2012 se llama `puedif` en 2018 (ambas significan "pueblo de pertenencia del difunto").
- **Valores cambian de formato**: En un ano dicen `Casado(a)`, en otro `CASADO`, en otro `casado`.
- **Columnas aparecen y desaparecen**: `areag` (area geografica) no existe en todos los anos.
- **Codigos numericos vs texto**: Algunos anos usan `1` para "Masculino", otros ya traen el texto.
- **Ano 2016 no disponible**: El INE no publico datos para la mayoria de datasets ese ano.

Sin armonizacion, no se pueden combinar los datos de multiples anos para hacer analisis de series temporales.

### 1.2 Arquitectura del Pipeline

El pipeline tiene 4 etapas secuenciales:

```
ESCANEO -> ETIQUETADO -> ARMONIZACION -> ESCRITURA
```

#### Etapa 1: Escaneo (scanner.py)

**Que hace**: Descubre archivos `.sav` en `rowdata/` y los clasifica automaticamente.

**Como clasifica**: Cada tipo de dataset tiene columnas unicas que lo identifican (firmas). Por ejemplo, `defunciones` siempre tiene las columnas `caudef`, `edadif`, `sexo`, `perdif`. Si un archivo tiene esas columnas, es de defunciones.

```python
# schemas.py - Firmas de identificacion
DATASET_SIGNATURES = {
    "defunciones": ["caudef", "edadif", "sexo", "perdif"],
    "nacimientos": ["edadm", "pesona", "tallona", "tippar"],
    # ...
}
```

**Como extrae el ano**: Primero intenta sacarlo del nombre del archivo con regex (busca 4 digitos como 2019). Si no encuentra, lee las primeras 5 filas del archivo y busca una columna `anio` o `ano`.

**Por que pyreadstat en modo metadata-only**: Para clasificar no necesita leer todos los datos (un archivo puede tener 500K filas). Solo lee los nombres de columnas y la cantidad de filas, lo cual es instantaneo.

#### Etapa 2: Etiquetado Inteligente (labels.py)

**El problema**: Los archivos SPSS almacenan codigos numericos internamente (1, 2, 3...) pero tienen "etiquetas de valor" que dicen que significa cada codigo (1="Masculino", 2="Femenino"). Pero no todos los codigos tienen etiqueta, y no todas las variables son categoricas.

**La regla del 50%**: Si mas del 50% de los valores de una columna tienen etiqueta, la columna es **categorica** y se reemplazan los codigos por las etiquetas. Si menos del 50% tienen etiqueta, la columna es **numerica** y los codigos con etiqueta son valores sentinela (como 999="Ignorado") que se convierten en NULL.

**Ejemplo**:
- Columna `sexo`: 100% de valores tienen etiqueta (1="Masculino", 2="Femenino") -> Se aplica como categorica.
- Columna `edadif` (edad): Solo el codigo 999 tiene etiqueta ("Ignorado") -> Se trata como numerica; 999 se convierte en NULL.

**Por que funciona**: En datos del INE, las variables categoricas siempre tienen etiquetas para todos sus valores posibles. Las variables numericas solo tienen etiquetas para sus codigos de "dato faltante" (9, 99, 999).

#### Etapa 3: Armonizacion (harmonizer.py)

**Que hace**: Transforma cada DataFrame a un esquema canonico uniforme. Despues de esta etapa, un archivo de 2012 y uno de 2023 tienen exactamente las mismas columnas, en el mismo orden, con los mismos nombres.

**Pasos en orden**:

1. **Lowercase**: `SEXO` -> `sexo`, `CauDef` -> `caudef`. Elimina problemas de mayusculas.

2. **Rename aliases**: Usa el mapa de aliases de `schemas.py`. Si una columna se llama `getdif`, la renombra a `puedif` (su nombre canonico).

3. **Agregar columnas faltantes**: Si el esquema canonico de defunciones tiene 28 columnas pero el archivo solo tiene 25, las 3 faltantes se agregan como columnas de puros NULL.

4. **Normalizar texto**: Dos fases:
   - Fase A: Limpiar espacios y convertir a Title Case (`GUATEMALA` -> `Guatemala`).
   - Fase B: Aplicar reglas de `VALUE_NORMALIZATION` (`Casado(A)` -> `Casado`, `320.0` -> `Guatemala`, `Garifuna` -> `Garifuna`).

5. **Agregar metadata**: Columnas `_year` (Int16) y `_source_file` (nombre del archivo original).

**Por que Title Case**: Es un compromiso. Los datos vienen en formatos mixtos (MAYUSCULAS, minusculas, Mixto). Title Case unifica todo: `GUATEMALA` y `guatemala` ambos se convierten en `Guatemala`. Luego la normalizacion fina corrige casos especiales (`Casado(A)` -> `Casado`).

**Por que Polars y no Pandas**: Polars es mas rapido para transformaciones columnares. En un dataset de 4M+ filas (nacimientos), Polars procesa en segundos lo que Pandas tardaria minutos. Ademas, Polars es lazy por defecto y optimiza el plan de ejecucion.

#### Etapa 4: Escritura (writer.py)

**Tres backends en paralelo**:

1. **Parquet** (`data/parquet/{tipo}/{ano}.parquet`): Formato columnar comprimido con zstd. Eficiente para lectura analitica porque solo lee las columnas que necesitas. Un archivo de 500K filas ocupa ~15MB en vez de ~200MB en CSV.

2. **DuckDB** (`data/ine.duckdb`): No copia los datos; crea VISTAS (views) sobre los archivos Parquet. La query `SELECT * FROM defunciones` internamente hace `read_parquet('data/parquet/defunciones/*.parquet')`. Esto significa que no hay duplicacion de datos.

3. **MongoDB** (opcional): Inserta los registros como documentos JSON en batches de 5,000. Util para aplicaciones web o APIs REST, pero no necesario para el analisis en notebook.

### 1.3 El API de Consulta (query/api.py)

**Por que existe**: El notebook no deberia preocuparse por rutas de archivos, conexiones a DuckDB, ni registrar vistas. El API encapsula todo eso.

```python
# En el notebook, asi de simple:
from src.query.api import load, agg, yearly_counts

# Cargar 150K registros de defunciones
df = load("defunciones", sample_n=150_000)

# Query SQL directa
yearly = agg("SELECT _year, COUNT(*) as n FROM defunciones GROUP BY _year")

# Conteos por ano (push-down a DuckDB)
counts = yearly_counts("defunciones")
```

**Push-down**: `yearly_counts()` ejecuta la agregacion directamente en DuckDB sobre los Parquet files, sin cargar los 950K registros en memoria. Solo devuelve 12 filas (una por ano).

**USING SAMPLE**: Cuando llamas `load("defunciones", sample_n=150_000)`, DuckDB toma una muestra aleatoria de 150K filas directamente al leer el Parquet. Esto significa que puedes trabajar con datasets de millones de filas sin agotar la RAM.

---

## Parte 2: Los Esquemas (schemas.py)

### 2.1 DATASET_SIGNATURES

Cada tipo de dataset tiene columnas unicas. Estas firmas permiten clasificar un archivo sin saber su nombre:

| Dataset | Columnas firma |
|---------|---------------|
| defunciones | caudef, edadif, sexo, perdif |
| nacimientos | edadm, pesona, tallona, tippar |
| matrimonios | ecoam, natm |
| divorcios | numhij |
| defunciones_fetales | pesofe, tallafe, edadm |

### 2.2 CANONICAL_SCHEMAS

Define las 28 columnas canonicas de defunciones (por ejemplo), con sus alias historicos:

```python
"defunciones": [
    {"canonical": "puedif", "aliases": ["getdif"], "description": "Pueblo de pertenencia del difunto"},
    {"canonical": "ciuohom", "aliases": ["ocuhom"], "description": "Municipio de ocurrencia del hecho"},
    # ...28 columnas total
]
```

Cuando el harmonizer encuentra `getdif`, sabe que debe renombrarla a `puedif` gracias a este mapa.

### 2.3 VALUE_NORMALIZATION

Reglas de limpieza de texto organizadas por columna:

- **Departamentos**: `Quiche` -> `Quiche` (con tilde), `Sacatepequez` -> `Sacatepequez`
- **Estado civil**: `Casado(a)` -> `Casado`, `Soltero(a)` -> `Soltero`
- **Codigos ISO de pais**: `320.0` -> `Guatemala`, `222.0` -> `El Salvador`, `484.0` -> `Mexico`
- **Etnia**: `Mestizo / Ladino` -> `Ladino/Mestizo`, `Garifuna` -> `Garifuna`, `Xinca` -> `Xinka`

**Por que se necesita esto**: Los datos del INE no son consistentes entre anos. En un ano pueden decir `Casado(a)` y en otro `Casado`. Sin normalizacion, al hacer `df['ecivil'].value_counts()` aparecerian como categorias separadas.

---

## Parte 3: El Analisis Exploratorio (01_eda.ipynb)

### 3.1 Setup y Carga (Celdas 1-9)

**Seed de reproducibilidad**: `RANDOM_SEED = 42` y `np.random.seed(42)`. Esto garantiza que cada vez que ejecutes el kernel, los resultados sean identicos. Sin seed, el muestreo de DuckDB y el K-Means darian resultados ligeramente diferentes cada vez.

**Orden determinista**: `df = df.sort_values('_year').reset_index(drop=True)`. DuckDB lee Parquet files en paralelo y puede devolver las filas en orden diferente cada vez. El sort asegura orden consistente.

**Carga completa**: Se cargan los 950,793 registros de defunciones (sin muestreo) porque el analisis necesita datos completos para las pruebas estadisticas. Con sample_n solo se obtendria una muestra que podria sesgar los resultados de chi-cuadrado.

**Clasificacion de variables**: Se separan automaticamente en `num_cols` (4 numericas: anioreg, diaocu, anioocu, edadif) y `cat_cols` (24 categoricas: sexo, areag, asist, etc.). Esto permite aplicar analisis diferentes a cada tipo.

### 3.2 Variables Numericas (Celdas 10-18)

#### Estadisticas Descriptivas

Solo hay 2 variables numericas significativas para analisis:
- `edadif` (edad del difunto): La variable mas importante del dataset.
- `diaocu` (dia de ocurrencia): Uniforme, sin patrones interesantes.

Las otras 2 (`anioreg`, `anioocu`) son anos, no tienen sentido analizarlas estadisticamente.

**Resultados clave de edadif**:
- Media: ~57 anos (promedio de edad al morir)
- Mediana: ~62 anos (el 50% muere antes de 62)
- Asimetria: -0.536 (sesgada a la izquierda = cola hacia edades jovenes)
- Interpretacion: La mayoria muere en edad avanzada, pero hay un pico en la infancia (mortalidad neonatal)

#### Histogramas

**Que muestran**: Distribucion bimodal de edadif:
- Pico 1 (edades 0-1): Mortalidad infantil/neonatal
- Valle (edades 5-30): Baja mortalidad en juventud
- Pico 2 (edades 60-85): Mortalidad por edad avanzada

**Por que es bimodal**: Es tipico de paises en desarrollo. En paises desarrollados, el pico infantil casi desaparece. En Guatemala persiste porque la mortalidad neonatal sigue siendo alta, especialmente en areas rurales.

#### Boxplots

**Que muestran**: Distribucion de cuartiles. No se detectaron outliers con IQR (1.5x) porque el rango de edad (0-110+) es biologicamente coherente. No hay "edades imposibles" en los datos.

#### Test de Normalidad (Shapiro-Wilk)

**Por que se hace**: Muchas pruebas estadisticas (como ANOVA) asumen que los datos siguen una distribucion normal. Si no la siguen, hay que usar alternativas no parametricas.

**Resultado**: edadif tiene W=0.9322, p=6.88e-43. El p-valor es basicamente 0, lo que **rechaza la normalidad**. Esto significa que debemos usar Kruskal-Wallis (no parametrica) en vez de ANOVA para comparar grupos.

**Nota sobre la muestra de 5,000**: Shapiro-Wilk tiene un limite de ~5,000 observaciones. Con mas, se vuelve inestable. Por eso se toma una submuestra aleatoria.

### 3.3 Variables Categoricas (Celdas 19-22)

#### Tablas de Frecuencia

Las mas relevantes:

| Variable | Categoria principal | % |
|----------|-------------------|---|
| sexo | Hombre | 54.4% |
| areag | Urbano | 54.5% |
| asist (asistencia) | Ninguna | 50.7% |
| escodif (escolaridad) | Ninguna | 52.2% |
| puedif (etnia) | Ladino/Mestizo | 47.2% |

**Hallazgo critico**: El 50.7% de las personas muere sin NINGUNA asistencia medica. Esto revela un problema estructural masivo en el sistema de salud.

**Nota sobre areag**: Tiene 50.3% de nulos porque no todos los anos registran esta variable. Cuando se analiza (excluyendo nulos), la distribucion es Urbano 54.5%, Rural 43.7%, Ignorado 1.9%.

### 3.4 Correlaciones y Cruces (Celdas 23-28)

**Por que la correlacion es limitada**: Solo hay 2 variables numericas significativas. La correlacion entre edadif y diaocu es cercana a 0 (no hay relacion entre la edad al morir y el dia del mes). El analisis mas rico viene de las tablas cruzadas entre categoricas.

**Tablas cruzadas relevantes**:
- sexo x asist: Los hombres tienen ligeramente mas acceso a asistencia medica que las mujeres.
- areag x asist: Brecha abismal. Rural: 85.6% sin asistencia. Urbano: 23.1% sin asistencia.
- escodif x areag: Personas sin escolaridad se concentran en area rural.

### 3.5 Analisis Temporal (Celdas 29-31)

**Tendencia anual**: Se observa un incremento gradual en defunciones registradas, con un pico notable en 2020-2021 (pandemia COVID-19).

**Estacionalidad mensual**: No hay patron estacional fuerte. Las muertes se distribuyen relativamente uniforme a lo largo del ano.

**Nota tecnica**: Los conteos anuales vienen de `yearly_counts()`, que hace push-down a DuckDB sobre todos los registros (no sobre la muestra). Esto garantiza que las cifras son exactas.

### 3.6 Outliers (Celdas 32-33)

**Metodo IQR**: Para cada variable numerica:
- Calcula Q1 (percentil 25) y Q3 (percentil 75)
- IQR = Q3 - Q1
- Limites: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
- Valores fuera de los limites = outliers

**Resultado para edadif**: Q1=35, Q3=78, IQR=43, limites=[-29.5, 142.5]. No hay outliers porque todas las edades de muerte (0-110) caen dentro de los limites. Esto tiene sentido: morir a cualquier edad es biologicamente posible.

---

## Parte 4: Las 5 Hipotesis

### 4.1 Estructura de cada hipotesis

Cada hipotesis sigue la misma estructura formal:

1. **Supuesto**: La creencia inicial que se quiere validar.
2. **H0 (hipotesis nula)**: Lo que asumimos como verdadero hasta que se demuestre lo contrario.
3. **H1 (hipotesis alternativa)**: Lo que queremos probar.
4. **Prueba estadistica**: El test que se aplica.
5. **Nivel de significancia**: alpha = 0.05 (5% de probabilidad de rechazar H0 cuando es verdadera).
6. **Resultado**: Estadistico de prueba + p-valor.
7. **Decision**: Si p < 0.05, se rechaza H0.
8. **Discusion**: Interpretacion del resultado en contexto.

### 4.2 H1: Impacto del COVID-19 por departamento

**Supuesto**: "Los departamentos con menos infraestructura de salud tuvieron mas muertes COVID sin asistencia medica."

**Datos usados**: 24,579 defunciones con codigo CIE-10 que empieza con `U07` (COVID-19). Se filtran con `WHERE caudef LIKE 'U07%'` directamente en DuckDB.

**Prueba**: Chi-cuadrado de independencia. Se construye una tabla de contingencia: departamento x tipo de asistencia (con/sin). H0 dice que la proporcion de muertes sin asistencia es la misma en todos los departamentos.

**Por que chi-cuadrado**: Es la prueba estandar para evaluar si dos variables categoricas son independientes. Si el resultado es significativo, la asistencia depende del departamento.

**Resultado**: chi2 = 3,322.04, gl = 21, p < 0.001. Se rechaza H0.

**Interpretacion**:
- Solola: 35.0% de muertes COVID sin asistencia (el mas alto)
- Guatemala capital: 0.3% sin asistencia (el mas bajo)
- Los departamentos del altiplano occidental (Solola, Huehuetenango, Totonicapan) absorbieron la pandemia con recursos minimos.

**Por que tiene sentido**: Estos departamentos tienen alta poblacion Maya, son rurales y tienen la menor densidad de hospitales. La infraestructura de salud se concentra en la capital.

### 4.3 H2: Escolaridad y edad de fallecimiento

**Supuesto**: "Hay una brecha de al menos 10 anos en la mediana de muerte entre personas sin escolaridad y con nivel universitario."

**Prueba**: Kruskal-Wallis H (no parametrica). Se usa esta en vez de ANOVA porque Shapiro-Wilk demostro que edadif no es normal. Kruskal-Wallis compara las medianas de multiples grupos.

**Resultado**: H = 18,992.25, p < 0.001. Hay diferencias significativas.

**Pero la sorpresa**: La relacion NO es lineal como se esperaba:

| Nivel | Mediana |
|-------|---------|
| Post Grado | 70 |
| Ninguna | 66 |
| Primaria | 62 |
| Universitario | 62 |
| Diversificado | 51 |
| Basico | 39 |

Las personas **sin escolaridad** viven MAS que las que tienen educacion basica (66 vs 39 anos). Esto es una paradoja.

**Explicacion de la paradoja**: Es un efecto de composicion (Simpson's paradox):
- "Basico" incluye muchos hombres jovenes que murieron por violencia (homicidios, accidentes). Estos jovenes tenian escolaridad basica y murieron a los 20-30 anos, bajando la mediana drasticamente.
- "Ninguna" incluye mayoritariamente adultos mayores rurales. Estas personas nunca fueron a la escuela pero vivieron hasta los 60-80 anos porque vivian en comunidades rurales alejadas de la violencia urbana.

**Leccion**: La correlacion escolaridad-longevidad existe pero esta confundida por la mortalidad violenta en jovenes. Para aislar el efecto real de la educacion, habria que controlar por causa de muerte.

### 4.4 H3: Mortalidad infantil en descenso

**Supuesto**: "La mortalidad infantil ha disminuido entre 2012 y 2021."

**Definicion de mortalidad infantil**: Menores de 1 ano. En los datos, se identifican por `perdif IN ('Menos De Un Mes', '1 A 11 Meses')`.

**Prueba**: Chi-cuadrado de proporciones. Se comparan dos proporciones:
- 2012: 7,121 muertes infantiles / 72,657 totales = 9.80%
- 2021: 5,075 muertes infantiles / 96,001 totales = 5.29%

Se construye una tabla 2x2 (ano x infantil/no-infantil) y se aplica chi-cuadrado.

**Resultado**: chi2 = 1,255.77, p = 4.63e-275. Reduccion confirmada.

**Interpretacion**: La reduccion de 9.80% a 5.29% es un logro real de salud publica. Sin embargo, las disparidades departamentales persisten: departamentos como Alta Verapaz y Huehuetenango concentran la mayor parte de las muertes infantiles.

### 4.5 H4: Muertes en via publica y hombres jovenes

**Supuesto**: "Mas del 70% de las muertes en via publica son hombres entre 15 y 40 anos."

**Datos**: 25,064 defunciones donde el lugar de ocurrencia contiene "Publica" (filtro: `ocur.str.contains('Publica')`).

**Prueba**: Test z de proporcion (una cola). H0: proporcion <= 50%. H1: proporcion > 50%.

**Formula del z-test**:
```
z = (p_observada - p_0) / sqrt(p_0 * (1 - p_0) / n)
```
Donde p_observada = 0.595 (59.5%), p_0 = 0.50, n = 25,064.

**Resultado**: z = 30.05, p < 0.001. La proporcion es significativamente mayor al 50%.

**Matiz**: El 59.5% no llega al 70% supuesto inicialmente. Pero si solo contamos hombres en via publica (85.4% del total), el 69.6% de esos hombres son jovenes (15-40). La hipotesis original mezclaba dos filtros.

**Significado**: Hay un problema grave de mortalidad violenta masculina en edad productiva. Estos hombres probablemente mueren por homicidios, accidentes de transito y otras causas externas.

### 4.6 H5: Brecha rural-urbana en asistencia medica

**Supuesto**: "La proporcion de muertes sin asistencia medica en area rural es al menos el doble que en area urbana."

**Datos**: Se filtran registros con areag in ['Rural', 'Urbano'] (excluyendo 'Ignorado' y nulos).

**NOTA IMPORTANTE**: La categoria en los datos es `'Urbano'`, no `'Urbana'`. Esto causo un bug inicial donde el filtro no encontraba datos.

**Prueba**: Chi-cuadrado de independencia. Tabla cruzada: areag x tipo de asistencia (sin/con).

**Resultado**: chi2 = 178,929.08, p < 0.001.

| Area | Sin asistencia | Con asistencia |
|------|---------------|----------------|
| Rural | 85.6% | 9.2% |
| Urbano | 23.1% | 75.6% |

**Interpretacion**: La brecha es de 3.7 veces (85.6% / 23.1%), SUPERANDO el factor 2x supuesto. En zonas rurales, 9 de cada 10 personas mueren sin ver a un medico. Esto es el hallazgo mas contundente del analisis y evidencia un problema estructural critico en la infraestructura de salud del pais.

### 4.7 Sobre los p-valores "basicamente 0"

Todas las pruebas dan p < 0.001 (practicamente 0). Esto es **normal y esperado** con n=950,000. Con muestras tan grandes, incluso diferencias pequenas se vuelven estadisticamente significativas. Lo importante no es solo el p-valor, sino el **tamano del efecto**:
- H5: 3.7x de brecha (efecto enorme)
- H4: 59.5% vs 50% (efecto moderado)
- H2: La paradoja es mas interesante que el p-valor

---

## Parte 5: Clustering

### 5.1 Preparacion de datos

**Variables usadas** (10 total):
- Numericas (2): diaocu, edadif
- Categoricas (8): sexo, perdif (periodo edad), puedif (etnia), ecivil, escodif, asist, ocur (lugar), certif

**One-Hot Encoding**: Las categoricas se convierten a variables binarias (0/1). `sexo` se convierte en `sexo_Hombre` y `sexo_Mujer`.

**StandardScaler**: Se estandarizan todas las variables a media=0, desviacion=1. Sin esto, `edadif` (rango 0-110) dominaria sobre las variables binarias (rango 0-1).

**Muestra para K optimo**: Se usan 50,000 registros para evaluar diferentes K (el paso mas costoso computacionalmente). Sin embargo, el clustering FINAL con K=5 se aplica a TODOS los registros (~945K), no a la submuestra. Solo la busqueda de K usa muestra.

### 5.2 Seleccion de K

**Metodo del codo**: Grafica la inercia (suma de distancias al centroide) vs K. Se busca el "codo" donde la reduccion de inercia se desacelera.

**Coeficiente de silueta**: Mide que tan bien separados estan los clusters. Rango: -1 a 1. Mayor es mejor. Se evalua para K=2 a K=8:

| K | Silueta |
|---|---------|
| 2 | 0.1190 |
| 3 | 0.1332 |
| 4 | 0.1353 |
| **5** | **0.1499** |
| 6 | 0.1274 |
| 7 | 0.1037 |
| 8 | 0.1045 |

**K=5 maximiza la silueta** (0.1499).

**Por que la silueta es baja (0.15)**: En datos sociodemograficos, los clusters no tienen fronteras discretas. Una persona de 60 anos que murio en domicilio sin asistencia podria pertenecer al cluster 1 o al cluster 2. Los datos humanos son inherentemente difusos. Un coeficiente de 0.15 es aceptable para este tipo de datos.

### 5.3 Interpretacion de los 5 clusters

| Cluster | Nombre | % | Perfil |
|---------|--------|---|--------|
| 0 | Mortalidad Infantil/Neonatal | 8.2% | Edad 6.1, hospital publico, con asistencia medica. 100% solteros, 99.8% sin escolaridad (logico: son bebes). |
| 1 | Adultos Con Asistencia | 39.7% | Edad 60.6, fallecen en hospitales o domicilio CON asistencia medica profesional. Grupo con acceso a salud. |
| 2 | Adultos Mayores Rurales | 40.3% | Edad 63.7, fallecen en domicilio SIN asistencia (90.3%). El cluster mas grande. Perfil tipico rural. |
| 3 | Jovenes Muerte Violenta | 8.6% | Edad 38.5, 82.2% hombres, via publica o lugar ignorado. Confirma H4. |
| 4 | Adultos No Indigenas | 3.3% | Edad 58.5, 100% identificados como "No Indigena". Asistencia medica mixta (50/50). |

**Conexion con hipotesis**:
- Cluster 2 confirma H5 (brecha rural, 40.3% del total)
- Cluster 3 confirma H4 (mortalidad violenta masculina)
- Cluster 0 se conecta con H3 (mortalidad infantil)
- Clusters 2+3 juntos = 48.9% de muertes sin acceso a salud

### 5.4 Nombramiento dinamico

El notebook tiene una funcion `auto_label()` que nombra los clusters automaticamente basandose en sus metricas:
- Si edad media < 10 -> "Mortalidad Infantil"
- Si % hombres > 75% y edad < 45 -> "Jovenes Muerte Violenta"
- Si % sin asistencia > 80% -> "Sin Asistencia"
- Etc.

Esto hace que si los datos cambian (por ejemplo, al re-ejecutar con seed diferente), los nombres se adaptan automaticamente.

---

## Parte 6: Decisiones Tecnicas Clave

### Por que DuckDB y no SQLite/PostgreSQL?

- **DuckDB es OLAP** (analitico), optimizado para SELECT con agregaciones sobre millones de filas.
- **No requiere servidor**: Es un archivo local, como SQLite.
- **Lee Parquet nativamente**: `SELECT * FROM read_parquet('*.parquet')` sin importar datos.
- **USING SAMPLE**: Muestreo eficiente a nivel de motor, no carga todo en RAM.
- **union_by_name=true**: Maneja schemas que varian entre archivos Parquet (columnas faltantes en algunos anos).

### Por que Polars para ETL y Pandas para notebooks?

- **Polars**: Mas rapido que Pandas para transformaciones masivas (4M+ filas). API funcional e inmutable. Ideal para el pipeline automatizado.
- **Pandas**: Mas compatible con seaborn, scipy, sklearn. La mayoria de tutoriales y documentacion usan Pandas. En el notebook, con 150K-950K filas, la diferencia de velocidad no importa.

### Por que Parquet y no CSV?

| Aspecto | CSV | Parquet |
|---------|-----|---------|
| Tamano nacimientos | ~800 MB | ~120 MB |
| Lectura 3 columnas | Lee todo | Solo lee 3 columnas |
| Tipos de datos | Todo es string | Preserva tipos |
| Compresion | Ninguna | zstd integrada |

### Por que no se usan paquetes de EDA automatico?

Las instrucciones del curso dicen: "El uso de paquetes y modulos para hacer Analisis Exploratorio de forma automatica, no esta permitido." Esto descarta herramientas como `pandas-profiling`, `sweetviz`, `dtale`, etc. Todo el analisis se hizo manualmente con pandas, matplotlib, seaborn, scipy y sklearn.

---

## Parte 7: Preguntas Frecuentes

### "Por que los p-valores son todos 0?"
Con n=950,000, las pruebas estadisticas tienen un poder enorme. Incluso diferencias pequenas son detectables. Lo importante es el tamano del efecto, no solo la significancia.

### "Por que la silueta es tan baja?"
0.1499 es esperado en datos sociodemograficos. Las personas no se agrupan en categorias discretas. Un anciano rural que murio con asistencia no encaja perfectamente en ningun cluster.

### "Por que Kruskal-Wallis y no ANOVA?"
Porque Shapiro-Wilk demostro que edadif NO es normal. ANOVA asume normalidad. Kruskal-Wallis es la alternativa no parametrica que no hace esa asuncion.

### "Que es Title Case y por que se usa?"
Title Case: cada palabra empieza con mayuscula (`Guatemala`, `Casado`, `Masculino`). Se usa porque los datos del INE vienen en formatos mixtos (MAYUSCULAS, minusculas). Title Case unifica todo antes de aplicar normalizaciones finas.

### "Por que falta 2016?"
El INE no publico los microdatos de la mayoria de datasets para el ano 2016 en su portal. No es un error del pipeline.

### "Que significa caudef LIKE 'U07%'?"
CIE-10 es la Clasificacion Internacional de Enfermedades. U07.1 y U07.2 son los codigos para COVID-19. El filtro `LIKE 'U07%'` captura todas las variantes.

### "Por que se usan 950K registros y no una muestra?"
Para las pruebas estadisticas (chi-cuadrado, Kruskal-Wallis), necesitamos las frecuencias reales. Una muestra introduciria sesgo. Para el clustering, se usa una submuestra de 50K para calcular K optimo, pero el clustering final se aplica a todos los registros.

### "Que es areag y por que tiene 50% nulos?"
`areag` = area geografica (Urbano/Rural). No todos los anos del INE registran esta variable. Los anos que no la tienen quedan como NULL. Cuando analizamos areag, excluimos los nulos.

### "Por que el harmonizer agrega columnas con NULL?"
Para que todos los anos tengan el mismo esquema. Si 2012 no tiene la columna `areag`, se agrega como NULL. Esto permite hacer `SELECT * FROM defunciones` y obtener una tabla uniforme con 28 columnas, independientemente del ano.
