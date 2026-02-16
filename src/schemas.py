"""
Esquemas canónicos, firmas de clasificación y normalización de valores.
Este archivo centraliza todo el conocimiento sobre la estructura de los datos INE.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. DATASET_SIGNATURES: patrones de columnas para clasificar archivos .sav
# ---------------------------------------------------------------------------
DATASET_SIGNATURES: dict[str, list[list[str]]] = {
    "nacimientos": [
        ["libras", "onzas", "sexo", "tipar"],
        ["libras", "onzas", "sexo", "naciop"],
        ["libras", "onzas", "sexo", "edadp", "edadm"],
    ],
    "defunciones": [
        ["caudef", "edadif", "sexo", "perdif"],
    ],
    "defunciones_fetales": [
        ["semges", "caudef", "tipar", "edadm"],
    ],
    "matrimonios": [
        ["clauni", "edadhom", "edadmuj"],
    ],
    "divorcios": [
        ["edadhom", "edadmuj", "ciuohom", "ciuomuj"],
        ["edadhom", "edadmuj", "ocuhom", "ocumuj"],
    ],
}

# ---------------------------------------------------------------------------
# 2. CANONICAL_SCHEMAS: esquema canónico por tipo de dataset
#    Cada entry: nombre_canonico → {aliases: [...], description: str}
#    Los aliases son nombres alternativos usados en distintos años.
# ---------------------------------------------------------------------------

CANONICAL_SCHEMAS: dict[str, dict[str, dict]] = {
    "nacimientos": {
        "depreg":    {"aliases": [], "description": "Departamento de registro"},
        "mupreg":    {"aliases": [], "description": "Municipio de registro"},
        "mesreg":    {"aliases": [], "description": "Mes de registro"},
        "anioreg":   {"aliases": ["añoreg"], "description": "Año de registro"},
        "tipoins":   {"aliases": [], "description": "Tipo de inscripción"},
        "depocu":    {"aliases": [], "description": "Departamento de ocurrencia"},
        "mupocu":    {"aliases": [], "description": "Municipio de ocurrencia"},
        "libras":    {"aliases": [], "description": "Peso en libras"},
        "onzas":     {"aliases": [], "description": "Peso en onzas"},
        "diaocu":    {"aliases": [], "description": "Día de ocurrencia"},
        "mesocu":    {"aliases": [], "description": "Mes de ocurrencia"},
        "anioocu":   {"aliases": ["añoocu"], "description": "Año de ocurrencia"},
        "sexo":      {"aliases": [], "description": "Sexo del nacido"},
        "tipar":     {"aliases": [], "description": "Tipo de parto"},
        "viapar":    {"aliases": [], "description": "Vía del parto"},
        "edadp":     {"aliases": [], "description": "Edad del padre"},
        "paisrep":   {"aliases": [], "description": "País de residencia del padre"},
        "deprep":    {"aliases": [], "description": "Departamento de residencia del padre"},
        "muprep":    {"aliases": [], "description": "Municipio de residencia del padre"},
        "pueblopp":  {"aliases": ["gretnp"], "description": "Pueblo de pertenencia del padre"},
        "escivp":    {"aliases": [], "description": "Estado civil del padre"},
        "paisnacp":  {"aliases": [], "description": "País de nacimiento del padre"},
        "depnap":    {"aliases": [], "description": "Departamento de nacimiento del padre"},
        "munpnap":   {"aliases": ["mupnap"], "description": "Municipio de nacimiento del padre"},
        "naciop":    {"aliases": [], "description": "Nacionalidad del padre"},
        "escolap":   {"aliases": [], "description": "Escolaridad del padre"},
        "ocupap":    {"aliases": ["ciuopad"], "description": "Ocupación del padre"},
        "edadm":     {"aliases": [], "description": "Edad de la madre"},
        "paisrem":   {"aliases": [], "description": "País de residencia de la madre"},
        "deprem":    {"aliases": [], "description": "Departamento de residencia de la madre"},
        "muprem":    {"aliases": [], "description": "Municipio de residencia de la madre"},
        "pueblopm":  {"aliases": ["grupetma"], "description": "Pueblo de pertenencia de la madre"},
        "escivm":    {"aliases": [], "description": "Estado civil de la madre"},
        "paisnacm":  {"aliases": [], "description": "País de nacimiento de la madre"},
        "depnam":    {"aliases": [], "description": "Departamento de nacimiento de la madre"},
        "mupnam":    {"aliases": ["munnam"], "description": "Municipio de nacimiento de la madre"},
        "naciom":    {"aliases": [], "description": "Nacionalidad de la madre"},
        "escolam":   {"aliases": [], "description": "Escolaridad de la madre"},
        "ocupam":    {"aliases": ["ciuomad"], "description": "Ocupación de la madre"},
        "asisrec":   {"aliases": [], "description": "Asistencia recibida"},
        "sitioocu":  {"aliases": [], "description": "Sitio de ocurrencia"},
        "tohite":    {"aliases": [], "description": "Total hijos tenidos"},
        "tohinm":    {"aliases": [], "description": "Total hijos nacidos muertos"},
        "tohivi":    {"aliases": [], "description": "Total hijos vivos"},
    },
    "defunciones": {
        "depreg":    {"aliases": [], "description": "Departamento de registro"},
        "mupreg":    {"aliases": [], "description": "Municipio de registro"},
        "mesreg":    {"aliases": [], "description": "Mes de registro"},
        "anioreg":   {"aliases": ["añoreg"], "description": "Año de registro"},
        "depocu":    {"aliases": [], "description": "Departamento de ocurrencia"},
        "mupocu":    {"aliases": [], "description": "Municipio de ocurrencia"},
        "areag":     {"aliases": [], "description": "Área geográfica"},
        "sexo":      {"aliases": [], "description": "Sexo del difunto"},
        "diaocu":    {"aliases": [], "description": "Día de ocurrencia"},
        "mesocu":    {"aliases": [], "description": "Mes de ocurrencia"},
        "anioocu":   {"aliases": ["añoocu"], "description": "Año de ocurrencia"},
        "edadif":    {"aliases": [], "description": "Edad del difunto"},
        "perdif":    {"aliases": [], "description": "Período de edad del difunto"},
        "puedif":    {"aliases": ["getdif"], "description": "Pueblo de pertenencia del difunto"},
        "ecidif":    {"aliases": [], "description": "Estado civil del difunto"},
        "escodif":   {"aliases": [], "description": "Escolaridad del difunto"},
        "ciuodif":   {"aliases": ["ocudif"], "description": "Ocupación del difunto"},
        "pnadif":    {"aliases": [], "description": "País de nacimiento del difunto"},
        "dnadif":    {"aliases": [], "description": "Departamento de nacimiento del difunto"},
        "mnadif":    {"aliases": [], "description": "Municipio de nacimiento del difunto"},
        "nacdif":    {"aliases": [], "description": "Nacionalidad del difunto"},
        "predif":    {"aliases": [], "description": "País de residencia del difunto"},
        "dredif":    {"aliases": [], "description": "Departamento de residencia del difunto"},
        "mredif":    {"aliases": [], "description": "Municipio de residencia del difunto"},
        "caudef":    {"aliases": [], "description": "Causa de defunción"},
        "asist":     {"aliases": [], "description": "Asistencia médica"},
        "ocur":      {"aliases": [], "description": "Lugar de ocurrencia"},
        "cerdef":    {"aliases": [], "description": "Certificado de defunción"},
    },
    "defunciones_fetales": {
        "depreg":    {"aliases": [], "description": "Departamento de registro"},
        "mupreg":    {"aliases": [], "description": "Municipio de registro"},
        "mesreg":    {"aliases": [], "description": "Mes de registro"},
        "anioreg":   {"aliases": ["añoreg"], "description": "Año de registro"},
        "depocu":    {"aliases": [], "description": "Departamento de ocurrencia"},
        "mupocu":    {"aliases": [], "description": "Municipio de ocurrencia"},
        "areag":     {"aliases": [], "description": "Área geográfica"},
        "sexo":      {"aliases": [], "description": "Sexo"},
        "diaocu":    {"aliases": [], "description": "Día de ocurrencia"},
        "mesocu":    {"aliases": [], "description": "Mes de ocurrencia"},
        "anioocu":   {"aliases": ["añoocu"], "description": "Año de ocurrencia"},
        "tipar":     {"aliases": [], "description": "Tipo de parto"},
        "clapar":    {"aliases": [], "description": "Clase de parto"},
        "viapar":    {"aliases": [], "description": "Vía del parto"},
        "semges":    {"aliases": [], "description": "Semanas de gestación"},
        "edadm":     {"aliases": [], "description": "Edad de la madre"},
        "paisrem":   {"aliases": [], "description": "País de residencia de la madre"},
        "deprem":    {"aliases": [], "description": "Departamento de residencia de la madre"},
        "muprem":    {"aliases": [], "description": "Municipio de residencia de la madre"},
        "pueblopm":  {"aliases": ["gretnm"], "description": "Pueblo de pertenencia de la madre"},
        "escivm":    {"aliases": [], "description": "Estado civil de la madre"},
        "nacionm":   {"aliases": ["naciom"], "description": "Nacionalidad de la madre"},
        "escolam":   {"aliases": [], "description": "Escolaridad de la madre"},
        "ocupam":    {"aliases": [], "description": "Ocupación de la madre"},
        "ciuomad":   {"aliases": [], "description": "Ocupación de la madre (alt)"},
        "caudef":    {"aliases": [], "description": "Causa de defunción"},
        "asisrec":   {"aliases": [], "description": "Asistencia recibida"},
        "sitioocu":  {"aliases": [], "description": "Sitio de ocurrencia"},
        "tohite":    {"aliases": [], "description": "Total hijos tenidos"},
        "tohinm":    {"aliases": [], "description": "Total hijos nacidos muertos"},
        "tohivi":    {"aliases": [], "description": "Total hijos vivos"},
    },
    "matrimonios": {
        "depreg":    {"aliases": [], "description": "Departamento de registro"},
        "mupreg":    {"aliases": [], "description": "Municipio de registro"},
        "mesreg":    {"aliases": [], "description": "Mes de registro"},
        "anioreg":   {"aliases": ["añoreg"], "description": "Año de registro"},
        "anioocu":   {"aliases": ["añoocu"], "description": "Año de ocurrencia"},
        "clauni":    {"aliases": [], "description": "Clase de unión"},
        "nunuho":    {"aliases": [], "description": "Número de nupcias del hombre"},
        "nunumu":    {"aliases": [], "description": "Número de nupcias de la mujer"},
        "edadhom":   {"aliases": [], "description": "Edad del hombre"},
        "edadmuj":   {"aliases": [], "description": "Edad de la mujer"},
        "puehom":    {"aliases": ["gethom"], "description": "Pueblo de pertenencia del hombre"},
        "puemuj":    {"aliases": ["getmuj"], "description": "Pueblo de pertenencia de la mujer"},
        "nachom":    {"aliases": [], "description": "Nacionalidad del hombre"},
        "nacmuj":    {"aliases": [], "description": "Nacionalidad de la mujer"},
        "eschom":    {"aliases": [], "description": "Escolaridad del hombre"},
        "escmuj":    {"aliases": [], "description": "Escolaridad de la mujer"},
        "ciuohom":   {"aliases": ["ocuhom"], "description": "Ocupación del hombre"},
        "ciuomuj":   {"aliases": ["ocumuj"], "description": "Ocupación de la mujer"},
        "depocu":    {"aliases": [], "description": "Departamento de ocurrencia"},
        "mupocu":    {"aliases": [], "description": "Municipio de ocurrencia"},
        "diaocu":    {"aliases": [], "description": "Día de ocurrencia"},
        "mesocu":    {"aliases": [], "description": "Mes de ocurrencia"},
        "areagocu":  {"aliases": ["areag"], "description": "Área geográfica de ocurrencia"},
    },
    "divorcios": {
        "depreg":    {"aliases": [], "description": "Departamento de registro"},
        "mupreg":    {"aliases": [], "description": "Municipio de registro"},
        "mesreg":    {"aliases": [], "description": "Mes de registro"},
        "anioreg":   {"aliases": ["añoreg"], "description": "Año de registro"},
        "diaocu":    {"aliases": [], "description": "Día de ocurrencia"},
        "mesocu":    {"aliases": [], "description": "Mes de ocurrencia"},
        "anioocu":   {"aliases": ["añoocu"], "description": "Año de ocurrencia"},
        "depocu":    {"aliases": [], "description": "Departamento de ocurrencia"},
        "mupocu":    {"aliases": [], "description": "Municipio de ocurrencia"},
        "edadhom":   {"aliases": [], "description": "Edad del hombre"},
        "edadmuj":   {"aliases": [], "description": "Edad de la mujer"},
        "puehom":    {"aliases": ["pperhom", "gethom"], "description": "Pueblo de pertenencia del hombre"},
        "puemuj":    {"aliases": ["ppermuj", "getmuj"], "description": "Pueblo de pertenencia de la mujer"},
        "nachom":    {"aliases": [], "description": "Nacionalidad del hombre"},
        "nacmuj":    {"aliases": [], "description": "Nacionalidad de la mujer"},
        "eschom":    {"aliases": [], "description": "Escolaridad del hombre"},
        "escmuj":    {"aliases": [], "description": "Escolaridad de la mujer"},
        "ciuohom":   {"aliases": ["ocuhom"], "description": "Ocupación del hombre"},
        "ciuomuj":   {"aliases": ["ocumuj"], "description": "Ocupación de la mujer"},
    },
}


# ---------------------------------------------------------------------------
# 3. Helpers: build alias→canonical lookup
# ---------------------------------------------------------------------------

def build_alias_map(dataset_type: str) -> dict[str, str]:
    """Retorna {alias_lower: canonical_name} para un tipo de dataset."""
    schema = CANONICAL_SCHEMAS.get(dataset_type, {})
    alias_map: dict[str, str] = {}
    for canonical, info in schema.items():
        alias_map[canonical] = canonical
        for alias in info["aliases"]:
            alias_map[alias] = canonical
    return alias_map


def get_canonical_columns(dataset_type: str) -> list[str]:
    """Retorna lista ordenada de columnas canónicas para un tipo."""
    return list(CANONICAL_SCHEMAS.get(dataset_type, {}).keys())


def get_column_descriptions(dataset_type: str) -> dict[str, str]:
    """Retorna {canonical_name: description} para un tipo de dataset."""
    schema = CANONICAL_SCHEMAS.get(dataset_type, {})
    return {name: info["description"] for name, info in schema.items()}


# ---------------------------------------------------------------------------
# 4. VALUE_NORMALIZATION: normalización de variantes de string por columna
# ---------------------------------------------------------------------------

VALUE_NORMALIZATION: dict[str, dict[str, str]] = {
    # Estado civil
    "ecidif": {
        "Casado(a)": "Casado",
        "Soltero(a)": "Soltero",
        "Divorciado(a)": "Divorciado",
        "Viudo(a)": "Viudo",
        "Unido(a)": "Unido",
        "Separado(a)": "Separado",
    },
    "escivp": {
        "Casado(a)": "Casado",
        "Soltero(a)": "Soltero",
        "Divorciado(a)": "Divorciado",
        "Viudo(a)": "Viudo",
        "Unido(a)": "Unido",
        "Separado(a)": "Separado",
    },
    "escivm": {
        "Casado(a)": "Casado",
        "Soltero(a)": "Soltero",
        "Divorciado(a)": "Divorciado",
        "Viudo(a)": "Viudo",
        "Unido(a)": "Unido",
        "Separado(a)": "Separado",
    },
    # Pueblo de pertenencia
    "puedif": {
        "Garífuna": "Garifuna",
        "Garifuna": "Garifuna",
        "Xinca": "Xinka",
        "Xinka": "Xinka",
    },
    "pueblopp": {
        "Garífuna": "Garifuna",
        "Xinca": "Xinka",
    },
    "pueblopm": {
        "Garífuna": "Garifuna",
        "Xinca": "Xinka",
    },
    "puehom": {
        "Garífuna": "Garifuna",
        "Xinca": "Xinka",
    },
    "puemuj": {
        "Garífuna": "Garifuna",
        "Xinca": "Xinka",
    },
    # Sexo
    "sexo": {
        "Hombre": "Hombre",
        "Mujer": "Mujer",
        "Masculino": "Hombre",
        "Femenino": "Mujer",
    },
}
