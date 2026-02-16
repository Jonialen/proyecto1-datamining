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
# ---------------------------------------------------------------------------

CANONICAL_SCHEMAS: dict[str, dict[str, dict]] = {
    "nacimientos": {
        "depreg": {"aliases": [], "description": "Departamento de registro"},
        "mupreg": {"aliases": [], "description": "Municipio de registro"},
        "mesreg": {"aliases": [], "description": "Mes de registro"},
        "anioreg": {"aliases": ["añoreg"], "description": "Año de registro"},
        "tipoins": {"aliases": [], "description": "Tipo de inscripción"},
        "depocu": {"aliases": [], "description": "Departamento de ocurrencia"},
        "mupocu": {"aliases": [], "description": "Municipio de ocurrencia"},
        "libras": {"aliases": [], "description": "Peso en libras"},
        "onzas": {"aliases": [], "description": "Peso en onzas"},
        "diaocu": {"aliases": [], "description": "Día de ocurrencia"},
        "mesocu": {"aliases": [], "description": "Mes de ocurrencia"},
        "anioocu": {"aliases": ["añoocu"], "description": "Año de ocurrencia"},
        "sexo": {"aliases": [], "description": "Sexo del nacido"},
        "tipar": {"aliases": [], "description": "Tipo de parto"},
        "viapar": {"aliases": [], "description": "Vía del parto"},
        "edadp": {"aliases": [], "description": "Edad del padre"},
        "paisrep": {"aliases": [], "description": "País de residencia del padre"},
        "deprep": {
            "aliases": [],
            "description": "Departamento de residencia del padre",
        },
        "muprep": {"aliases": [], "description": "Municipio de residencia del padre"},
        "pueblopp": {
            "aliases": ["gretnp"],
            "description": "Pueblo de pertenencia del padre",
        },
        "escivp": {"aliases": [], "description": "Estado civil del padre"},
        "paisnacp": {"aliases": [], "description": "País de nacimiento del padre"},
        "depnap": {
            "aliases": [],
            "description": "Departamento de nacimiento del padre",
        },
        "munpnap": {
            "aliases": ["mupnap"],
            "description": "Municipio de nacimiento del padre",
        },
        "naciop": {"aliases": [], "description": "Nacionalidad del padre"},
        "escolap": {"aliases": [], "description": "Escolaridad del padre"},
        "ocupap": {"aliases": ["ciuopad"], "description": "Ocupación del padre"},
        "edadm": {"aliases": [], "description": "Edad de la madre"},
        "paisrem": {"aliases": [], "description": "País de residencia de la madre"},
        "deprem": {
            "aliases": [],
            "description": "Departamento de residencia de la madre",
        },
        "muprem": {"aliases": [], "description": "Municipio de residencia de la madre"},
        "pueblopm": {
            "aliases": ["grupetma"],
            "description": "Pueblo de pertenencia de la madre",
        },
        "escivm": {"aliases": [], "description": "Estado civil de la madre"},
        "paisnacm": {"aliases": [], "description": "País de nacimiento de la madre"},
        "depnam": {
            "aliases": [],
            "description": "Departamento de nacimiento de la madre",
        },
        "mupnam": {
            "aliases": ["munnam"],
            "description": "Municipio de nacimiento de la madre",
        },
        "naciom": {"aliases": [], "description": "Nacionalidad de la madre"},
        "escolam": {"aliases": [], "description": "Escolaridad de la madre"},
        "ocupam": {"aliases": ["ciuomad"], "description": "Ocupación de la madre"},
        "asisrec": {"aliases": [], "description": "Asistencia recibida"},
        "sitioocu": {"aliases": [], "description": "Sitio de ocurrencia"},
        "tohite": {"aliases": [], "description": "Total hijos tenidos"},
        "tohinm": {"aliases": [], "description": "Total hijos nacidos muertos"},
        "tohivi": {"aliases": [], "description": "Total hijos vivos"},
    },
    "defunciones": {
        "depreg": {"aliases": [], "description": "Departamento de registro"},
        "mupreg": {"aliases": [], "description": "Municipio de registro"},
        "mesreg": {"aliases": [], "description": "Mes de registro"},
        "anioreg": {"aliases": ["añoreg"], "description": "Año de registro"},
        "depocu": {"aliases": [], "description": "Departamento de ocurrencia"},
        "mupocu": {"aliases": [], "description": "Municipio de ocurrencia"},
        "areag": {"aliases": [], "description": "Área geográfica"},
        "sexo": {"aliases": [], "description": "Sexo del difunto"},
        "diaocu": {"aliases": [], "description": "Día de ocurrencia"},
        "mesocu": {"aliases": [], "description": "Mes de ocurrencia"},
        "anioocu": {"aliases": ["añoocu"], "description": "Año de ocurrencia"},
        "edadif": {"aliases": [], "description": "Edad del difunto"},
        "perdif": {"aliases": [], "description": "Período de edad del difunto"},
        "puedif": {
            "aliases": ["getdif"],
            "description": "Pueblo de pertenencia del difunto",
        },
        "ecidif": {"aliases": [], "description": "Estado civil del difunto"},
        "escodif": {"aliases": [], "description": "Escolaridad del difunto"},
        "ciuodif": {"aliases": ["ocudif"], "description": "Ocupación del difunto"},
        "pnadif": {"aliases": [], "description": "País de nacimiento del difunto"},
        "dnadif": {
            "aliases": [],
            "description": "Departamento de nacimiento del difunto",
        },
        "mnadif": {"aliases": [], "description": "Municipio de nacimiento del difunto"},
        "nacdif": {"aliases": [], "description": "Nacionalidad del difunto"},
        "predif": {"aliases": [], "description": "País de residencia del difunto"},
        "dredif": {
            "aliases": [],
            "description": "Departamento de residencia del difunto",
        },
        "mredif": {"aliases": [], "description": "Municipio de residencia del difunto"},
        "caudef": {"aliases": [], "description": "Causa de defunción"},
        "asist": {"aliases": [], "description": "Asistencia médica"},
        "ocur": {"aliases": [], "description": "Lugar de ocurrencia"},
        "cerdef": {"aliases": [], "description": "Certificado de defunción"},
    },
    "defunciones_fetales": {
        "depreg": {"aliases": [], "description": "Departamento de registro"},
        "mupreg": {"aliases": [], "description": "Municipio de registro"},
        "mesreg": {"aliases": [], "description": "Mes de registro"},
        "anioreg": {"aliases": ["añoreg"], "description": "Año de registro"},
        "depocu": {"aliases": [], "description": "Departamento de ocurrencia"},
        "mupocu": {"aliases": [], "description": "Municipio de ocurrencia"},
        "areag": {"aliases": [], "description": "Área geográfica"},
        "sexo": {"aliases": [], "description": "Sexo"},
        "diaocu": {"aliases": [], "description": "Día de ocurrencia"},
        "mesocu": {"aliases": [], "description": "Mes de ocurrencia"},
        "anioocu": {"aliases": ["añoocu"], "description": "Año de ocurrencia"},
        "tipar": {"aliases": [], "description": "Tipo de parto"},
        "clapar": {"aliases": [], "description": "Clase de parto"},
        "viapar": {"aliases": [], "description": "Vía del parto"},
        "semges": {"aliases": [], "description": "Semanas de gestación"},
        "edadm": {"aliases": [], "description": "Edad de la madre"},
        "paisrem": {"aliases": [], "description": "País de residencia de la madre"},
        "deprem": {
            "aliases": [],
            "description": "Departamento de residencia de la madre",
        },
        "muprem": {"aliases": [], "description": "Municipio de residencia de la madre"},
        "pueblopm": {
            "aliases": ["gretnm"],
            "description": "Pueblo de pertenencia de la madre",
        },
        "escivm": {"aliases": [], "description": "Estado civil de la madre"},
        "nacionm": {"aliases": ["naciom"], "description": "Nacionalidad de la madre"},
        "escolam": {"aliases": [], "description": "Escolaridad de la madre"},
        "ocupam": {"aliases": [], "description": "Ocupación de la madre"},
        "ciuomad": {"aliases": [], "description": "Ocupación de la madre (alt)"},
        "caudef": {"aliases": [], "description": "Causa de defunción"},
        "asisrec": {"aliases": [], "description": "Asistencia recibida"},
        "sitioocu": {"aliases": [], "description": "Sitio de ocurrencia"},
        "tohite": {"aliases": [], "description": "Total hijos tenidos"},
        "tohinm": {"aliases": [], "description": "Total hijos nacidos muertos"},
        "tohivi": {"aliases": [], "description": "Total hijos vivos"},
    },
    "matrimonios": {
        "depreg": {"aliases": [], "description": "Departamento de registro"},
        "mupreg": {"aliases": [], "description": "Municipio de registro"},
        "mesreg": {"aliases": [], "description": "Mes de registro"},
        "anioreg": {"aliases": ["añoreg"], "description": "Año de registro"},
        "anioocu": {"aliases": ["añoocu"], "description": "Año de ocurrencia"},
        "clauni": {"aliases": [], "description": "Clase de unión"},
        "nunuho": {"aliases": [], "description": "Número de nupcias del hombre"},
        "nunumu": {"aliases": [], "description": "Número de nupcias de la mujer"},
        "edadhom": {"aliases": [], "description": "Edad del hombre"},
        "edadmuj": {"aliases": [], "description": "Edad de la mujer"},
        "puehom": {
            "aliases": ["gethom"],
            "description": "Pueblo de pertenencia del hombre",
        },
        "puemuj": {
            "aliases": ["getmuj"],
            "description": "Pueblo de pertenencia de la mujer",
        },
        "nachom": {"aliases": [], "description": "Nacionalidad del hombre"},
        "nacmuj": {"aliases": [], "description": "Nacionalidad de la mujer"},
        "eschom": {"aliases": [], "description": "Escolaridad del hombre"},
        "escmuj": {"aliases": [], "description": "Escolaridad de la mujer"},
        "ciuohom": {"aliases": ["ocuhom"], "description": "Ocupación del hombre"},
        "ciuomuj": {"aliases": ["ocumuj"], "description": "Ocupación de la mujer"},
        "depocu": {"aliases": [], "description": "Departamento de ocurrencia"},
        "mupocu": {"aliases": [], "description": "Municipio de ocurrencia"},
        "diaocu": {"aliases": [], "description": "Día de ocurrencia"},
        "mesocu": {"aliases": [], "description": "Mes de ocurrencia"},
        "areagocu": {
            "aliases": ["areag"],
            "description": "Área geográfica de ocurrencia",
        },
    },
    "divorcios": {
        "depreg": {"aliases": [], "description": "Departamento de registro"},
        "mupreg": {"aliases": [], "description": "Municipio de registro"},
        "mesreg": {"aliases": [], "description": "Mes de registro"},
        "anioreg": {"aliases": ["añoreg"], "description": "Año de registro"},
        "diaocu": {"aliases": [], "description": "Día de ocurrencia"},
        "mesocu": {"aliases": [], "description": "Mes de ocurrencia"},
        "anioocu": {"aliases": ["añoocu"], "description": "Año de ocurrencia"},
        "depocu": {"aliases": [], "description": "Departamento de ocurrencia"},
        "mupocu": {"aliases": [], "description": "Municipio de ocurrencia"},
        "edadhom": {"aliases": [], "description": "Edad del hombre"},
        "edadmuj": {"aliases": [], "description": "Edad de la mujer"},
        "puehom": {
            "aliases": ["pperhom", "gethom"],
            "description": "Pueblo de pertenencia del hombre",
        },
        "puemuj": {
            "aliases": ["ppermuj", "getmuj"],
            "description": "Pueblo de pertenencia de la mujer",
        },
        "nachom": {"aliases": [], "description": "Nacionalidad del hombre"},
        "nacmuj": {"aliases": [], "description": "Nacionalidad de la mujer"},
        "eschom": {"aliases": [], "description": "Escolaridad del hombre"},
        "escmuj": {"aliases": [], "description": "Escolaridad de la mujer"},
        "ciuohom": {"aliases": ["ocuhom"], "description": "Ocupación del hombre"},
        "ciuomuj": {"aliases": ["ocumuj"], "description": "Ocupación de la mujer"},
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

# ---------------------------------------------------------------------------
# Mapeo de códigos ISO 3166-1 numéricos → nombre de país (INE usa estos códigos)
# ---------------------------------------------------------------------------
_ISO_COUNTRY_CODES: dict[str, str] = {
    "320.0": "Guatemala", "320": "Guatemala",
    "222.0": "El Salvador", "222": "El Salvador",
    "340.0": "Honduras", "340": "Honduras",
    "484.0": "México", "484": "México",
    "840.0": "Estados Unidos De América", "840": "Estados Unidos De América",
    "84.0": "Belice", "84": "Belice",
    "170.0": "Colombia", "170": "Colombia",
    "188.0": "Costa Rica", "188": "Costa Rica",
    "192.0": "Cuba", "192": "Cuba",
    "276.0": "Alemania", "276": "Alemania",
    "380.0": "Italia", "380": "Italia",
    "156.0": "China", "156": "China",
    "410.0": "República De Corea", "410": "República De Corea",
    "124.0": "Canadá", "124": "Canadá",
    "604.0": "Perú", "604": "Perú",
    "724.0": "España", "724": "España",
    "862.0": "Venezuela", "862": "Venezuela",
    "558.0": "Nicaragua", "558": "Nicaragua",
    "076.0": "Brasil", "076": "Brasil",
    "858.0": "Uruguay", "858": "Uruguay",
    "032.0": "Argentina", "032": "Argentina",
    "152.0": "Chile", "152": "Chile",
    "218.0": "Ecuador", "218": "Ecuador",
    "591.0": "Panamá", "591": "Panamá",
    "214.0": "República Dominicana", "214": "República Dominicana",
    "332.0": "Haití", "332": "Haití",
    "388.0": "Jamaica", "388": "Jamaica",
    "826.0": "Gran Bretaña", "826": "Gran Bretaña",
    "250.0": "Francia", "250": "Francia",
    "392.0": "Japón", "392": "Japón",
    "356.0": "India", "356": "India",
    "643.0": "Rusia", "643": "Rusia",
    "458.0": "Malasia", "458": "Malasia",
    "720.0": "Yemen", "720": "Yemen",
    "721.0": "El Salvador", "721": "El Salvador",
    "723.0": "El Salvador", "723": "El Salvador",
    "275.0": "Palestina", "275": "Palestina",
    "642.0": "Rumania", "642": "Rumania",
    "40.0": "Austria", "40": "Austria",
    "1020.0": "Ignorado", "1020": "Ignorado",
    "1026.0": "Ignorado", "1026": "Ignorado",
    "1045.0": "Ignorado", "1045": "Ignorado",
    "9999.0": "Ignorado", "9999": "Ignorado",
}

VALUE_NORMALIZATION: dict[str, dict[str, str]] = {
    # Departamentos (Unificar tildes y variantes)
    "dnadif": {"Quiche": "Quiché", "Quiché": "Quiché", "9999.0": "Ignorado", "9999": "Ignorado"},
    "depocu": {"Quiche": "Quiché", "Quiché": "Quiché"},
    "depreg": {"Quiche": "Quiché", "Quiché": "Quiché"},
    # Estado civil
    "ecidif": {
        "Casado(a)": "Casado",
        "Soltero(a)": "Soltero",
        "Divorciado(a)": "Divorciado",
        "Viudo(a)": "Viudo",
        "Unido(a)": "Unido",
        "Separado(a)": "Separado",
    },
    # País de residencia del difunto
    "predif": _ISO_COUNTRY_CODES,
    # País de nacimiento del difunto
    "pnadif": _ISO_COUNTRY_CODES,
    # País de residencia (nacimientos, defunciones fetales)
    "paisrem": _ISO_COUNTRY_CODES,
    "paisrep": _ISO_COUNTRY_CODES,
    # País de nacimiento (nacimientos)
    "paisnacp": _ISO_COUNTRY_CODES,
    "paisnacm": _ISO_COUNTRY_CODES,
    # Pueblo (Fusión de categorías redundantes)
    "puedif": {
        "Mestizo, Ladino": "Ladino/Mestizo",
        "Mestizo / Ladino": "Ladino/Mestizo",
        "Mestizo/Ladino": "Ladino/Mestizo",
        "No Indigena": "No Indígena",
        "Indigena": "Indígena",
        "Garífuna": "Garifuna",
        "Xinka": "Xinka",
        "Xinca": "Xinka",
    },
    # Período de edad (Consistencia para gráficas)
    "perdif": {
        "1 Año Y Más": "1 Año Y Más",
        "1 Año Y Má": "1 Año Y Más",
        "Menos De Un Mes": "Menos De Un Mes",
        "1 A 11 Meses": "1 A 11 Meses",
    },
    # Certificado y Asistencia
    "cerdef": {
        "Medico": "Médico",
        "Médico": "Médico",
        "Paramedico": "Paramédico",
        "Paramédico": "Paramédico",
    },
    # Sexo
    "sexo": {
        "Hombre": "Hombre",
        "Mujer": "Mujer",
        "Masculino": "Hombre",
        "Femenino": "Mujer",
    },
}
