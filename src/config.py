from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAWDATA_DIR = PROJECT_ROOT / "rowdata"

# MongoDB
MONGO_URI = "mongodb://admin:admin123@localhost:27017/"
DB_NAME = "ine_guatemala"

# Datos disponibles: 2012-2022 (sin 2016 en ningún tipo, sin 2014 en def. fetales)
# Todos los tipos tienen datos extra de 2023
AVAILABLE_YEARS = [2012, 2013, 2014, 2015, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

# Dataset type keywords mapping: column patterns -> dataset type
# These are matched against column names (case-insensitive) to classify each .sav file
# Based on actual INE Guatemala .sav column names
DATASET_SIGNATURES = {
    "nacimientos": [
        # 42 cols (2017+): Libras, Onzas, Tipar, Sexo, Edadm, Edadp, etc.
        ["libras", "onzas", "sexo", "tipar"],
        # 41 cols (2012-2013): same but with Naciop/Naciom
        ["libras", "onzas", "sexo", "naciop"],
        # 39 cols (2014): similar
        ["libras", "onzas", "sexo", "edadp", "edadm"],
    ],
    "defunciones": [
        # 27-28 cols: Caudef, Edadif, Sexo, Perdif
        ["caudef", "edadif", "sexo", "perdif"],
    ],
    "defunciones_fetales": [
        # 29-30 cols: SEMGES, CAUDEF, TIPAR, EDADM
        ["semges", "caudef", "tipar", "edadm"],
    ],
    "matrimonios": [
        # 20-23 cols: CLAUNI, EDADHOM, EDADMUJ
        ["clauni", "edadhom", "edadmuj"],
    ],
    "divorcios": [
        # 18-19 cols: EDADHOM, EDADMUJ, CIUOHOM, CIUOMUJ (sin CLAUNI)
        ["edadhom", "edadmuj", "ciuohom", "ciuomuj"],
        # 2012 format: EDADHOM, EDADMUJ, OCUHOM, OCUMUJ (sin CLAUNI)
        ["edadhom", "edadmuj", "ocuhom", "ocumuj"],
    ],
}
