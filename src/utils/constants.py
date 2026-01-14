"""
Constantes usadas en el dataset LAPOP Bolivia 2023
"""

# Las siguientes constantes corresponden a las contantes y variables detalladas en el cuestionario base en la documentación de LAPOP.

# ============================================
# CÓDIGOS DE MISSING
# ============================================
from pathlib import Path


LAPOP_MISSING_CODES = {
    888888: "No sabe",
    988888: "No responde",
    999999: "Inaplicable"
}

# ============================================
# VARIABLES DEMOGRÁFICAS SELECCIONADAS
# ============================================

# Variables de relevancia ALTA (peso 0.80-1.00)
VARIABLES_ALTA = {
    'q2': {
        'nombre': 'Edad',
        'tipo': 'numerico',
        'rango': (18, 120),
        'peso': 1.00
    },
    'edre': {
        'nombre': 'Nivel educativo',
        'tipo': 'categorico_ordinal',
        'rango': (0, 6),
        'peso': 1.00,
        'categorias': {
            0: 'Ninguna',
            1: 'Primaria incompleta',
            2: 'Primaria completa',
            3: 'Secundaria incompleta',
            4: 'Secundaria completa',
            5: 'Universidad incompleta',
            6: 'Universidad completa'
        }
    },
    'q10inc': {
        'nombre': 'Ingreso familiar',
        'tipo': 'categorico_ordinal',
        'rango': (1001, 1015),
        'peso': 1.00,
        'categorias': {
            1001: '0-600 Bs',
            1002: '601-1400 Bs',
            1003: '1401-2200 Bs',
            1004: '2201-3000 Bs',
            1005: '3001-3700 Bs',
            1006: '3701-4300 Bs',
            1007: '4301-5100 Bs',
            1008: '5101-5600 Bs',
            1009: '5601-6200 Bs',
            1010: '6201-7000 Bs',
            1011: '7001-7700 Bs',
            1012: '7701-8500 Bs',
            1013: '8501-9300 Bs',
            1014: '9301-10300 Bs',
            1015: 'Más de 10300 Bs'
        }
    },
    'etid': {
        'nombre': 'Identidad étnica',
        'tipo': 'categorico_nominal',
        'rango': (1, 7),
        'peso': 0.90,
        'categorias': {
            1: 'Blanca',
            2: 'Mestiza',
            3: 'Indígena',
            4: 'Negra',
            5: 'Mulata',
            7: 'Otra'
        }
    },
    'ur': {
        'nombre': 'Urbano/Rural',
        'tipo': 'categorico_nominal',
        'rango': (1, 2),
        'peso': 0.85,
        'categorias': {
            1: 'Urbano',
            2: 'Rural'
        }
    },
    'ocupoit': {
        'nombre': 'Ocupación',
        'tipo': 'categorico_nominal',
        'rango': (1, 10),
        'peso': 0.95,
        'categorias': {
            1: 'Directores y gerentes',
            2: 'Profesionales científicos',
            3: 'Técnicos nivel medio',
            4: 'Personal administrativo',
            5: 'Servicios y vendedores',
            6: 'Agricultores',
            7: 'Oficiales y artesanos',
            8: 'Operadores de máquinas',
            9: 'Ocupaciones elementales',
            10: 'Ocupaciones militares'
        }
    }
}

# Variables de relevancia MODERADA (peso 0.45-0.65)
VARIABLES_MODERADA = {
    # Observación: El cuestinario original tiene 'q1tc' para el genero, pero en el dataset es 'q1tc_r'
    'q1tc_r': {
        'nombre': 'Género',
        'tipo': 'categorico_nominal',
        'rango': (1, 3),
        'peso': 0.50,
        'categorias': {
            1: 'Hombre',
            2: 'Mujer',
            3: 'No binario'
        }
    },
    'q11n': {
        'nombre': 'Estado civil',
        'tipo': 'categorico_nominal',
        'rango': (1, 6),
        'peso': 0.55,
        'categorias': {
            1: 'Soltero',
            2: 'Casado',
            3: 'Unión libre',
            4: 'Divorciado',
            5: 'Separado',
            6: 'Viudo'
        }
    },
    'q12cn': {
        'nombre': 'Tamaño del hogar',
        'tipo': 'numerico',
        'rango': (1, 25),
        'peso': 0.50
    }
}

# Variables de relevancia CONTEXTUAL (peso 0.50-0.75)
VARIABLES_CONTEXTUAL = {
    'boletidnew': {
        'nombre': 'Pertenencia indígena',
        'tipo': 'categorico_nominal',
        'rango': (1, 2),
        'peso': 0.75,
        'categorias': {
            1: 'Sí',
            2: 'No'
        }
    },
    'q3cn': {
        'nombre': 'Religión',
        'tipo': 'categorico_nominal',
        'rango': (1, 11),
        'peso': 0.80,
        'categorias': {
            1: 'Católico',
            2: 'Protestante no evangélico',
            3: 'Religiones orientales',
            4: 'Ninguna (cree en ser superior)',
            5: 'Evangélica y Pentecostal',
            7: 'Religiones tradicionales',
            11: 'Agnóstico/ateo',
            77: 'Otra'
        }
    }
}

TODAS_VARIABLES = {**VARIABLES_ALTA, **VARIABLES_MODERADA, **VARIABLES_CONTEXTUAL}

# ============================================
# CONFIGURACIÓN DE PREPROCESAMIENTO
# ============================================
ESTRATEGIA_MISSING = {
    'numerico': 'median',           # Mediana
    'categorico_ordinal': 'mode',   # Moda
    'categorico_nominal': 'mode'    # Moda
}

METODO_NORMALIZACION = 'minmax'    # 'minmax' o 'standard'

# ============================================
# RUTAS
# ============================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/utils/constants.py → electocluster/

RAW_DATA_PATH = str(PROJECT_ROOT / "data" / "raw" / "lapop_bolivia_2023.dta")
PROCESSED_DATA_PATH = str(PROJECT_ROOT / "data" / "processed" / "lapop_bolivia_2023_processed.csv")
SYNTHETIC_DATA_DIRECTORY_PATH = str(PROJECT_ROOT / "data" / "synthetic")
SYNTHETIC_DATA_PATH = str(PROJECT_ROOT / "data" / "synthetic" / "lapop_bolivia_2023.dta")
RESULTS_PATH = str(PROJECT_ROOT / "results")