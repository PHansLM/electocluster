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

# Variables de relevancia ALTA según LAPOP Bolivia
VARIABLES_ALTA = {
    'q2': {
        'nombre': 'Edad',
        'tipo': 'numerico',
        'rango': (18, 120),
        'peso': 1.00,
        'justificacion': 'Generación del votante'
    },
    'edre': {
        'nombre': 'Nivel educativo',
        'tipo': 'categorico_ordinal',
        'rango': (0, 6),
        'peso': 1.00,
        'justificacion': 'Determinante sobre tendencias',
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
        'justificacion': 'Condición socioeconómica',
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
        'peso': 0.95,
        'justificacion': 'Factor de peso en el país',
        'categorias': {
            1: 'Blanca',
            2: 'Mestiza',
            3: 'Indígena',
            4: 'Negra',
            5: 'Mulata',
            7: 'Otra'
        }
    },
    'boletidnew': {
        'nombre': 'Pertenencia indígena',
        'tipo': 'binario',
        'rango': (1, 2),
        'peso': 0.95,
        'justificacion': 'Identidad política diferenciada',
        'mapeo_binario': {1: 1, 2: 0},  # 1=Sí→1, 2=No→0
        'categorias': {
            0: 'No',
            1: 'Sí'
        }
    },
    'boletidnewb': {
        'nombre': 'Pueblo indígena específico',
        'tipo': 'categorico_nominal',
        'rango': (1, 8),
        'peso': 0.95,
        'justificacion': 'Identidad política diferenciada',
        'categorias': {
            1: 'Quechua',
            2: 'Aymara',
            3: 'Guaraní',
            4: 'Chiquitano',
            5: 'Mojeño',
            6: 'Afroboliviano',
            7: 'Otro',
            8: 'No especifica'
        }
    },
    'ur': {
        'nombre': 'Ubicación Urbano/Rural',
        'tipo': 'binario',
        'rango': (1, 2),
        'peso': 0.90,
        'justificacion': 'Patrones regionales para el voto',
        'mapeo_binario': {1: 1, 2: 0},  # 1=Urbano→1, 2=Rural→0
        'categorias': {
            0: 'Rural',
            1: 'Urbano'
        }
    },
    'ocupoit': {
        'nombre': 'Ocupación',
        'tipo': 'categorico_nominal',
        'rango': (1, 10),
        'peso': 0.90,
        'justificacion': 'Clase social y sector económico',
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

# Variables de relevancia MEDIA según LAPOP Bolivia
VARIABLES_MEDIA = {
    'q1tc_r': {
        'nombre': 'Género',
        'tipo': 'categorico_nominal',
        'rango': (1, 3),
        'peso': 0.70,
        'justificacion': 'Brecha de género política',
        'categorias': {
            1: 'Hombre',
            2: 'Mujer',
            3: 'No binario'
        }
    },
    'q3cn': {
        'nombre': 'Religión',
        'tipo': 'categorico_nominal',
        'rango': (1, 77),
        'peso': 0.70,
        'justificacion': 'Valores y posiciones morales',
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
    },
    'q5b': {
        'nombre': 'Importancia de la religión',
        'tipo': 'categorico_ordinal',
        'rango': (1, 4),
        'peso': 0.70,
        'justificacion': 'Valores y posiciones morales',
        'categorias': {
            1: 'Muy importante',
            2: 'Algo importante',
            3: 'Poco importante',
            4: 'Nada importante'
        }
    },
    # Bienes del hogar (R3-R27)
    'r3': {
        'nombre': 'Refrigerador',
        'tipo': 'binario',
        'rango': (0, 1),
        'peso': 0.65,
        'justificacion': 'Nivel socioeconómico objetivo',
        'categorias': {0: 'No', 1: 'Sí'}
    },
    'r4a': {
        'nombre': 'Teléfono celular',
        'tipo': 'binario',
        'rango': (0, 1),
        'peso': 0.65,
        'justificacion': 'Nivel socioeconómico objetivo',
        'categorias': {0: 'No', 1: 'Sí'}
    },
    'r6': {
        'nombre': 'Lavadora',
        'tipo': 'binario',
        'rango': (0, 1),
        'peso': 0.65,
        'justificacion': 'Nivel socioeconómico objetivo',
        'categorias': {0: 'No', 1: 'Sí'}
    },
    'r7': {
        'nombre': 'Microondas',
        'tipo': 'binario',
        'rango': (0, 1),
        'peso': 0.65,
        'justificacion': 'Nivel socioeconómico objetivo',
        'categorias': {0: 'No', 1: 'Sí'}
    },
    'r12': {
        'nombre': 'Agua potable',
        'tipo': 'binario',
        'rango': (0, 1),
        'peso': 0.65,
        'justificacion': 'Nivel socioeconómico objetivo',
        'categorias': {0: 'No', 1: 'Sí'}
    },
    'r15': {
        'nombre': 'Computadora',
        'tipo': 'binario',
        'rango': (0, 1),
        'peso': 0.65,
        'justificacion': 'Nivel socioeconómico objetivo',
        'categorias': {0: 'No', 1: 'Sí'}
    },
    'r18n': {
        'nombre': 'Internet banda ancha',
        'tipo': 'binario',
        'rango': (0, 1),
        'peso': 0.65,
        'justificacion': 'Nivel socioeconómico objetivo',
        'categorias': {0: 'No', 1: 'Sí'}
    },
    'r18': {
        'nombre': 'Internet (general)',
        'tipo': 'binario',
        'rango': (0, 1),
        'peso': 0.65,
        'justificacion': 'Nivel socioeconómico objetivo',
        'categorias': {0: 'No', 1: 'Sí'}
    },
    'r16': {
        'nombre': 'TV pantalla plana',
        'tipo': 'binario',
        'rango': (0, 1),
        'peso': 0.65,
        'justificacion': 'Nivel socioeconómico objetivo',
        'categorias': {0: 'No', 1: 'Sí'}
    },
    'r27': {
        'nombre': 'TV cable/satelital',
        'tipo': 'binario',
        'rango': (0, 1),
        'peso': 0.65,
        'justificacion': 'Nivel socioeconómico objetivo',
        'categorias': {0: 'No', 1: 'Sí'}
    },
    'leng1': {
        'nombre': 'Lengua materna',
        'tipo': 'categorico_nominal',
        'rango': (1001, 1006),
        'peso': 0.65,
        'justificacion': 'Identidad cultural',
        'categorias': {
            1001: 'Castellano/español',
            1002: 'Quechua',
            1003: 'Aymara',
            1006: 'Guaraní',
            1004: 'Otro nativo',
            1005: 'Otro extranjero'
        }
    },
    'formal': {
        'nombre': 'Formalidad laboral',
        'tipo': 'binario',
        'rango': (1, 2),
        'peso': 0.65,
        'justificacion': 'Inserción económica formal',
        'mapeo_binario': {1: 1, 2: 0},  # 1=Sí→1, 2=No→0
        'categorias': {
            0: 'No',
            1: 'Sí'
        }
    },
    'wf1': {
        'nombre': 'Programas sociales (general)',
        'tipo': 'binario',
        'rango': (1, 2),
        'peso': 0.65,
        'justificacion': 'Dependencia del Estado',
        'mapeo_binario': {1: 1, 2: 0},  # 1=Sí→1, 2=No→0
        'categorias': {
            0: 'No',
            1: 'Sí'
        }
    },
    'bolcct1a': {
        'nombre': 'Renta Dignidad',
        'tipo': 'binario',
        'rango': (1, 2),
        'peso': 0.65,
        'justificacion': 'Dependencia del Estado',
        'mapeo_binario': {1: 1, 2: 0},
        'categorias': {
            0: 'No',
            1: 'Sí'
        }
    },
    'bolcct1b': {
        'nombre': 'Bono Juancito Pinto',
        'tipo': 'binario',
        'rango': (1, 2),
        'peso': 0.65,
        'justificacion': 'Dependencia del Estado',
        'mapeo_binario': {1: 1, 2: 0},
        'categorias': {
            0: 'No',
            1: 'Sí'
        }
    },
    'bolcct1c': {
        'nombre': 'Bono Juana Azurduy',
        'tipo': 'binario',
        'rango': (1, 2),
        'peso': 0.65,
        'justificacion': 'Dependencia del Estado',
        'mapeo_binario': {1: 1, 2: 0},
        'categorias': {
            0: 'No',
            1: 'Sí'
        }
    },
    'estratosec': {
        'nombre': 'Tamaño de municipio',
        'tipo': 'categorico_ordinal',
        'rango': (1, 3),
        'peso': 0.60,
        'justificacion': 'Contexto urbano/rural',
        'categorias': {
            1: 'Grande (>100,000)',
            2: 'Mediana (25,000-100,000)',
            3: 'Pequeña (<25,000)'
        }
    },
    'q12cn': {
        'nombre': 'Tamaño del hogar',
        'tipo': 'numerico',
        'rango': (1, 25),
        'peso': 0.60,
        'justificacion': 'Estructura familiar'
    },
    'q12bn': {
        'nombre': 'Niños en el hogar',
        'tipo': 'numerico',
        'rango': (0, 25),
        'peso': 0.60,
        'justificacion': 'Estructura familiar'
    },
    'wealth_index': {
        'nombre': 'Índice de riqueza material',
        'tipo': 'numerico',
        'rango': (0, 10),          # 0 = ningún bien, 10 = todos los bienes R
        'peso': 0.65,
        'justificacion': (
            'Índice compuesto que expresa la riqueza material del hogar - clase aspiracional. '
            'Suma de 10 bienes del hogar (r3, r4a, r6, r7, r12, r15, r16, r18, r18n, r27). '
            'Reemplaza las variables R individuales para evitar sobrerepresentación. '
        )
    },
}

# Variables de relevancia CONTEXTUAL según LAPOP Bolivia
VARIABLES_CONTEXTUAL = {
    'gi0n': {
        'nombre': 'Consumo de noticias',
        'tipo': 'categorico_ordinal',
        'rango': (1, 5),
        'peso': 0.55,
        'justificacion': 'Acceso a información',
        'categorias': {
            1: 'Diariamente',
            2: 'Algunas veces a la semana',
            3: 'Algunas veces al mes',
            4: 'Algunas veces al año',
            5: 'Nunca'
        }
    },
    'smedia3n': {
        'nombre': 'Consumo de medios digitales',
        'tipo': 'categorico_ordinal',
        'rango': (1, 5),
        'peso': 0.55,
        'justificacion': 'Acceso a información',
        'categorias': {
            1: 'Diariamente',
            2: 'Algunas veces a la semana',
            3: 'Algunas veces al mes',
            4: 'Algunas veces al año',
            5: 'Nunca'
        }
    },
    'cp6': {
        'nombre': 'Participación social (religiosa)',
        'tipo': 'categorico_ordinal',
        'rango': (1, 4),
        'peso': 0.50,
        'justificacion': 'Capital social',
        'categorias': {
            1: 'Una vez a la semana',
            2: 'Una o dos veces al mes',
            3: 'Una o dos veces al año',
            4: 'Nunca'
        }
    },
    'cp7': {
        'nombre': 'Participación social (padres de familia)',
        'tipo': 'categorico_ordinal',
        'rango': (1, 4),
        'peso': 0.50,
        'justificacion': 'Capital social',
        'categorias': {
            1: 'Una vez a la semana',
            2: 'Una o dos veces al mes',
            3: 'Una o dos veces al año',
            4: 'Nunca'
        }
    },
    'cp8': {
        'nombre': 'Participación social (comunitaria)',
        'tipo': 'categorico_ordinal',
        'rango': (1, 4),
        'peso': 0.50,
        'justificacion': 'Capital social',
        'categorias': {
            1: 'Una vez a la semana',
            2: 'Una o dos veces al mes',
            3: 'Una o dos veces al año',
            4: 'Nunca'
        }
    },
    'cp13': {
        'nombre': 'Participación social (política)',
        'tipo': 'categorico_ordinal',
        'rango': (1, 4),
        'peso': 0.50,
        'justificacion': 'Capital social',
        'categorias': {
            1: 'Una vez a la semana',
            2: 'Una o dos veces al mes',
            3: 'Una o dos veces al año',
            4: 'Nunca'
        }
    },
    'q10e': {
        'nombre': 'Cambio en ingreso',
        'tipo': 'categorico_ordinal',
        'rango': (1, 3),
        'peso': 0.50,
        'justificacion': 'Movilidad económica percibida',
        'categorias': {
            1: 'Aumentó',
            2: 'Permaneció igual',
            3: 'Disminuyó'
        }
    },
    'q11n': {
        'nombre': 'Estado civil',
        'tipo': 'categorico_nominal',
        'rango': (1, 6),
        'peso': 0.50,
        'justificacion': 'Estructura familiar',
        'categorias': {
            1: 'Soltero',
            2: 'Casado',
            3: 'Unión libre',
            4: 'Divorciado',
            5: 'Separado',
            6: 'Viudo'
        }
    },
    'leng4': {
        'nombre': 'Idioma de padres',
        'tipo': 'categorico_nominal',
        'rango': (1, 5),
        'peso': 0.50,
        'justificacion': 'Transmisión cultural',
        'categorias': {
            1: 'Solo castellano',
            2: 'Castellano e idioma nativo',
            3: 'Solo idioma nativo',
            4: 'Castellano e idioma extranjero',
            5: 'Solo idioma extranjero'
        }
    },
    'civic_index': {
        'nombre': 'Índice de participación cívica',
        'tipo': 'numerico',
        'rango': (4, 16),          # 4 = nunca participa en nada, 16 = participa semanalmente en todo
        'peso': 0.50,
        'justificacion': (
            'Índice compuesto que expresa la participación en organizaciones sociales. '
            'Suma invertida de frecuencia de asistencia a 4 organizaciones '
            '(cp6 religiosa, cp7 padres de familia, cp8 comunitaria, cp13 política). '
            'Reemplaza las variables CP individuales para evitar sobrerepresentación.'
        )
    },
}

TODAS_VARIABLES = {**VARIABLES_ALTA, **VARIABLES_MEDIA, **VARIABLES_CONTEXTUAL}

# ============================================
# CONFIGURACIÓN DE PREPROCESAMIENTO
# ============================================
ESTRATEGIA_MISSING = {
    'numerico': 'median',           # Mediana
    'categorico_ordinal': 'mode',   # Moda
    'categorico_nominal': 'mode',   # Moda
    'binario': 'mode'               # Moda (valor más frecuente: 0 o 1)
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