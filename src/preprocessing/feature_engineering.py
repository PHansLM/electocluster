"""
Módulo para la construcción de caracteristicas compuestas

Caracteristicas tratadas:
    1. wealth_index          — variables R (bienes del hogar → Riqueza material - clase social aspiracional)
    2. civic_index           — variables CP (asistencia a organizaciones → participacion social)
"""

import pandas as pd
import numpy as np
from typing import Optional
from sklearn.decomposition import PCA


# Variables relacionadas (coincide con constants.py)

# Bienes del hogar — binarios 0/1
WEALTH_VARS = ['r3', 'r4a', 'r6', 'r7', 'r12', 'r15', 'r16', 'r18', 'r18n', 'r27']

# Participación en organizaciones — ordinales 1=semanal … 4=nunca (escala inversa)
CIVIC_VARS = ['cp6', 'cp7', 'cp8', 'cp13']

class FeatureEngineer:
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: Dataset limpio (post-cleaner, pre-transformer).
        """
        self.df = df.copy()
        self.metadata = {
            'variables_creadas': [],
            'variables_eliminadas': [],
            'variables_mantenidas_separadas': [],
            'advertencias': []
        }

    # 1. ÍNDICE DE RIQUEZA MATERIAL (variables R)
    def build_wealth_index(self, drop_originals: bool = True) -> pd.DataFrame:
        """
        Construye wealth_index como suma de bienes del hogar presentes.

        Justificación:
            - Cada bien tiene peso igual en términos de indicar acceso a capital material.
            - La interpretación es directa (0 = ningún bien, 10 = todos).
            - Alternativa PCA disponible via build_wealth_index_pca().

        Args:
            drop_originals: Si True, elimina las variables R originales.

        Return:
            DataFrame con columna 'wealth_index' añadida (rango 0-10).
        """
        vars_presentes = [v for v in WEALTH_VARS if v in self.df.columns]
        vars_ausentes  = [v for v in WEALTH_VARS if v not in self.df.columns]

        if vars_ausentes:
            msg = f"wealth_index: variables no encontradas en el dataset → {vars_ausentes}"
            self.metadata['advertencias'].append(msg)
            print(f"  ⚠ {msg}")

        if not vars_presentes:
            print("  ✗ wealth_index: ninguna variable R disponible. Índice no creado.")
            return self.df

        self.df['wealth_index'] = self.df[vars_presentes].sum(axis=1, skipna=True)

        if drop_originals:
            self.df.drop(columns=vars_presentes, inplace=True)
            self.metadata['variables_eliminadas'].extend(vars_presentes)

        self.metadata['variables_creadas'].append({
            'nombre': 'wealth_index',
            'variables_fuente': vars_presentes,
            'n_variables': len(vars_presentes),
            'rango_teorico': f'[0, {len(vars_presentes)}]',
            'metodo': 'suma_simple',
            'interpretacion': (
                'Número de bienes del hogar presentes. '
                'Proxy de capital material y clase social aspiracional. '
                f'Reemplaza {len(vars_presentes)} variables R individuales.'
            )
        })

        print(f"  ✓ wealth_index creado "
              f"({len(vars_presentes)} variables R → 1 índice, "
              f"rango [0, {len(vars_presentes)}])")
        return self.df

    def build_wealth_index_pca(self, drop_originals: bool = True) -> pd.DataFrame:
        """
        Alternativa: wealth_index usando el primer componente de PCA.

        Útil para bienes con pesos distintos (ej: una cocina es mas común que internet de banda ancha).

        Args:
            drop_originals: Si True, elimina las variables R originales.

        Return:
            DataFrame con columna 'wealth_index_pca' añadida.
        """

        vars_presentes = [v for v in WEALTH_VARS if v in self.df.columns]
        if not vars_presentes:
            print("  ✗ wealth_index_pca: ninguna variable R disponible.")
            return self.df

        X = self.df[vars_presentes].fillna(0).values
        pca = PCA(n_components=1)
        score = pca.fit_transform(X).flatten()
        varianza = pca.explained_variance_ratio_[0]

        self.df['wealth_index_pca'] = score

        if drop_originals:
            self.df.drop(columns=vars_presentes, inplace=True)
            self.metadata['variables_eliminadas'].extend(vars_presentes)

        self.metadata['variables_creadas'].append({
            'nombre': 'wealth_index_pca',
            'variables_fuente': vars_presentes,
            'varianza_explicada_pc1': round(float(varianza), 4),
            'metodo': 'pca_primer_componente',
            'interpretacion': 'Primer componente principal de bienes del hogar.'
        })

        print(f"  ✓ wealth_index_pca creado "
              f"(varianza explicada PC1: {varianza:.1%})")
        return self.df

    # ----------------------------------------------------------
    # 2. ÍNDICE DE PARTICIPACIÓN CÍVICA (variables CP)
    # ----------------------------------------------------------
    def build_civic_index(self, drop_originals: bool = True) -> pd.DataFrame:
        """
        Construye civic_index a partir de la frecuencia de participación en organizaciones.

        La escala original es INVERSA (1=semanal=máximo, 4=nunca=mínimo),
        por lo que se invierte antes de sumar.
            civic_index ALTO → mayor participación comunitaria.

        Organizaciones:
            CP6  → religiosa
            CP7  → asociación de padres de familia
            CP8  → junta comunal / vecinal
            CP13 → partido o movimiento político

        Args:
            drop_originals: Si True, elimina las variables CP originales.

        Return:
            DataFrame con columna 'civic_index' añadida (rango 4-16).
        """
        vars_presentes = [v for v in CIVIC_VARS if v in self.df.columns]
        vars_ausentes  = [v for v in CIVIC_VARS if v not in self.df.columns]

        if vars_ausentes:
            msg = f"civic_index: variables no encontradas → {vars_ausentes}"
            self.metadata['advertencias'].append(msg)
            print(f"  ⚠ {msg}")

        if not vars_presentes:
            print("  ✗ civic_index: ninguna variable CP disponible. Índice no creado.")
            return self.df

        # Invertir escala: 1→4, 2→3, 3→2, 4→1
        mapeo_inversion = {1: 4, 2: 3, 3: 2, 4: 1}
        cp_invertidas = self.df[vars_presentes].apply(
            lambda col: col.map(mapeo_inversion)
        )

        self.df['civic_index'] = cp_invertidas.sum(axis=1, skipna=True)

        if drop_originals:
            self.df.drop(columns=vars_presentes, inplace=True)
            self.metadata['variables_eliminadas'].extend(vars_presentes)

        n = len(vars_presentes)
        self.metadata['variables_creadas'].append({
            'nombre': 'civic_index',
            'variables_fuente': vars_presentes,
            'n_variables': n,
            'rango_teorico': f'[{n}, {4 * n}]',
            'metodo': 'suma_invertida',
            'nota_escala': 'Escala original invertida. Mayor valor = mayor participación.',
            'interpretacion': (
                'Nivel de participación comunitaria y organizacional. '
                f'Reemplaza {n} variables CP individuales.'
            )
        })

        print(f"  ✓ civic_index creado "
              f"({len(vars_presentes)} variables CP → 1 índice, "
              f"rango [{n}, {4*n}], escala invertida: mayor = más participativo)")
        return self.df

    # REPORTE
    def get_report(self) -> dict:
        """
        Genera reporte de las transformaciones de feature engineering.

        Return:
            dict: Reporte con índices creados, variables eliminadas y advertencias.
        """
        return {
            'n_variables_creadas': len(self.metadata['variables_creadas']),
            'n_variables_eliminadas': len(self.metadata['variables_eliminadas']),
            'variables_creadas': self.metadata['variables_creadas'],
            'variables_eliminadas': self.metadata['variables_eliminadas'],
            'variables_mantenidas_separadas': self.metadata['variables_mantenidas_separadas'],
            'advertencias': self.metadata['advertencias'],
            'dimensiones_finales': self.df.shape
        }


# FUNCIÓN DE CONVENIENCIA
def engineer_features(df: pd.DataFrame,
                      build_wealth: bool = True,
                      wealth_method: str = 'sum',
                      build_civic: bool = True,
                      drop_originals: bool = True) -> tuple:
    """
    Función de conveniencia para ejecutar el pipeline de feature engineering.

    Args:
        df: Dataset limpio (post-cleaner).
        build_wealth: Si construir el índice de riqueza.
        wealth_method: 'sum' (suma simple) o 'pca' (primer componente).
        build_civic: Si construir el índice de participación cívica.
        social_programs_strategy: 'separate', 'count' o 'both'.
        drop_originals: Si eliminar variables originales tras crear índices.

    Return:
        tuple: (DataFrame transformado, engineer, reporte)
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING: construcción de índices compuestos")
    print("=" * 60)

    engineer = FeatureEngineer(df)

    if build_wealth:
        print("\n[1/2] Índice de riqueza material (variables R):")
        if wealth_method == 'pca':
            engineer.build_wealth_index_pca(drop_originals=drop_originals)
        else:
            engineer.build_wealth_index(drop_originals=drop_originals)

    if build_civic:
        print("\n[2/2] Índice de participación cívica (variables CP):")
        engineer.build_civic_index(drop_originals=drop_originals)

    report = engineer.get_report()

    print(f"\n{'=' * 60}")
    print(f"✓ Feature engineering completado:")
    print(f"  Índices creados:        {report['n_variables_creadas']}")
    print(f"  Variables eliminadas:   {report['n_variables_eliminadas']}")
    print(f"  Dimensiones finales:    {report['dimensiones_finales']}")
    if report['advertencias']:
        print(f"  Advertencias:          {len(report['advertencias'])}")
    print("=" * 60)

    return engineer.df, engineer, report