"""
Módulo para limpieza de datos del dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.constants import (
    TODAS_VARIABLES,
    ESTRATEGIA_MISSING
)


class DataCleaner:
    """
    Clase para limpieza de datos: manejo de missings, detección de outliers,
    y validación de rangos.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df (pd.DataFrame): Dataset a limpiar.
        """
        self.df = df.copy()
        self.metadata = {
            'registros_originales': len(df),
            'registros_finales': len(df),  
            'registros_eliminados': 0,
            'variables_imputadas': []
        }
    
    def validate_ranges(self) -> pd.DataFrame:
        """
        Valida que los valores estén dentro de los rangos esperados (definidos en la documentación).
        Convierte valores fuera de rango a NaN.

        Return:
            pd.DataFrame: Dataset con valores validados en rango.
        """
        print("Validando rangos de variables...")
        
        for var_code, var_info in TODAS_VARIABLES.items():
            if var_code not in self.df.columns:
                continue
            
            if 'rango' in var_info:
                min_val, max_val = var_info['rango']
                
                # Contar valores fuera de rango
                fuera_rango = ((self.df[var_code] < min_val) | 
                               (self.df[var_code] > max_val)) & \
                              (self.df[var_code].notna())
                n_fuera = fuera_rango.sum()
                
                if n_fuera > 0:
                    print(f"  ⚠ {var_code}: {n_fuera} valores fuera de rango [{min_val}, {max_val}] → NaN")
                    self.df.loc[fuera_rango, var_code] = np.nan
        
        print("✓ Validación de rangos completada")
        return self.df
    
    def handle_missing_values(self, strategy: dict = None) -> pd.DataFrame:
        """
        Maneja valores faltantes según estrategia definida.
        
        Args:
            strategy (dict): Estrategia de imputación por tipo de variable.
                            Si es None, usa ESTRATEGIA_MISSING.
        
        Return:
            pd.DataFrame: Dataset con missings imputados.
        """
        if strategy is None:
            strategy = ESTRATEGIA_MISSING
        
        print("Imputando valores faltantes...")
        
        for var_code, var_info in TODAS_VARIABLES.items():
            if var_code not in self.df.columns:
                continue
            
            # Verificar si hay missings
            n_missing = self.df[var_code].isnull().sum()
            if n_missing == 0:
                continue
            
            tipo_var = var_info['tipo']
            estrategia = strategy.get(tipo_var, 'drop')
            
            if estrategia == 'median':
                valor_imputar = self.df[var_code].median()
                self.df[var_code].fillna(valor_imputar, inplace=True)
                print(f"  ✓ {var_code}: {n_missing} missings imputados con mediana ({valor_imputar:.1f})")
                
            elif estrategia == 'mode':
                valor_imputar = self.df[var_code].mode()[0] if not self.df[var_code].mode().empty else np.nan
                self.df[var_code].fillna(valor_imputar, inplace=True)
                print(f"  ✓ {var_code}: {n_missing} missings imputados con moda ({valor_imputar})")
            
            self.metadata['variables_imputadas'].append(var_code)
        
        self.metadata['registros_finales'] = len(self.df) 

        print("✓ Imputación completada")
        return self.df
    
    def remove_incomplete_records(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Elimina registros con más de un umbral de valores faltantes.
        
        Args:
            threshold (float): Proporción máxima de missings permitida (0-1).
        
        Return:
            pd.DataFrame: Dataset sin registros incompletos.
        """
        print(f"Eliminando registros con >{threshold*100:.0f}% de valores faltantes...")
        
        # Calcular % de missings por fila
        n_vars = len(self.df.columns)
        missings_por_fila = self.df.isnull().sum(axis=1) / n_vars
        
        # Identificar filas a eliminar
        filas_eliminar = missings_por_fila > threshold
        n_eliminar = filas_eliminar.sum()
        
        if n_eliminar > 0:
            self.df = self.df[~filas_eliminar].copy()
            self.metadata['registros_eliminados'] = n_eliminar
            print(f"  ✓ {n_eliminar} registros eliminados")
        else:
            print(f"  ✓ No hay registros que cumplan el criterio (lo suficientemente incompletos)")
        
        self.metadata['registros_finales'] = len(self.df)
        
        return self.df
    
    # REVISAR: Vale la pena este método?
    def detect_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> dict:
        """
        Detecta valores atípicos en variables numéricas.
        
        Args:
            method (str): Método de detección ('iqr' o 'zscore').
            threshold (float): Umbral para considerar outlier.
        
        Return:
            dict: Diccionario con outliers detectados por variable.
        """
        print(f"Detectando outliers con método '{method}'...")
        
        outliers = {}
        
        for var_code, var_info in TODAS_VARIABLES.items():
            if var_code not in self.df.columns or var_info['tipo'] != 'numerico':
                continue
            
            if method == 'iqr':
                Q1 = self.df[var_code].quantile(0.25)
                Q3 = self.df[var_code].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = ((self.df[var_code] < lower_bound) | 
                               (self.df[var_code] > upper_bound))
                
            elif method == 'zscore':
                z_scores = np.abs((self.df[var_code] - self.df[var_code].mean()) / 
                                 self.df[var_code].std())
                outlier_mask = z_scores > threshold
            
            n_outliers = outlier_mask.sum()
            if n_outliers > 0:
                outliers[var_code] = {
                    'count': n_outliers,
                    'percentage': (n_outliers / len(self.df)) * 100,
                    'indices': self.df[outlier_mask].index.tolist()
                }
                print(f"  ⚠ {var_code}: {n_outliers} outliers detectados ({outliers[var_code]['percentage']:.1f}%)")
        
        self.metadata['outliers'] = outliers
        
        if not outliers:
            print("  ✓ No se detectaron outliers significativos")
        
        return outliers
    
    def get_cleaning_report(self) -> dict:
        """
        Genera un reporte de la limpieza realizada.
        
        Return:
            dict: Reporte con estadísticas de limpieza.
        """
        return self.metadata


# ============================================
# FUNCIÓN DE LIMPIEZA COMPLETA
# ============================================
def clean_dataset(df: pd.DataFrame, 
                  validate_ranges: bool = True,
                  handle_missing: bool = True,
                  remove_incomplete: bool = False,
                  detect_outliers_flag: bool = True) -> tuple:
    """
    Función de conveniencia para ejecutar el pipeline de limpieza completo.
    
    Args:
        df (pd.DataFrame): Dataset a limpiar.
        validate_ranges (bool): Si validar rangos de variables.
        handle_missing (bool): Si imputar valores faltantes.
        remove_incomplete (bool): Si eliminar registros incompletos.
        detect_outliers_flag (bool): Si detectar outliers.
    
    Return:
        tuple: (DataFrame limpio, reporte de limpieza)
    """
    cleaner = DataCleaner(df)
    
    if validate_ranges:
        cleaner.validate_ranges()
    
    if remove_incomplete:
        cleaner.remove_incomplete_records()
        
    if handle_missing:
        cleaner.handle_missing_values()
    
    if detect_outliers_flag:
        cleaner.detect_outliers()
    
    return cleaner.df, cleaner.get_cleaning_report()