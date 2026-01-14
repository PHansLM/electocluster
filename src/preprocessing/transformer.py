"""
Módulo para transformación de datos del dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.constants import (
    TODAS_VARIABLES,
    METODO_NORMALIZACION
)


class DataTransformer:
    """
    Clase para transformación de datos: normalización de numéricas,
    encoding de categóricas, y preparación para clustering.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df (pd.DataFrame): Dataset a transformar.
        """
        self.df = df.copy()
        self.scalers = {}
        self.encoders = {}
        self.metadata = {
            'variables_normalizadas': [],
            'variables_codificadas': [],
            'metodo_normalizacion': None
        }
    
    def normalize_numeric_features(self, method: str = None) -> pd.DataFrame:
        """
        Normaliza variables numéricas.
        
        Args:
            method (str): Método de normalización ('minmax' o 'standard').
                         Si es None, usa METODO_NORMALIZACION.
        
        Return:
            pd.DataFrame: Dataset con variables numéricas normalizadas.
        """
        if method is None:
            method = METODO_NORMALIZACION
        
        self.metadata['metodo_normalizacion'] = method
        
        print(f"Normalizando variables numéricas con método '{method}'...")
        
        # Seleccionar scaler
        if method == 'minmax':
            scaler_class = MinMaxScaler
        elif method == 'standard':
            scaler_class = StandardScaler
        else:
            raise ValueError(f"Método desconocido: {method}")
        
        # Normalizar cada variable numérica
        for var_code, var_info in TODAS_VARIABLES.items():
            if var_code not in self.df.columns or var_info['tipo'] != 'numerico':
                continue
            
            # Aplicar scaler
            scaler = scaler_class()
            values = self.df[[var_code]].values
            self.df[var_code] = scaler.fit_transform(values)
            
            self.scalers[var_code] = scaler
            self.metadata['variables_normalizadas'].append(var_code)
            
            print(f"  ✓ {var_code} normalizado")
        
        print(f"✓ Normalización completada ({len(self.metadata['variables_normalizadas'])} variables)")
        return self.df
    
    def encode_categorical_features(self, method: str = 'ordinal') -> pd.DataFrame:
        """
        Codifica variables categóricas a formato numérico.
        
        Args:
            method (str): Método de codificación ('ordinal' o 'onehot').
        
        Return:
            pd.DataFrame: Dataset con variables categóricas codificadas.
        """
        print(f"Codificando variables categóricas con método '{method}'...")
        
        for var_code, var_info in TODAS_VARIABLES.items():
            if var_code not in self.df.columns:
                continue
            
            tipo = var_info['tipo']
            
            # Categóricas ordinales: Se mantienen sus numeros ordenados, se valida que sean enteros
            if tipo == 'categorico_ordinal':
                self.df[var_code] = self.df[var_code].astype(int)
                self.metadata['variables_codificadas'].append(var_code)
                print(f"  ✓ {var_code} (ordinal) mantenido como int")
            
            # Categóricas nominales: usar LabelEncoder
            elif tipo == 'categorico_nominal':
                encoder = LabelEncoder()
                self.df[var_code] = encoder.fit_transform(self.df[var_code].astype(int))
                self.encoders[var_code] = encoder
                self.metadata['variables_codificadas'].append(var_code)
                print(f"  ✓ {var_code} (nominal) codificado con LabelEncoder")
        
        print(f"✓ Codificación completada ({len(self.metadata['variables_codificadas'])} variables)")
        return self.df
    
    def get_feature_matrix(self) -> np.ndarray:
        """
        Obtiene la matriz de características lista para clustering.
        
        Return:
            np.ndarray: Matriz de características (n_samples, n_features).
        """
        return self.df.values
    
    def get_feature_names(self) -> list:
        """
        Obtiene los nombres de las características en el orden de la matriz.
        
        Return:
            list: Lista de nombres de características.
        """
        return self.df.columns.tolist()
    
    def inverse_transform_numeric(self, var_code: str, values: np.ndarray) -> np.ndarray:
        """
        Revierte la normalización de una variable numérica.
        
        Args:
            var_code (str): Código de la variable.
            values (np.ndarray): Valores normalizados.
        
        Return:
            np.ndarray: Valores en escala original.
        """
        if var_code not in self.scalers:
            raise ValueError(f"No hay scaler guardado para {var_code}")
        
        return self.scalers[var_code].inverse_transform(values.reshape(-1, 1)).flatten()
    
    def get_transformation_report(self) -> dict:
        """
        Genera un reporte de las transformaciones realizadas.
        
        Return:
            dict: Reporte con estadísticas de transformación.
        """
        return self.metadata


# ============================================
# FUNCIÓN DE TRANSFORMACIÓN COMPLETA
# ============================================
def transform_dataset(df: pd.DataFrame,
                      normalize: bool = True,
                      encode: bool = True,
                      normalization_method: str = None) -> tuple:
    """
    Función de conveniencia para ejecutar el pipeline de transformación completo.
    
    Args:
        df (pd.DataFrame): Dataset a transformar.
        normalize (bool): Si normalizar variables numéricas.
        encode (bool): Si codificar variables categóricas.
        normalization_method (str): Método de normalización.
    
    Return:
        tuple: (DataFrame transformado, transformer, reporte)
    """
    transformer = DataTransformer(df)
    
    if encode:
        transformer.encode_categorical_features()
    
    if normalize:
        transformer.normalize_numeric_features(method=normalization_method)
    
    return transformer.df, transformer, transformer.get_transformation_report()
