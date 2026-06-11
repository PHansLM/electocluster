"""
Módulo para carga
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.constants import (
    LAPOP_MISSING_CODES,
    TODAS_VARIABLES,
    RAW_DATA_PATH
)


class DatasetLoader:
    """
    Clase para cargar el dataset y hacer validaciones base
    """
    
    def __init__(self, file_path: str = None):
        """
        Args:
            file_path (str): Ruta al dataset en .dta o .csv
        """
        self.file_path = file_path or RAW_DATA_PATH
        self.df = None
        self.metadata = {}
        
    def load(self) -> pd.DataFrame:
        """
        Carga el dataset desde el archivo .dta o .csv
        
        Return:
            pd.DataFrame: Dataset cargado.
            
        Exceptions:
            FileNotFoundError: Si el archivo no existe.
            Exception: Si hay error en la carga.
        """
        try:
            print(f"Cargando dataset desde: {self.file_path}")
            suffix = Path(self.file_path).suffix.lower()
            if suffix == ".dta":
                self.df = pd.read_stata(self.file_path, convert_categoricals=False)
            elif suffix == ".csv":
                self.df = pd.read_csv(self.file_path)
            else:
                raise ValueError("Formato no soportado. Usa archivos .dta o .csv.")
            
            # Metadata básica
            self.metadata['n_registros'] = len(self.df)
            self.metadata['n_variables_total'] = len(self.df.columns)
            self.metadata['columnas'] = list(self.df.columns)
            
            print(f"✓ Dataset cargado: {self.metadata['n_registros']} registros × "
                  f"{self.metadata['n_variables_total']} variables")
            
            return self.df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Archivo no encontrado: {self.file_path}")
        except Exception as e:
            raise Exception(f"Error al cargar dataset: {str(e)}")
    
    def select_variables(self) -> pd.DataFrame:
        """
        Selecciona solo las variables esperadas (Las seleccionadas previamente y definidas en constants.py)
        
        Return:
            pd.DataFrame: Dataset con variables seleccionadas.
        """
        if self.df is None:
            raise ValueError("Dataset no cargado: Debe cargar el dataset primero con load()")
        
        # Variables a seleccionar (códigos en minúsculas)
        variables_requeridas = list(TODAS_VARIABLES.keys())
        
        # Verificar cuáles existen en el dataset
        variables_existentes = [v for v in variables_requeridas if v in self.df.columns]
        variables_faltantes = [v for v in variables_requeridas if v not in self.df.columns]
        
        if variables_faltantes:
            print(f"⚠ Variables no encontradas en el dataset: {variables_faltantes}")
        
        # Filtrar para tener solo las variables que interesan
        self.df = self.df[variables_existentes].copy()
        
        self.metadata['n_variables_seleccionadas'] = len(variables_existentes)
        self.metadata['variables_seleccionadas'] = variables_existentes
        
        print(f"✓ Variables seleccionadas: {len(variables_existentes)}/{len(variables_requeridas)}")
        
        return self.df
    
    def replace_missing_codes(self) -> pd.DataFrame:
        """
        Reemplaza códigos especiales de LAPOP (888888, 988888, 999999) por NaN (para evitar que interfieran al analisis posterior).
        
        Return:
            pd.DataFrame: Dataset con códigos especiales reemplazados por NaN.
        """
        if self.df is None:
            raise ValueError("Debe cargar el dataset primero con load()")
        
        print("Reemplazando códigos especiales de LAPOP por NaN...")
        
        # Reemplazar cada código especial
        for codigo in LAPOP_MISSING_CODES.keys():
            self.df.replace(codigo, np.nan, inplace=True)
        
        # Contar missings por variable
        missings_count = self.df.isnull().sum()
        missings_pct = (missings_count / len(self.df) * 100).round(1)
        
        self.metadata['missings'] = {
            'count': missings_count.to_dict(),
            'percentage': missings_pct.to_dict()
        }
        
        print(f"✓ Códigos especiales reemplazados por NaN")
        
        return self.df
    
    def get_summary(self) -> dict:
        """
        Genera un resumen del dataset cargado.
        
        Return:
            dict: Diccionario con estadísticas del dataset.
        """
        if self.df is None:
            raise ValueError("Debe cargar el dataset primero con load()")
        
        summary = {
            'shape': self.df.shape,
            'n_registros': len(self.df),
            'n_variables': len(self.df.columns),
            'missings_totales': self.df.isnull().sum().sum(),
            'missings_por_variable': self.df.isnull().sum().to_dict(),
            'dtypes': self.df.dtypes.to_dict()
        }
        
        return summary
    
    def print_summary(self):
        """
        Imprime un resumen del dataset cargado.
        """
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("RESUMEN DEL DATASET CARGADO")
        print("=" * 70)
        print(f"Dimensiones: {summary['n_registros']} registros × {summary['n_variables']} variables")
        print(f"Missings totales: {summary['missings_totales']}")
        print("\nMissings por variable:")
        for var, count in summary['missings_por_variable'].items():
            if count > 0:
                pct = (count / summary['n_registros']) * 100
                print(f"  {var:15} {count:5} ({pct:5.1f}%)")
        print("=" * 70)


# ============================================
# FUNCIÓN DE CARGA COMPLETA
# ============================================
def load_dataset(file_path: str = None) -> tuple:
    """
    Función de conveniencia para cargar el dataset completo.
    
    Args:
        file_path (str): Ruta al archivo .dta o .csv.
        
    Return:
        tuple: (DataFrame procesado, metadata del loader)
    """
    loader = DatasetLoader(file_path)
    loader.load()
    loader.select_variables()
    loader.replace_missing_codes()
    loader.print_summary()
    
    return loader.df, loader.metadata
