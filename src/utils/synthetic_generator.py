import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.constants import LAPOP_MISSING_CODES


class SyntheticDataGenerator:
    """
    Genera datasets sintéticos con problemas controlados para testing.
    """
    
    def __init__(self, n_samples: int = 500, seed: int = 42):
        """
        Inicializa el generador.
        
        Args:
            n_samples: Nro de registros a generar.
            seed: Semill.
        """
        self.n_samples = n_samples
        self.seed = seed
        np.random.seed(seed)
        self.df = None
    
    def generate_clean_data(self) -> pd.DataFrame:
        """
        Genera dataset sintético SIN problemas (baseline).
        
        Return:
            Dataset limpio con estructura LAPOP Bolivia.
        """
        data = {}
        
        # Variables numéricas
        data['q2'] = np.random.randint(18, 86, self.n_samples)
        data['q12cn'] = np.random.randint(1, 11, self.n_samples)
        
        # Variables ordinales
        data['edre'] = np.random.randint(0, 7, self.n_samples)
        data['q10inc'] = np.random.randint(1001, 1016, self.n_samples)
        
        # Variables nominales
        data['etid'] = np.random.choice([1, 2, 3, 4, 5, 7], self.n_samples)
        data['ur'] = np.random.choice([1, 2], self.n_samples, p=[0.68, 0.32])
        data['ocupoit'] = np.random.randint(1, 11, self.n_samples)
        data['q1tc_r'] = np.random.randint(1, 3, self.n_samples)
        data['q11n'] = np.random.randint(1, 7, self.n_samples)
        data['boletidnew'] = np.random.choice([1, 2], self.n_samples)
        data['q3cn'] = np.random.choice([1, 2, 3, 4, 5, 7, 11, 77], self.n_samples)
        
        self.df = pd.DataFrame(data)
        return self.df
    
    def inject_missing_codes(self, variables: Optional[List[str]] = None, 
                            percentage: float = 0.05) -> pd.DataFrame:
        """
        Inyecta códigos especiales de LAPOP (888888, 988888, 999999).
        
        Args:
            variables: Variables a afectar. Si None, afecta a todas.
            percentage: Porcentaje de registros a afectar por código.
        
        Return:
            Dataset con códigos especiales.
        """
        if self.df is None:
            self.generate_clean_data()
        
        if variables is None:
            variables = self.df.columns.tolist()
        
        for var in variables:
            if var not in self.df.columns:
                continue
            
            n_affect = int(self.n_samples * percentage)
            
            for code in LAPOP_MISSING_CODES.keys():
                indices = np.random.choice(self.n_samples, n_affect, replace=False)
                self.df.loc[indices, var] = code
        
        return self.df
    
    def inject_out_of_range_values(self, variable_ranges: Optional[Dict] = None) -> pd.DataFrame:
        """
        Inyecta valores fuera de rango válido.
        
        Args:
            variable_ranges: {variable: (pct, invalid_value)}
        
        Return:
            Dataset con valores fuera de rango.
        """
        if self.df is None:
            self.generate_clean_data()
        
        if variable_ranges is None:
            variable_ranges = {
                'q2': (0.03, 200),
                'edre': (0.02, 10),
                'q12cn': (0.04, 30),
                'q3cn': (0.02, 99)
            }
        
        for var, (pct, invalid_val) in variable_ranges.items():
            if var not in self.df.columns:
                continue
            
            n_affect = int(self.n_samples * pct)
            indices = np.random.choice(self.n_samples, n_affect, replace=False)
            self.df.loc[indices, var] = invalid_val
        
        return self.df
    
    def inject_true_nans(self, variables: Optional[List[str]] = None, 
                        percentage: float = 0.10) -> pd.DataFrame:
        """
        Inyecta valores NaN verdaderos (no códigos especiales).
        
        Args:
            variables: Variables a afectar.
            percentage: Porcentaje de NaN a inyectar.
        
        Return:
            Dataset con NaN.
        """
        if self.df is None:
            self.generate_clean_data()
        
        if variables is None:
            variables = ['q10inc', 'ocupoit', 'edre']
        
        for var in variables:
            if var not in self.df.columns:
                continue
            
            n_affect = int(self.n_samples * percentage)
            indices = np.random.choice(self.n_samples, n_affect, replace=False)
            self.df.loc[indices, var] = np.nan
        
        return self.df
    
    def inject_outliers(self, variable_outliers: Optional[Dict] = None) -> pd.DataFrame:
        """
        Inyecta outliers en variables numéricas.
        
        Args:
            variable_outliers: {variable: (pct, outlier_value)}
        
        Return:
            Dataset con outliers.
        """
        if self.df is None:
            self.generate_clean_data()
        
        if variable_outliers is None:
            variable_outliers = {
                'q2': (0.02, 120),
                'q12cn': (0.05, [15, 18, 20, 25])
            }
        
        for var, (pct, outlier_vals) in variable_outliers.items():
            if var not in self.df.columns:
                continue
            
            n_affect = int(self.n_samples * pct)
            indices = np.random.choice(self.n_samples, n_affect, replace=False)
            
            if isinstance(outlier_vals, list):
                values = np.random.choice(outlier_vals, n_affect)
            else:
                values = outlier_vals
            
            self.df.loc[indices, var] = values
        
        return self.df
    
    def generate_problematic_dataset(self, problem_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Genera dataset con múltiples problemas combinados.
        
        Args:
            problem_types: Lista de problemas a inyectar.
        
        Return:
            Dataset con varios problemas.
        """
        if problem_types is None:
            problem_types = ['missing_codes', 'out_of_range', 'nans', 'outliers']
        
        self.generate_clean_data()
        
        if 'missing_codes' in problem_types:
            self.inject_missing_codes(percentage=0.08)
        
        if 'out_of_range' in problem_types:
            self.inject_out_of_range_values()
        
        if 'nans' in problem_types:
            self.inject_true_nans(percentage=0.12)
        
        if 'outliers' in problem_types:
            self.inject_outliers()
        
        return self.df
    
    def save_dataset(self, output_path: str):
        """
        Guarda el dataset generado en formato .dta (Stata).
        
        Args:
            output_path: Ruta de salida (debe terminar en .dta)
        """
        if self.df is None:
            raise ValueError("Debe generar un dataset primero")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar en formato Stata
        self.df.to_stata(output_path, write_index=False)
        print(f"✓ Dataset sintético guardado en: {output_path}")
    
    def print_summary(self):
        """Imprime resumen del dataset generado."""
        if self.df is None:
            print("No hay dataset generado")
            return
        
        print("\n" + "=" * 70)
        print("RESUMEN DEL DATASET SINTÉTICO")
        print("=" * 70)
        print(f"Dimensiones: {len(self.df)} registros × {len(self.df.columns)} variables")
        
        print("\nValores faltantes (NaN):")
        missing_counts = self.df.isnull().sum()
        has_missing = False
        for var, count in missing_counts.items():
            if count > 0:
                pct = (count / len(self.df)) * 100
                print(f"  {var:15} {count:4} ({pct:5.1f}%)")
                has_missing = True
        if not has_missing:
            print("  (ninguno)")
        
        print("\nCódigos especiales LAPOP:")
        has_codes = False
        for code, label in LAPOP_MISSING_CODES.items():
            count = (self.df == code).sum().sum()
            if count > 0:
                print(f"  {label:20} {count:4}")
                has_codes = True
        if not has_codes:
            print("  (ninguno)")
        
        print("=" * 70)