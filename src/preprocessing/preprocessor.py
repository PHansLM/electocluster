"""
Módulo orquestador del preprocesamiento completo
"""

import json
from datetime import datetime
from pathlib import Path

from src.preprocessing.feature_engineering import engineer_features
from src.utils.constants import PROCESSED_DATA_PATH

from .loader import load_dataset
from .cleaner import clean_dataset
from .transformer import transform_dataset
from .reporter import PreprocessingReporter


class DataPreprocessor:
    """Clase principal que orquesta todo el preprocesamiento"""
    
    def __init__(self):
        self.reporter = PreprocessingReporter()
        self.df_raw = None
        self.df_clean = None
        self.df_engineered = None
        self.df_transformed = None
        
    def run_pipeline(self, filepath: str) -> 'pd.DataFrame':
        """
        Ejecuta preprocesamiento completo: load → clean → transform
        
        Args:
            filepath: Ruta al archivo .dta
            
        Return:
            pd.DataFrame: Dataset transformado listo para clustering
        """
        # PASO 1: CARGA
        self.df_raw, load_metadata = load_dataset(filepath)
        self.reporter.add_section('loading', load_metadata)
        
        # PASO 2: LIMPIEZA
        self.df_clean, clean_report = clean_dataset(
            self.df_raw,
            validate_ranges=True,
            handle_missing=True,
            remove_incomplete=False,
            detect_outliers_flag=True
        )
        self.reporter.add_section('cleaning', clean_report)
        
        # PASO 3: FEATURE ENGINEERING
        self.df_engineered, _, engineering_report = engineer_features(
            self.df_clean,
            build_wealth=True,
            wealth_method='sum',            # Alternativa: 'pca'
            build_civic=True,
            drop_originals=True
        )
        self.reporter.add_section('feature_engineering', engineering_report)

        # PASO 4: TRANSFORMACIÓN
        self.df_transformed, transformer, transform_report = transform_dataset(
            self.df_engineered,
            normalize=True,
            encode=True,
            normalization_method='minmax'
        )
        self.reporter.add_section('transformation', transform_report)
        
        output_path = Path(PROCESSED_DATA_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df_transformed.to_csv(output_path, index=False)
        print(f"\n✓ Dataset procesado guardado en: {output_path}")
        
        return self.df_transformed
    
    def generate_report(self, output_path: str = None) -> dict:
        """Genera reporte consolidado final"""
        return self.reporter.generate_full_report(output_path)
    
    def get_validation_summary(self) -> dict:
        """Retorna resumen de validación"""
        return self.reporter.get_summary()


def run_pipeline(filepath: str, output_report_path: str = None):
    """
    Funcion de conveniencia compatible con imports antiguos.

    Args:
        filepath: Ruta al archivo .dta de LAPOP.
        output_report_path: Ruta opcional para guardar el reporte JSON.

    Returns:
        DataFrame procesado listo para clustering.
    """
    preprocessor = DataPreprocessor()
    df = preprocessor.run_pipeline(filepath)
    if output_report_path is not None:
        preprocessor.generate_report(output_report_path)
    return df
