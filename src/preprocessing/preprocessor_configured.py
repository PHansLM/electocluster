"""
Función configurable para el pipeline de preprocesamiento.
Complementa DataPreprocessor sin modificarlo.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable
from .loader import load_dataset
from .cleaner import clean_dataset_configured
from .transformer import transform_dataset
from .feature_engineering import engineer_features
from .reporter import PreprocessingReporter
from src.utils.constants import PROCESSED_DATA_PATH


@dataclass
class PreprocessingConfig:
    validate_ranges: bool = True
    handle_missing: bool = True
    remove_incomplete: bool = False
    incomplete_threshold: float = 0.5
    detect_outliers: bool = True
    outlier_method: str = 'iqr'
    outlier_threshold: float = 1.5
    build_wealth: bool = True
    wealth_method: str = 'sum'
    build_civic: bool = True
    drop_originals: bool = True
    normalize: bool = True
    encode: bool = True
    normalization_method: str = 'minmax'


def run_pipeline_configured(
    filepath: str,
    config: PreprocessingConfig = None,
    save_report_path: str = None,
    report_metadata: dict = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> tuple:
    """
    Ejecuta el pipeline de preprocesamiento con configuración parametrizable.
    Orquesta las mismas funciones de conveniencia que DataPreprocessor.run_pipeline(),
    sin modificar esa clase.

    Args:
        filepath:         Ruta al archivo .dta
        config:           PreprocessingConfig con los parámetros. None usa defaults.
        save_report_path: Si se provee, guarda el reporte JSON en esa ruta.
        report_metadata: Metadata adicional de trazabilidad para la interfaz.

    Returns:
        tuple: (df_procesado, reporte_dict)
    """
    if config is None:
        config = PreprocessingConfig()

    reporter = PreprocessingReporter()
    total_steps = 5

    def notify(completed: int, message: str) -> None:
        if on_progress is not None:
            on_progress(completed, total_steps, message)

    # PASO 1: CARGA
    notify(0, "Cargando y verificando el dataset...")
    df_raw, load_metadata = load_dataset(filepath)
    reporter.add_section('loading', load_metadata)
    notify(1, "Dataset cargado; iniciando limpieza...")

    # PASO 2: LIMPIEZA
    df_clean, clean_report = clean_dataset_configured(
        df_raw,
        validate_ranges=config.validate_ranges,
        handle_missing=config.handle_missing,
        remove_incomplete=config.remove_incomplete,
        incomplete_threshold=config.incomplete_threshold,
        detect_outliers_flag=config.detect_outliers,
        outlier_method=config.outlier_method,
        outlier_threshold=config.outlier_threshold,
    )
    reporter.add_section('cleaning', clean_report)
    notify(2, "Limpieza completada; construyendo variables derivadas...")

    # PASO 3: FEATURE ENGINEERING
    df_engineered, _, engineering_report = engineer_features(
        df_clean,
        build_wealth=config.build_wealth,
        wealth_method=config.wealth_method,
        build_civic=config.build_civic,
        drop_originals=config.drop_originals,
    )
    reporter.add_section('feature_engineering', engineering_report)
    notify(3, "Variables derivadas listas; aplicando transformaciones...")

    # PASO 4: TRANSFORMACIÓN
    df_transformed, _, transform_report = transform_dataset(
        df_engineered,
        normalize=config.normalize,
        encode=config.encode,
        normalization_method=config.normalization_method,
    )
    reporter.add_section('transformation', transform_report)
    notify(4, "Transformaciones completadas; guardando resultados...")

    # GUARDAR CSV
    output_path = Path(PROCESSED_DATA_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_transformed.to_csv(output_path, index=False)

    report = reporter.generate_full_report()
    if report_metadata:
        report.update(reporter._convert_to_serializable(report_metadata))

    if save_report_path:
        output_report_path = Path(save_report_path)
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    notify(5, "Dataset procesado y reporte guardados.")

    return df_transformed, report
