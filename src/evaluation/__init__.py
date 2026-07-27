"""
Modulo de evaluacion de clustering.

Expone utilidades seguras de importar sin cargar algoritmos opcionales en el
arranque de la app. Los componentes pesados o sensibles a versionado, como el
optimizador basado en K-Medoids, deben importarse desde su submodulo concreto.
"""

from .comparability import assess_run_comparability, run_evaluation_summary
from .feature_space import (
    ALL_SAMPLES_V1,
    CLUSTERED_WITHOUT_NOISE_V1,
    COMMON_PROCESSED_V1,
    DIRECT_WEIGHTED_DISTANCE_V1,
    PRECOMPUTED_DISTANCE_V1,
    WEIGHTED_FEATURES_V1,
    WEIGHTED_PCA_V1,
    build_evaluation_context,
)
from .metrics import ClusteringMetrics, calculate_metrics, calculate_metrics_weighted
from .provenance import build_run_provenance, sha256_file, stored_dataset_sha256
from .report_generator import ReportGenerator
from .results_manager import ResultsManager

__all__ = [
    "ClusteringMetrics",
    "calculate_metrics",
    "calculate_metrics_weighted",
    "ResultsManager",
    "ReportGenerator",
    "build_run_provenance",
    "sha256_file",
    "stored_dataset_sha256",
    "ALL_SAMPLES_V1",
    "CLUSTERED_WITHOUT_NOISE_V1",
    "COMMON_PROCESSED_V1",
    "DIRECT_WEIGHTED_DISTANCE_V1",
    "PRECOMPUTED_DISTANCE_V1",
    "WEIGHTED_FEATURES_V1",
    "WEIGHTED_PCA_V1",
    "build_evaluation_context",
    "assess_run_comparability",
    "run_evaluation_summary",
]
