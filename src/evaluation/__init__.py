"""
Módulo de evaluación de clustering.

Proporciona métricas de validación interna y herramientas de comparación.
"""

from .metrics import ClusteringMetrics, calculate_metrics, calculate_metrics_weighted
from .parameter_optimizer import (
    ParameterSearchResult,
    optimize_wdbscan_params,
    optimize_whierarchical_params,
    optimize_wkmedoids_params,
)
from .results_manager import ResultsManager
from .report_generator import ReportGenerator
from .provenance import build_run_provenance, sha256_file, stored_dataset_sha256
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

__all__ = [
    'ClusteringMetrics',
    'calculate_metrics',
    'calculate_metrics_weighted',
    'ParameterSearchResult',
    'optimize_wdbscan_params',
    'optimize_whierarchical_params',
    'optimize_wkmedoids_params',
    'ResultsManager',
    'ReportGenerator',
    'build_run_provenance',
    'sha256_file',
    'stored_dataset_sha256',
    'ALL_SAMPLES_V1',
    'CLUSTERED_WITHOUT_NOISE_V1',
    'COMMON_PROCESSED_V1',
    'DIRECT_WEIGHTED_DISTANCE_V1',
    'PRECOMPUTED_DISTANCE_V1',
    'WEIGHTED_FEATURES_V1',
    'WEIGHTED_PCA_V1',
    'build_evaluation_context',
]
