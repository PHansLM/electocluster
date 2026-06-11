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
]
