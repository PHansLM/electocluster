"""
Módulo de evaluación de clustering.

Proporciona métricas de validación interna y herramientas de comparación.
"""

from .metrics import ClusteringMetrics, calculate_metrics

__all__ = [
    'ClusteringMetrics',
    'calculate_metrics'
]