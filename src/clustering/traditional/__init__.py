"""
Módulo de algoritmos de clustering tradicionales (sin ponderación).

Proporciona implementaciones de K-Medoids, Clustering Jerárquico y DBSCAN
como línea base para comparación con versiones ponderadas.
"""

from .kmedoids import KMedoids
from .hierarchical import HierarchicalClustering
from .dbscan import DBSCAN

__all__ = [
    'KMedoids',
    'HierarchicalClustering',
    'DBSCAN'
]