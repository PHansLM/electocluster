"""
Implementación de DBSCAN tradicional
Clustering basado en densidad
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN as SKLearnDBSCAN
from typing import Optional

from src.weighting.weight_manager import WeightManager

from ..base import BaseClusterer 
from ._weights import resolve_feature_weights


class WDBSCAN(BaseClusterer):
    """
    DBSCAN 
    
    Agrupa puntos densamente cercanos, marcando como outliers
    los puntos en regiones de baja densidad
    
    Attributes:
        eps (float): Distancia máxima entre dos puntos para ser vecinos
        min_samples (int): Mínimo de puntos para formar región densa
        core_sample_indices_ (np.ndarray): Índices de puntos core
        n_noise_points_ (int): Número de puntos marcados como ruido (-1)
        
        NOTA: metric y algorithm son fijas en constrate con DBSCAN estandar
    """
    
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = 'precomputed',
        algorithm: str = 'brute',
        leaf_size: int = 30,
        n_jobs: Optional[int] = -1,
        weight_manager: Optional[WeightManager] = None
    ):
        """
        Inicializa W-DBSCAN
        
        Args:
            eps: Radio epsilon de vecindad (default: 0.5)
            min_samples: Mínimo de vecinos para punto core (default: 5)
            metric: Métrica de distancia (default: 'euclidean')
            algorithm: Algoritmo de búsqueda ('auto', 'ball_tree', 'kd_tree', 'brute')
            leaf_size: Tamaño de hoja para ball_tree/kd_tree
            n_jobs: Número de trabajos paralelos (-1 = todos los cores)
            'weight_manager': Modulo para ponderacion 
        """
        params = {
            'eps': eps,
            'min_samples': min_samples,
            'metric': 'precomputed',
            'algorithm': 'brute',
            'leaf_size': leaf_size,
            'n_jobs': n_jobs,
            'weight_manager':weight_manager
        }
        super().__init__(**params)
        
        self.eps = eps
        self.min_samples = min_samples
        self.metric = 'precomputed'
        self.algorithm = 'brute'
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.weight_manager = weight_manager
        
        # Inicializar modelo
        self._model = SKLearnDBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='precomputed',
            algorithm='brute',
            leaf_size=leaf_size,
            n_jobs=n_jobs
        )
        
        self.core_sample_indices_ = None
        self.n_noise_points_ = None
        
    def fit(self, X: pd.DataFrame) -> 'WDBSCAN':
        """
        Entrena W-DBSCAN sobre los datos
        
        Args:
            X: DataFrame con datos preprocesados
            
        Return:
            self: Modelo entrenado
        """
        print(f"Entrenando DBSCAN (eps={self.eps}, min_samples={self.min_samples})...")
        
        X_array, weights_array, _ = resolve_feature_weights(
            X,
            self.weight_manager,
        )

        diff = X_array[:, np.newaxis, :] - X_array[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(weights_array * diff**2, axis=2))
        self._validate_precomputed_distances(dist_matrix)

        #self._model = SKLearnDBSCAN(eps=self.eps, min_samples=self.min_samples,
        #                            metric='precomputed', algorithm='brute')
        
        self._model.fit(dist_matrix)
        
        
        # Extraer resultados
        self.labels_ = self._model.labels_
        self.core_sample_indices_ = self._model.core_sample_indices_
        
        # Contar clusters y noise
        unique_labels = set(self.labels_)
        self.n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)
        self.n_noise_points_ = list(self.labels_).count(-1)
        
        self.fitted_ = True
        
        print(f"✓ WDBSCAN entrenado:")
        print(f"  Clusters identificados: {self.n_clusters_}")
        print(f"  Puntos de ruido: {self.n_noise_points_} ({(self.n_noise_points_/len(X_array))*100:.2f}%)")
        print(f"  Puntos core: {len(self.core_sample_indices_)}")
        
        return self

    @staticmethod
    def _validate_precomputed_distances(dist_matrix: np.ndarray) -> None:
        """Valida el contrato requerido por DBSCAN con metric precomputed."""
        if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
            raise ValueError("La matriz de distancias debe ser cuadrada.")
        if not np.all(np.isfinite(dist_matrix)):
            raise ValueError("La matriz de distancias debe contener valores finitos.")
        if np.any(dist_matrix < 0):
            raise ValueError("La matriz de distancias no puede contener valores negativos.")
        if not np.allclose(dist_matrix, dist_matrix.T):
            raise ValueError("La matriz de distancias debe ser simetrica.")
        if not np.allclose(np.diag(dist_matrix), 0.0):
            raise ValueError("La diagonal de la matriz de distancias debe ser cero.")

    # DBSCAN no tiene un predict perse, opera sobre el mismo dataset de entrenamiento
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        
        """
        Se retorna un fit_predict        
        
        Args:
            X: Datos a predecir
            
        Return:
            Etiquetas de cluster
        """
        print("⚠ Warning: DBSCAN no tiene predict. Usando fit_predict.")
        return self.fit_predict(X)
    
    def get_noise_points(self) -> np.ndarray:
        """
        Retorna índices de puntos marcados como ruido
        
        Return:
            Array con índices de outliers
        """
        self._check_fitted()
        return np.where(self.labels_ == -1)[0]
    
    def get_core_points(self) -> np.ndarray:
        """
        Retorna índices de puntos core
        
        Return:
            Array con índices de puntos core
        """
        self._check_fitted()
        return self.core_sample_indices_
    
    def get_cluster_sizes(self) -> dict:
        """
        Sobreescritura agregando manejo a noise points (-1)
        
        Return:
            Dict con tamaños por cluster (incluyendo -1 para noise)
        """
        self._check_fitted()
        
        unique, counts = np.unique(self.labels_, return_counts=True)
        sizes = dict(zip(unique.astype(int), counts.astype(int)))
        
        # Renombrar -1 a 'noise' para claridad
        if -1 in sizes:
            sizes['noise'] = sizes.pop(-1)
        
        return sizes
