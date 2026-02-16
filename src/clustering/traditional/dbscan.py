"""
Implementación de DBSCAN tradicional
Clustering basado en densidad
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN as SKLearnDBSCAN
from typing import Optional

from ..base import BaseClusterer 


class DBSCAN(BaseClusterer):
    """
    DBSCAN 
    
    Agrupa puntos densamente cercanos, marcando como outliers
    los puntos en regiones de baja densidad
    
    Attributes:
        eps (float): Distancia máxima entre dos puntos para ser vecinos
        min_samples (int): Mínimo de puntos para formar región densa
        metric (str): Métrica de distancia
        core_sample_indices_ (np.ndarray): Índices de puntos core
        n_noise_points_ (int): Número de puntos marcados como ruido (-1)
    """
    
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = 'euclidean',
        algorithm: str = 'auto',
        leaf_size: int = 30,
        n_jobs: Optional[int] = -1
    ):
        """
        Inicializa DBSCAN
        
        Args:
            eps: Radio epsilon de vecindad (default: 0.5)
            min_samples: Mínimo de vecinos para punto core (default: 5)
            metric: Métrica de distancia (default: 'euclidean')
            algorithm: Algoritmo de búsqueda ('auto', 'ball_tree', 'kd_tree', 'brute')
            leaf_size: Tamaño de hoja para ball_tree/kd_tree
            n_jobs: Número de trabajos paralelos (-1 = todos los cores)
        """
        params = {
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric,
            'algorithm': algorithm,
            'leaf_size': leaf_size,
            'n_jobs': n_jobs
        }
        super().__init__(**params)
        
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        
        # Inicializar modelo
        self._model = SKLearnDBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=n_jobs
        )
        
        self.core_sample_indices_ = None
        self.n_noise_points_ = None
        
    def fit(self, X: pd.DataFrame) -> 'DBSCAN':
        """
        Entrena DBSCAN sobre los datos
        
        Args:
            X: DataFrame con datos preprocesados
            
        Return:
            self: Modelo entrenado
        """
        print(f"Entrenando DBSCAN (eps={self.eps}, min_samples={self.min_samples})...")
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Entrenar modelo
        self._model.fit(X_array)
        
        # Extraer resultados
        self.labels_ = self._model.labels_
        self.core_sample_indices_ = self._model.core_sample_indices_
        
        # Contar clusters y noise
        unique_labels = set(self.labels_)
        self.n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)
        self.n_noise_points_ = list(self.labels_).count(-1)
        
        self.fitted_ = True
        
        print(f"✓ DBSCAN entrenado:")
        print(f"  Clusters identificados: {self.n_clusters_}")
        print(f"  Puntos de ruido: {self.n_noise_points_} ({(self.n_noise_points_/len(X_array))*100:.2f}%)")
        print(f"  Puntos core: {len(self.core_sample_indices_)}")
        
        return self

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