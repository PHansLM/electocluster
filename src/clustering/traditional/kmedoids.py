"""
Implementación de K-Medoids tradicional
Emplea scikit-learn para la implementación base.
"""

import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids as SKLearnKMedoids
from typing import Optional

from ..base import BaseClusterer 


class KMedoids(BaseClusterer):
    """
    K-Medoids tradicional
    
    Attributes:
        n_clusters (int): Número de clusters a encontrar
        metric (str): Métrica de distancia ('euclidean', 'manhattan')
        method (str): Método de inicialización ('pam', 'alternate', 'build')
        max_iter (int): Número max de iteraciones
        random_state (int): Semilla para reproducibilidad
        medoid_indices_ (np.ndarray): Índices de los medoides finales
        inertia_ (float): Suma de distancias al medoide más cercano (compacidad)
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        metric: str = 'manhattan',
        method: str = 'pam',
        max_iter: int = 300,
        random_state: Optional[int] = 42
    ):
        """
        Inicializa K-Medoids
        
        Args:
            n_clusters: Número de clusters (default: 3)
            metric: Métrica de distancia (default: 'euclidean')
            method: Método de inicialización (default: 'pam')
            max_iter: Iteraciones máximas (default: 300)
            random_state: Semilla aleatoria (default: 42)
        """
        params = {
            'n_clusters': n_clusters,
            'metric': metric,
            'method': method,
            'max_iter': max_iter,
            'random_state': random_state
        }
        super().__init__(**params)
        
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Inicialización del modelo base de scikit-learn
        self._model = SKLearnKMedoids(
            n_clusters=n_clusters,
            metric=metric,
            method=method,
            max_iter=max_iter,
            random_state=random_state
        )
        
        self.medoid_indices_ = None
        self.inertia_ = None
        
    def fit(self, X: pd.DataFrame) -> 'KMedoids':
        """
        Entrenar K-Medoids
        
        Args:
            X: DataFrame con datos preprocesados
            
        Return:
            self: Modelo entrenado
        """
        print(f"Entrenando K-Medoids con {self.n_clusters} clusters...")
        
        # Conversion a array
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Entrenamiento del modelo
        self._model.fit(X_array)
        
        # Extraer resultados
        self.labels_ = self._model.labels_
        self.medoid_indices_ = self._model.medoid_indices_
        self.inertia_ = self._model.inertia_
        self.n_clusters_ = len(np.unique(self.labels_))
        self.fitted_ = True
        
        print(f"✓ K-Medoids entrenado: {self.n_clusters_} clusters identificados")
        print(f"  Inercia: {self.inertia_:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice clusters para nuevos datos
        
        Args:
            X: DataFrame con datos a predecir
            
        Return:
            Array con etiquetas de cluster
        """
        self._check_fitted()
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self._model.predict(X_array)
    
    def get_medoids(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna los medoides (centros de clusters)
        
        Args:
            X: DataFrame original usado en fit()
            
        Return:
            DataFrame con los medoides
        """
        self._check_fitted()
        
        if isinstance(X, pd.DataFrame):
            return X.iloc[self.medoid_indices_].copy()
        else:
            return pd.DataFrame(
                X[self.medoid_indices_],
                columns=[f'feature_{i}' for i in range(X.shape[1])]
            )