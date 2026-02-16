"""
Clase base abstracta para algoritmos de clustering
Una interfaz común que entre los algoritmos
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class BaseClusterer(ABC):
    """
    Clase base para algoritmos de clustering
    
    Attributes:
        params (Dict[str, Any]): Parámetros del algoritmo
        labels_ (np.ndarray): Etiquetas de cluster asignadas a cada punto
        n_clusters_ (int): Número de clusters identificados
        fitted_ (bool): Indica si el modelo ha sido entrenado
    """
    
    def __init__(self, **params):
        """
        Inicializa el clusterer con parámetros
        
        Args:
            **params: Parámetros específicos del algoritmo
        """
        self.params = params
        self.labels_ = None
        self.n_clusters_ = None
        self.fitted_ = False
        
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> 'BaseClusterer':
        """
        Entrena el modelo de clustering
        
        Args:
            X: DataFrame con datos preprocesados
            
        Return:
            self: Instancia entrenada
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice clusters en los datos ingresados
        
        Args:
            X: DataFrame con datos a predecir
            
        Return:
            Array con etiquetas de cluster
        """
        pass
    
    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Entrena el modelo y retorna las etiquetas
        
        Args:
            X: DataFrame con datos preprocesados
            
        Return:
            Array con etiquetas de cluster asignado
        """
        self.fit(X)
        return self.labels_
    
    def get_params(self) -> Dict[str, Any]:
        """Retorna los parámetros del algoritmo"""
        return self.params.copy()
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """
        Retorna el tamaño de cada cluster
        
        Return:
            Diccionario {cluster_id: tamaño}
        """
        if self.labels_ is None:
            raise ValueError("Modelo no entrenado")
        
        unique, counts = np.unique(self.labels_, return_counts=True)
        return dict(zip(unique.astype(int), counts.astype(int)))
    
    def get_cluster_distribution(self) -> pd.DataFrame:
        """
        Retorna distribución detallada de clusters
        
        Return:
            DataFrame con estadísticas por cluster
        """
        if self.labels_ is None:
            raise ValueError("Modelo no entrenado")
        
        sizes = self.get_cluster_sizes()
        total = len(self.labels_)
        
        distribution = []
        for cluster_id, size in sizes.items():
            distribution.append({
                'cluster': cluster_id,
                'size': size,
                'percentage': (size / total) * 100
            })
        
        return pd.DataFrame(distribution).sort_values('cluster')
    
    def _check_fitted(self):
        """Verifica modelo esté entrenado"""
        if not self.fitted_:
            raise ValueError(
                f"{self.__class__.__name__} no entrenado. "
                "Ejecuta fit() antes de predict()."
            )
    
    def __repr__(self):
        """Representación en string del clusterer"""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"