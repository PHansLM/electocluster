"""
Implementación de Clustering Jerárquico tradicional
Usa scipy y sklearn para clustering aglomerativo.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Optional, Tuple
import matplotlib.pyplot as plt

from ..base import BaseClusterer 


class HierarchicalClustering(BaseClusterer):
    """
    Clustering Jerárquico Aglomerativo tradicional no ponderación
    
    Construye un árbol jerárquico (dendrograma) fusionando clusters
    progresivamente según su similitud (parte de muchos a pocos) 
    
    Attributes:
        n_clusters (int): Número de clusters finales
        linkage (str): Método de linkage ('ward', 'complete', 'average', 'single')
        metric (str): Mé-trica de distancia
        linkage_matrix_ (np.ndarray): Matriz de linkage para dendrograma
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        linkage: str = 'ward',
        metric: str = 'euclidean'
    ):
        """
        Inicializa Clustering Jerárquico
        
        Args:
            n_clusters: Número de clusters finales (default: 3)
            linkage: Método de linkage (default: 'ward')
                - 'ward': minimiza varianza dentro de clusters
                - 'complete': máxima distancia entre clusters
                - 'average': promedio de distancias
                - 'single': mínima distancia entre clusters
            metric: Métrica de distancia (default: 'euclidean')
        """
        params = {
            'n_clusters': n_clusters,
            'linkage': linkage,
            'metric': metric
        }
        super().__init__(**params)
        
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        
        # Validar linkage y metric
        if linkage == 'ward' and metric != 'euclidean':
            raise ValueError("Ward linkage requiere métrica euclidean")
        
        # Inicializar modelo
        self._model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric
        )
        
        self.linkage_matrix_ = None
        
    def fit(self, X: pd.DataFrame) -> 'HierarchicalClustering':
        """
        Entrena clustering jerárquico
        
        Args:
            X: DataFrame con datos preprocesados
            
        Return:
            self: Modelo entrenado
        """
        print(f"Entrenando Clustering Jerárquico ({self.linkage})...")
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Entrenar modelo
        self._model.fit(X_array)
        self.labels_ = self._model.labels_
        self.n_clusters_ = len(np.unique(self.labels_))
        
        # Calcular linkage matrix para dendrograma
        self.linkage_matrix_ = linkage(X_array, method=self.linkage)
        
        self.fitted_ = True
        
        print(f"✓ Clustering Jerárquico entrenado: {self.n_clusters_} clusters")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Asigna al cluster del punto más cercano.
        
        Args:
            X: Datos a predecir
            
        Return:
            Etiquetas de cluster
        """
        self._check_fitted()
        
        # Clustering jeraquico solo opera sobre el dataset usado en su entrenamiento, por lo que su predict es distinto
        # No se puede predecir con entradas diferentes 
        print("⚠ Warning: Hierarchical clustering requiere refit para nuevos datos")
        
        return self._model.fit_predict(X.values if isinstance(X, pd.DataFrame) else X)
    
    def plot_dendrogram(
        self,
        figsize: Tuple[int, int] = (12, 6),
        truncate_mode: Optional[str] = None,
        p: int = 30,
        save_path: Optional[str] = None
    ):
        """
        Genera dendrograma del clustering jerárquico
        
        Args:
            figsize: Tamaño de la figura (ancho, alto)
            truncate_mode: Modo de truncamiento ('lastp', 'level', None)
            p: Parámetro de truncamiento
            save_path: Ruta para guardar figura (opcional)
        """
        self._check_fitted()
        
        plt.figure(figsize=figsize)
        
        dendrogram(
            self.linkage_matrix_,
            truncate_mode=truncate_mode,
            p=p,
            leaf_font_size=8,
            show_leaf_counts=True
        )
        
        plt.title(f'Dendrograma - Clustering Jerárquico ({self.linkage})')
        plt.xlabel('Índice de muestra o (tamaño del cluster)')
        plt.ylabel('Distancia')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Dendrograma guardado en: {save_path}")
        
        plt.tight_layout()
        return plt.gcf()