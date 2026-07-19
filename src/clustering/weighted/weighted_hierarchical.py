"""
Implementación de Clustering Jerárquico tradicional
Usa scipy y sklearn para clustering aglomerativo.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from typing import Optional, Tuple
import matplotlib.pyplot as plt

from src.weighting.weight_manager import WeightManager

from ..base import BaseClusterer 
from ._weights import resolve_feature_weights


class WeightedHierarchicalClustering(BaseClusterer):
    """
    Clustering Jerárquico Ponderado
    
    Construye un árbol jerárquico (dendrograma) fusionando clusters
    progresivamente según su similitud (parte de muchos a pocos) 
    
    Attributes:
        n_clusters (int): Número de clusters finales
        linkage (str): Método de linkage ('complete', 'average')
        metric (str): Mé-trica de distancia
        linkage_matrix_ (np.ndarray): Matriz de linkage para dendrograma
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        linkage: str = 'average',
        metric: str = 'precomputed',
        weight_manager: Optional[WeightManager] = None
    ):
        """
        Inicializa Weighted Clustering Jerárquico
        
        Args:
            n_clusters: Número de clusters finales (default: 3)
            linkage: Método de linkage (default: 'average')
                - 'complete': máxima distancia entre clusters
                - 'average': promedio de distancias
                - 'single': mínima distancia entre clusters
            'metric': Métrica de distancia ('precomputed')
            'weight_manager': Modulo para ponderacion 
        """
        valid_linkages = {'complete', 'average', 'single'}
        if linkage not in valid_linkages:
            raise ValueError(
                f"linkage debe ser uno de {sorted(valid_linkages)}. Recibido: {linkage}"
            )
        if metric != 'precomputed':
            raise ValueError(
                "WeightedHierarchicalClustering requiere metric='precomputed' "
                "para usar la distancia ponderada documentada."
            )

        params = {
            'n_clusters': n_clusters,
            'linkage': linkage,
            'metric': 'precomputed',
            'weight_manager': weight_manager
        }
        super().__init__(**params)
        
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = 'precomputed'
        self.weight_manager = weight_manager
        
        # Inicializar modelo
        self._model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric='precomputed'
        )
        
        self.linkage_matrix_ = None
        
    def fit(self, X: pd.DataFrame) -> 'WeightedHierarchicalClustering':
        """
        Entrena clustering jerárquico ponderado
        
        Args:
            X: DataFrame con datos preprocesados
            
        Return:
            self: Modelo entrenado
        """
        print(f"Entrenando Clustering Jerárquico Ponderado ({self.linkage})...")
        
        X_array, weights_array, _ = resolve_feature_weights(
            X,
            self.weight_manager,
        )

        # Matriz de distancias ponderada NxN
        diff = X_array[:, np.newaxis, :] - X_array[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(weights_array * diff**2, axis=2))

        # Modelo necesita metric='precomputed'
        #self._model = AgglomerativeClustering(n_clusters=self.n_clusters, metric='precomputed', linkage='average')
        self._model.fit(dist_matrix)

        self.labels_ = self._model.labels_
        self.n_clusters_ = len(np.unique(self.labels_))

        # Dendrograma también usa la matriz
        condensed = squareform(dist_matrix)
        self.linkage_matrix_ = linkage(condensed, method=self.linkage)

        self.fitted_ = True
        
        print(f"✓ Clustering Jerárquico entrenado: {self.n_clusters_} clusters")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> None:
        """
        No soportado. Clustering jerárquico opera únicamente sobre 
        el dataset de entrenamiento. Use fit() sobre el dataset completo.
        """
        self._check_fitted()
        raise NotImplementedError(
            "predict() no está soportado en WeightedHierarchicalClustering. "
            "El clustering jerárquico no generaliza a nuevos datos. "
            "Use fit() sobre el dataset completo."
    )
    
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
