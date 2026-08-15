"""
Métricas de validación interna para clustering.

Implementa Silhouette Score, Davies-Bouldin Index y Calinski-Harabasz Index.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    pairwise_distances,
)
from typing import Dict, Any, Optional
import warnings

from src.weighting.weight_manager import WeightManager


class ClusteringMetrics:
    """
    Calculador de métricas de validación interna para clustering.
    
    Implementa tres métricas:
    - Silhouette Score: Cohesión y separación (mayor es mejor, rango [-1, 1])
    - Davies-Bouldin Index: Ratio de dispersión dentro y entre clusters (menor es mejor)
    - Calinski-Harabasz Index: Ratio de varianza dentor y entre clusters (mayor es mejor)
    
    Attributes:
        scores_ (Dict[str, float]): Diccionario con scores calculados
        n_samples_ (int): Número de muestras evaluadas
        n_clusters_ (int): Número de clusters en las etiquetas
    """
    
    def __init__(self):
        self.scores_ = {}
        self.n_samples_ = None
        self.n_clusters_ = None
        
    def calculate_all(
        self,
        X: pd.DataFrame,
        labels: np.ndarray,
        metric: str = 'euclidean'
    ) -> Dict[str, float]:
        """
        Calcula todas las métricas de validación.
        
        Args:
            X: DataFrame con datos (features normalizados)
            labels: Array con etiquetas de cluster
            metric: Métrica de distancia para Silhouette (default: 'euclidean')
            
        Return:
            Diccionario con todas las métricas calculadas
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        labels = np.array(labels)
        
        # Validaciones
        self._validate_inputs(X_array, labels)
        
        # Almacenar metadata
        self.n_samples_ = len(labels)
        self.n_clusters_ = len(np.unique(labels))
        
        print(f"Calculando métricas de validación...")
        print(f"  Muestras: {self.n_samples_}")
        print(f"  Clusters: {self.n_clusters_}")
        
        # Calcular cada métrica
        self.scores_ = {
            'silhouette': self.silhouette(X_array, labels, metric),
            'davies_bouldin': self.davies_bouldin(X_array, labels),
            'calinski_harabasz': self.calinski_harabasz(X_array, labels)
        }
        
        return self.scores_
    
    def silhouette(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        metric: str = 'euclidean'
    ) -> float:
        """
        Calcula Silhouette Score.
        
        Se mide qué tan similar es un elemento a su propio cluster comparado con
        otros clusters. Valores cercanos a 1 indican clustering apropiado,
        valores cercanos a 0 indican puntos en el borde entre clusters,
        valores negativos indican una posible asignación incorrecta.
        
        Args:
            X: Datos (n_samples, n_features)
            labels: Etiquetas de cluster
            metric: Métrica de distancia
            
        Return:
            Silhouette Score en rango [-1, 1]
        """
        try:
            score = silhouette_score(X, labels, metric=metric)
        except ValueError as e:
            warnings.warn(f"Error calculando Silhouette: {e}")
            return np.nan
        print(f"  [OK] Silhouette Score: {score:.4f}")
        return float(score)
    
    def davies_bouldin(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Calcula Davies-Bouldin Index.
        
        Se mide el promedio de similitud entre cada cluster y su cluster más similar
        Valores bajos indican mejor clustering (clusters separados y compactos)
        Rango: [0, ∞), óptimo: 0.
        
        Args:
            X: Datos (n_samples, n_features)
            labels: Etiquetas de cluster
            
        Return:
            Davies-Bouldin Index (menor es mejor)
        """
        try:
            score = davies_bouldin_score(X, labels)
        except ValueError as e:
            warnings.warn(f"Error calculando Davies-Bouldin: {e}")
            return np.nan
        print(f"  [OK] Davies-Bouldin Index: {score:.4f}")
        return float(score)
    
    def calinski_harabasz(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Calcula Calinski-Harabasz Index (Variance Ratio Criterion).
        
        Mide el ratio entre la dispersión inter cluster y la dispersión
        intra cluster. Valores altos indican clusters mejor definidos.
        Rango: [0, ∞), óptimo: máximo posible.
        
        Args:
            X: Datos (n_samples, n_features)
            labels: Etiquetas de cluster
            
        Return:
            Calinski-Harabasz Index (mayor es mejor)
        """
        try:
            score = calinski_harabasz_score(X, labels)
        except ValueError as e:
            warnings.warn(f"Error calculando Calinski-Harabasz: {e}")
            return np.nan
        print(f"  [OK] Calinski-Harabasz Index: {score:.4f}")
        return float(score)
    
    def _validate_inputs(self, X: np.ndarray, labels: np.ndarray):
        """
        Valida entradas antes de calcular métricas.
        
        Args:
            X: Datos
            labels: Etiquetas
            
        Raises:
            ValueError: Si entradas son inválidos
        """
        if len(X) != len(labels):
            raise ValueError(
                f"X y labels deben tener mismo tamaño: "
                f"X={len(X)}, labels={len(labels)}"
            )
        
        n_clusters = len(np.unique(labels))
        
        if n_clusters < 2:
            raise ValueError(
                f"Se requieren al menos 2 clusters para métricas. "
                f"Encontrados: {n_clusters}"
            )
        
        if n_clusters >= len(X):
            raise ValueError(
                f"Número de clusters ({n_clusters}) debe ser menor que "
                f"número de muestras ({len(X)})"
            )
    
    def get_scores(self) -> Dict[str, float]:
        """
        Retorna los scores calculados.
        
        Return:
            Diccionario con métricas
            
        Raises:
            ValueError: Si no se han calculado métricas
        """
        if not self.scores_:
            raise ValueError(
                "No hay scores calculados. Ejecuta calculate_all() primero."
            )
        return self.scores_.copy()
    
    def get_summary(self) -> pd.DataFrame:
        """
        Retorna resumen de métricas en DataFrame.
        
        Return:
            DataFrame con métricas y sus interpretaciones
        """
        if not self.scores_:
            raise ValueError(
                "No hay scores calculados. Ejecuta calculate_all() primero."
            )
        
        summary = []
        
        # Silhouette
        sil = self.scores_['silhouette']
        sil_interp = self._interpret_silhouette(sil)
        summary.append({
            'Métrica': 'Silhouette Score',
            'Valor': f"{sil:.4f}",
            'Rango': '[-1, 1]',
            'Óptimo': 'Cercano a 1',
            'Interpretación': sil_interp
        })
        
        # Davies-Bouldin
        db = self.scores_['davies_bouldin']
        db_interp = self._interpret_davies_bouldin(db)
        summary.append({
            'Métrica': 'Davies-Bouldin Index',
            'Valor': f"{db:.4f}",
            'Rango': '[0, ∞)',
            'Óptimo': 'Cercano a 0',
            'Interpretación': db_interp
        })
        
        # Calinski-Harabasz
        ch = self.scores_['calinski_harabasz']
        ch_interp = self._interpret_calinski_harabasz(ch)
        summary.append({
            'Métrica': 'Calinski-Harabasz Index',
            'Valor': f"{ch:.4f}",
            'Rango': '[0, ∞)',
            'Óptimo': 'Mayor posible',
            'Interpretación': ch_interp
        })
        
        return pd.DataFrame(summary)
    
    @staticmethod
    def _interpret_silhouette(score: float) -> str:
        """Interpreta Silhouette Score."""
        if score >= 0.7:
            return "Excelente estructura de clusters"
        elif score >= 0.5:
            return "Estructura razonable de clusters"
        elif score >= 0.25:
            return "Estructura débil, clusters se solapan"
        else:
            return "Sin estructura clara, clustering inadecuado"
    
    @staticmethod
    def _interpret_davies_bouldin(score: float) -> str:
        """Interpreta Davies-Bouldin Index."""
        if score <= 0.5:
            return "Clusters muy bien separados y compactos"
        elif score <= 1.0:
            return "Clusters razonablemente separados"
        elif score <= 1.5:
            return "Separación moderada entre clusters"
        else:
            return "Clusters se solapan significativamente"
    
    @staticmethod
    def _interpret_calinski_harabasz(score: float) -> str:
        """Interpreta Calinski-Harabasz Index."""
        if score >= 1000:
            return "Clusters muy bien definidos"
        elif score >= 500:
            return "Clusters bien definidos"
        elif score >= 100:
            return "Clusters moderadamente definidos"
        else:
            return "Clusters débilmente definidos"
    
    def compare_algorithms(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Compara métricas de múltiples algoritmos.
        
        Args:
            results: Dict de la forma {
                'Algoritmo1': {'silhouette': 0.5, 'davies_bouldin': 1.2, ...},
                'Algoritmo2': {...},
                ...
            }
            
        Return:
            DataFrame comparativo con rankings
        """
        comparison = []
        
        for algo_name, scores in results.items():
            comparison.append({
                'Algoritmo': algo_name,
                'Silhouette': scores.get('silhouette', np.nan),
                'Davies-Bouldin': scores.get('davies_bouldin', np.nan),
                'Calinski-Harabasz': scores.get('calinski_harabasz', np.nan)
            })
        
        df = pd.DataFrame(comparison)
        
        # Agregar rankings (1 = mejor)
        df['Rank_Silhouette'] = df['Silhouette'].rank(ascending=False)
        df['Rank_Davies-Bouldin'] = df['Davies-Bouldin'].rank(ascending=True)  # Menor es mejor
        df['Rank_Calinski-Harabasz'] = df['Calinski-Harabasz'].rank(ascending=False)
        
        # Ranking promedio
        df['Rank_Promedio'] = df[[
            'Rank_Silhouette',
            'Rank_Davies-Bouldin',
            'Rank_Calinski-Harabasz'
        ]].mean(axis=1)
        
        return df.sort_values('Rank_Promedio')


def calculate_metrics(
    X: pd.DataFrame,
    labels: np.ndarray,
    metric: str = 'euclidean'
) -> Dict[str, float]:
    """
    Función helper para calcular todas las métricas rápidamente.
    
    Args:
        X: DataFrame con datos
        labels: Etiquetas de cluster
        metric: Métrica de distancia
        
    Return:
        Dict con todas las métricas
    """
    evaluator = ClusteringMetrics()
    return evaluator.calculate_all(X, labels, metric)


def calculate_weighted_geometry_metrics(
    X: pd.DataFrame,
    labels: np.ndarray,
    weight_manager: WeightManager,
) -> Dict[str, float]:
    """Calcula metricas en la geometria ponderada usada por WK/WH.

    La distancia de los algoritmos ponderados es
    ``sqrt(sum(w_i * (x_i - y_i)^2))``. Para Silhouette se utiliza la
    matriz de distancias exacta; Davies-Bouldin y Calinski-Harabasz se
    calculan sobre la representacion euclidiana equivalente ``X * sqrt(w)``.

    Esta funcion es complementaria a ``calculate_metrics`` y no reemplaza
    las metricas historicas calculadas sobre el espacio procesado comun.
    """
    X_array = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    try:
        weights = np.asarray(
            weight_manager.get_weights_array(
                feature_order=X.columns.tolist() if isinstance(X, pd.DataFrame) else None
            ),
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("No se pudieron alinear los pesos con las columnas de X.") from exc

    if X_array.ndim != 2 or len(X_array) != len(labels):
        raise ValueError("X y labels deben tener dimensiones compatibles.")
    if weights.ndim != 1 or weights.size != X_array.shape[1]:
        raise ValueError("La cantidad de pesos debe coincidir con las columnas de X.")
    if not np.all(np.isfinite(X_array)) or not np.all(np.isfinite(weights)):
        raise ValueError("X y los pesos deben contener valores finitos.")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Los pesos deben ser no negativos y al menos uno debe ser positivo.")

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        raise ValueError("Se requieren entre 2 y n-1 clusters para calcular metricas.")

    # ||(x - y) * sqrt(w)||_2 es exactamente la distancia ponderada utilizada
    # por WKMedoids y el aglomerativo ponderado. Se evita construir un tensor
    # n x n x d, cuya memoria crece mucho más que la matriz necesaria para
    # Silhouette.
    weighted_coordinates = X_array * np.sqrt(weights)
    distance_matrix = pairwise_distances(weighted_coordinates, metric="euclidean")

    return {
        "silhouette": float(
            silhouette_score(distance_matrix, labels, metric="precomputed")
        ),
        "davies_bouldin": float(
            davies_bouldin_score(weighted_coordinates, labels)
        ),
        "calinski_harabasz": float(
            calinski_harabasz_score(weighted_coordinates, labels)
        ),
    }

def calculate_metrics_weighted(
    X: pd.DataFrame,
    labels: np.ndarray,
    weight_manager: WeightManager,
    metric: str = 'euclidean'
) -> Dict[str, float]:
    """
    Helper historico para calcular metricas en ``weighted_features_v1``.
    Útil para mostrar como es que el algoritmo ponderado percibe los elementos del dataset y la distancia entre ellos, justificado las agrupaciones
    No aplicable a todos los algoritmos ni comparaciones

    Conserva la transformacion X * weights usada por las iteraciones previas.
    No equivale a sustituir las variables por X * sqrt(weights).
    """
    weights = weight_manager.get_weights_array(feature_order=X.columns.tolist())
    X_weighted = X * weights
    evaluator = ClusteringMetrics()
    return evaluator.calculate_all(X_weighted, labels, metric)
