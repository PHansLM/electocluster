"""
Helpers de ejecucion integrada para la iteracion 5.

Centraliza la logica que antes quedaria repartida en Streamlit: carga del
dataset procesado, aplicacion de pesos, entrenamiento, calculo de metricas y
persistencia del run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.evaluation.feature_space import (
    ALL_SAMPLES_V1,
    CLUSTERED_WITHOUT_NOISE_V1,
    COMMON_PROCESSED_V1,
    DIRECT_WEIGHTED_DISTANCE_V1,
    GEOMETRY_METRICS_SCHEMA,
    HISTORICAL_READING_V1,
    PRECOMPUTED_DISTANCE_V1,
    WEIGHTED_PCA_V1,
    WEIGHTED_GEOMETRY_READING_V1,
    build_evaluation_context,
    build_metric_reading,
)
from src.evaluation.metrics import calculate_metrics, calculate_weighted_geometry_metrics
from src.evaluation.provenance import build_run_provenance
from src.evaluation.results_manager import ResultsManager
from src.utils.constants import PROCESSED_DATA_PATH
from src.weighting.weight_manager import WeightManager


ALGORITHMS = ["WKMedoids", "W-Hierarchical Clustering", "W-DBSCAN"]

# Los identificadores se conservan porque forman parte de resultados y controles
# históricos. Este catálogo describe con precisión lo que ejecuta el código.
ALGORITHM_IMPLEMENTATIONS = {
    "WKMedoids": {
        "implementation_name": "K-Medoids con distancia euclidiana global ponderada",
        "weighting_strategy": "Distancia precomputada sqrt(sum(w_i * (x_i - y_i)^2)) sobre las variables procesadas.",
        "method_family": "particional basado en medoides",
        "not_equivalent_to": [],
    },
    "W-Hierarchical Clustering": {
        "implementation_name": "Clustering aglomerativo con distancia euclidiana global ponderada",
        "weighting_strategy": "Matriz de distancia ponderada global con enlace configurable.",
        "method_family": "aglomerativo",
        "not_equivalent_to": ["Ward_p"],
    },
    "W-DBSCAN": {
        "implementation_name": "DBSCAN sobre PCA de variables ponderadas (pipeline histórico)",
        "weighting_strategy": "Transformación histórica X * weights, PCA y distancia euclidiana precomputada sobre las componentes.",
        "method_family": "clustering basado en densidad",
        "not_equivalent_to": ["W-DBSCANR"],
    },
}


class _UniformWeightManager:
    """WeightManager minimo para espacios ya ponderados como PCA."""

    def __init__(self, feature_order: list[str]):
        self._weights = {feature: 1.0 for feature in feature_order}

    def get_weights_array(self, feature_order: list[str] | None = None) -> list[float]:
        order = feature_order or list(self._weights)
        return [self._weights[feature] for feature in order]


@dataclass
class ExecutionResult:
    run_id: str | None
    algorithm: str
    params: dict
    metrics: dict
    labels: np.ndarray
    metadata: dict
    model: Any
    data: pd.DataFrame
    evaluation_data: pd.DataFrame
    evaluation_labels: np.ndarray


def load_processed_dataset(path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Carga el dataset procesado y valida que este listo para clustering."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No existe dataset procesado en {dataset_path}. Ejecuta preprocesamiento primero."
        )

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError("El dataset procesado esta vacio.")
    if df.isna().sum().sum() > 0:
        raise ValueError("El dataset procesado contiene valores faltantes.")
    return df


def run_clustering_experiment(
    algorithm: str,
    params: dict | None = None,
    dataset_path: str = PROCESSED_DATA_PATH,
    weight_manager: WeightManager | None = None,
    results_manager: ResultsManager | None = None,
    persist: bool = True,
) -> ExecutionResult:
    """
    Ejecuta un algoritmo ponderado y opcionalmente persiste el resultado.

    Para W-DBSCAN se replica la decision metodologica de la iteracion 4:
    primero se aplica ponderacion, luego PCA, y las metricas se calculan sin
    puntos de ruido.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algoritmo no soportado: {algorithm}")

    params = params.copy() if params else default_params(algorithm)
    weight_manager = weight_manager or WeightManager()
    results_manager = results_manager or ResultsManager()
    df = load_processed_dataset(dataset_path)

    if algorithm == "WKMedoids":
        result = _run_wkmedoids(df, params, weight_manager)
    elif algorithm == "W-Hierarchical Clustering":
        result = _run_whierarchical(df, params, weight_manager)
    else:
        result = _run_wdbscan(df, params, weight_manager)

    weights_snapshot = weight_manager.get_weights()
    result["metadata"] = {
        **result["metadata"],
        "implementation": describe_algorithm_implementation(algorithm),
        "provenance": build_run_provenance(
            dataset_path=dataset_path,
            df=df,
            algorithm=algorithm,
            params=result["params"],
            weights=weights_snapshot,
        ),
    }

    run_id = None
    if persist:
        run_id = results_manager.save_run(
            algorithm=algorithm,
            params=result["params"],
            weights=weights_snapshot,
            metrics=result["metrics"],
            labels=result["labels"].tolist(),
            metadata=result["metadata"],
        )

    return ExecutionResult(
        run_id=run_id,
        algorithm=algorithm,
        params=result["params"],
        metrics=result["metrics"],
        labels=result["labels"],
        metadata=result["metadata"],
        model=result["model"],
        data=df,
        evaluation_data=result["evaluation_data"],
        evaluation_labels=result["evaluation_labels"],
    )


def default_params(algorithm: str) -> dict:
    """Parametros canonicos usados como punto de partida en la interfaz."""
    if algorithm == "WKMedoids":
        return {"n_clusters": 13, "random_state": 42}
    if algorithm == "W-Hierarchical Clustering":
        return {"n_clusters": 2, "linkage": "complete"}
    if algorithm == "W-DBSCAN":
        return {"eps": 0.606, "min_samples": 34, "pca_components": 17}
    raise ValueError(f"Algoritmo no soportado: {algorithm}")


def canonical_experiments() -> list[tuple[str, dict]]:
    """Configuraciones finales documentadas en la iteracion 4."""
    return [(algorithm, default_params(algorithm)) for algorithm in ALGORITHMS]


def describe_algorithm_implementation(algorithm: str) -> dict:
    """Devuelve metadatos metodológicos sin cambiar identificadores históricos."""
    implementation = ALGORITHM_IMPLEMENTATIONS.get(algorithm)
    if implementation is None:
        raise ValueError(f"Algoritmo no soportado: {algorithm}")
    return {"identifier": algorithm, **implementation}


def _metric_readings(
    *,
    historical_metrics: dict,
    historical_space: str,
    historical_population: str,
    historical_silhouette_definition: str,
    historical_coordinate_definition: str,
    geometry_metrics: dict,
    geometry_space: str,
    geometry_population: str,
    geometry_silhouette_definition: str,
    geometry_coordinate_definition: str,
    geometry_relationship: str,
) -> dict:
    """Construye las dos lecturas sin cambiar el contrato histórico de metrics."""
    return {
        HISTORICAL_READING_V1: build_metric_reading(
            reading=HISTORICAL_READING_V1,
            metric_space=historical_space,
            population=historical_population,
            metrics=historical_metrics,
            silhouette_definition=historical_silhouette_definition,
            coordinate_definition=historical_coordinate_definition,
            relationship_to_historical="self",
        ),
        WEIGHTED_GEOMETRY_READING_V1: build_metric_reading(
            reading=WEIGHTED_GEOMETRY_READING_V1,
            metric_space=geometry_space,
            population=geometry_population,
            metrics=geometry_metrics,
            silhouette_definition=geometry_silhouette_definition,
            coordinate_definition=geometry_coordinate_definition,
            relationship_to_historical=geometry_relationship,
        ),
    }


def _legacy_geometry_metrics(reading: dict) -> dict:
    """Mantiene el campo geometry_metrics de artefactos de transición."""
    return {
        "schema": GEOMETRY_METRICS_SCHEMA,
        "space": reading["space"]["id"],
        "population": reading["population"]["id"],
        "metrics": reading["metrics"],
        "silhouette_definition": reading["silhouette_definition"],
        "coordinate_definition": reading["coordinate_definition"],
    }


def _run_wkmedoids(
    df: pd.DataFrame,
    params: dict,
    weight_manager: WeightManager,
) -> dict:
    from src.clustering.weighted.weighted_kmedoids import WKMedoids

    clean_params = {
        "n_clusters": int(params.get("n_clusters", 13)),
        "random_state": int(params.get("random_state", 42)),
    }
    model = WKMedoids(
        n_clusters=clean_params["n_clusters"],
        random_state=clean_params["random_state"],
        weight_manager=weight_manager,
    )
    model.fit(df)
    labels = np.asarray(model.labels_, dtype=int)
    metrics = _safe_metrics(df, labels)
    geometry_metrics = calculate_weighted_geometry_metrics(df, labels, weight_manager)
    metric_readings = _metric_readings(
        historical_metrics=metrics,
        historical_space=COMMON_PROCESSED_V1,
        historical_population=ALL_SAMPLES_V1,
        historical_silhouette_definition="euclidean_processed_dataset",
        historical_coordinate_definition="processed_features",
        geometry_metrics=geometry_metrics,
        geometry_space=DIRECT_WEIGHTED_DISTANCE_V1,
        geometry_population=ALL_SAMPLES_V1,
        geometry_silhouette_definition="precomputed_weighted_distance",
        geometry_coordinate_definition="processed_features_times_sqrt_weights",
        geometry_relationship="distinct_weighted_geometry",
    )
    metadata = {
        "evaluation_space": "processed_dataset",
        "excluded_noise": False,
        "n_evaluated": int(len(labels)),
        "inertia": float(model.inertia_) if model.inertia_ is not None else None,
        "metric_readings": metric_readings,
        "geometry_metrics": _legacy_geometry_metrics(
            metric_readings[WEIGHTED_GEOMETRY_READING_V1]
        ),
        "evaluation_context": build_evaluation_context(
            clustering_space=DIRECT_WEIGHTED_DISTANCE_V1,
            model_input_space=PRECOMPUTED_DISTANCE_V1,
            metric_space=COMMON_PROCESSED_V1,
            population=ALL_SAMPLES_V1,
            n_total=len(labels),
            n_evaluated=len(labels),
            evaluated_indices=df.index.tolist(),
        ),
    }
    return _result_dict(model, clean_params, metrics, labels, metadata, df, labels)


def _run_whierarchical(
    df: pd.DataFrame,
    params: dict,
    weight_manager: WeightManager,
) -> dict:
    from src.clustering.weighted.weighted_hierarchical import (
        WeightedHierarchicalClustering,
    )

    clean_params = {
        "n_clusters": int(params.get("n_clusters", 2)),
        "linkage": params.get("linkage", "complete"),
    }
    model = WeightedHierarchicalClustering(
        n_clusters=clean_params["n_clusters"],
        linkage=clean_params["linkage"],
        weight_manager=weight_manager,
    )
    model.fit(df)
    labels = np.asarray(model.labels_, dtype=int)
    metrics = _safe_metrics(df, labels)
    geometry_metrics = calculate_weighted_geometry_metrics(df, labels, weight_manager)
    metric_readings = _metric_readings(
        historical_metrics=metrics,
        historical_space=COMMON_PROCESSED_V1,
        historical_population=ALL_SAMPLES_V1,
        historical_silhouette_definition="euclidean_processed_dataset",
        historical_coordinate_definition="processed_features",
        geometry_metrics=geometry_metrics,
        geometry_space=DIRECT_WEIGHTED_DISTANCE_V1,
        geometry_population=ALL_SAMPLES_V1,
        geometry_silhouette_definition="precomputed_weighted_distance",
        geometry_coordinate_definition="processed_features_times_sqrt_weights",
        geometry_relationship="distinct_weighted_geometry",
    )
    metadata = {
        "evaluation_space": "processed_dataset",
        "excluded_noise": False,
        "n_evaluated": int(len(labels)),
        "metric_readings": metric_readings,
        "geometry_metrics": _legacy_geometry_metrics(
            metric_readings[WEIGHTED_GEOMETRY_READING_V1]
        ),
        "evaluation_context": build_evaluation_context(
            clustering_space=DIRECT_WEIGHTED_DISTANCE_V1,
            model_input_space=PRECOMPUTED_DISTANCE_V1,
            metric_space=COMMON_PROCESSED_V1,
            population=ALL_SAMPLES_V1,
            n_total=len(labels),
            n_evaluated=len(labels),
            evaluated_indices=df.index.tolist(),
        ),
    }
    return _result_dict(model, clean_params, metrics, labels, metadata, df, labels)


def _run_wdbscan(
    df: pd.DataFrame,
    params: dict,
    weight_manager: WeightManager,
) -> dict:
    from src.clustering.weighted.weighted_dbscan import WDBSCAN

    clean_params = {
        "eps": float(params.get("eps", 0.606)),
        "min_samples": int(params.get("min_samples", 34)),
        "pca_components": int(params.get("pca_components", 17)),
    }

    weights = np.asarray(
        weight_manager.get_weights_array(feature_order=df.columns.tolist()),
        dtype=float,
    )
    weighted_df = df * weights
    n_components = min(clean_params["pca_components"], weighted_df.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca_values = pca.fit_transform(weighted_df)
    df_pca = pd.DataFrame(
        pca_values,
        columns=[f"PCA_{i + 1}" for i in range(pca_values.shape[1])],
        index=df.index,
    )

    model = WDBSCAN(
        eps=clean_params["eps"],
        min_samples=clean_params["min_samples"],
        weight_manager=_UniformWeightManager(df_pca.columns.tolist()),
    )
    model.fit(df_pca)
    labels = np.asarray(model.labels_, dtype=int)
    eval_mask = labels != -1
    eval_data = df_pca.loc[eval_mask].copy()
    eval_labels = labels[eval_mask]
    metrics = _safe_metrics(eval_data, eval_labels)
    metric_readings = _metric_readings(
        historical_metrics=metrics,
        historical_space=WEIGHTED_PCA_V1,
        historical_population=CLUSTERED_WITHOUT_NOISE_V1,
        historical_silhouette_definition="euclidean_weighted_pca",
        historical_coordinate_definition="historical_weighted_features_then_pca",
        geometry_metrics=metrics,
        geometry_space=WEIGHTED_PCA_V1,
        geometry_population=CLUSTERED_WITHOUT_NOISE_V1,
        geometry_silhouette_definition="euclidean_weighted_pca",
        geometry_coordinate_definition="historical_weighted_features_then_pca",
        geometry_relationship="identical_space_and_population",
    )

    metadata = {
        "evaluation_space": "weighted_pca_without_noise",
        "excluded_noise": True,
        "n_evaluated": int(eval_mask.sum()),
        "pca_components": int(n_components),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "n_noise": int((labels == -1).sum()),
        "noise_percentage": float((labels == -1).sum() / len(labels) * 100),
        "metric_readings": metric_readings,
        "geometry_metrics": _legacy_geometry_metrics(
            metric_readings[WEIGHTED_GEOMETRY_READING_V1]
        ),
        "evaluation_context": build_evaluation_context(
            clustering_space=WEIGHTED_PCA_V1,
            model_input_space=PRECOMPUTED_DISTANCE_V1,
            metric_space=WEIGHTED_PCA_V1,
            population=CLUSTERED_WITHOUT_NOISE_V1,
            n_total=len(labels),
            n_evaluated=int(eval_mask.sum()),
            evaluated_indices=df.index[eval_mask].tolist(),
        ),
    }
    return _result_dict(model, clean_params, metrics, labels, metadata, eval_data, eval_labels)


def _safe_metrics(df: pd.DataFrame, labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(labels) == 0 or len(unique) < 2 or len(unique) >= len(labels):
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }

    try:
        return calculate_metrics(df, labels)
    except ValueError:
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }


def _result_dict(
    model: Any,
    params: dict,
    metrics: dict,
    labels: np.ndarray,
    metadata: dict,
    evaluation_data: pd.DataFrame,
    evaluation_labels: np.ndarray,
) -> dict:
    return {
        "model": model,
        "params": params,
        "metrics": metrics,
        "labels": labels,
        "metadata": metadata,
        "evaluation_data": evaluation_data,
        "evaluation_labels": np.asarray(evaluation_labels, dtype=int),
    }
