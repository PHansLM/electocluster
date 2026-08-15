"""Identificadores de espacios historicos usados en la evaluacion integrada."""

from __future__ import annotations

import hashlib
import json


EVALUATION_CONTEXT_SCHEMA = "iteration5-evaluation-context-v1"
INTERNAL_METRICS_V1 = "sklearn_internal_metrics_v1"
METRIC_READING_SCHEMA = "iteration5-metric-reading-v1"
GEOMETRY_METRICS_SCHEMA = "iteration5-geometry-metrics-v1"

HISTORICAL_READING_V1 = "historical"
WEIGHTED_GEOMETRY_READING_V1 = "weighted_geometry"

COMMON_PROCESSED_V1 = "common_processed_v1"
DIRECT_WEIGHTED_DISTANCE_V1 = "direct_weighted_distance_v1"
WEIGHTED_FEATURES_V1 = "weighted_features_v1"
WEIGHTED_PCA_V1 = "weighted_pca_v1"
PRECOMPUTED_DISTANCE_V1 = "precomputed_distance_v1"

ALL_SAMPLES_V1 = "all_samples_v1"
CLUSTERED_WITHOUT_NOISE_V1 = "clustered_samples_without_noise_v1"


FEATURE_SPACES = {
    COMMON_PROCESSED_V1: {
        "label": "Dataset procesado comun",
        "description": "Matriz numerica de 28 variables producida por el preprocesamiento historico.",
    },
    DIRECT_WEIGHTED_DISTANCE_V1: {
        "label": "Distancia ponderada directa",
        "description": "sqrt(sum(w_i * (x_i - y_i)^2)), sin transformar el dataset base.",
    },
    WEIGHTED_FEATURES_V1: {
        "label": "Variables ponderadas historicas",
        "description": "Transformacion historica X * weights, conservada sin sustitucion por sqrt(weights).",
    },
    WEIGHTED_PCA_V1: {
        "label": "PCA ponderado historico",
        "description": "PCA aplicado despues de la transformacion historica X * weights.",
    },
    PRECOMPUTED_DISTANCE_V1: {
        "label": "Matriz de distancias precomputada",
        "description": "Entrada cuadrada entregada al algoritmo con metric='precomputed'.",
    },
}


EVALUATION_POPULATIONS = {
    ALL_SAMPLES_V1: {
        "label": "Todas las observaciones",
        "description": "Las metricas se calculan sobre todas las etiquetas producidas.",
    },
    CLUSTERED_WITHOUT_NOISE_V1: {
        "label": "Observaciones agrupadas sin ruido",
        "description": "Las metricas excluyen las observaciones etiquetadas como ruido (-1).",
    },
}


def describe_feature_space(space_id: str) -> dict:
    """Devuelve una descripcion serializable de un espacio registrado."""
    if space_id not in FEATURE_SPACES:
        raise ValueError(f"Espacio de caracteristicas desconocido: {space_id}")
    return {"id": space_id, **FEATURE_SPACES[space_id]}


def describe_evaluation_population(population_id: str) -> dict:
    """Devuelve una descripcion serializable de una poblacion registrada."""
    if population_id not in EVALUATION_POPULATIONS:
        raise ValueError(f"Poblacion de evaluacion desconocida: {population_id}")
    return {"id": population_id, **EVALUATION_POPULATIONS[population_id]}


def build_evaluation_context(
    *,
    clustering_space: str,
    model_input_space: str,
    metric_space: str,
    population: str,
    n_total: int,
    n_evaluated: int,
    evaluated_indices: list | None = None,
) -> dict:
    """Registra donde se agrupo y donde se calcularon las metricas."""
    n_total = int(n_total)
    n_evaluated = int(n_evaluated)
    if n_total < 0 or not 0 <= n_evaluated <= n_total:
        raise ValueError(
            "n_evaluated debe estar entre cero y n_total: "
            f"n_total={n_total}, n_evaluated={n_evaluated}."
        )
    return {
        "schema": EVALUATION_CONTEXT_SCHEMA,
        "clustering_space": describe_feature_space(clustering_space),
        "model_input_space": describe_feature_space(model_input_space),
        "metric_space": describe_feature_space(metric_space),
        "population": describe_evaluation_population(population),
        "metric_definition": {
            "id": INTERNAL_METRICS_V1,
            "metrics": ["silhouette", "davies_bouldin", "calinski_harabasz"],
        },
        "n_total": n_total,
        "n_evaluated": n_evaluated,
        "evaluation_index_sha256": _sequence_sha256(evaluated_indices),
    }


def build_metric_reading(
    *,
    reading: str,
    metric_space: str,
    population: str,
    metrics: dict,
    silhouette_definition: str,
    coordinate_definition: str,
    relationship_to_historical: str,
) -> dict:
    """Describe una lectura de métricas sin mezclarla con otra geometría."""
    if reading not in {HISTORICAL_READING_V1, WEIGHTED_GEOMETRY_READING_V1}:
        raise ValueError(f"Lectura de métricas desconocida: {reading}")
    return {
        "schema": METRIC_READING_SCHEMA,
        "reading": reading,
        "space": describe_feature_space(metric_space),
        "population": describe_evaluation_population(population),
        "metric_definition": {
            "id": INTERNAL_METRICS_V1,
            "metrics": ["silhouette", "davies_bouldin", "calinski_harabasz"],
        },
        "metrics": dict(metrics),
        "silhouette_definition": silhouette_definition,
        "coordinate_definition": coordinate_definition,
        "relationship_to_historical": relationship_to_historical,
    }


def _sequence_sha256(values: list | None) -> str | None:
    if values is None:
        return None
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
