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

from src.clustering.weighted import WKMedoids, WDBSCAN, WeightedHierarchicalClustering
from src.evaluation.feature_space import (
    ALL_SAMPLES_V1,
    CLUSTERED_WITHOUT_NOISE_V1,
    COMMON_PROCESSED_V1,
    DIRECT_WEIGHTED_DISTANCE_V1,
    PRECOMPUTED_DISTANCE_V1,
    WEIGHTED_PCA_V1,
    build_evaluation_context,
)
from src.evaluation.metrics import calculate_metrics
from src.evaluation.provenance import build_run_provenance
from src.evaluation.results_manager import ResultsManager
from src.utils.constants import PROCESSED_DATA_PATH
from src.weighting.weight_manager import WeightManager


ALGORITHMS = ["WKMedoids", "W-Hierarchical Clustering", "W-DBSCAN"]


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


def _run_wkmedoids(
    df: pd.DataFrame,
    params: dict,
    weight_manager: WeightManager,
) -> dict:
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
    metadata = {
        "evaluation_space": "processed_dataset",
        "excluded_noise": False,
        "n_evaluated": int(len(labels)),
        "inertia": float(model.inertia_) if model.inertia_ is not None else None,
        "evaluation_context": build_evaluation_context(
            clustering_space=DIRECT_WEIGHTED_DISTANCE_V1,
            model_input_space=PRECOMPUTED_DISTANCE_V1,
            metric_space=COMMON_PROCESSED_V1,
            population=ALL_SAMPLES_V1,
            n_total=len(labels),
            n_evaluated=len(labels),
        ),
    }
    return _result_dict(model, clean_params, metrics, labels, metadata, df, labels)


def _run_whierarchical(
    df: pd.DataFrame,
    params: dict,
    weight_manager: WeightManager,
) -> dict:
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
    metadata = {
        "evaluation_space": "processed_dataset",
        "excluded_noise": False,
        "n_evaluated": int(len(labels)),
        "evaluation_context": build_evaluation_context(
            clustering_space=DIRECT_WEIGHTED_DISTANCE_V1,
            model_input_space=PRECOMPUTED_DISTANCE_V1,
            metric_space=COMMON_PROCESSED_V1,
            population=ALL_SAMPLES_V1,
            n_total=len(labels),
            n_evaluated=len(labels),
        ),
    }
    return _result_dict(model, clean_params, metrics, labels, metadata, df, labels)


def _run_wdbscan(
    df: pd.DataFrame,
    params: dict,
    weight_manager: WeightManager,
) -> dict:
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

    metadata = {
        "evaluation_space": "weighted_pca_without_noise",
        "excluded_noise": True,
        "n_evaluated": int(eval_mask.sum()),
        "pca_components": int(n_components),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "n_noise": int((labels == -1).sum()),
        "noise_percentage": float((labels == -1).sum() / len(labels) * 100),
        "evaluation_context": build_evaluation_context(
            clustering_space=WEIGHTED_PCA_V1,
            model_input_space=PRECOMPUTED_DISTANCE_V1,
            metric_space=WEIGHTED_PCA_V1,
            population=CLUSTERED_WITHOUT_NOISE_V1,
            n_total=len(labels),
            n_evaluated=int(eval_mask.sum()),
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
