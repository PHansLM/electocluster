"""
Busqueda ligera de parametros para la interfaz de configuracion.

La busqueda se ejecuta sobre una muestra controlada para evitar que WKMedoids
y W-Hierarchical agoten memoria al construir matrices de distancia completas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from src.clustering.weighted import WKMedoids, WDBSCAN, WeightedHierarchicalClustering
from src.evaluation.execution import _UniformWeightManager
from src.evaluation.metrics import calculate_metrics
from src.weighting.weight_manager import WeightManager


@dataclass
class ParameterSearchResult:
    algorithm: str
    best_params: dict
    rows: list[dict]
    sample_size: int
    criterion: str
    notes: list[str]


def sample_dataset(
    df: pd.DataFrame,
    sample_size: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """Devuelve una muestra reproducible sin alterar el indice original."""
    if sample_size >= len(df):
        return df.copy()
    return df.sample(n=sample_size, random_state=random_state).sort_index().copy()


def optimize_wkmedoids_params(
    df: pd.DataFrame,
    weight_manager: WeightManager,
    k_values: Iterable[int],
    sample_size: int = 300,
    random_state: int = 42,
) -> ParameterSearchResult:
    """Busca k para WKMedoids maximizando Silhouette en muestra."""
    sample = sample_dataset(df, sample_size=sample_size, random_state=random_state)
    rows = []

    for k in sorted(set(int(value) for value in k_values)):
        if k < 2 or k >= len(sample):
            continue
        model = WKMedoids(
            n_clusters=k,
            random_state=random_state,
            weight_manager=weight_manager,
        )
        model.fit(sample)
        metrics = _safe_metrics(sample, model.labels_)
        rows.append({
            "n_clusters": k,
            "random_state": random_state,
            "silhouette": metrics.get("silhouette"),
            "davies_bouldin": metrics.get("davies_bouldin"),
            "calinski_harabasz": metrics.get("calinski_harabasz"),
            "inertia": float(model.inertia_) if model.inertia_ is not None else None,
        })

    best = _best_row(rows)
    best_params = {
        "n_clusters": int(best["n_clusters"]),
        "random_state": random_state,
    } if best else {"n_clusters": 3, "random_state": random_state}

    return ParameterSearchResult(
        algorithm="WKMedoids",
        best_params=best_params,
        rows=rows,
        sample_size=len(sample),
        criterion="Mayor Silhouette; Davies-Bouldin como apoyo",
        notes=[
            "Busqueda calculada sobre muestra para controlar memoria.",
            "Confirma el resultado con una ejecucion final sobre el dataset completo.",
        ],
    )


def optimize_whierarchical_params(
    df: pd.DataFrame,
    weight_manager: WeightManager,
    k_values: Iterable[int],
    linkages: Iterable[str],
    sample_size: int = 300,
    random_state: int = 42,
) -> ParameterSearchResult:
    """Busca k y linkage para W-Hierarchical maximizando Silhouette."""
    sample = sample_dataset(df, sample_size=sample_size, random_state=random_state)
    rows = []

    for linkage in linkages:
        for k in sorted(set(int(value) for value in k_values)):
            if k < 2 or k >= len(sample):
                continue
            model = WeightedHierarchicalClustering(
                n_clusters=k,
                linkage=linkage,
                weight_manager=weight_manager,
            )
            model.fit(sample)
            metrics = _safe_metrics(sample, model.labels_)
            rows.append({
                "n_clusters": k,
                "linkage": linkage,
                "silhouette": metrics.get("silhouette"),
                "davies_bouldin": metrics.get("davies_bouldin"),
                "calinski_harabasz": metrics.get("calinski_harabasz"),
            })

    best = _best_row(rows)
    best_params = {
        "n_clusters": int(best["n_clusters"]),
        "linkage": best["linkage"],
    } if best else {"n_clusters": 2, "linkage": "complete"}

    return ParameterSearchResult(
        algorithm="W-Hierarchical Clustering",
        best_params=best_params,
        rows=rows,
        sample_size=len(sample),
        criterion="Mayor Silhouette; Davies-Bouldin como apoyo",
        notes=[
            "Busqueda calculada sobre muestra para controlar memoria.",
            "Las etiquetas y el dendrograma usan el mismo enlace seleccionado.",
        ],
    )


def optimize_wdbscan_params(
    df: pd.DataFrame,
    weight_manager: WeightManager,
    sample_size: int = 500,
    random_state: int = 42,
    variance_target: float = 0.85,
    min_samples_values: Iterable[int] | None = None,
) -> ParameterSearchResult:
    """
    Recomienda eps, min_samples y PCA para W-DBSCAN.

    Metodo:
      1. Aplica pesos.
      2. Escoge componentes PCA por varianza acumulada.
      3. Detecta codos k-distance.
      4. Evalua candidatos alrededor del codo, excluyendo ruido en metricas.
    """
    sample = sample_dataset(df, sample_size=sample_size, random_state=random_state)
    weighted = _apply_weights(sample, weight_manager)

    pca_full = PCA(random_state=random_state)
    pca_full.fit(weighted)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_target) + 1)
    n_components = max(2, min(n_components, weighted.shape[1], len(sample) - 1))

    pca = PCA(n_components=n_components, random_state=random_state)
    pca_values = pca.fit_transform(weighted)
    df_pca = pd.DataFrame(
        pca_values,
        columns=[f"PCA_{i + 1}" for i in range(pca_values.shape[1])],
        index=sample.index,
    )

    if min_samples_values is None:
        min_samples_values = _default_min_samples(n_components, len(sample))

    rows = []
    for min_samples in sorted(set(int(value) for value in min_samples_values)):
        if min_samples < 2 or min_samples >= len(df_pca):
            continue
        distances = _kdistance(df_pca.values, min_samples)
        elbow_idx, elbow_eps = _find_elbow(distances)
        eps_candidates = sorted({
            round(float(elbow_eps * factor), 3)
            for factor in (0.85, 1.0, 1.15)
            if elbow_eps * factor > 0
        })

        for eps in eps_candidates:
            model = WDBSCAN(
                eps=eps,
                min_samples=min_samples,
                weight_manager=_UniformWeightManager(df_pca.columns.tolist()),
            )
            model.fit(df_pca)
            labels = np.asarray(model.labels_, dtype=int)
            mask = labels != -1
            metrics = _safe_metrics(df_pca.loc[mask], labels[mask])
            noise_pct = float((labels == -1).sum() / len(labels) * 100)
            rows.append({
                "eps": eps,
                "min_samples": min_samples,
                "pca_components": n_components,
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                "elbow_eps": float(elbow_eps),
                "elbow_index": int(elbow_idx),
                "n_clusters": int(len(set(label for label in labels if label != -1))),
                "n_noise": int((labels == -1).sum()),
                "noise_percentage": noise_pct,
                "silhouette": metrics.get("silhouette"),
                "davies_bouldin": metrics.get("davies_bouldin"),
                "calinski_harabasz": metrics.get("calinski_harabasz"),
            })

    best = _best_row(rows)
    if best:
        best_params = {
            "eps": float(best["eps"]),
            "min_samples": int(best["min_samples"]),
            "pca_components": int(best["pca_components"]),
        }
    else:
        fallback_min_samples = max(2, min(2 * n_components, len(sample) - 1))
        fallback_eps = float(_find_elbow(_kdistance(df_pca.values, fallback_min_samples))[1])
        best_params = {
            "eps": round(fallback_eps, 3),
            "min_samples": fallback_min_samples,
            "pca_components": n_components,
        }

    return ParameterSearchResult(
        algorithm="W-DBSCAN",
        best_params=best_params,
        rows=rows,
        sample_size=len(sample),
        criterion="Mayor Silhouette sin puntos de ruido; codo k-distance como base",
        notes=[
            "Los pesos se aplican antes de PCA.",
            "Las metricas se calculan excluyendo ruido.",
            "DBSCAN puede preferir soluciones con mucho ruido; revisa n_noise antes de aceptar.",
        ],
    )


def _apply_weights(df: pd.DataFrame, weight_manager: WeightManager) -> pd.DataFrame:
    weights = np.asarray(
        weight_manager.get_weights_array(feature_order=df.columns.tolist()),
        dtype=float,
    )
    return df * weights


def _safe_metrics(df: pd.DataFrame, labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    if len(labels) == 0:
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    try:
        return calculate_metrics(df, labels)
    except ValueError:
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}


def _best_row(rows: list[dict]) -> dict | None:
    valid_rows = [row for row in rows if _finite(row.get("silhouette"))]
    if not valid_rows:
        return None
    return sorted(
        valid_rows,
        key=lambda row: (
            row["silhouette"],
            -row["davies_bouldin"] if _finite(row.get("davies_bouldin")) else -math.inf,
            row["calinski_harabasz"] if _finite(row.get("calinski_harabasz")) else -math.inf,
        ),
        reverse=True,
    )[0]


def _finite(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _default_min_samples(n_components: int, n_samples: int) -> list[int]:
    candidates = [
        max(5, n_components),
        max(5, int(n_components * 1.5)),
        max(5, n_components * 2),
    ]
    return [min(value, n_samples - 1) for value in candidates]


def _kdistance(values: np.ndarray, k: int) -> np.ndarray:
    n_neighbors = min(k + 1, len(values))
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    distances, _ = neighbors.fit(values).kneighbors(values)
    kth_index = min(k, distances.shape[1] - 1)
    return np.sort(distances[:, kth_index])[::-1]


def _find_elbow(distances: np.ndarray) -> tuple[int, float]:
    n = len(distances)
    if n == 0:
        return 0, 0.0
    if n < 3:
        return 0, float(distances[0])

    p1 = np.array([0, distances[0]])
    p2 = np.array([n - 1, distances[-1]])
    line_vec = p2 - p1
    norm = np.linalg.norm(line_vec)
    if norm == 0:
        return 0, float(distances[0])
    line_vec_norm = line_vec / norm

    perp_distances = []
    for i, distance in enumerate(distances):
        point = np.array([i, distance])
        vec_to_point = point - p1
        projection = np.dot(vec_to_point, line_vec_norm) * line_vec_norm
        perp_distances.append(np.linalg.norm(vec_to_point - projection))

    elbow_idx = int(np.argmax(perp_distances))
    return elbow_idx, float(distances[elbow_idx])
