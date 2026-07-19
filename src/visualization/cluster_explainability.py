"""
Analisis explicativo para perfiles de clusters.

Estas funciones agregan vistas interpretativas sin alterar las salidas
historicas de clustering_plots.py usadas por notebooks previos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .profile_interpreter import feature_display_name, interpret_feature_value


EXPLANATORY_DIMENSIONS = {
    "Socioeconomia": ["edre", "q10inc", "ocupoit", "formal", "wealth_index"],
    "Territorio": ["estratopri", "estratosec", "ur"],
    "Identidad cultural": ["etid", "boletidnew", "boletidnewb", "leng1", "leng4"],
    "Religion y valores": ["q3cn", "q5b"],
    "Hogar y ciclo de vida": ["q2", "q1tc_r", "q11n", "q12cn", "q12bn"],
    "Participacion e informacion": ["gi0n", "smedia3n", "civic_index", "q10e"],
    "Relacion con Estado": ["wf1", "bolcct1a", "bolcct1b", "bolcct1c"],
}


def cluster_profile_deviation(
    df: pd.DataFrame,
    labels,
    exclude_noise: bool = True,
) -> pd.DataFrame:
    """
    Calcula diferencia entre cada perfil de cluster y el promedio global.

    Retorna formato largo con una fila por cluster-variable. Los valores se
    mantienen en la escala procesada del prototipo.
    """
    profile_matrix, global_mean = _cluster_profile_matrix(df, labels, exclude_noise)
    if profile_matrix.empty:
        return pd.DataFrame()

    rows = []
    for cluster_id, cluster_values in profile_matrix.iterrows():
        for feature in profile_matrix.columns:
            cluster_value = float(cluster_values[feature])
            global_value = float(global_mean[feature])
            deviation = cluster_value - global_value
            rows.append({
                "cluster": str(int(cluster_id)),
                "cluster_id": int(cluster_id),
                "feature": feature,
                "feature_name": feature_display_name(feature),
                "cluster_value": cluster_value,
                "global_value": global_value,
                "deviation": deviation,
                "abs_deviation": abs(deviation),
            })

    return pd.DataFrame(rows)


def top_distinctive_features(
    df: pd.DataFrame,
    labels,
    top_n: int = 5,
    exclude_noise: bool = True,
) -> pd.DataFrame:
    """
    Obtiene las variables mas distintivas de cada cluster.

    La distincion se mide como desviacion absoluta respecto al promedio global.
    Se agregan interpretaciones legibles para el valor del cluster y el global.
    """
    deviations = cluster_profile_deviation(df, labels, exclude_noise)
    if deviations.empty:
        return deviations

    rows = []
    for cluster_id, group in deviations.groupby("cluster_id", sort=True):
        ranked = group.sort_values("abs_deviation", ascending=False).head(top_n)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            cluster_interpretation = interpret_feature_value(
                row["feature"],
                row["cluster_value"],
            )
            global_interpretation = interpret_feature_value(
                row["feature"],
                row["global_value"],
            )
            rows.append({
                "cluster": str(int(cluster_id)),
                "rank": rank,
                "feature": row["feature"],
                "feature_name": row["feature_name"],
                "cluster_value": row["cluster_value"],
                "global_value": row["global_value"],
                "deviation": row["deviation"],
                "abs_deviation": row["abs_deviation"],
                "cluster_interpretation": cluster_interpretation.text,
                "global_interpretation": global_interpretation.text,
            })

    return pd.DataFrame(rows)


def cluster_profile_distance_matrix(
    df: pd.DataFrame,
    labels,
    exclude_noise: bool = True,
) -> pd.DataFrame:
    """
    Calcula distancias euclidianas entre perfiles promedio de clusters.

    Menor distancia implica perfiles demograficos mas parecidos en la escala
    procesada del prototipo.
    """
    profile_matrix, _ = _cluster_profile_matrix(df, labels, exclude_noise)
    if profile_matrix.empty:
        return pd.DataFrame()

    values = profile_matrix.to_numpy(dtype=float)
    diff = values[:, None, :] - values[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    labels_str = [str(int(cluster_id)) for cluster_id in profile_matrix.index]
    return pd.DataFrame(distances, index=labels_str, columns=labels_str)


def cluster_dimension_scores(
    df: pd.DataFrame,
    labels,
    dimensions: dict[str, list[str]] | None = None,
    exclude_noise: bool = True,
) -> pd.DataFrame:
    """
    Resume perfiles por dimensiones conceptuales agregadas.

    Cada dimension es el promedio de las variables disponibles en la escala
    procesada [0, 1]. Es una vista de lectura, no una metrica de validacion.
    """
    profile_matrix, global_mean = _cluster_profile_matrix(df, labels, exclude_noise)
    if profile_matrix.empty:
        return pd.DataFrame()

    dimensions = dimensions or EXPLANATORY_DIMENSIONS
    rows = []
    for dimension, features in dimensions.items():
        present_features = [feature for feature in features if feature in profile_matrix.columns]
        if not present_features:
            continue

        global_score = float(global_mean[present_features].mean())
        for cluster_id, cluster_values in profile_matrix.iterrows():
            cluster_score = float(cluster_values[present_features].mean())
            rows.append({
                "cluster": str(int(cluster_id)),
                "cluster_id": int(cluster_id),
                "dimension": dimension,
                "cluster_value": cluster_score,
                "global_value": global_score,
                "deviation": cluster_score - global_score,
                "features": ", ".join(present_features),
            })

    return pd.DataFrame(rows)


def _cluster_profile_matrix(
    df: pd.DataFrame,
    labels,
    exclude_noise: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    labels_array = np.asarray(labels, dtype=int)
    numeric_df = df.select_dtypes(include="number").copy()
    if numeric_df.empty or len(numeric_df) != len(labels_array):
        return pd.DataFrame(), pd.Series(dtype=float)

    numeric_df["cluster"] = labels_array
    if exclude_noise:
        numeric_df = numeric_df[numeric_df["cluster"] != -1]
    if numeric_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    feature_columns = [column for column in numeric_df.columns if column != "cluster"]
    profile_matrix = numeric_df.groupby("cluster")[feature_columns].mean()
    global_mean = numeric_df[feature_columns].mean()
    return profile_matrix, global_mean
