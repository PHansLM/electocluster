"""Vistas multidimensionales derivadas de perfiles semanticos.

Estas funciones no modifican etiquetas ni metricas historicas. Transforman las
estadisticas tipadas ya calculadas en matrices aptas para visualizacion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .semantic_profiles import SemanticProfileTables


HEATMAP_SUPPORTED_TYPES = {"numerico", "categorico_ordinal", "binario"}


@dataclass(frozen=True)
class SemanticHeatmapData:
    """Matriz y metadatos para un heatmap de perfiles multidimensionales."""

    matrix: pd.DataFrame
    features: pd.DataFrame
    excluded_nominal_features: tuple[str, ...]


def semantic_profile_heatmap(
    profiles: SemanticProfileTables,
    *,
    top_n: int = 10,
) -> SemanticHeatmapData:
    """Construye una matriz de desviaciones semanticas por cluster.

    Las variables nominales se excluyen: sus codigos no representan una escala
    sobre la cual sea valido calcular diferencias. Para las restantes, cada
    columna se expresa entre -1 y 1 respecto a su maxima desviacion absoluta;
    el color comunica direccion y magnitud relativa dentro de cada variable,
    no una comparacion de unidades entre variables.
    """
    if not isinstance(profiles, SemanticProfileTables):
        raise TypeError("profiles debe ser una instancia de SemanticProfileTables.")
    if top_n < 1:
        raise ValueError("top_n debe ser mayor o igual a 1.")

    summaries = profiles.summaries.copy()
    required = {
        "cluster_id",
        "feature_code",
        "feature_name",
        "feature_type",
        "difference_from_global",
        "difference_unit",
    }
    if summaries.empty or not required <= set(summaries.columns):
        return _empty_heatmap()

    nominal_features = tuple(sorted(
        summaries.loc[
            summaries["feature_type"] == "categorico_nominal",
            "feature_code",
        ].drop_duplicates().astype(str).tolist()
    ))
    supported = summaries[summaries["feature_type"].isin(HEATMAP_SUPPORTED_TYPES)].copy()
    supported["difference_from_global"] = pd.to_numeric(
        supported["difference_from_global"], errors="coerce"
    )
    supported = supported.dropna(subset=["difference_from_global"])
    if supported.empty:
        return _empty_heatmap(nominal_features)

    # Las diferencias binarias se almacenan en puntos porcentuales; se llevan
    # a proporcion antes de medir que variables diferencian mas los perfiles.
    supported["scaled_difference"] = supported["difference_from_global"]
    binary_mask = supported["difference_unit"] == "percentage_points"
    supported.loc[binary_mask, "scaled_difference"] /= 100.0

    raw_matrix = supported.pivot(
        index="cluster_id",
        columns="feature_code",
        values="scaled_difference",
    ).sort_index()
    if raw_matrix.empty:
        return _empty_heatmap(nominal_features)

    dispersion = raw_matrix.std(axis=0, ddof=0).fillna(0.0)
    selected_codes = sorted(
        dispersion.sort_values(ascending=False, kind="stable").head(top_n).index.tolist(),
        key=lambda code: (-float(dispersion[code]), str(code)),
    )
    selected_raw = raw_matrix.reindex(columns=selected_codes).fillna(0.0)
    scale = selected_raw.abs().max(axis=0).replace(0.0, 1.0)
    matrix = selected_raw.divide(scale, axis="columns")
    matrix.index = [f"Cluster {int(cluster_id)}" for cluster_id in matrix.index]

    feature_info = (
        supported.drop_duplicates("feature_code")
        .set_index("feature_code")
        .reindex(selected_codes)
        .reset_index()[["feature_code", "feature_name", "feature_type"]]
    )
    feature_info["relative_dispersion"] = [float(dispersion[code]) for code in selected_codes]
    return SemanticHeatmapData(
        matrix=matrix,
        features=feature_info,
        excluded_nominal_features=nominal_features,
    )


def _empty_heatmap(
    nominal_features: tuple[str, ...] = (),
) -> SemanticHeatmapData:
    return SemanticHeatmapData(
        matrix=pd.DataFrame(),
        features=pd.DataFrame(
            columns=["feature_code", "feature_name", "feature_type", "relative_dispersion"]
        ),
        excluded_nominal_features=nominal_features,
    )
