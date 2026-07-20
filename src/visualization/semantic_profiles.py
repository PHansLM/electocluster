"""
Estadisticas semanticas aditivas para perfiles de clusters.

El modulo opera exclusivamente sobre el dataset procesado y las etiquetas ya
calculadas. No transforma las variables, no modifica las entradas y no
reemplaza los perfiles promedio historicos de ``cluster_profiles``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.utils.constants import TODAS_VARIABLES
from src.evaluation.feature_space import ALL_SAMPLES_V1, CLUSTERED_WITHOUT_NOISE_V1


SEMANTIC_PROFILE_EXPORT_SCHEMA = "iteration5-semantic-profiles-v1"

SUMMARY_COLUMNS = [
    "cluster",
    "cluster_id",
    "cluster_size",
    "feature_code",
    "feature_name",
    "feature_type",
    "category_mapping_status",
    "valid_count",
    "missing_count",
    "representative_statistic",
    "representative_value",
    "representative_label",
    "representative_percentage",
    "mean",
    "median",
    "std",
    "q1",
    "q3",
    "minimum",
    "maximum",
    "global_reference",
    "global_reference_label",
    "global_reference_percentage",
    "difference_from_global",
    "difference_unit",
]

DISTRIBUTION_COLUMNS = [
    "cluster",
    "cluster_id",
    "cluster_size",
    "feature_code",
    "feature_name",
    "feature_type",
    "category_mapping_status",
    "valid_count",
    "category_value",
    "category_label",
    "count",
    "percentage",
    "global_percentage",
    "percentage_point_difference",
]

_NUMERIC = "numerico"
_ORDINAL = "categorico_ordinal"
_NOMINAL = "categorico_nominal"
_BINARY = "binario"
_CATEGORICAL_TYPES = {_ORDINAL, _NOMINAL, _BINARY}
_SUPPORTED_TYPES = {_NUMERIC, *_CATEGORICAL_TYPES}


@dataclass(frozen=True)
class SemanticProfileTables:
    """Tablas derivadas para resumir e inspeccionar perfiles semanticos."""

    summaries: pd.DataFrame
    distributions: pd.DataFrame


def build_semantic_profile_export(
    profiles: SemanticProfileTables,
    *,
    run_id: str,
    algorithm: str | None,
    run_timestamp: Any = None,
    dataset_sha256: str | None = None,
    execution_sha256: str | None = None,
    exclude_noise: bool = True,
) -> dict[str, Any]:
    """Construye un documento JSON trazable sin modificar ni persistir tablas."""
    if not isinstance(profiles, SemanticProfileTables):
        raise TypeError("profiles debe ser una instancia de SemanticProfileTables.")
    if not str(run_id).strip():
        raise ValueError("run_id es obligatorio para exportar perfiles semanticos.")

    summaries = profiles.summaries
    distributions = profiles.distributions
    feature_order = (
        summaries["feature_code"].drop_duplicates().astype(str).tolist()
        if "feature_code" in summaries
        else []
    )
    cluster_ids = (
        sorted(int(value) for value in summaries["cluster_id"].drop_duplicates())
        if "cluster_id" in summaries
        else []
    )
    processed_only_features = (
        sorted(
            summaries.loc[
                summaries["category_mapping_status"] == "processed_only",
                "feature_code",
            ]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
        if {"category_mapping_status", "feature_code"} <= set(summaries.columns)
        else []
    )

    return {
        "schema": SEMANTIC_PROFILE_EXPORT_SCHEMA,
        "source": {
            "run_id": str(run_id),
            "algorithm": None if algorithm is None else str(algorithm),
            "run_timestamp": _json_scalar(run_timestamp),
            "dataset_sha256": dataset_sha256,
            "execution_sha256": execution_sha256,
        },
        "analysis": {
            "exclude_noise": bool(exclude_noise),
            "population_id": (
                CLUSTERED_WITHOUT_NOISE_V1 if exclude_noise else ALL_SAMPLES_V1
            ),
            "cluster_ids": cluster_ids,
            "cluster_count": len(cluster_ids),
            "feature_order": feature_order,
            "feature_count": len(feature_order),
            "summary_rows": len(summaries),
            "distribution_rows": len(distributions),
            "processed_only_features": processed_only_features,
        },
        "tables": {
            "summaries": _dataframe_records(summaries),
            "distributions": _dataframe_records(distributions),
        },
    }


def cluster_semantic_profiles(
    df: pd.DataFrame,
    labels,
    exclude_noise: bool = True,
    feature_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> SemanticProfileTables:
    """
    Calcula estadisticas por cluster respetando el tipo de cada variable.

    - nominales: moda y distribucion por categoria;
    - ordinales: mediana, cuartiles y distribucion;
    - binarias: prevalencia de la categoria positiva y distribucion;
    - numericas: media, mediana, desviacion poblacional, cuartiles y rango.

    ``difference_from_global`` usa la misma poblacion incluida en el analisis:
    excluye ruido cuando ``exclude_noise=True``. Para variables binarias se
    expresa en puntos porcentuales; para numericas y ordinales, en la escala
    procesada. Los codigos nominales nunca se restan entre si.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un DataFrame de pandas.")

    labels_array = np.asarray(labels)
    if labels_array.ndim != 1:
        raise ValueError("labels debe ser un vector unidimensional.")
    if len(labels_array) != len(df):
        raise ValueError("df y labels deben contener el mismo numero de filas.")

    try:
        numeric_labels = labels_array.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels debe contener identificadores enteros.") from exc
    if not np.isfinite(numeric_labels).all() or not np.equal(
        numeric_labels,
        np.floor(numeric_labels),
    ).all():
        raise ValueError("labels debe contener identificadores enteros finitos.")
    labels_array = numeric_labels.astype(int, copy=False)

    metadata = TODAS_VARIABLES if feature_metadata is None else feature_metadata
    included = labels_array != -1 if exclude_noise else np.ones(len(labels_array), dtype=bool)
    if not included.any() or df.shape[1] == 0:
        return _empty_profile_tables()

    analysis_df = df.iloc[np.flatnonzero(included)]
    analysis_labels = labels_array[included]
    cluster_ids = np.sort(np.unique(analysis_labels))

    summary_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []

    for column in df.columns:
        feature_code = str(column)
        info = dict(metadata.get(feature_code, {}) or {})
        feature_name = str(info.get("nombre") or feature_code)
        feature_type = str(info.get("tipo") or _NUMERIC)
        if feature_type not in _SUPPORTED_TYPES:
            feature_type = _NUMERIC

        global_values = _numeric_values(analysis_df[column])
        global_counts = global_values.value_counts(dropna=True).sort_index()
        global_valid_count = int(global_values.notna().sum())
        category_values = global_counts.index.to_list()
        category_mapping_status = _category_mapping_status(
            feature_type,
            category_values,
            info,
        )

        for cluster_id in cluster_ids:
            cluster_mask = analysis_labels == cluster_id
            cluster_values = _numeric_values(analysis_df.loc[cluster_mask, column])
            cluster_size = int(cluster_mask.sum())
            valid_count = int(cluster_values.notna().sum())
            missing_count = cluster_size - valid_count
            cluster_name = _cluster_name(cluster_id)

            base = {
                "cluster": cluster_name,
                "cluster_id": int(cluster_id),
                "cluster_size": cluster_size,
                "feature_code": feature_code,
                "feature_name": feature_name,
                "feature_type": feature_type,
                "category_mapping_status": category_mapping_status,
                "valid_count": valid_count,
                "missing_count": missing_count,
            }
            summary_rows.append(
                _semantic_summary(
                    base=base,
                    values=cluster_values.dropna(),
                    global_values=global_values.dropna(),
                    feature_type=feature_type,
                    info=info,
                    category_mapping_status=category_mapping_status,
                )
            )

            if feature_type in _CATEGORICAL_TYPES:
                counts = cluster_values.value_counts(dropna=True)
                for category_value in category_values:
                    count = int(counts.get(category_value, 0))
                    percentage = _percentage(count, valid_count)
                    global_count = int(global_counts.get(category_value, 0))
                    global_percentage = _percentage(global_count, global_valid_count)
                    distribution_rows.append(
                        {
                            **base,
                            "category_value": float(category_value),
                            "category_label": _category_label(
                                category_value,
                                info,
                                category_mapping_status,
                            ),
                            "count": count,
                            "percentage": percentage,
                            "global_percentage": global_percentage,
                            "percentage_point_difference": percentage - global_percentage,
                        }
                    )

    return SemanticProfileTables(
        summaries=pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS),
        distributions=pd.DataFrame(distribution_rows, columns=DISTRIBUTION_COLUMNS),
    )


def _semantic_summary(
    *,
    base: dict[str, Any],
    values: pd.Series,
    global_values: pd.Series,
    feature_type: str,
    info: Mapping[str, Any],
    category_mapping_status: str,
) -> dict[str, Any]:
    row = {
        **base,
        "representative_statistic": None,
        "representative_value": np.nan,
        "representative_label": None,
        "representative_percentage": np.nan,
        "mean": np.nan,
        "median": np.nan,
        "std": np.nan,
        "q1": np.nan,
        "q3": np.nan,
        "minimum": np.nan,
        "maximum": np.nan,
        "global_reference": np.nan,
        "global_reference_label": None,
        "global_reference_percentage": np.nan,
        "difference_from_global": np.nan,
        "difference_unit": None,
    }
    if values.empty:
        return row

    if feature_type == _NUMERIC:
        mean = float(values.mean())
        global_mean = float(global_values.mean()) if not global_values.empty else np.nan
        row.update(
            {
                "representative_statistic": "mean",
                "representative_value": mean,
                "mean": mean,
                "median": float(values.median()),
                "std": float(values.std(ddof=0)),
                "q1": float(values.quantile(0.25)),
                "q3": float(values.quantile(0.75)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "global_reference": global_mean,
                "difference_from_global": mean - global_mean,
                "difference_unit": "processed_scale",
            }
        )
        return row

    if feature_type == _ORDINAL:
        median = float(values.median())
        global_median = float(global_values.median()) if not global_values.empty else np.nan
        row.update(
            {
                "representative_statistic": "median",
                "representative_value": median,
                "representative_label": _category_label(
                    median,
                    info,
                    category_mapping_status,
                ),
                "median": median,
                "q1": float(values.quantile(0.25)),
                "q3": float(values.quantile(0.75)),
                "global_reference": global_median,
                "global_reference_label": _category_label(
                    global_median,
                    info,
                    category_mapping_status,
                ),
                "difference_from_global": median - global_median,
                "difference_unit": "processed_scale",
            }
        )
        return row

    if feature_type == _NOMINAL:
        mode = float(values.mode().iloc[0])
        mode_count = int(np.isclose(values.to_numpy(dtype=float), mode).sum())
        global_mode = float(global_values.mode().iloc[0]) if not global_values.empty else np.nan
        global_mode_count = (
            int(np.isclose(global_values.to_numpy(dtype=float), global_mode).sum())
            if not global_values.empty
            else 0
        )
        row.update(
            {
                "representative_statistic": "mode",
                "representative_value": mode,
                "representative_label": _category_label(
                    mode,
                    info,
                    category_mapping_status,
                ),
                "representative_percentage": _percentage(mode_count, len(values)),
                "global_reference": global_mode,
                "global_reference_label": _category_label(
                    global_mode,
                    info,
                    category_mapping_status,
                ),
                "global_reference_percentage": _percentage(global_mode_count, len(global_values)),
                "difference_unit": "not_applicable",
            }
        )
        return row

    positive_value = _positive_binary_value(info, global_values)
    prevalence = _percentage(
        int(np.isclose(values.to_numpy(dtype=float), positive_value).sum()),
        len(values),
    )
    global_prevalence = _percentage(
        int(np.isclose(global_values.to_numpy(dtype=float), positive_value).sum()),
        len(global_values),
    )
    row.update(
        {
            "representative_statistic": "prevalence",
            "representative_value": prevalence,
            "representative_label": _category_label(
                positive_value,
                info,
                category_mapping_status,
            ),
            "representative_percentage": prevalence,
            "global_reference": global_prevalence,
            "global_reference_label": _category_label(
                positive_value,
                info,
                category_mapping_status,
            ),
            "global_reference_percentage": global_prevalence,
            "difference_from_global": prevalence - global_prevalence,
            "difference_unit": "percentage_points",
        }
    )
    return row


def _numeric_values(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return numeric.replace([np.inf, -np.inf], np.nan)


def _percentage(count: int, total: int) -> float:
    return float(count / total * 100) if total else np.nan


def _cluster_name(cluster_id: int) -> str:
    return "noise" if int(cluster_id) == -1 else str(int(cluster_id))


def _positive_binary_value(info: Mapping[str, Any], values: pd.Series) -> float:
    points = _category_points(info)
    if points:
        return float(points[-1][0])
    if not values.empty:
        return float(values.max())
    return 1.0


def _category_label(
    value: float,
    info: Mapping[str, Any],
    mapping_status: str,
) -> str:
    if not np.isfinite(value):
        return "Sin valor"

    if mapping_status != "configured_exact":
        return f"Valor procesado {float(value):g}"

    points = _category_points(info)
    if not points:
        return f"{float(value):g}"

    for point, label in points:
        if np.isclose(value, point):
            return label

    ordered = sorted(points, key=lambda item: item[0])
    for left, right in zip(ordered, ordered[1:]):
        if left[0] < value < right[0]:
            return f"Entre {left[1]} y {right[1]}"

    nearest = min(ordered, key=lambda item: abs(item[0] - value))
    return nearest[1]


def _category_mapping_status(
    feature_type: str,
    observed_values: list[float],
    info: Mapping[str, Any],
) -> str:
    if feature_type not in _CATEGORICAL_TYPES:
        return "not_applicable"

    points = _category_points(info)
    if not points or not observed_values:
        return "processed_only"

    configured_values = np.asarray([point for point, _ in points], dtype=float)
    for observed in observed_values:
        if not np.isclose(float(observed), configured_values).any():
            return "processed_only"
    return "configured_exact"


def _category_points(info: Mapping[str, Any]) -> list[tuple[float, str]]:
    categories = info.get("categorias") or {}
    if not categories:
        return []

    feature_type = info.get("tipo")
    if feature_type == _BINARY:
        return sorted(
            [(float(key), str(label)) for key, label in categories.items()],
            key=lambda item: item[0],
        )

    if feature_type == _NOMINAL:
        ordered_keys = sorted(categories, key=lambda key: str(key))
        return _even_category_points(ordered_keys, categories)

    ordered_keys = sorted(categories, key=float)
    value_range = info.get("rango")
    if not value_range or len(value_range) != 2:
        return _even_category_points(ordered_keys, categories)

    lower, upper = float(value_range[0]), float(value_range[1])
    span = upper - lower
    if span <= 0:
        return _even_category_points(ordered_keys, categories)
    return [
        ((float(key) - lower) / span, str(categories[key]))
        for key in ordered_keys
    ]


def _even_category_points(
    ordered_keys: list[Any],
    categories: Mapping[Any, Any],
) -> list[tuple[float, str]]:
    if not ordered_keys:
        return []
    if len(ordered_keys) == 1:
        return [(0.0, str(categories[ordered_keys[0]]))]
    denominator = len(ordered_keys) - 1
    return [
        (index / denominator, str(categories[key]))
        for index, key in enumerate(ordered_keys)
    ]


def _empty_profile_tables() -> SemanticProfileTables:
    return SemanticProfileTables(
        summaries=pd.DataFrame(columns=SUMMARY_COLUMNS),
        distributions=pd.DataFrame(columns=DISTRIBUTION_COLUMNS),
    )


def _dataframe_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Convierte NaN/inf y escalares numpy a JSON estandar (sin valores NaN)."""
    return json.loads(
        dataframe.to_json(
            orient="records",
            force_ascii=False,
            double_precision=15,
        )
    )


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value if isinstance(value, (str, int, float, bool)) else str(value)
