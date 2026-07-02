"""
Interpretacion legible de valores procesados para perfiles de cluster.

Los perfiles se calculan sobre datos transformados a escala numerica. Este
modulo aproxima esos valores contra la configuracion de variables para mostrar
una lectura humana sin modificar los resultados numericos persistidos.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from src.utils.constants import TODAS_VARIABLES


PROFILE_METADATA_COLUMNS = {"cluster", "size", "percentage"}


@dataclass(frozen=True)
class FeatureValueInterpretation:
    """Resultado de interpretacion para una celda de perfil."""

    text: str
    raw_value: float | None
    feature_name: str
    feature_type: str


def feature_display_name(feature_code: str) -> str:
    """Retorna nombre corto para encabezados de perfiles."""
    info = TODAS_VARIABLES.get(feature_code, {})
    name = info.get("nombre")
    return f"{name} ({feature_code})" if name else feature_code


def interpret_feature_value(feature_code: str, value: Any) -> FeatureValueInterpretation:
    """
    Traduce un valor procesado de perfil a una aproximacion legible.

    Para categoricas se ubica el valor entre las categorias configuradas y se
    calcula una proporcion lineal entre los dos puntos mas cercanos. Para
    numericas se invierte la normalizacion min-max usando el rango configurado.
    """
    raw_value = _to_float(value)
    info = TODAS_VARIABLES.get(feature_code)
    if raw_value is None:
        return FeatureValueInterpretation("Sin valor", None, feature_code, "desconocido")
    if not info:
        return FeatureValueInterpretation(
            "Sin configuracion",
            raw_value,
            feature_code,
            "desconocido",
        )

    feature_type = info.get("tipo", "desconocido")
    feature_name = info.get("nombre", feature_code)

    if feature_type == "numerico":
        text = _interpret_numeric(raw_value, info)
    elif feature_type in {"categorico_ordinal", "categorico_nominal", "binario"}:
        text = _interpret_categorical(raw_value, info)
    else:
        text = "Valor procesado"

    return FeatureValueInterpretation(text, raw_value, feature_name, feature_type)


def _to_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _interpret_numeric(value: float, info: dict) -> str:
    value = _clamp(value, 0.0, 1.0)
    rango = info.get("rango")
    if not rango or len(rango) != 2:
        return f"Valor procesado {_format_number(value)}"

    min_value, max_value = float(rango[0]), float(rango[1])
    original = min_value + value * (max_value - min_value)
    return f"Aprox. {_format_number(original)} en escala original"


def _interpret_categorical(value: float, info: dict) -> str:
    categories = info.get("categorias") or {}
    if not categories:
        return f"Valor procesado {_format_number(value)}"

    points = _category_points(info, categories)
    if not points:
        return f"Valor procesado {_format_number(value)}"

    value = _clamp(value, points[0][0], points[-1][0])

    for point, label in points:
        if abs(value - point) <= 1e-9:
            return str(label)

    lower = points[0]
    upper = points[-1]
    for index in range(len(points) - 1):
        left = points[index]
        right = points[index + 1]
        if left[0] <= value <= right[0]:
            lower, upper = left, right
            break

    lower_point, lower_label = lower
    upper_point, upper_label = upper
    span = upper_point - lower_point
    if span <= 0:
        return str(lower_label)

    upper_weight = (value - lower_point) / span
    lower_weight = 1.0 - upper_weight
    lower_pct = round(lower_weight * 100)
    upper_pct = round(upper_weight * 100)

    if abs(lower_weight - upper_weight) <= 0.005:
        return f"Entre {lower_label} y {upper_label} (50% / 50%)"
    if upper_weight > lower_weight:
        return f"Mas cercano a {upper_label} ({upper_pct}%) que a {lower_label} ({lower_pct}%)"
    return f"Mas cercano a {lower_label} ({lower_pct}%) que a {upper_label} ({upper_pct}%)"


def _category_points(info: dict, categories: dict) -> list[tuple[float, str]]:
    feature_type = info.get("tipo")
    if feature_type == "categorico_nominal":
        ordered_keys = sorted(categories, key=lambda item: str(item))
        return _even_points(ordered_keys, categories)

    ordered_keys = sorted(categories, key=lambda item: float(item))
    if feature_type == "binario":
        return [(float(key), str(categories[key])) for key in ordered_keys]

    rango = info.get("rango")
    if not rango or len(rango) != 2:
        return _even_points(ordered_keys, categories)

    min_code, max_code = float(rango[0]), float(rango[1])
    span = max_code - min_code
    if span <= 0:
        return _even_points(ordered_keys, categories)

    return [
        ((float(key) - min_code) / span, str(categories[key]))
        for key in ordered_keys
    ]


def _even_points(ordered_keys: list, categories: dict) -> list[tuple[float, str]]:
    if not ordered_keys:
        return []
    if len(ordered_keys) == 1:
        return [(0.0, str(categories[ordered_keys[0]]))]
    divisor = len(ordered_keys) - 1
    return [
        (index / divisor, str(categories[key]))
        for index, key in enumerate(ordered_keys)
    ]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.1f}"
