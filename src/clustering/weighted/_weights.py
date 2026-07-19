"""Validacion comun de pesos para los algoritmos ponderados."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def resolve_feature_weights(
    X: pd.DataFrame | np.ndarray,
    weight_manager: Any | None,
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    """Convierte los datos y resuelve un vector de pesos valido una sola vez."""
    if isinstance(X, pd.DataFrame):
        feature_order = X.columns.tolist()
        X_array = X.to_numpy(dtype=float)
    else:
        feature_order = None
        X_array = np.asarray(X, dtype=float)

    if X_array.ndim != 2:
        raise ValueError("X debe ser una matriz bidimensional.")
    if X_array.shape[1] == 0:
        raise ValueError("X debe contener al menos una caracteristica.")

    if weight_manager is None:
        weights_array = np.ones(X_array.shape[1], dtype=float)
    else:
        weights_array = np.asarray(
            weight_manager.get_weights_array(feature_order=feature_order),
            dtype=float,
        )

    if weights_array.ndim != 1 or len(weights_array) != X_array.shape[1]:
        raise ValueError(
            "La cantidad de pesos debe coincidir con las caracteristicas: "
            f"pesos={weights_array.size}, caracteristicas={X_array.shape[1]}."
        )
    if not np.all(np.isfinite(weights_array)):
        raise ValueError("Los pesos deben contener solo valores finitos.")
    if np.any(weights_array < 0):
        raise ValueError("Los pesos no pueden ser negativos.")
    if not np.any(weights_array > 0):
        raise ValueError("Al menos un peso debe ser mayor que cero.")

    return X_array, weights_array, feature_order
