"""
Utilidades de inspeccion de datasets candidatos para preprocesamiento.

Estas funciones son aditivas para la interfaz: no alteran el pipeline historico
ni cambian la carga validada en los notebooks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.constants import TODAS_VARIABLES


COMPOSITE_VARIABLES = {"wealth_index", "civic_index"}


def read_supported_dataset(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Lee datasets .dta o .csv para inspeccion previa."""
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()

    if suffix == ".dta":
        return pd.read_stata(dataset_path, convert_categoricals=False)
    if suffix == ".csv":
        return pd.read_csv(dataset_path, nrows=nrows)

    raise ValueError("Formato no soportado. Usa archivos .dta o .csv.")


def inspect_expected_variables(path: str | Path) -> dict:
    """
    Informa que variables contempladas por el proyecto estan presentes.

    Se separan variables base y compuestas porque los datasets crudos deberian
    contener las primeras, mientras que las compuestas se generan en el pipeline.
    """
    df = read_supported_dataset(path)
    columns = list(df.columns)
    columns_set = set(columns)

    base_expected = [
        code for code in TODAS_VARIABLES
        if code not in COMPOSITE_VARIABLES
    ]
    composite_expected = [
        code for code in TODAS_VARIABLES
        if code in COMPOSITE_VARIABLES
    ]

    present_base = [code for code in base_expected if code in columns_set]
    missing_base = [code for code in base_expected if code not in columns_set]
    present_composite = [code for code in composite_expected if code in columns_set]
    known_expected = set(base_expected) | set(composite_expected)
    extra_columns = [column for column in columns if column not in known_expected]

    variable_rows = []
    for code in base_expected:
        info = TODAS_VARIABLES.get(code, {})
        variable_rows.append({
            "codigo": code,
            "nombre": info.get("nombre", ""),
            "tipo": info.get("tipo", ""),
            "peso": info.get("peso", ""),
            "estado": "Presente" if code in columns_set else "Faltante",
        })

    return {
        "path": str(path),
        "n_rows": int(len(df)),
        "n_columns": int(len(columns)),
        "columns": columns,
        "base_expected": base_expected,
        "present_base": present_base,
        "missing_base": missing_base,
        "present_composite": present_composite,
        "extra_columns": extra_columns,
        "coverage": len(present_base) / len(base_expected) if base_expected else 0,
        "variable_rows": variable_rows,
    }
