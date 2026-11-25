"""Metadatos de lectura derivados de la transformación real, sin cambiar datos."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.provenance import sha256_file
from src.utils.constants import TODAS_VARIABLES, RESULTS_PATH


def build_profile_metadata(original: pd.DataFrame, processed: pd.DataFrame) -> dict:
    """Relaciona valores antes/después usando las mismas filas del pipeline."""
    metadata = {}
    for column in processed:
        info = dict(TODAS_VARIABLES.get(column, {}))
        info["processed_categories"] = []
        info["numeric_inverse"] = None
        pairs = pd.DataFrame({"original": original[column], "processed": processed[column]})
        if info.get("tipo") == "numerico":
            valid = pairs.dropna().drop_duplicates()
            if not valid.empty:
                lower = valid.loc[valid.processed.idxmin()]
                upper = valid.loc[valid.processed.idxmax()]
                span = upper.processed - lower.processed
                scale = float((upper.original - lower.original) / span) if span else 0.0
                offset = float(lower.original - scale * lower.processed)
                if np.allclose(valid.original, scale * valid.processed + offset):
                    info["numeric_inverse"] = {"scale": scale, "offset": offset}
        else:
            categories = info.get("categorias", {})
            binary = info.get("mapeo_binario", {}) if info.get("tipo") == "binario" else {}
            for value, group in pairs.dropna(subset=["processed"]).groupby("processed"):
                labels = set()
                for raw in group.original.unique():
                    if pd.isna(raw):
                        labels.add("Sin dato")
                    else:
                        key = binary.get(raw, raw)
                        labels.add(str(categories.get(key, f"Código {raw}")))
                if len(labels) == 1:
                    info["processed_categories"].append([float(value), labels.pop()])
                    if binary and any(binary.get(raw, raw) == 1 for raw in group.original.dropna()):
                        info["positive_processed_value"] = float(value)
        metadata[column] = info
    return metadata


def load_profile_metadata(dataset_path, run=None, report_path=None) -> dict:
    """Solo usa metadatos vinculados a la huella del CSV que se va a leer."""
    digest = sha256_file(dataset_path)
    saved = ((run or {}).get("metadata") or {}).get("profile_reading") or {}
    if saved.get("dataset_sha256") == digest and saved.get("features"):
        return saved["features"]
    path = Path(report_path or Path(RESULTS_PATH) / "preprocessing" / "reporte_real.json")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("processed_dataset_sha256") == digest:
            return report.get("sections", {}).get("transformation", {}).get("feature_metadata", {})
    except (OSError, ValueError, TypeError):
        pass
    return {}


def unverified_profile_metadata(columns) -> dict:
    """Sin trazabilidad se conserva el valor procesado, sin deducir etiquetas."""
    return {column: {**TODAS_VARIABLES.get(column, {}),
                     "processed_categories": [], "numeric_inverse": None}
            for column in columns}
