"""Reglas para comparar ejecuciones sin mezclar universos incompatibles."""

from __future__ import annotations

from typing import Any

from .provenance import stored_dataset_sha256


COMPARABILITY_SCHEMA = "iteration5-comparability-v1"

SIGNATURE_FIELDS = {
    "dataset_sha256": "dataset",
    "metric_space": "espacio de metricas",
    "population": "tipo de poblacion evaluada",
    "evaluation_index_sha256": "poblacion exacta evaluada",
    "metric_definition": "definicion de metricas",
}


def run_evaluation_summary(run: dict) -> dict:
    """Normaliza cobertura y contexto, incluyendo runs historicos incompletos."""
    metadata = run.get("metadata") or {}
    context = metadata.get("evaluation_context") or {}
    n_total = _as_int(context.get("n_total"), run.get("n_samples"), default=0)
    n_noise = _as_int(run.get("n_noise"), metadata.get("n_noise"), default=0)
    fallback_evaluated = n_total - n_noise if metadata.get("excluded_noise") else n_total
    n_evaluated = _as_int(
        context.get("n_evaluated"),
        metadata.get("n_evaluated"),
        fallback_evaluated,
        default=0,
    )
    coverage = n_evaluated / n_total if n_total else None
    noise_percentage = n_noise / n_total * 100 if n_total else None

    metric_space = _nested_id(context, "metric_space") or metadata.get("evaluation_space")
    population = _nested_id(context, "population")
    metric_definition = _nested_id(context, "metric_definition")

    return {
        "dataset_sha256": stored_dataset_sha256(run),
        "metric_space": metric_space,
        "population": population,
        "evaluation_index_sha256": context.get("evaluation_index_sha256"),
        "metric_definition": metric_definition,
        "n_total": n_total,
        "n_evaluated": n_evaluated,
        "coverage": coverage,
        "coverage_percentage": coverage * 100 if coverage is not None else None,
        "n_noise": n_noise,
        "noise_percentage": noise_percentage,
    }


def assess_run_comparability(runs: list[dict]) -> dict:
    """Determina si un conjunto de runs admite un ranking directo."""
    if len(runs) < 2:
        return {
            "schema": COMPARABILITY_SCHEMA,
            "status": "insufficient",
            "directly_comparable": False,
            "reasons": ["Se requieren al menos dos ejecuciones para establecer un ranking."],
        }

    summaries = [run_evaluation_summary(run) for run in runs]
    missing = []
    for index, (run, summary) in enumerate(zip(runs, summaries), start=1):
        run_id = run.get("run_id") or f"run_{index}"
        missing_fields = [
            label for field, label in SIGNATURE_FIELDS.items() if not summary.get(field)
        ]
        if missing_fields:
            missing.append(f"{run_id}: falta {', '.join(missing_fields)}")

    if missing:
        return {
            "schema": COMPARABILITY_SCHEMA,
            "status": "unverifiable",
            "directly_comparable": False,
            "reasons": missing,
        }

    differences = []
    for field, label in SIGNATURE_FIELDS.items():
        values = {summary[field] for summary in summaries}
        if len(values) > 1:
            differences.append(f"Las ejecuciones difieren en {label}.")

    if differences:
        return {
            "schema": COMPARABILITY_SCHEMA,
            "status": "incompatible",
            "directly_comparable": False,
            "reasons": differences,
        }

    return {
        "schema": COMPARABILITY_SCHEMA,
        "status": "comparable",
        "directly_comparable": True,
        "reasons": ["Dataset, espacio, poblacion y definicion de metricas coinciden."],
    }


def _nested_id(mapping: dict, key: str) -> Any:
    value = mapping.get(key)
    if isinstance(value, dict):
        return value.get("id")
    return value


def _as_int(*values, default: int) -> int:
    for value in values:
        if value is not None:
            return int(value)
    return default
