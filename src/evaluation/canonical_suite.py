"""Ejecucion y evidencia consolidada de la bateria canonica de Iteracion 5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any

from .comparability import assess_run_comparability, run_evaluation_summary
from .execution import ALGORITHMS, ExecutionResult, default_params, run_clustering_experiment
from .results_manager import ResultsManager
from src.visualization.semantic_profiles import (
    build_semantic_profile_export,
    cluster_semantic_profiles,
)


CANONICAL_SUITE_SCHEMA = "iteration5-canonical-suite-v1"
CANONICAL_METRIC_BASELINES = {
    "WKMedoids": {
        "silhouette": 0.10623499888238168,
        "davies_bouldin": 2.452128159703432,
        "calinski_harabasz": 94.95935333260046,
    },
    "W-Hierarchical Clustering": {
        "silhouette": 0.14090720223511682,
        "davies_bouldin": 2.3074104414738756,
        "calinski_harabasz": 254.6930434573174,
    },
    "W-DBSCAN": {
        "silhouette": 0.29576400377933976,
        "davies_bouldin": 1.3593183247417573,
        "calinski_harabasz": 120.30614949115017,
    },
}
CANONICAL_METRIC_TOLERANCE = 1e-8


@dataclass
class CanonicalSuiteResult:
    """Resultado de una ejecucion integrada sin perder los runs individuales."""

    suite_id: str | None
    artifact: dict[str, Any]
    results: list[ExecutionResult]


def evaluate_canonical_regression(
    algorithm: str,
    metrics: dict[str, Any],
    *,
    tolerance: float = CANONICAL_METRIC_TOLERANCE,
) -> dict[str, Any]:
    """Contrasta metricas contra el control historico sin alterar el resultado."""
    expected = CANONICAL_METRIC_BASELINES.get(algorithm)
    if expected is None:
        return {"status": "not_registered", "passed": False, "metrics": {}}

    checks: dict[str, dict[str, Any]] = {}
    passed = True
    for metric, expected_value in expected.items():
        actual_value = metrics.get(metric)
        metric_passed = (
            actual_value is not None
            and abs(float(actual_value) - expected_value) <= tolerance
        )
        checks[metric] = {
            "expected": expected_value,
            "actual": actual_value,
            "difference": (
                None if actual_value is None else float(actual_value) - expected_value
            ),
            "passed": metric_passed,
        }
        passed = passed and metric_passed

    return {
        "status": "passed" if passed else "mismatch",
        "passed": passed,
        "tolerance": tolerance,
        "metrics": checks,
    }


def run_canonical_suite(
    *,
    dataset_path: str | None = None,
    results_manager: ResultsManager | None = None,
    persist: bool = True,
) -> CanonicalSuiteResult:
    """Ejecuta la terna canonica y produce la fuente consolidada del capitulo 8."""
    manager = results_manager or ResultsManager()
    started_at = datetime.now().isoformat()
    suite_id = manager.generate_suite_id() if persist else None
    results: list[ExecutionResult] = []
    suite_runs: list[dict[str, Any]] = []

    for algorithm in ALGORITHMS:
        started = perf_counter()
        kwargs = {
            "algorithm": algorithm,
            "params": default_params(algorithm),
            "results_manager": manager,
            "persist": persist,
        }
        if dataset_path is not None:
            kwargs["dataset_path"] = dataset_path
        result = run_clustering_experiment(**kwargs)
        elapsed_seconds = perf_counter() - started
        results.append(result)
        suite_runs.append(
            _suite_run_record(
                result,
                profile_run_id=result.run_id or f"transient_{algorithm}",
                elapsed_seconds=elapsed_seconds,
            )
        )

    comparison_runs = [run["comparison_record"] for run in suite_runs]
    artifact = {
        "schema": CANONICAL_SUITE_SCHEMA,
        "suite_id": suite_id,
        "generated_at": started_at,
        "source": {
            "configuration": "canonical_iteration4_v1",
            "algorithms": list(ALGORITHMS),
            "purpose": "evidence_consolidation_iteration5",
        },
        "runs": [
            {key: value for key, value in run.items() if key != "comparison_record"}
            for run in suite_runs
        ],
        "comparability": assess_run_comparability(comparison_runs),
        "regression": {
            "all_passed": all(run["regression"]["passed"] for run in suite_runs),
            "checks": {
                run["algorithm"]: run["regression"] for run in suite_runs
            },
        },
    }

    if persist:
        manager.save_suite(artifact, suite_id=suite_id)

    return CanonicalSuiteResult(suite_id=suite_id, artifact=artifact, results=results)


def _suite_run_record(
    result: ExecutionResult,
    *,
    profile_run_id: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    labels = result.labels
    metadata = result.metadata
    provenance = metadata.get("provenance", {})
    evaluation = run_evaluation_summary(
        {
            "run_id": result.run_id or profile_run_id,
            "n_samples": len(labels),
            "n_noise": int((labels == -1).sum()),
            "metadata": metadata,
        }
    )
    profiles = cluster_semantic_profiles(result.data, labels)
    semantic_profiles = build_semantic_profile_export(
        profiles,
        run_id=profile_run_id,
        algorithm=result.algorithm,
        run_timestamp=provenance.get("generated_at"),
        dataset_sha256=provenance.get("dataset", {}).get("sha256"),
        execution_sha256=provenance.get("configuration", {}).get("execution_sha256"),
        exclude_noise=True,
    )
    comparison_record = {
        "run_id": result.run_id or profile_run_id,
        "n_samples": len(labels),
        "n_noise": int((labels == -1).sum()),
        "metadata": metadata,
    }
    return {
        "run_id": result.run_id,
        "algorithm": result.algorithm,
        "params": result.params,
        "metrics": result.metrics,
        "elapsed_seconds": elapsed_seconds,
        "n_clusters": int(len(set(int(label) for label in labels if label != -1))),
        "n_noise": int((labels == -1).sum()),
        "n_samples": int(len(labels)),
        "evaluation": evaluation,
        "regression": evaluate_canonical_regression(result.algorithm, result.metrics),
        "semantic_profiles": semantic_profiles,
        "comparison_record": comparison_record,
    }
