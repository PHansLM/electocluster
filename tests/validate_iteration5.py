"""
Smoke tests for Iteration 5 without mutating historical notebook outputs.

Usage:
    python tests/validate_iteration5.py --quick
    python tests/validate_iteration5.py --sample
    python tests/validate_iteration5.py --full
    python tests/validate_iteration5.py --streamlit
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



def _canonical_baselines() -> dict:
    """Carga la única línea histórica cuando una prueba la necesita."""
    from src.evaluation.canonical_suite import CANONICAL_METRIC_BASELINES

    return CANONICAL_METRIC_BASELINES


def _canonical_geometry_baselines() -> dict:
    """Carga la línea geométrica independiente cuando una prueba la necesita."""
    from src.evaluation.canonical_suite import CANONICAL_GEOMETRY_BASELINES

    return CANONICAL_GEOMETRY_BASELINES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run fast compatibility checks.")
    parser.add_argument("--sample", action="store_true", help="Run algorithms on a temporary sample dataset.")
    parser.add_argument("--sample-size", type=int, default=300, help="Rows used by --sample.")
    parser.add_argument("--full", action="store_true", help="Run canonical algorithms with persist=False.")
    parser.add_argument("--streamlit", action="store_true", help="Run Streamlit page smoke tests.")
    args = parser.parse_args()

    if not (args.quick or args.sample or args.full or args.streamlit):
        args.quick = True

    try:
        run_quick_checks()
        if args.sample:
            run_sample_checks(args.sample_size)
        if args.full:
            run_full_checks()
        if args.streamlit:
            run_streamlit_checks()
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1

    print("[OK] Iteration 5 validation completed.")
    return 0


def run_quick_checks() -> None:
    print("[quick] Checking syntax for src/*.py")
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")

    print("[quick] Importing compatibility modules")
    modules = [
        "src.preprocessing",
        "src.preprocessing.preprocessor",
        "src.preprocessing.preprocessor_configured",
        "src.preprocessing.dataset_inspector",
        "src.clustering.weighted",
        "src.evaluation",
        "src.evaluation.execution",
        "src.evaluation.feature_space",
        "src.evaluation.comparability",
        "src.evaluation.parameter_optimizer",
        "src.evaluation.provenance",
        "src.evaluation.results_manager",
        "src.evaluation.report_generator",
        "src.visualization.clustering_plots",
        "src.visualization.semantic_profiles",
    ]
    for module in modules:
        importlib.import_module(module)

    from src.clustering.weighted import WDBSCAN, WKMedoids, WeightedHierarchicalClustering
    from src.evaluation.report_generator import ReportGenerator
    from src.evaluation.results_manager import ResultsManager

    assert WKMedoids and WeightedHierarchicalClustering and WDBSCAN

    print("[quick] Checking WK-Medoids weight contract")
    _check_wkmedoids_contract(WKMedoids)

    print("[quick] Checking weighted hierarchical linkage contract")
    _check_weighted_hierarchical_contract(WeightedHierarchicalClustering)

    print("[quick] Checking W-DBSCAN precomputed distance contract")
    _check_wdbscan_contract(WDBSCAN)

    print("[quick] Checking metric key compatibility")
    legacy = {
        "silhouette_score": 0.1,
        "davies_bouldin_score": 2.0,
        "calinski_harabasz_score": 30.0,
    }
    normalized = ResultsManager._normalize_metrics(legacy)
    assert normalized["silhouette"] == normalized["silhouette_score"] == 0.1
    assert ReportGenerator._metric_value(legacy, "davies_bouldin") == 2.0

    print("[quick] Checking implementation nomenclature")
    _check_implementation_nomenclature()

    print("[quick] Checking additive weighted-geometry metrics")
    _check_weighted_geometry_metrics()

    print("[quick] Checking iteration 5 provenance")
    _check_provenance()

    print("[quick] Checking historical feature-space identifiers")
    _check_feature_spaces()

    print("[quick] Checking safe report comparability")
    _check_comparability()

    print("[quick] Checking canonical suite evidence contract")
    _check_canonical_suite()

    print("[quick] Checking additive semantic cluster profiles")
    _check_semantic_profiles()

    print("[quick] Checking persisted canonical runs")
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    paths = sorted(metrics_dir.glob("*.json"))
    assert len(paths) >= 3, "Expected at least three persisted metric files in results/metrics."
    algorithms = set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        algorithms.add(data.get("algorithm"))
        metrics = data.get("metrics", {})
        for key in ("silhouette", "davies_bouldin", "calinski_harabasz"):
            assert key in metrics, f"{path.name} is missing metric key {key}."
    assert {"WKMedoids", "W-Hierarchical Clustering", "W-DBSCAN"} <= algorithms


def _check_provenance() -> None:
    import hashlib
    import pandas as pd

    from src.evaluation.provenance import (
        PROVENANCE_SCHEMA,
        build_run_provenance,
        sha256_file,
        stored_dataset_sha256,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "processed.csv"
        dataset_path.write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
        df = pd.read_csv(dataset_path)
        provenance = build_run_provenance(
            dataset_path=dataset_path,
            df=df,
            algorithm="WKMedoids",
            params={"random_state": 42, "n_clusters": 2},
            weights={"x": 0.4, "y": 0.6},
        )

        expected_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        assert sha256_file(dataset_path) == expected_hash
        assert provenance["schema"] == PROVENANCE_SCHEMA
        assert provenance["dataset"]["sha256"] == expected_hash
        assert provenance["dataset"]["rows"] == 2
        assert provenance["dataset"]["columns"] == 2
        assert provenance["dataset"]["feature_order"] == ["x", "y"]
        assert len(provenance["configuration"]["execution_sha256"]) == 64
        assert stored_dataset_sha256({"metadata": {"provenance": provenance}}) == expected_hash
        assert stored_dataset_sha256({"metadata": {}}) is None


def _check_weighted_geometry_metrics() -> None:
    import numpy as np
    import pandas as pd

    from src.evaluation.metrics import (
        calculate_metrics,
        calculate_weighted_geometry_metrics,
    )

    class StaticWeightManager:
        def __init__(self, weights):
            self.weights = dict(weights)

        def get_weights_array(self, feature_order=None):
            order = feature_order or list(self.weights)
            return [self.weights[name] for name in order]

    data = pd.DataFrame(
        {
            "x": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2],
            "y": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2],
        }
    )
    labels = np.array([0, 0, 0, 1, 1, 1])
    manager = StaticWeightManager({"x": 1.0, "y": 1.0})
    historical = calculate_metrics(data, labels)
    geometry = calculate_weighted_geometry_metrics(data, labels, manager)

    for key in ("silhouette", "davies_bouldin", "calinski_harabasz"):
        assert np.isclose(historical[key], geometry[key]), (
            f"Weighted geometry with unit weights changed {key}."
        )

    weighted = calculate_weighted_geometry_metrics(
        pd.DataFrame(
            {
                "x": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2],
                "y": [0.0, 3.0, 0.2, 5.0, 2.0, 5.1],
            }
        ),
        labels,
        StaticWeightManager({"x": 1.0, "y": 0.01}),
    )
    assert set(weighted) == {"silhouette", "davies_bouldin", "calinski_harabasz"}
    assert all(np.isfinite(value) for value in weighted.values())

    anisotropic_historical = calculate_metrics(
        pd.DataFrame(
            {
                "x": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2],
                "y": [0.0, 3.0, 0.2, 5.0, 2.0, 5.1],
            }
        ),
        labels,
    )
    assert any(
        not np.isclose(anisotropic_historical[key], weighted[key])
        for key in weighted
    ), "La ponderación anisotrópica no modificó ninguna métrica geométrica."

    invalid_inputs = (
        (data, labels[:-1], StaticWeightManager({"x": 1.0, "y": 1.0})),
        (data, labels, StaticWeightManager({"x": -1.0, "y": 1.0})),
        (data, labels, StaticWeightManager({"x": 0.0, "y": 0.0})),
        (data.assign(x=np.nan), labels, StaticWeightManager({"x": 1.0, "y": 1.0})),
        (data, labels, StaticWeightManager({"x": 1.0})),
    )
    for invalid_data, invalid_labels, invalid_manager in invalid_inputs:
        try:
            calculate_weighted_geometry_metrics(
                invalid_data,
                invalid_labels,
                invalid_manager,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("La geometría ponderada aceptó una entrada inválida.")


def _check_implementation_nomenclature() -> None:
    from src.evaluation.execution import (
        ALGORITHM_IMPLEMENTATIONS,
        ALGORITHMS,
        describe_algorithm_implementation,
    )

    assert set(ALGORITHM_IMPLEMENTATIONS) == set(ALGORITHMS)
    for algorithm in ALGORITHMS:
        implementation = describe_algorithm_implementation(algorithm)
        assert implementation["identifier"] == algorithm
        assert implementation["implementation_name"]
        assert implementation["weighting_strategy"]
        assert implementation["method_family"]
    assert "W-DBSCANR" in describe_algorithm_implementation("W-DBSCAN")[
        "not_equivalent_to"
    ]
    assert "Ward_p" in describe_algorithm_implementation(
        "W-Hierarchical Clustering"
    )["not_equivalent_to"]
    try:
        describe_algorithm_implementation("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Se describió un algoritmo no soportado.")


def _check_feature_spaces() -> None:
    from src.evaluation.feature_space import (
        ALL_SAMPLES_V1,
        COMMON_PROCESSED_V1,
        DIRECT_WEIGHTED_DISTANCE_V1,
        HISTORICAL_READING_V1,
        PRECOMPUTED_DISTANCE_V1,
        WEIGHTED_GEOMETRY_READING_V1,
        build_evaluation_context,
        build_metric_reading,
    )

    context = build_evaluation_context(
        clustering_space=DIRECT_WEIGHTED_DISTANCE_V1,
        model_input_space=PRECOMPUTED_DISTANCE_V1,
        metric_space=COMMON_PROCESSED_V1,
        population=ALL_SAMPLES_V1,
        n_total=6,
        n_evaluated=6,
        evaluated_indices=list(range(6)),
    )
    assert context["clustering_space"]["id"] == DIRECT_WEIGHTED_DISTANCE_V1
    assert context["metric_space"]["id"] == COMMON_PROCESSED_V1
    assert context["population"]["id"] == ALL_SAMPLES_V1
    assert context["n_evaluated"] == context["n_total"] == 6
    assert context["evaluation_index_sha256"]

    reading = build_metric_reading(
        reading=HISTORICAL_READING_V1,
        metric_space=COMMON_PROCESSED_V1,
        population=ALL_SAMPLES_V1,
        metrics={"silhouette": 0.1, "davies_bouldin": 2.0, "calinski_harabasz": 3.0},
        silhouette_definition="euclidean_processed_dataset",
        coordinate_definition="processed_features",
        relationship_to_historical="self",
    )
    assert reading["reading"] == HISTORICAL_READING_V1
    assert reading["space"]["id"] == COMMON_PROCESSED_V1
    assert reading["population"]["id"] == ALL_SAMPLES_V1
    try:
        build_metric_reading(
            reading="unknown",
            metric_space=COMMON_PROCESSED_V1,
            population=ALL_SAMPLES_V1,
            metrics={},
            silhouette_definition="unknown",
            coordinate_definition="unknown",
            relationship_to_historical="unknown",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Se aceptó una lectura métrica desconocida.")


def _check_comparability() -> None:
    from src.evaluation.comparability import assess_run_comparability
    from src.evaluation.feature_space import (
        ALL_SAMPLES_V1,
        COMMON_PROCESSED_V1,
        DIRECT_WEIGHTED_DISTANCE_V1,
        PRECOMPUTED_DISTANCE_V1,
        WEIGHTED_PCA_V1,
        build_evaluation_context,
    )
    from src.evaluation.report_generator import ReportGenerator
    from src.evaluation.results_manager import ResultsManager

    def metadata(metric_space: str) -> dict:
        return {
            "provenance": {"dataset": {"sha256": "a" * 64}},
            "evaluation_context": build_evaluation_context(
                clustering_space=DIRECT_WEIGHTED_DISTANCE_V1,
                model_input_space=PRECOMPUTED_DISTANCE_V1,
                metric_space=metric_space,
                population=ALL_SAMPLES_V1,
                n_total=6,
                n_evaluated=6,
                evaluated_indices=list(range(6)),
            ),
        }

    base_run = {"run_id": "run_a", "n_samples": 6, "n_noise": 0, "metadata": metadata(COMMON_PROCESSED_V1)}
    comparable_run = {"run_id": "run_b", "n_samples": 6, "n_noise": 0, "metadata": metadata(COMMON_PROCESSED_V1)}
    incompatible_run = {"run_id": "run_c", "n_samples": 6, "n_noise": 0, "metadata": metadata(WEIGHTED_PCA_V1)}
    assert assess_run_comparability([base_run])["status"] == "insufficient"
    assert assess_run_comparability([base_run, comparable_run])["directly_comparable"]
    assert not assess_run_comparability([base_run, incompatible_run])["directly_comparable"]
    assert assess_run_comparability([base_run, {"run_id": "legacy"}])["status"] == "unverifiable"

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ResultsManager(results_path=tmpdir)
        common_args = {
            "params": {"n_clusters": 2},
            "weights": {"x": 1.0},
            "labels": [0, 0, 0, 1, 1, 1],
        }
        manager.save_run(
            algorithm="Algorithm A",
            metrics={"silhouette": 0.2, "davies_bouldin": 1.5, "calinski_harabasz": 10.0},
            metadata=metadata(COMMON_PROCESSED_V1),
            run_id="run_a",
            **common_args,
        )
        manager.save_run(
            algorithm="Algorithm B",
            metrics={"silhouette": 0.3, "davies_bouldin": 1.2, "calinski_harabasz": 12.0},
            metadata=metadata(COMMON_PROCESSED_V1),
            run_id="run_b",
            **common_args,
        )
        manager.save_run(
            algorithm="Algorithm C",
            metrics={"silhouette": 0.4, "davies_bouldin": 1.0, "calinski_harabasz": 14.0},
            metadata=metadata(WEIGHTED_PCA_V1),
            run_id="run_c",
            **common_args,
        )
        generator = ReportGenerator(manager)
        valid_report = generator.compare_runs(["run_a", "run_b"], "valid")
        assert valid_report["comparability"]["directly_comparable"]
        assert valid_report["best_by_metric"]["silhouette"] == "run_b"
        invalid_report = generator.compare_runs(["run_a", "run_c"], "invalid")
        assert not invalid_report["comparability"]["directly_comparable"]
        assert invalid_report["best_by_metric"] == {}


def _check_canonical_suite() -> None:
    from src.evaluation.canonical_suite import (
        CANONICAL_GEOMETRY_BASELINES,
        CANONICAL_METRIC_BASELINES,
        CANONICAL_SUITE_SCHEMA,
        evaluate_canonical_regression,
        evaluate_geometry_regression,
    )
    from src.evaluation.results_manager import ResultsManager

    metrics = CANONICAL_METRIC_BASELINES["WKMedoids"]
    assert evaluate_canonical_regression("WKMedoids", metrics)["passed"]
    changed = {**metrics, "silhouette": metrics["silhouette"] + 0.01}
    assert not evaluate_canonical_regression("WKMedoids", changed)["passed"]
    geometry = CANONICAL_GEOMETRY_BASELINES["WKMedoids"]
    assert evaluate_geometry_regression("WKMedoids", geometry)["passed"]
    changed_geometry = {**geometry, "silhouette": geometry["silhouette"] + 0.01}
    assert not evaluate_geometry_regression("WKMedoids", changed_geometry)["passed"]

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ResultsManager(results_path=tmpdir)
        suite_id = manager.save_suite({
            "schema": CANONICAL_SUITE_SCHEMA,
            "generated_at": "2026-07-24T00:00:00",
            "runs": [{"algorithm": "WKMedoids"}],
            "comparability": {"status": "incompatible"},
            "regression": {"all_passed": True},
            "geometry_regression": {"all_passed": True},
        }, suite_id="suite_test")
        assert suite_id == "suite_test"
        assert manager.load_suite(suite_id)["schema"] == CANONICAL_SUITE_SCHEMA
        assert manager.list_suites()[0]["regression_passed"]
        assert manager.list_suites()[0]["geometry_regression_passed"]


def _check_semantic_profiles() -> None:
    import numpy as np
    import pandas as pd
    from pandas.testing import assert_frame_equal

    from src.visualization.clustering_plots import cluster_profiles
    from src.visualization.semantic_profiles import (
        SEMANTIC_PROFILE_EXPORT_SCHEMA,
        build_semantic_profile_export,
        cluster_semantic_profiles,
    )

    df = pd.DataFrame(
        {
            "numeric": [0.0, 0.5, 1.0, 0.25, 0.25, 0.9],
            "nominal": [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            "ordinal": [0.0, 0.5, 1.0, 0.5, 0.5, 1.0],
            "binary": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "ambiguous": [0.0, 0.5, 1.0, 0.5, 1.0, 0.0],
        }
    )
    labels = np.array([0, 0, 0, 1, 1, -1])
    metadata = {
        "numeric": {"nombre": "Numerica", "tipo": "numerico", "rango": (0, 100)},
        "nominal": {
            "nombre": "Nominal",
            "tipo": "categorico_nominal",
            "categorias": {1: "A", 2: "B"},
        },
        "ordinal": {
            "nombre": "Ordinal",
            "tipo": "categorico_ordinal",
            "rango": (1, 3),
            "categorias": {1: "Bajo", 2: "Medio", 3: "Alto"},
        },
        "binary": {
            "nombre": "Binaria",
            "tipo": "binario",
            "categorias": {0: "No", 1: "Si"},
        },
        "ambiguous": {
            "nombre": "Nominal ambigua",
            "tipo": "categorico_nominal",
            "categorias": {1: "A", 2: "B", 3: "C", 4: "D"},
        },
    }

    original_df = df.copy(deep=True)
    original_labels = labels.copy()
    legacy_before = cluster_profiles(df, labels)
    semantic = cluster_semantic_profiles(df, labels, feature_metadata=metadata)
    legacy_after = cluster_profiles(df, labels)

    assert_frame_equal(df, original_df)
    assert np.array_equal(labels, original_labels)
    assert_frame_equal(legacy_after, legacy_before)
    assert set(semantic.summaries["cluster_id"]) == {0, 1}
    assert -1 not in set(semantic.summaries["cluster_id"])

    def summary(cluster_id: int, feature_code: str):
        matches = semantic.summaries[
            (semantic.summaries["cluster_id"] == cluster_id)
            & (semantic.summaries["feature_code"] == feature_code)
        ]
        assert len(matches) == 1
        return matches.iloc[0]

    numeric = summary(0, "numeric")
    assert numeric["representative_statistic"] == "mean"
    assert math.isclose(float(numeric["mean"]), 0.5)
    assert math.isclose(float(numeric["median"]), 0.5)
    assert math.isclose(float(numeric["std"]), math.sqrt(1 / 6))
    assert numeric["difference_unit"] == "processed_scale"

    nominal = summary(0, "nominal")
    assert nominal["representative_statistic"] == "mode"
    assert nominal["representative_label"] == "A"
    assert math.isclose(float(nominal["representative_percentage"]), 200 / 3)
    assert math.isnan(float(nominal["difference_from_global"]))

    ordinal = summary(0, "ordinal")
    assert ordinal["representative_statistic"] == "median"
    assert ordinal["representative_label"] == "Medio"
    assert math.isclose(float(ordinal["median"]), 0.5)

    binary = summary(0, "binary")
    assert binary["representative_statistic"] == "prevalence"
    assert binary["representative_label"] == "Si"
    assert math.isclose(float(binary["representative_percentage"]), 200 / 3)
    assert math.isclose(float(binary["global_reference_percentage"]), 60.0)
    assert math.isclose(float(binary["difference_from_global"]), 20 / 3)
    assert binary["difference_unit"] == "percentage_points"

    ambiguous = summary(0, "ambiguous")
    assert ambiguous["category_mapping_status"] == "processed_only"
    assert ambiguous["representative_label"] == "Valor procesado 0"

    nominal_b = semantic.distributions[
        (semantic.distributions["cluster_id"] == 0)
        & (semantic.distributions["feature_code"] == "nominal")
        & (semantic.distributions["category_label"] == "B")
    ].iloc[0]
    assert int(nominal_b["count"]) == 1
    assert math.isclose(float(nominal_b["percentage"]), 100 / 3)
    assert math.isclose(float(nominal_b["global_percentage"]), 60.0)

    export = build_semantic_profile_export(
        semantic,
        run_id="semantic_test_run",
        algorithm="Algorithm Test",
        run_timestamp="2026-07-20T12:00:00",
        dataset_sha256="a" * 64,
        execution_sha256="b" * 64,
    )
    assert export["schema"] == SEMANTIC_PROFILE_EXPORT_SCHEMA
    assert export["source"]["run_id"] == "semantic_test_run"
    assert export["source"]["dataset_sha256"] == "a" * 64
    assert export["analysis"]["population_id"] == "clustered_samples_without_noise_v1"
    assert export["analysis"]["cluster_ids"] == [0, 1]
    assert export["analysis"]["feature_count"] == len(df.columns)
    assert export["analysis"]["processed_only_features"] == ["ambiguous"]
    assert len(export["tables"]["summaries"]) == len(semantic.summaries)
    assert len(export["tables"]["distributions"]) == len(semantic.distributions)
    json.dumps(export, ensure_ascii=False, allow_nan=False)

    with_noise = cluster_semantic_profiles(
        df,
        labels,
        exclude_noise=False,
        feature_metadata=metadata,
    )
    assert -1 in set(with_noise.summaries["cluster_id"])
    assert "noise" in set(with_noise.summaries["cluster"])

    try:
        cluster_semantic_profiles(df, labels[:-1], feature_metadata=metadata)
    except ValueError:
        pass
    else:
        raise AssertionError("Semantic profiles accepted labels with an invalid length.")

    try:
        cluster_semantic_profiles(
            df,
            np.array([0, 0, 0, 1, 1, 1.5]),
            feature_metadata=metadata,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Semantic profiles accepted non-integer labels.")


def _check_wkmedoids_contract(WKMedoids) -> None:
    import numpy as np
    import pandas as pd

    class StaticWeightManager:
        def __init__(self, weights):
            self.weights = weights
            self.calls = 0

        def get_weights_array(self, feature_order=None):
            self.calls += 1
            assert feature_order == ["x", "y"]
            return self.weights

    data = pd.DataFrame(
        {
            "x": [0.0, 0.1, 0.2, 4.8, 4.9, 5.0],
            "y": [0.0, 0.2, 0.1, 5.0, 4.8, 4.9],
        }
    )

    manager = StaticWeightManager([0.4, 0.6])
    weighted = WKMedoids(n_clusters=2, random_state=42, weight_manager=manager)
    with contextlib.redirect_stdout(io.StringIO()):
        weighted.fit(data)
    assert manager.calls == 1, "WKMedoids debe resolver los pesos una vez en fit()."

    with contextlib.redirect_stdout(io.StringIO()):
        predicted = weighted.predict(data)
    assert manager.calls == 2, "WKMedoids debe resolver los pesos una vez en predict()."
    assert len(predicted) == len(data)

    implicit_uniform = WKMedoids(n_clusters=2, random_state=42, weight_manager=None)
    explicit_uniform = WKMedoids(
        n_clusters=2,
        random_state=42,
        weight_manager=StaticWeightManager([1.0, 1.0]),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        implicit_uniform.fit(data)
        explicit_uniform.fit(data)
    implicit_partition = implicit_uniform.labels_[:, None] == implicit_uniform.labels_[None, :]
    explicit_partition = explicit_uniform.labels_[:, None] == explicit_uniform.labels_[None, :]
    assert np.array_equal(implicit_partition, explicit_partition)

    for invalid_weights in ([1.0], [1.0, np.nan], [1.0, -0.1], [0.0, 0.0]):
        invalid_model = WKMedoids(
            n_clusters=2,
            random_state=42,
            weight_manager=StaticWeightManager(invalid_weights),
        )
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                invalid_model.fit(data)
        except ValueError:
            pass
        else:
            raise AssertionError(f"WKMedoids acepto pesos invalidos: {invalid_weights}")


def _check_weighted_hierarchical_contract(WeightedHierarchicalClustering) -> None:
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import fcluster, linkage as scipy_linkage
    from scipy.spatial.distance import pdist

    class StaticWeightManager:
        def __init__(self, weights):
            self.weights = weights
            self.calls = 0

        def get_weights_array(self, feature_order=None):
            self.calls += 1
            assert feature_order == ["x", "y"]
            return self.weights

    data = pd.DataFrame(
        {
            "x": [0.0, 0.1, 0.2, 4.8, 4.9, 5.0],
            "y": [0.0, 0.2, 0.1, 5.0, 4.8, 4.9],
        }
    )

    for linkage_method in ("complete", "average", "single"):
        model = WeightedHierarchicalClustering(
            n_clusters=2,
            linkage=linkage_method,
            weight_manager=None,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(data)

        expected_linkage = scipy_linkage(pdist(data.to_numpy()), method=linkage_method)
        assert np.allclose(model.linkage_matrix_, expected_linkage)

        scipy_labels = fcluster(model.linkage_matrix_, t=2, criterion="maxclust")
        model_partition = model.labels_[:, None] == model.labels_[None, :]
        scipy_partition = scipy_labels[:, None] == scipy_labels[None, :]
        assert np.array_equal(model_partition, scipy_partition)

    manager = StaticWeightManager([0.4, 0.6])
    weighted = WeightedHierarchicalClustering(
        n_clusters=2,
        linkage="complete",
        weight_manager=manager,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        weighted.fit(data)
    assert manager.calls == 1

    invalid = WeightedHierarchicalClustering(
        n_clusters=2,
        linkage="complete",
        weight_manager=StaticWeightManager([1.0]),
    )
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            invalid.fit(data)
    except ValueError:
        pass
    else:
        raise AssertionError("WeightedHierarchicalClustering acepto pesos invalidos.")


def _check_wdbscan_contract(WDBSCAN) -> None:
    import numpy as np
    import pandas as pd

    class StaticWeightManager:
        def __init__(self, weights):
            self.weights = weights
            self.calls = 0

        def get_weights_array(self, feature_order=None):
            self.calls += 1
            assert feature_order == ["x", "y"]
            return self.weights

    data = pd.DataFrame(
        {
            "x": [0.0, 0.1, 0.2, 4.8, 4.9, 5.0],
            "y": [0.0, 0.2, 0.1, 5.0, 4.8, 4.9],
        }
    )

    manager = StaticWeightManager([0.4, 0.6])
    model = WDBSCAN(
        eps=0.5,
        min_samples=2,
        metric="euclidean",
        algorithm="auto",
        weight_manager=manager,
    )
    assert model.metric == "precomputed"
    assert model.algorithm == "brute"
    assert model._model.metric == "precomputed"
    assert model._model.algorithm == "brute"
    assert model.get_params()["metric"] == "precomputed"
    assert model.get_params()["algorithm"] == "brute"

    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(data)
    assert manager.calls == 1
    assert model.n_clusters_ == 2
    assert model.n_noise_points_ == 0

    implicit_uniform = WDBSCAN(eps=0.5, min_samples=2, weight_manager=None)
    explicit_uniform = WDBSCAN(
        eps=0.5,
        min_samples=2,
        weight_manager=StaticWeightManager([1.0, 1.0]),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        implicit_uniform.fit(data)
        explicit_uniform.fit(data)
    implicit_partition = implicit_uniform.labels_[:, None] == implicit_uniform.labels_[None, :]
    explicit_partition = explicit_uniform.labels_[:, None] == explicit_uniform.labels_[None, :]
    assert np.array_equal(implicit_partition, explicit_partition)

    invalid = WDBSCAN(
        eps=0.5,
        min_samples=2,
        weight_manager=StaticWeightManager([1.0]),
    )
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            invalid.fit(data)
    except ValueError:
        pass
    else:
        raise AssertionError("WDBSCAN acepto pesos invalidos.")

    invalid_data = data.copy()
    invalid_data.loc[0, "x"] = np.nan
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            WDBSCAN(eps=0.5, min_samples=2).fit(invalid_data)
    except ValueError as exc:
        assert "finitos" in str(exc)
    else:
        raise AssertionError("WDBSCAN acepto una matriz de distancias no finita.")


def run_full_checks() -> None:
    print("[full] Checking historical preprocessor call with temporary output")
    import src.preprocessing.preprocessor as preprocessor_module
    from src.preprocessing.preprocessor import DataPreprocessor
    from src.utils.constants import RAW_DATA_PATH

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_output = Path(tmpdir) / "processed.csv"
        original_output = preprocessor_module.PROCESSED_DATA_PATH
        preprocessor_module.PROCESSED_DATA_PATH = str(temp_output)
        try:
            preprocessor = DataPreprocessor()
            with contextlib.redirect_stdout(io.StringIO()):
                df = preprocessor.run_pipeline(RAW_DATA_PATH)
            assert df.shape == (1706, 28), f"Unexpected preprocessed shape: {df.shape}"
            assert int(df.isna().sum().sum()) == 0, "Preprocessed dataset has missing values."
            assert temp_output.exists(), "Temporary preprocessing output was not created."
        finally:
            preprocessor_module.PROCESSED_DATA_PATH = original_output

    print("[full] Running canonical experiments with persist=False")
    from src.evaluation.execution import default_params, run_clustering_experiment

    gc.collect()
    for algorithm, expected in _canonical_baselines().items():
        print(f"[full] Running {algorithm}")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_clustering_experiment(
                    algorithm,
                    default_params(algorithm),
                    persist=False,
                )
        except MemoryError as exc:
            raise RuntimeError(
                f"{algorithm} needs enough RAM for the full distance matrix. "
                "Close other apps or use --sample for a lighter smoke test."
            ) from exc
        assert result.run_id is None, f"{algorithm} unexpectedly persisted a run."
        for key, expected_value in expected.items():
            actual = result.metrics.get(key)
            assert actual is not None, f"{algorithm} metric {key} is None."
            assert math.isclose(actual, expected_value, rel_tol=1e-8, abs_tol=1e-8), (
                f"{algorithm} metric {key} changed: expected {expected_value}, got {actual}"
            )
        geometry_metrics = result.metadata.get("metric_readings", {}).get(
            "weighted_geometry", {}
        ).get("metrics", {})
        for key, expected_value in _canonical_geometry_baselines()[algorithm].items():
            actual = geometry_metrics.get(key)
            assert actual is not None, f"{algorithm} geometry metric {key} is None."
            assert math.isclose(actual, expected_value, rel_tol=1e-8, abs_tol=1e-8), (
                f"{algorithm} geometry metric {key} changed: "
                f"expected {expected_value}, got {actual}"
            )
        del result
        gc.collect()


def run_sample_checks(sample_size: int) -> None:
    print(f"[sample] Running algorithms on {sample_size} temporary rows")
    import pandas as pd

    from src.evaluation.canonical_suite import (
        CANONICAL_SUITE_SCHEMA,
        run_canonical_suite,
    )
    from src.utils.constants import PROCESSED_DATA_PATH

    df = pd.read_csv(PROCESSED_DATA_PATH).head(sample_size)
    assert len(df) == sample_size, f"Processed dataset has fewer than {sample_size} rows."

    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "processed_sample.csv"
        df.to_csv(sample_path, index=False)

        expected_historical_metric_spaces = {
            "WKMedoids": "common_processed_v1",
            "W-Hierarchical Clustering": "common_processed_v1",
            "W-DBSCAN": "weighted_pca_v1",
        }
        expected_geometry_metric_spaces = {
            "WKMedoids": "direct_weighted_distance_v1",
            "W-Hierarchical Clustering": "direct_weighted_distance_v1",
            "W-DBSCAN": "weighted_pca_v1",
        }
        expected_populations = {
            "WKMedoids": "all_samples_v1",
            "W-Hierarchical Clustering": "all_samples_v1",
            "W-DBSCAN": "clustered_samples_without_noise_v1",
        }
        print("[sample] Running consolidated canonical suite")
        with contextlib.redirect_stdout(io.StringIO()):
            suite = run_canonical_suite(dataset_path=str(sample_path), persist=False)
        assert suite.suite_id is None
        assert suite.artifact["schema"] == CANONICAL_SUITE_SCHEMA
        assert len(suite.artifact["runs"]) == len(_canonical_baselines())
        assert suite.artifact["comparability"]["status"] == "incompatible"
        from src.evaluation.results_manager import ResultsManager

        temporary_manager = ResultsManager(
            results_path=Path(tmpdir) / "transient_suite_results"
        )
        temporary_suite_id = temporary_manager.save_suite(suite.artifact)
        restored_suite = temporary_manager.load_suite(temporary_suite_id)
        assert restored_suite["schema"] == CANONICAL_SUITE_SCHEMA
        assert restored_suite["geometry_regression"] == suite.artifact[
            "geometry_regression"
        ]
        assert (
            temporary_manager.list_suites()[0]["geometry_regression_passed"]
            == suite.artifact["geometry_regression"]["all_passed"]
        )
        suite_records = {item["algorithm"]: item for item in suite.artifact["runs"]}
        for result in suite.results:
            algorithm = result.algorithm
            assert result.run_id is None, f"{algorithm} unexpectedly persisted a run."
            assert len(result.labels) == sample_size, f"{algorithm} returned wrong label count."
            assert result.model is not None, f"{algorithm} did not return a model."
            provenance = result.metadata.get("provenance", {})
            assert provenance.get("dataset", {}).get("rows") == sample_size
            assert provenance.get("dataset", {}).get("feature_order") == df.columns.tolist()
            assert provenance.get("configuration", {}).get("execution_sha256")
            implementation = result.metadata.get("implementation", {})
            assert implementation.get("identifier") == algorithm
            assert implementation.get("implementation_name")
            evaluation_context = result.metadata.get("evaluation_context", {})
            assert (
                evaluation_context.get("metric_space", {}).get("id")
                == expected_historical_metric_spaces[algorithm]
            )
            assert evaluation_context.get("population", {}).get("id") == expected_populations[algorithm]
            geometry_metrics = result.metadata.get("geometry_metrics", {})
            metric_readings = result.metadata.get("metric_readings", {})
            historical_reading = metric_readings.get("historical", {})
            geometry_reading = metric_readings.get("weighted_geometry", {})
            assert historical_reading.get("reading") == "historical"
            assert geometry_reading.get("reading") == "weighted_geometry"
            assert historical_reading.get("space", {}).get("id") == (
                expected_historical_metric_spaces[algorithm]
            )
            assert geometry_reading.get("space", {}).get("id") == (
                expected_geometry_metric_spaces[algorithm]
            )
            assert historical_reading.get("population", {}).get("id") == (
                expected_populations[algorithm]
            )
            assert geometry_reading.get("population", {}).get("id") == (
                expected_populations[algorithm]
            )
            assert historical_reading.get("metrics") == result.metrics
            assert geometry_reading.get("metrics") == geometry_metrics.get("metrics")
            assert geometry_metrics.get("schema") == "iteration5-geometry-metrics-v1"
            assert geometry_metrics.get("space") == expected_geometry_metric_spaces[algorithm]
            assert geometry_metrics.get("population") == expected_populations[algorithm]
            assert set(geometry_metrics.get("metrics", {})) == {
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
            }
            suite_record = suite_records[algorithm]
            assert suite_record.get("implementation") == implementation
            assert suite_record.get("metric_readings") == metric_readings
            assert "geometry_regression" in suite_record
            assert suite_record.get("geometry_metrics", {}).get("schema") == (
                "iteration5-geometry-metrics-v1"
            )
            assert suite_record["semantic_profiles"]["source"]["algorithm"] == algorithm
            assert "elapsed_seconds" in suite_record
            del result
            gc.collect()


def run_streamlit_checks() -> None:
    print("[streamlit] Checking dependencies")
    importlib.import_module("plotly")
    from streamlit.testing.v1 import AppTest

    pages = [
        "src/app.py",
        "src/pages/0_preprocesamiento.py",
        "src/pages/1_ponderacion.py",
        "src/pages/2_configuracion.py",
        "src/pages/3_ejecucion.py",
        "src/pages/4_resultados.py",
    ]
    for page in pages:
        app = AppTest.from_file(str(PROJECT_ROOT / page))
        app.run(timeout=20)
        assert not app.exception, f"{page} raised Streamlit exceptions: {app.exception}"

        if page == "src/pages/4_resultados.py":
            tab_labels = [tab.label for tab in app.tabs]
            assert "Perfiles" not in tab_labels, (
                f"Unexpected legacy tab found. Available tabs: {tab_labels}"
            )
            assert "Analisis de perfiles" in tab_labels, (
                f"Semantic profiles tab missing. Available tabs: {tab_labels}"
            )
            download_labels = [
                button.label for button in app.get("download_button")
            ]
            button_labels = [button.label for button in app.button]
            assert "Descargar perfiles CSV" in download_labels, (
                f"Legacy profiles download missing. Downloads: {download_labels}"
            )
            assert {
                "Resumen semantico CSV",
                "Distribuciones semanticas CSV",
                "Perfiles semanticos JSON",
            } <= set(download_labels)
            assert "Guardar figuras exportables" in button_labels, (
                f"Figure export button missing. Buttons: {button_labels}"
            )
            assert "Descargar evidencia integrada JSON" not in download_labels, (
                "Deprecated integrated evidence download is still exposed."
            )

            selectboxes = {selectbox.label: selectbox for selectbox in app.selectbox}
            assert "Cluster del perfil semantico" in selectboxes, (
                f"Cluster selector missing. Selectors: {list(selectboxes)}"
            )
            assert "Tipo de variable del perfil semantico" in selectboxes, (
                f"Feature-type selector missing. Selectors: {list(selectboxes)}"
            )

            for feature_type in (
                "categorico_nominal",
                "categorico_ordinal",
                "binario",
            ):
                current_selectboxes = {
                    selectbox.label: selectbox for selectbox in app.selectbox
                }
                current_selectboxes["Tipo de variable del perfil semantico"].select(
                    feature_type
                ).run(timeout=20)
                assert not app.exception, (
                    f"Semantic {feature_type} profile raised Streamlit exceptions: "
                    f"{app.exception}"
                )
                updated_labels = [selectbox.label for selectbox in app.selectbox]
                assert "Variable para inspeccionar su distribucion" in updated_labels, (
                    f"Distribution selector missing for {feature_type}. "
                    f"Selectors: {updated_labels}"
                )

                if feature_type == "categorico_nominal":
                    assert any("leng1" in warning.value for warning in app.warning), (
                        "Expected semantic warning for leng1 was not rendered. "
                        f"Warnings: {[warning.value for warning in app.warning]}"
                    )
                    semantic_summary = next(
                        dataframe.value
                        for dataframe in app.dataframe
                        if "porcentaje representativo" in dataframe.value.columns
                    )
                    semantic_distribution = next(
                        dataframe.value
                        for dataframe in app.dataframe
                        if "porcentaje del cluster" in dataframe.value.columns
                    )
                    for column in (
                        "porcentaje representativo",
                        "porcentaje total",
                    ):
                        assert semantic_summary[column].map(
                            lambda value: value == "N/A" or str(value).endswith("%")
                        ).all(), (
                            f"Invalid percentage formatting in summary column {column}: "
                            f"{semantic_summary[column].tolist()}"
                        )
                    for column in (
                        "porcentaje del cluster",
                        "porcentaje total",
                    ):
                        assert semantic_distribution[column].map(
                            lambda value: value == "N/A" or str(value).endswith("%")
                        ).all(), (
                            f"Invalid percentage formatting in distribution column {column}: "
                            f"{semantic_distribution[column].tolist()}"
                        )
                    assert semantic_distribution[
                        "diferencia en puntos porcentuales"
                    ].map(
                        lambda value: value == "N/A" or str(value).endswith(" pp")
                    ).all(), (
                        "Invalid percentage-point formatting: "
                        f"{semantic_distribution['diferencia en puntos porcentuales'].tolist()}"
                    )


if __name__ == "__main__":
    raise SystemExit(main())
