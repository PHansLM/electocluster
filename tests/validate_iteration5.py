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


EXPECTED_CANONICAL = {
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
        "src.evaluation.parameter_optimizer",
        "src.evaluation.results_manager",
        "src.evaluation.report_generator",
        "src.visualization.clustering_plots",
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
    for algorithm, expected in EXPECTED_CANONICAL.items():
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
        del result
        gc.collect()


def run_sample_checks(sample_size: int) -> None:
    print(f"[sample] Running algorithms on {sample_size} temporary rows")
    import pandas as pd

    from src.evaluation.execution import default_params, run_clustering_experiment
    from src.utils.constants import PROCESSED_DATA_PATH

    df = pd.read_csv(PROCESSED_DATA_PATH).head(sample_size)
    assert len(df) == sample_size, f"Processed dataset has fewer than {sample_size} rows."

    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "processed_sample.csv"
        df.to_csv(sample_path, index=False)

        for algorithm in EXPECTED_CANONICAL:
            print(f"[sample] Running {algorithm}")
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_clustering_experiment(
                    algorithm,
                    default_params(algorithm),
                    dataset_path=str(sample_path),
                    persist=False,
                )
            assert result.run_id is None, f"{algorithm} unexpectedly persisted a run."
            assert len(result.labels) == sample_size, f"{algorithm} returned wrong label count."
            assert result.model is not None, f"{algorithm} did not return a model."
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


if __name__ == "__main__":
    raise SystemExit(main())
