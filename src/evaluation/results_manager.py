"""
ResultsManager: persistencia de resultados de clustering por ejecucion.
Guarda metricas, asignaciones y configuracion en results/ con timestamp.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .comparability import run_evaluation_summary

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = PROJECT_ROOT / "results"


class ResultsManager:
    """
    Gestiona el almacenamiento de resultados de cada ejecucion de clustering.

    Cada ejecucion genera un ID unico basado en timestamp y se almacena en:
      - results/metrics/     -> metricas + configuracion en JSON
      - results/assignments/ -> asignaciones de cluster en CSV
      - results/comparisons/ -> comparaciones entre ejecuciones
      - results/reports/     -> reportes legibles
    """

    def __init__(self, results_path: str = None):
        self.results_path = Path(results_path) if results_path else RESULTS_PATH
        self._ensure_directories()

    def _ensure_directories(self):
        for subdir in ["metrics", "assignments", "comparisons", "reports", "figures", "suites"]:
            (self.results_path / subdir).mkdir(parents=True, exist_ok=True)

    def _generate_run_id(self, algorithm: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_algorithm = algorithm.lower().replace(" ", "_").replace("-", "")
        return f"{safe_algorithm}_{ts}"

    def generate_suite_id(self) -> str:
        """Genera un identificador para una bateria integrada de ejecuciones."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"canonical_suite_{ts}"

    def save_suite(self, artifact: dict, suite_id: str = None) -> str:
        """Persiste un artefacto consolidado sin duplicar los runs individuales."""
        suite_id = suite_id or self.generate_suite_id()
        record = {**self._json_safe(artifact), "suite_id": suite_id}
        path = self.results_path / "suites" / f"{suite_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, allow_nan=False)
        return suite_id

    def list_suites(self) -> list[dict]:
        """Lista artefactos consolidados por fecha descendente."""
        suites = []
        for path in (self.results_path / "suites").glob("*.json"):
            with open(path, encoding="utf-8") as f:
                suite = json.load(f)
            suites.append({
                "suite_id": suite.get("suite_id"),
                "generated_at": suite.get("generated_at"),
                "regression_passed": suite.get("regression", {}).get("all_passed"),
                "geometry_regression_passed": suite.get("geometry_regression", {}).get(
                    "all_passed"
                ),
                "comparability_status": suite.get("comparability", {}).get("status"),
                "algorithms": [run.get("algorithm") for run in suite.get("runs", [])],
            })
        return sorted(suites, key=lambda suite: suite.get("generated_at") or "", reverse=True)

    def load_suite(self, suite_id: str) -> dict:
        path = self.results_path / "suites" / f"{suite_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No existe suite con suite_id: {suite_id}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def save_run(
        self,
        algorithm: str,
        params: dict,
        weights: dict,
        metrics: dict,
        labels: list,
        metadata: dict = None,
        run_id: str = None,
    ) -> str:
        """
        Guarda una ejecucion completa.

        Args:
            algorithm: Nombre del algoritmo (ej. 'WKMedoids').
            params: Parametros de configuracion.
            weights: Esquema de pesos usado {feature: weight}.
            metrics: Resultado de ClusteringMetrics.calculate_all().
            labels: Etiquetas asignadas a cada registro.
            metadata: Informacion adicional (PCA, muestras evaluadas, etc.).
            run_id: ID opcional; si es None se genera automaticamente.

        Returns:
            run_id generado o provisto.
        """
        if run_id is None:
            run_id = self._generate_run_id(algorithm)

        labels = [int(label) for label in labels]
        clean_metrics = self._normalize_metrics(metrics)

        record = {
            "run_id": run_id,
            "algorithm": algorithm,
            "timestamp": datetime.now().isoformat(),
            "params": self._json_safe(params),
            "weights_snapshot": self._json_safe(weights),
            "metrics": clean_metrics,
            "n_clusters": len(set(label for label in labels if label != -1)),
            "n_noise": int(sum(1 for label in labels if label == -1)),
            "n_samples": len(labels),
            "metadata": self._json_safe(metadata or {}),
        }

        metrics_path = self.results_path / "metrics" / f"{run_id}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        assignments_path = self.results_path / "assignments" / f"{run_id}.csv"
        with open(assignments_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "cluster"])
            for i, label in enumerate(labels):
                writer.writerow([i, label])

        return run_id

    def list_runs(self) -> list[dict]:
        """
        Devuelve todas las ejecuciones guardadas, ordenadas por timestamp desc.
        """
        runs = []
        metrics_dir = self.results_path / "metrics"
        for filepath in sorted(metrics_dir.glob("*.json"), reverse=True):
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            metrics = data.get("metrics", {})
            evaluation = run_evaluation_summary(data)
            runs.append({
                "run_id": data.get("run_id"),
                "algorithm": data.get("algorithm"),
                "timestamp": data.get("timestamp"),
                "n_clusters": data.get("n_clusters"),
                "n_noise": data.get("n_noise"),
                "silhouette": self._metric_value(metrics, "silhouette"),
                "davies_bouldin": self._metric_value(metrics, "davies_bouldin"),
                "calinski_harabasz": self._metric_value(metrics, "calinski_harabasz"),
                "n_total": evaluation["n_total"],
                "n_evaluated": evaluation["n_evaluated"],
                "coverage": evaluation["coverage"],
                "coverage_percentage": evaluation["coverage_percentage"],
                "noise_percentage": evaluation["noise_percentage"],
                "metric_space": evaluation["metric_space"],
                "population": evaluation["population"],
                "dataset_sha256": evaluation["dataset_sha256"],
            })
        return sorted(runs, key=lambda run: run.get("timestamp") or "", reverse=True)

    def load_run(self, run_id: str) -> dict:
        """Carga el registro completo de una ejecucion por su run_id."""
        path = self.results_path / "metrics" / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No existe ejecucion con run_id: {run_id}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def load_assignments(self, run_id: str) -> list[int]:
        """Carga las etiquetas de cluster de una ejecucion."""
        path = self.results_path / "assignments" / f"{run_id}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No existen asignaciones para run_id: {run_id}")
        labels = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(int(row["cluster"]))
        return labels

    def delete_run(self, run_id: str):
        """Elimina los archivos de una ejecucion."""
        for subdir, ext in [("metrics", ".json"), ("assignments", ".csv")]:
            path = self.results_path / subdir / f"{run_id}{ext}"
            if path.exists():
                os.remove(path)

    @staticmethod
    def _metric_value(metrics: dict, key: str):
        legacy = {
            "silhouette": "silhouette_score",
            "davies_bouldin": "davies_bouldin_score",
            "calinski_harabasz": "calinski_harabasz_score",
        }
        value = metrics.get(key, metrics.get(legacy.get(key)))
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return value

    @classmethod
    def _normalize_metrics(cls, metrics: dict) -> dict:
        normalized = {
            "silhouette": cls._metric_value(metrics, "silhouette"),
            "davies_bouldin": cls._metric_value(metrics, "davies_bouldin"),
            "calinski_harabasz": cls._metric_value(metrics, "calinski_harabasz"),
        }
        normalized.update({
            "silhouette_score": normalized["silhouette"],
            "davies_bouldin_score": normalized["davies_bouldin"],
            "calinski_harabasz_score": normalized["calinski_harabasz"],
        })
        return normalized

    @classmethod
    def _json_safe(cls, value: Any):
        if isinstance(value, dict):
            return {str(k): cls._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return cls._json_safe(value.tolist())
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            if np.isnan(value):
                return None
            return float(value)
        if isinstance(value, float) and np.isnan(value):
            return None
        return value
