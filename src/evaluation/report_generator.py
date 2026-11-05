"""
ReportGenerator: genera reportes comparativos entre ejecuciones de clustering.
Produce JSON estructurado y CSV tabular en results/comparisons/ y results/reports/.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

from .comparability import assess_run_comparability, run_evaluation_summary
from .results_manager import ResultsManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = PROJECT_ROOT / "results"


class ReportGenerator:
    """
    Genera reportes comparativos entre dos o mas ejecuciones de clustering.
    """

    def __init__(self, results_manager: ResultsManager = None):
        self.rm = results_manager or ResultsManager()
        self.results_path = self.rm.results_path

    def compare_runs(self, run_ids: list[str], report_name: str = None) -> dict:
        """
        Compara metricas de multiples ejecuciones.

        Args:
            run_ids: Lista de run_ids a comparar (minimo 2).
            report_name: Nombre del reporte; si es None se genera automaticamente.

        Returns:
            Dict con el reporte estructurado.
        """
        if len(run_ids) < 2:
            raise ValueError("Se requieren al menos 2 ejecuciones para comparar.")

        runs = [self.rm.load_run(run_id) for run_id in run_ids]
        comparability = assess_run_comparability(runs)

        report = {
            "report_name": report_name or f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "runs_compared": run_ids,
            "summary": [],
            "comparability": comparability,
            "best_by_metric": {},
        }

        for run in runs:
            metrics = run.get("metrics", {})
            evaluation = run_evaluation_summary(run)
            report["summary"].append({
                "run_id": run["run_id"],
                "algorithm": run["algorithm"],
                "params": run["params"],
                "n_clusters": run["n_clusters"],
                "n_noise": run["n_noise"],
                "silhouette": self._metric_value(metrics, "silhouette"),
                "davies_bouldin": self._metric_value(metrics, "davies_bouldin"),
                "calinski_harabasz": self._metric_value(metrics, "calinski_harabasz"),
                **evaluation,
            })

        if comparability["directly_comparable"]:
            valid_sil = [
                run for run in runs
                if self._metric_value(run.get("metrics", {}), "silhouette") is not None
            ]
            valid_db = [
                run for run in runs
                if self._metric_value(run.get("metrics", {}), "davies_bouldin") is not None
            ]
            valid_ch = [
                run for run in runs
                if self._metric_value(run.get("metrics", {}), "calinski_harabasz") is not None
            ]

            if valid_sil:
                report["best_by_metric"]["silhouette"] = max(
                    valid_sil,
                    key=lambda run: self._metric_value(run["metrics"], "silhouette")
                )["run_id"]
            if valid_db:
                report["best_by_metric"]["davies_bouldin"] = min(
                    valid_db,
                    key=lambda run: self._metric_value(run["metrics"], "davies_bouldin")
                )["run_id"]
            if valid_ch:
                report["best_by_metric"]["calinski_harabasz"] = max(
                    valid_ch,
                    key=lambda run: self._metric_value(run["metrics"], "calinski_harabasz")
                )["run_id"]

        comp_path = self.results_path / "comparisons" / f"{report['report_name']}.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self._save_csv_report(report)
        return report

    def generate_run_summary(self, run_id: str) -> str:
        """Genera resumen legible de una sola ejecucion."""
        run = self.rm.load_run(run_id)
        metrics = run["metrics"]
        evaluation = run_evaluation_summary(run)
        lines = [
            f"EJECUCION: {run_id}",
            f"Algoritmo: {run['algorithm']}",
            f"Timestamp: {run['timestamp']}",
            f"Parametros: {run['params']}",
            f"Clusters identificados: {run['n_clusters']}",
            f"Puntos de ruido: {run['n_noise']}",
            f"Registros totales: {run['n_samples']}",
            f"Registros evaluados: {evaluation['n_evaluated']}",
            f"Cobertura: {self._format_percentage(evaluation['coverage_percentage'])}",
            f"Espacio de metricas: {evaluation['metric_space'] or 'No registrado'}",
            f"Poblacion: {evaluation['population'] or 'No registrada'}",
            "",
            "METRICAS:",
            f"  Silhouette Score:       {self._metric_value(metrics, 'silhouette')}",
            f"  Davies-Bouldin Index:   {self._metric_value(metrics, 'davies_bouldin')}",
            f"  Calinski-Harabasz:      {self._metric_value(metrics, 'calinski_harabasz')}",
        ]
        return "\n".join(lines)

    def _save_csv_report(self, report: dict):
        """Guarda las métricas comparadas en un CSV, una fila por ejecución."""
        comparability = report["comparability"]
        reasons = " | ".join(comparability.get("reasons", []))
        rows = []
        for summary in report["summary"]:
            rows.append({
                "reporte": report["report_name"],
                "generado_en": report["generated_at"],
                "ejecucion": summary["run_id"],
                "algoritmo": summary["algorithm"],
                "clusters": summary["n_clusters"],
                "registros_evaluados": summary["n_evaluated"],
                "cobertura_porcentaje": summary["coverage_percentage"],
                "ruido": summary["n_noise"],
                "silhouette": summary["silhouette"],
                "davies_bouldin": summary["davies_bouldin"],
                "calinski_harabasz": summary["calinski_harabasz"],
                "comparables_directamente": comparability["directly_comparable"],
                "motivo_sin_ranking": reasons,
            })

        report_path = self.results_path / "reports" / f"{report['report_name']}.csv"
        with open(report_path, "w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _metric_value(metrics: dict, key: str):
        legacy = {
            "silhouette": "silhouette_score",
            "davies_bouldin": "davies_bouldin_score",
            "calinski_harabasz": "calinski_harabasz_score",
        }
        return metrics.get(key, metrics.get(legacy.get(key)))

    @staticmethod
    def _format_metric(value, precision: int = 4) -> str:
        if value is None:
            return "N/A"
        return f"{value:.{precision}f}"

    @staticmethod
    def _format_percentage(value) -> str:
        if value is None:
            return "N/A"
        return f"{value:.1f}%"
