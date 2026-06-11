"""
ReportGenerator: genera reportes comparativos entre ejecuciones de clustering.
Produce JSON estructurado y texto plano en results/comparisons/ y results/reports/.
"""

import json
from datetime import datetime
from pathlib import Path

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

        report = {
            "report_name": report_name or f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "runs_compared": run_ids,
            "summary": [],
            "best_by_metric": {},
        }

        for run in runs:
            metrics = run.get("metrics", {})
            report["summary"].append({
                "run_id": run["run_id"],
                "algorithm": run["algorithm"],
                "params": run["params"],
                "n_clusters": run["n_clusters"],
                "n_noise": run["n_noise"],
                "silhouette": self._metric_value(metrics, "silhouette"),
                "davies_bouldin": self._metric_value(metrics, "davies_bouldin"),
                "calinski_harabasz": self._metric_value(metrics, "calinski_harabasz"),
            })

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

        self._save_text_report(report)
        return report

    def generate_run_summary(self, run_id: str) -> str:
        """Genera resumen legible de una sola ejecucion."""
        run = self.rm.load_run(run_id)
        metrics = run["metrics"]
        lines = [
            f"EJECUCION: {run_id}",
            f"Algoritmo: {run['algorithm']}",
            f"Timestamp: {run['timestamp']}",
            f"Parametros: {run['params']}",
            f"Clusters identificados: {run['n_clusters']}",
            f"Puntos de ruido: {run['n_noise']}",
            f"Registros totales: {run['n_samples']}",
            "",
            "METRICAS:",
            f"  Silhouette Score:       {self._metric_value(metrics, 'silhouette')}",
            f"  Davies-Bouldin Index:   {self._metric_value(metrics, 'davies_bouldin')}",
            f"  Calinski-Harabasz:      {self._metric_value(metrics, 'calinski_harabasz')}",
        ]
        return "\n".join(lines)

    def _save_text_report(self, report: dict):
        """Guarda un reporte comparativo legible en texto plano."""
        lines = [
            f"REPORTE COMPARATIVO - {report['report_name']}",
            f"Generado: {report['generated_at']}",
            "=" * 70,
            "",
            f"{'Algoritmo':<24} {'Clusters':>8} {'Ruido':>8} "
            f"{'Silhouette':>12} {'D-Bouldin':>12} {'Calinski-H':>12}",
            "-" * 70,
        ]

        for summary in report["summary"]:
            sil = self._format_metric(summary["silhouette"], precision=4)
            db = self._format_metric(summary["davies_bouldin"], precision=4)
            ch = self._format_metric(summary["calinski_harabasz"], precision=2)
            lines.append(
                f"{summary['algorithm']:<24} {summary['n_clusters']:>8} "
                f"{summary['n_noise']:>8} {sil:>12} {db:>12} {ch:>12}"
            )

        lines += [
            "-" * 70,
            "",
            "MEJOR POR METRICA:",
            f"  Silhouette (mayor es mejor):        {report['best_by_metric'].get('silhouette', 'N/A')}",
            f"  Davies-Bouldin (menor es mejor):    {report['best_by_metric'].get('davies_bouldin', 'N/A')}",
            f"  Calinski-Harabasz (mayor es mejor): {report['best_by_metric'].get('calinski_harabasz', 'N/A')}",
        ]

        report_path = self.results_path / "reports" / f"{report['report_name']}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

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
