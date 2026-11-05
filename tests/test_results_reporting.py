"""Contratos de legibilidad para la página de resultados."""

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PAGE = PROJECT_ROOT / "src" / "pages" / "4_resultados.py"


class ResultsReportingTest(unittest.TestCase):
    def test_reports_use_tables_and_keep_json_as_secondary_traceability(self):
        source = RESULTS_PAGE.read_text(encoding="utf-8")

        self.assertNotIn("st.json(", source)
        self.assertIn("Trazabilidad técnica de la ejecución", source)
        self.assertIn("Trazabilidad técnica de perfiles", source)
        self.assertIn("Lectura del reporte", source)
        self.assertIn("Descargar comparación métrica CSV", source)
        self.assertIn('file_name=f"{report[\'report_name\']}.csv"', source)
        self.assertIn("Comparación métrica", source)
        self.assertIn("Descargar registro técnico JSON", source)
        self.assertIn("Descargar perfiles semánticos JSON", source)
        self.assertIn("Detalle por grupo", source)
        self.assertIn("Descargar proyección PCA 2D (PNG)", source)

    def test_figure_export_is_a_browser_download(self):
        source = RESULTS_PAGE.read_text(encoding="utf-8")

        self.assertIn("Descargar figuras exportables (ZIP)", source)
        self.assertIn("build_run_figures_zip", source)
        self.assertNotIn("save_run_figures(", source)
        self.assertNotIn("results/figures", source)

    def test_profile_export_prefers_descriptions_and_offers_raw_values(self):
        source = RESULTS_PAGE.read_text(encoding="utf-8")

        self.assertIn("Descargar perfiles descriptivos CSV", source)
        self.assertIn("Descargar valores CSV", source)
        self.assertIn("_descriptive_profiles_export", source)
        self.assertNotIn("profile_options.popover", source)
        self.assertIn("default=profile_cluster_options", source)
        self.assertIn("default=profile_feature_options", source)
        self.assertIn("Los CSV exportados mostrarán exactamente", source)


if __name__ == "__main__":
    unittest.main()

