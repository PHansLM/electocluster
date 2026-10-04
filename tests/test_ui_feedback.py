"""Comprueba diálogos, decisiones de navegación y fallos de operaciones reales."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.execution import default_params
from src.evaluation.parameter_optimizer import ParameterSearchResult


class UIFeedbackTest(unittest.TestCase):
    def setUp(self):
        preview = pd.DataFrame({"x": range(100), "y": range(100, 200)})
        loader = patch("src.evaluation.execution.load_processed_dataset", return_value=preview)
        loader.start()
        self.addCleanup(loader.stop)
        self.app = AppTest.from_file(str(PROJECT_ROOT / "src" / "app.py"))
        self.app.run(timeout=20)
        self.app.switch_page("pages/2_configuracion.py").run(timeout=20)

    def button(self, label):
        return next(button for button in self.app.button if button.label == label)

    def click(self, label):
        self.button(label).click().run(timeout=20)
        self.assertFalse(self.app.exception, str(self.app.exception))
        # AppTest no actualiza su ruta cliente cuando la app llama st.switch_page.
        # El navegador sí la actualiza; sincronizamos antes de la siguiente acción.
        pages = {
            "Configuración": "pages/2_configuracion.py",
            "Ejecución": "pages/3_ejecucion.py",
        }
        section = self.app.session_state["ui_current_section"]
        if section in pages:
            self.app.switch_page(pages[section])

    def feedback(self, kind):
        self.assertEqual(self.app.session_state["operation_feedback"]["kind"], kind)
        self.assertTrue(self.app.get("dialog"))

    def edit(self, value=7):
        self.app.number_input[0].set_value(value).run(timeout=20)

    def open_assistant(self):
        self.click("Abrir exploración rápida")

    @staticmethod
    def search_result(algorithm="WKMedoids", valid=True):
        params = default_params(algorithm)
        if algorithm != "W-DBSCAN":
            params["n_clusters"] = 3
        else:
            params["eps"] = 0.806
        return ParameterSearchResult(
            algorithm=algorithm, best_params=params,
            rows=[{"silhouette": 0.2 if valid else None}],
            sample_size=100, criterion="Silhouette", notes=[],
        )

    @staticmethod
    def execution_result(algorithm="WKMedoids"):
        return SimpleNamespace(
            algorithm=algorithm, run_id=f"test_{algorithm}",
            metrics={"silhouette": 0.2, "davies_bouldin": 1.5, "calinski_harabasz": 12.0},
            labels=np.array([0, 0, 1, 1]), metadata={"n_evaluated": 4},
        )

    def execution_page(self):
        self.click("Ejecución")

    def test_sidebar_navigation_requires_choice_and_preserves_draft(self):
        self.edit()
        markers = [caption.value for caption in self.app.caption if "Sin guardar" in caption.value]
        self.assertEqual(len(markers), 1)
        self.assertIn("13 → En edición: 7", markers[0])
        self.click("Ejecución")
        self.assertEqual(self.app.title[0].value, "Configuración de algoritmos")
        self.assertTrue(self.app.get("dialog"))
        self.click("Seguir editando")
        self.assertFalse(self.app.get("dialog"))
        self.assertEqual(self.app.number_input[0].value, 7)
        self.click("Ejecución")
        self.click("Continuar sin guardar")
        self.assertEqual(self.app.title[0].value, "Ejecución de clustering")
        self.assertEqual(self.app.session_state["params_by_algorithm"]["WKMedoids"], default_params("WKMedoids"))
        self.click("Configuración")
        self.assertEqual(self.app.number_input[0].value, 7)
        self.click("Guardar configuración activa")
        self.feedback("success")
        self.click("Entendido")
        self.assertFalse(any("Sin guardar" in caption.value for caption in self.app.caption))
        self.click("Ejecución")
        self.assertFalse(self.app.get("dialog"))

    def test_algorithm_navigation_cannot_switch_until_confirmation(self):
        self.edit()
        self.app.radio[0].set_value("W-DBSCAN").run(timeout=20)
        self.assertEqual(self.app.session_state["algoritmo"], "WKMedoids")
        self.assertEqual(self.app.radio[0].value, "WKMedoids")
        self.click("Seguir editando")
        self.assertEqual(self.app.number_input[0].value, 7)
        self.app.radio[0].set_value("W-DBSCAN").run(timeout=20)
        self.click("Continuar sin guardar")
        self.assertEqual(self.app.session_state["algoritmo"], "W-DBSCAN")
        self.assertEqual(self.app.session_state["params_drafts_by_algorithm"]["WKMedoids"]["n_clusters"], 7)
        self.assertEqual(self.app.session_state["params_by_algorithm"]["WKMedoids"]["n_clusters"], 13)

    def test_assistant_success_for_each_algorithm_and_explicit_apply_save(self):
        targets = {
            "WKMedoids": "optimize_wkmedoids_params",
            "W-Hierarchical Clustering": "optimize_whierarchical_params",
            "W-DBSCAN": "optimize_wdbscan_params",
        }
        for algorithm, optimizer in targets.items():
            with self.subTest(algorithm=algorithm):
                self.app.radio[0].set_value(algorithm).run(timeout=20)
                if any(button.label == "Abrir exploración rápida" for button in self.app.button):
                    self.open_assistant()
                recommendation = self.search_result(algorithm)
                with patch(f"src.evaluation.parameter_optimizer.{optimizer}", return_value=recommendation):
                    self.click("Calcular recomendación")
                self.feedback("success")
                self.assertEqual(self.app.session_state["params_by_algorithm"][algorithm], default_params(algorithm))
                self.click("Entendido")
                self.assertFalse(self.app.get("dialog"))
                self.click("Aplicar esta recomendación")
                self.feedback("success")
                self.assertEqual(self.app.session_state["params_by_algorithm"][algorithm], default_params(algorithm))
                self.assertEqual(self.app.session_state["params_drafts_by_algorithm"][algorithm], recommendation.best_params)
                self.click("Entendido")
                self.click("Guardar configuración activa")
                self.click("Entendido")

    def test_assistant_failure_keeps_saved_and_edited_parameters(self):
        self.edit()
        self.open_assistant()
        with patch("src.evaluation.parameter_optimizer.optimize_wkmedoids_params", side_effect=RuntimeError("Fallo de prueba")):
            self.click("Calcular recomendación")
        self.feedback("error")
        self.assertIn("Fallo de prueba", self.app.session_state["operation_feedback"]["details"])
        self.assertEqual(self.app.session_state["params_by_algorithm"]["WKMedoids"]["n_clusters"], 13)
        self.assertEqual(self.app.session_state["params_drafts_by_algorithm"]["WKMedoids"]["n_clusters"], 7)
        self.click("Entendido")
        self.assertFalse(self.app.get("dialog"))

    def test_assistant_without_valid_metrics_disables_application(self):
        self.open_assistant()
        with patch("src.evaluation.parameter_optimizer.optimize_wkmedoids_params", return_value=self.search_result(valid=False)):
            self.click("Calcular recomendación")
        self.feedback("error")
        self.click("Entendido")
        self.assertTrue(self.button("Aplicar esta recomendación").disabled)

    def test_assistant_rejects_parameters_outside_manual_controls(self):
        self.open_assistant()
        recommendation = self.search_result()
        recommendation.best_params["n_clusters"] = 24
        with patch("src.evaluation.parameter_optimizer.optimize_wkmedoids_params", return_value=recommendation):
            self.click("Calcular recomendación")
        self.click("Entendido")
        self.click("Aplicar esta recomendación")
        self.feedback("error")
        self.assertEqual(self.app.number_input[0].value, 13)
        self.assertEqual(self.app.session_state["params_by_algorithm"]["WKMedoids"], default_params("WKMedoids"))

    def test_missing_assistant_dataset_shows_error_once_and_disables_search(self):
        with patch("src.evaluation.execution.load_processed_dataset", side_effect=FileNotFoundError("Dataset ausente")):
            self.open_assistant()
            self.feedback("error")
            self.assertTrue(self.button("Calcular recomendación").disabled)
            self.click("Entendido")
            self.assertFalse(self.app.get("dialog"))

    def test_missing_execution_dataset_shows_error_once(self):
        st.cache_data.clear()
        with patch("src.evaluation.execution.load_processed_dataset", side_effect=FileNotFoundError("Dataset ausente")):
            self.execution_page()
            self.feedback("error")
            self.click("Entendido")
            self.assertFalse(self.app.get("dialog"))

    def test_execution_success_survives_acknowledgement_without_running_twice(self):
        self.execution_page()
        with patch(
            "src.evaluation.execution.run_clustering_experiment",
            side_effect=lambda **kwargs: self.execution_result(kwargs["algorithm"]),
        ) as runner:
            self.click("Ejecutar seleccion")
            self.feedback("success")
            self.assertEqual(runner.call_count, 3)
            self.click("Entendido")
            self.assertEqual(runner.call_count, 3)
        self.assertFalse(self.app.get("dialog"))
        self.assertTrue(any("Última ejecución" in element.value for element in self.app.markdown))
        self.assertEqual(len(self.app.session_state["last_execution_summary"]["rows"]), 3)

    def test_execution_error_preserves_completed_runs(self):
        self.execution_page()
        with patch("src.evaluation.execution.run_clustering_experiment", side_effect=[self.execution_result(), RuntimeError("Fallo de algoritmo")]):
            self.click("Ejecutar seleccion")
        self.feedback("error")
        feedback = self.app.session_state["operation_feedback"]
        self.assertIn("Falló W-Hierarchical Clustering", feedback["message"])
        self.assertIn("1 de 3", feedback["message"])
        self.assertEqual(self.app.session_state["last_suite_run_ids"], ["test_WKMedoids"])
        self.click("Entendido")
        self.assertEqual(len(self.app.session_state["last_execution_summary"]["rows"]), 1)
        self.assertIn("Fallo de algoritmo", self.app.session_state["last_execution_summary"]["details"])

    def test_execution_failure_before_first_result(self):
        self.execution_page()
        self.app.session_state["last_run_id"] = "previous_saved_run"
        with patch("src.evaluation.execution.run_clustering_experiment", side_effect=OSError("No se pudo guardar")):
            self.click("Ejecutar seleccion")
        self.feedback("error")
        self.assertIn("0 de 3", self.app.session_state["operation_feedback"]["message"])
        self.assertEqual(self.app.session_state["last_suite_run_ids"], [])
        self.assertEqual(self.app.session_state["last_run_id"], "previous_saved_run")

    def test_canonical_suite_partial_failure_and_historical_warning(self):
        self.execution_page()
        self.click("Usar canonicas en todos")

        def fail_suite(*, persist, on_result):
            on_result(self.execution_result())
            raise RuntimeError("Fallo al consolidar")

        with patch("src.evaluation.canonical_suite.run_canonical_suite", side_effect=fail_suite):
            self.click("Ejecutar seleccion")
        self.feedback("error")
        self.assertEqual(self.app.session_state["last_suite_run_ids"], ["test_WKMedoids"])
        self.click("Entendido")
        results = [self.execution_result(name) for name in (
            "WKMedoids", "W-Hierarchical Clustering", "W-DBSCAN"
        )]
        suite = SimpleNamespace(suite_id="test_suite", results=results, artifact={
            "runs": [{"run_id": result.run_id, "algorithm": result.algorithm,
                      "metrics": result.metrics, "n_noise": 0, "elapsed_seconds": 1,
                      "regression": {"passed": False}} for result in results],
            "regression": {"all_passed": False},
        })
        with patch("src.evaluation.canonical_suite.run_canonical_suite", return_value=suite):
            self.click("Ejecutar seleccion")
        self.feedback("warning")
        self.assertIn("terminó y se guardó", self.app.session_state["operation_feedback"]["message"])
        self.click("Entendido")
        suite.artifact["regression"]["all_passed"] = True
        with patch("src.evaluation.canonical_suite.run_canonical_suite", return_value=suite):
            self.click("Ejecutar seleccion")
        self.feedback("success")


if __name__ == "__main__":
    unittest.main()
