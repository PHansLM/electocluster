"""Comprueba espera visible y confirmación final del preprocesamiento."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class PreprocessingFeedbackTest(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(PROJECT_ROOT / "src" / "app.py"))
        self.app.run(timeout=20)
        self.app.switch_page("pages/0_preprocesamiento.py").run(timeout=20)
        self.app.session_state["_preprocessing_step_min_seconds"] = 0
        self.assertFalse(self.app.exception, str(self.app.exception))

    def button(self, label):
        return next(button for button in self.app.button if button.label == label)

    @staticmethod
    def successful_pipeline(*_args, on_progress=None, **_kwargs):
        for completed, message in enumerate(
            [
                "Dataset cargado",
                "Limpieza completada",
                "Variables construidas",
                "Transformaciones completadas",
                "Resultados guardados",
            ],
            start=1,
        ):
            on_progress(completed, 5, message)
        return pd.DataFrame({"a": [1, 2], "b": [3, 4]}), {"sections": {}}

    def test_success_opens_confirmation_dialog(self):
        with patch(
            "src.preprocessing.preprocessor_configured.run_pipeline_configured",
            side_effect=self.successful_pipeline,
        ):
            self.button("Ejecutar pipeline").click().run(timeout=20)
        self.assertFalse(self.app.exception, str(self.app.exception))
        self.assertEqual(self.app.session_state["operation_feedback"]["kind"], "success")
        self.assertIn("guardó el dataset procesado", self.app.session_state["operation_feedback"]["message"])
        self.assertTrue(self.app.get("dialog"))
        progress_text = "\n".join(element.value for element in self.app.markdown)
        self.assertIn("Cargar y verificar el dataset", progress_text)
        self.assertIn("Guardar el dataset procesado y su reporte", progress_text)

    def test_pipeline_copy_and_controls_are_plain_language(self):
        page_text = "\n".join(
            element.value
            for collection in (self.app.markdown, self.app.caption)
            for element in collection
        )
        self.assertNotIn("iteración 1", page_text)
        self.assertNotIn("Primer componente PCA", page_text)
        self.assertIn("comportamiento documentado", page_text)
        self.assertIn("reporta más casos", page_text)

    def test_failure_does_not_claim_dataset_is_ready(self):
        with patch(
            "src.preprocessing.preprocessor_configured.run_pipeline_configured",
            side_effect=RuntimeError("Fallo controlado"),
        ):
            self.button("Ejecutar pipeline").click().run(timeout=20)
        self.assertFalse(self.app.exception, str(self.app.exception))
        feedback = self.app.session_state["operation_feedback"]
        self.assertEqual(feedback["kind"], "error")
        self.assertIn("no fue confirmado como listo", feedback["message"])
        self.assertIn("Fallo controlado", feedback["details"])
        self.assertTrue(self.app.get("dialog"))

    def test_completed_pipeline_with_missing_values_warns_it_is_not_ready(self):
        def incomplete_pipeline(*_args, on_progress=None, **_kwargs):
            on_progress(5, 5, "Resultados guardados")
            return pd.DataFrame({"a": [1, None]}), {"sections": {}}

        with patch(
            "src.preprocessing.preprocessor_configured.run_pipeline_configured",
            side_effect=incomplete_pipeline,
        ):
            self.button("Ejecutar pipeline").click().run(timeout=20)
        self.assertFalse(self.app.exception, str(self.app.exception))
        feedback = self.app.session_state["operation_feedback"]
        self.assertEqual(feedback["kind"], "warning")
        self.assertIn("1 valor faltante", feedback["message"])
        self.assertIn("no podrá utilizarlo", feedback["message"])


if __name__ == "__main__":
    unittest.main()

