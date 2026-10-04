"""Regresion del guardado explicito y la navegacion de parametros en la interfaz."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.execution import default_params


class ConfigurationSaveTest(unittest.TestCase):
    def setUp(self):
        # El plan solo necesita una vista previa; esta prueba no ejecuta clustering.
        preview = pd.DataFrame({"x": range(100), "y": range(100, 200)})
        self.loader = patch(
            "src.evaluation.execution.load_processed_dataset", return_value=preview
        )
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.app = AppTest.from_file(str(PROJECT_ROOT / "src" / "app.py"))
        self.app.run(timeout=20)
        self.navigate("2_configuracion.py")

    def navigate(self, filename):
        self.app.switch_page(f"pages/{filename}").run(timeout=20)
        if any(button.key == "config_navigation_continue" for button in self.app.button):
            self.button("Continuar sin guardar").click().run(timeout=20)
        self.assertFalse(self.app.exception, str(self.app.exception))

    def select_algorithm(self, algorithm):
        self.app.radio[0].set_value(algorithm).run(timeout=20)
        if any(button.key == "config_navigation_continue" for button in self.app.button):
            self.button("Continuar sin guardar").click().run(timeout=20)
        self.assertFalse(self.app.exception, str(self.app.exception))

    def button(self, label):
        return next(button for button in self.app.button if button.label == label)

    def number(self, label):
        return next(widget for widget in self.app.number_input if widget.label == label)

    def assert_plan(self, algorithm, expected):
        plan = next(
            element.value for element in self.app.dataframe
            if "fuente" in element.value.columns
        )
        row = plan[plan["algoritmo"] == algorithm].iloc[0]
        self.assertEqual(row["fuente"], "Activa")
        labels = {
            "n_clusters": "Clusters", "random_state": "Semilla", "linkage": "Linkage",
            "eps": "Eps", "min_samples": "Min. muestras", "pca_components": "PCA",
        }
        for key, value in expected.items():
            if isinstance(value, float):
                self.assertAlmostEqual(float(row[labels[key]]), value)
            else:
                self.assertEqual(str(row[labels[key]]), str(value))

    def test_edits_survive_navigation_and_only_save_changes_execution_plan(self):
        cases = {
            "WKMedoids": {
                "Número de clusters (k)": 7, "Semilla aleatoria (random_state)": 17,
            },
            "W-Hierarchical Clustering": {"Número de clusters": 4},
            "W-DBSCAN": {
                "Radio de vecindad (eps)": 0.806,
                "Mínimo de puntos núcleo": 20, "Componentes PCA previos": 5,
            },
        }
        drafts = {}
        for algorithm, edits in cases.items():
            with self.subTest(editing=algorithm):
                self.select_algorithm(algorithm)
                for label, value in edits.items():
                    self.number(label).set_value(value).run(timeout=20)
                    self.assertFalse(self.app.exception, str(self.app.exception))
                    self.assertEqual(
                        self.app.session_state["params_by_algorithm"][algorithm],
                        default_params(algorithm),
                    )
                if algorithm == "W-Hierarchical Clustering":
                    self.app.selectbox[0].set_value("average").run(timeout=20)
                drafts[algorithm] = dict(
                    self.app.session_state["params_drafts_by_algorithm"][algorithm]
                )
                self.assertTrue(any("sin guardar" in item.value for item in self.app.warning))

        committed = {algorithm: default_params(algorithm) for algorithm in cases}
        for algorithm in cases:
            with self.subTest(saving=algorithm):
                self.navigate("3_ejecucion.py")
                for other, params in committed.items():
                    self.assert_plan(other, params)
                self.navigate("2_configuracion.py")
                self.select_algorithm(algorithm)
                self.assertEqual(
                    self.app.session_state["params_drafts_by_algorithm"][algorithm],
                    drafts[algorithm],
                )
                self.button("Guardar configuración activa").click().run(timeout=20)
                self.button("Entendido").click().run(timeout=20)
                self.assertFalse(self.app.exception, str(self.app.exception))
                self.assertFalse(any("sin guardar" in item.value for item in self.app.warning))
                committed[algorithm] = drafts[algorithm]
                self.navigate("3_ejecucion.py")
                for other, params in committed.items():
                    self.assert_plan(other, params)
                self.navigate("2_configuracion.py")

        self.select_algorithm("WKMedoids")
        self.number("Número de clusters (k)").set_value(9).run(timeout=20)
        self.navigate("3_ejecucion.py")
        self.assert_plan("WKMedoids", committed["WKMedoids"])


if __name__ == "__main__":
    unittest.main()
