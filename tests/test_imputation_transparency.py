"""Contratos del registro y la presentación de imputaciones."""

import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.cleaner import DataCleaner


class ImputationTransparencyTest(unittest.TestCase):
    def test_records_statistical_imputation_with_rule_value_and_count(self):
        cleaner = DataCleaner(pd.DataFrame({"q2": [20.0, 40.0, np.nan]}))

        with redirect_stdout(StringIO()):
            cleaned = cleaner.handle_missing_values()
        record = cleaner.metadata["imputaciones_por_variable"]["q2"]

        self.assertEqual(cleaned.loc[2, "q2"], 30.0)
        self.assertEqual(record["strategy"], "median")
        self.assertEqual(record["value"], 30.0)
        self.assertEqual(record["count"], 1)
        self.assertEqual(record["percentage"], 33.33)

    def test_records_dependent_variable_consistency_separately(self):
        cleaner = DataCleaner(pd.DataFrame({
            "boletidnew": [2, 1, 1],
            "boletidnewb": [np.nan, 0, 1],
        }))

        with redirect_stdout(StringIO()):
            cleaned = cleaner.handle_missing_values()
        adjustments = cleaner.metadata["tratamientos_consistencia"]

        self.assertEqual(cleaned["boletidnewb"].tolist(), [0.0, 8.0, 1.0])
        self.assertEqual(len(adjustments), 2)
        self.assertEqual(adjustments[0]["value"], 0)
        self.assertEqual(adjustments[0]["count"], 1)
        self.assertEqual(adjustments[1]["value"], 8)
        self.assertEqual(adjustments[1]["count"], 1)

    def test_preprocessing_report_separates_both_kinds_of_treatment(self):
        source = (PROJECT_ROOT / "src" / "pages" / "0_preprocesamiento.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("Detalle de imputación por variable", source)
        self.assertIn("Ajustes de consistencia entre variables", source)
        self.assertIn("Son reglas deterministas entre variables relacionadas", source)
        self.assertIn("imputaciones estadísticas.", source)
        self.assertIn("Valor utilizado", source)
        self.assertIn("Valores imputados", source)


if __name__ == "__main__":
    unittest.main()
