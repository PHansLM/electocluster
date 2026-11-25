"""Regresiones de unidades originales y categorías realmente codificadas."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from src.preprocessing.profile_metadata import build_profile_metadata, load_profile_metadata
from src.preprocessing.transformer import transform_dataset
from src.visualization.profile_interpreter import interpret_feature_value
from src.visualization.semantic_profiles import cluster_semantic_profiles
from src.visualization.profile_differences import profile_difference_rows
from src.evaluation.provenance import sha256_file


class ProfileTransformMetadataTest(unittest.TestCase):
    def test_mother_tongue_uses_observed_encoder_classes_not_all_six_categories(self):
        original = pd.DataFrame({"leng1": [1001., 1002., 1003., 1004., 1006.]})
        expected = original.copy(deep=True)
        processed, _, _ = transform_dataset(original)
        metadata = build_profile_metadata(original, processed)
        profiles = cluster_semantic_profiles(processed, np.arange(5), feature_metadata=metadata)
        labels = profiles.summaries.representative_label.tolist()
        self.assertEqual(labels, ["Castellano/español", "Quechua", "Aymara", "Otro nativo", "Guaraní"])
        self.assertEqual(interpret_feature_value("leng1", .25, metadata).text, "Quechua")
        self.assertTrue((profiles.summaries.category_mapping_status == "configured_exact").all())
        assert_frame_equal(original, expected)

    def test_numeric_original_units_for_minmax_standard_and_no_normalization(self):
        original = pd.DataFrame({"q2": [18., 53., 88.]})
        for normalize, method in [(True, "minmax"), (True, "standard"), (False, "minmax")]:
            with self.subTest(normalize=normalize, method=method):
                processed, _, _ = transform_dataset(original, normalize=normalize, normalization_method=method)
                metadata = build_profile_metadata(original, processed)
                value = processed.q2.mean()
                self.assertEqual(interpret_feature_value("q2", value, metadata).text,
                                 "Aprox. 53 en escala original")
                profiles = cluster_semantic_profiles(processed, [0, 0, 0], feature_metadata=metadata)
                self.assertIn("53", profile_difference_rows(profiles).iloc[0].practical_value)

    def test_unencoded_category_uses_original_code_and_binary_positive_category(self):
        original = pd.DataFrame({"q10inc": [1003., 1003., 1004.], "boletidnew": [1., 1., 2.]})
        processed, _, _ = transform_dataset(original, encode=False)
        metadata = build_profile_metadata(original, processed)
        self.assertEqual(interpret_feature_value("q10inc", 1003, metadata).text, "1401-2200 Bs")
        profiles = cluster_semantic_profiles(processed, [0, 0, 0], feature_metadata=metadata)
        binary = profiles.summaries.query("feature_code == 'boletidnew'").iloc[0]
        self.assertEqual(binary.representative_label, "Sí")
        self.assertAlmostEqual(binary.representative_percentage, 200 / 3)

    def test_unverified_numeric_value_does_not_invent_original_units(self):
        reading = interpret_feature_value("q2", 0.487)
        self.assertIn("Valor procesado", reading.text)
        self.assertNotIn("escala original", reading.text)

    def test_report_context_is_rejected_when_dataset_hash_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "data.csv"
            report = Path(directory) / "report.json"
            csv.write_text("q2\n0\n1\n", encoding="utf-8")
            metadata = {"q2": {"numeric_inverse": {"scale": 70, "offset": 18}}}
            report.write_text(json.dumps({"processed_dataset_sha256": sha256_file(csv),
                "sections": {"transformation": {"feature_metadata": metadata}}}), encoding="utf-8")
            self.assertEqual(load_profile_metadata(csv, report_path=report), metadata)
            csv.write_text("q2\n0\n0.5\n", encoding="utf-8")
            self.assertEqual(load_profile_metadata(csv, report_path=report), {})

    def test_results_page_handles_missing_dataset_values_without_traceback(self):
        page = Path(__file__).resolve().parents[1] / "src/pages/4_resultados.py"
        with patch("src.evaluation.execution.load_processed_dataset",
                   side_effect=ValueError("El dataset procesado contiene valores faltantes.")):
            app = AppTest.from_file(str(page)).run(timeout=30)
        self.assertFalse(app.exception, str(app.exception))
        self.assertTrue(any("Revisa Preprocesamiento" in error.value for error in app.error))


if __name__ == "__main__":
    unittest.main()
