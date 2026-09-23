"""Regresión de unidades, categorías y visualización de diferencias."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.visualization.semantic_profiles import cluster_semantic_profiles
from src.visualization.plotly_config import HEATMAP_PLOT_CONFIG_ES, PLOT_CONFIG_ES
from src.visualization.profile_differences import (
    cluster_difference_summaries,
    difference_heatmap,
    pair_difference_details,
    pair_difference_summaries,
    profile_difference_rows,
)


class ProfileDifferencesTest(unittest.TestCase):
    def test_plotly_configuration_keeps_interactive_charts_available(self):
        self.assertEqual(PLOT_CONFIG_ES["locale"], "es")
        self.assertEqual(
            PLOT_CONFIG_ES["locales"]["es"]["dictionary"]["Zoom in"], "Acercar"
        )
        self.assertNotIn("modeBarButtons", PLOT_CONFIG_ES)
        self.assertEqual(HEATMAP_PLOT_CONFIG_ES["modeBarButtons"], [["toImage"]])

    def test_percentages_use_grouped_population_and_percentage_points(self):
        data = pd.DataFrame({"ur": [1, 1, 0, 0, 0]})
        rows = profile_difference_rows(cluster_semantic_profiles(data, [0, 0, 1, 1, -1]))
        group = rows[rows.cluster_id == 0].iloc[0]
        self.assertEqual(group.reference, "50,0 %")
        self.assertAlmostEqual(group.deviation, .5)
        self.assertEqual(group.reading, "50,0 puntos porcentuales por encima")

    def test_nominal_row_compares_one_category_not_mean_of_codes(self):
        metadata = {"category": {"tipo": "categorico_nominal", "nombre": "Categoría",
                                 "categorias": {1: "A", 2: "B", 3: "C"}}}
        data = pd.DataFrame({"category": [0., 1., .5, .5]})
        profiles = cluster_semantic_profiles(data, [0, 0, 1, 1], feature_metadata=metadata)
        rows = profile_difference_rows(profiles)
        self.assertEqual(rows.detail.nunique(), 1)
        self.assertEqual(rows.iloc[0].detail, "Categoría: B")
        self.assertEqual(rows.iloc[0].value, "0,0 %")
        self.assertEqual(rows.iloc[1].value, "100,0 %")

    def test_all_rows_and_group_labels_are_explicit_and_pan_is_disabled(self):
        data = pd.DataFrame({f"x{i}": [0., 0., 1., 1.] for i in range(28)})
        rows = profile_difference_rows(cluster_semantic_profiles(data, [0, 0, 1, 1]))
        codes = data.columns.tolist()
        fig = difference_heatmap(rows, codes)
        self.assertEqual(len(fig.layout.yaxis.tickvals), 28)
        self.assertEqual(list(fig.layout.xaxis.tickvals), ["Grupo 0", "Grupo 1"])
        self.assertGreaterEqual(fig.layout.height, 28 * 30)
        self.assertTrue(fig.layout.xaxis.fixedrange)
        self.assertTrue(fig.layout.yaxis.fixedrange)
        self.assertFalse(fig.layout.dragmode)
        self.assertEqual(fig.data[0].zmin, -fig.data[0].zmax)

    def test_tooltip_summaries_preserve_semantic_units(self):
        data = pd.DataFrame({"ur": [1, 1, 0, 0]})
        rows = profile_difference_rows(cluster_semantic_profiles(data, [0, 0, 1, 1]))

        clusters = cluster_difference_summaries(rows)
        pairs = pair_difference_summaries(rows)

        self.assertIn("puntos porcentuales", clusters[0])
        self.assertIn("Grupo 0", pairs[(0, 1)])
        self.assertIn("Grupo 1", pairs[(0, 1)])

    def test_practical_readings_respect_feature_type(self):
        data = pd.DataFrame({
            "q10e": [0.0, 0.0, 1.0, 1.0],
            "ur": [1.0, 1.0, 0.0, 0.0],
        })
        rows = profile_difference_rows(cluster_semantic_profiles(data, [0, 0, 1, 1]))

        ordinal = rows[(rows.feature == "q10e") & (rows.cluster_id == 0)].iloc[0]
        binary = rows[(rows.feature == "ur") & (rows.cluster_id == 0)].iloc[0]

        self.assertEqual(ordinal.practical_value, "Mediana: Aumentó")
        self.assertIn("Urbano", binary.practical_value)
        self.assertNotIn("más cercano", ordinal.practical_value.lower())

    def test_pair_detail_has_relative_intensity_scale(self):
        data = pd.DataFrame({
            "q10e": [0.0, 0.0, 1.0, 1.0],
            "ur": [1.0, 1.0, 0.0, 0.0],
        })
        rows = profile_difference_rows(cluster_semantic_profiles(data, [0, 0, 1, 1]))
        detail = pair_difference_details(rows, 0, 1)

        self.assertFalse(detail.empty)
        self.assertAlmostEqual(detail.intensidad.max(), 100.0)
        self.assertIn("Grupo 0", detail.columns)
        self.assertIn("Grupo 1", detail.columns)


if __name__ == "__main__":
    unittest.main()
