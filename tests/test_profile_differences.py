"""Regresión de unidades, categorías y visualización de diferencias."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.visualization.semantic_profiles import cluster_semantic_profiles
from src.visualization.profile_differences import profile_difference_rows, difference_heatmap


class ProfileDifferencesTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
