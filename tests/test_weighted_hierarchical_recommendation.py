"""Contratos de la recomendación exploratoria del jerárquico ponderado."""

import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.parameter_optimizer import optimize_whierarchical_params


class _WeightManager:
    def get_weights_array(self, feature_order=None):
        return np.ones(len(feature_order), dtype=float)


class _Clusterer:
    def __init__(self, n_clusters=2, linkage="complete", **_kwargs):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, data):
        self.labels_ = np.array([0, 0, 1, 1])
        return self


class WeightedHierarchicalRecommendationTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "q2": [0.1, 0.2, 0.8, 0.9],
            "wealth_index": [0.1, 0.2, 0.8, 0.9],
        })

    def test_candidates_are_ranked_with_weighted_geometry_metrics(self):
        weighted_metrics = {
            "silhouette": 0.42,
            "davies_bouldin": 0.71,
            "calinski_harabasz": 11.0,
        }
        with patch(
            "src.evaluation.parameter_optimizer.WeightedHierarchicalClustering",
            _Clusterer,
        ), patch(
            "src.evaluation.parameter_optimizer.calculate_weighted_geometry_metrics",
            return_value=weighted_metrics,
        ) as metric_call:
            result = optimize_whierarchical_params(
                self.data,
                weight_manager=_WeightManager(),
                k_values=[2, 3],
                linkages=["complete", "average"],
                sample_size=4,
            )

        self.assertEqual(metric_call.call_count, 4)
        self.assertEqual(result.criterion.split(";")[0], "Mayor Silhouette ponderado")
        self.assertTrue(all(row["estado"] == "Válido" for row in result.rows))
        self.assertEqual(
            sorted(row["orden de recomendación"] for row in result.rows),
            [1, 2, 3, 4],
        )
        self.assertIn("distancia ponderada", " ".join(result.notes))

    def test_degenerate_candidates_are_not_recommendations(self):
        class DegenerateClusterer(_Clusterer):
            def fit(self, data):
                self.labels_ = np.zeros(len(data), dtype=int)
                return self

        result = None
        with patch(
            "src.evaluation.parameter_optimizer.WeightedHierarchicalClustering",
            DegenerateClusterer,
        ):
            result = optimize_whierarchical_params(
                self.data,
                weight_manager=_WeightManager(),
                k_values=[2],
                linkages=["complete"],
                sample_size=4,
            )

        self.assertEqual(result.rows[0]["estado"], "Sin métricas válidas")
        self.assertIsNone(result.rows[0]["orden de recomendación"])
        self.assertEqual(result.best_params, {"n_clusters": 2, "linkage": "complete"})

    def test_configuration_exposes_context_and_invalidates_changed_searches(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "pages" / "2_configuracion.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("La recomendación ya no corresponde", source)
        self.assertIn("Condiciones de esta exploración", source)
        self.assertIn("_add_canonical_metric_deltas", source)
        self.assertIn("Referencia canónica", source)


if __name__ == "__main__":
    unittest.main()
