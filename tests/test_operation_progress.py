"""Comprueba que las búsquedas informen avance real por candidato."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluation.parameter_optimizer import (
    optimize_wdbscan_params,
    optimize_whierarchical_params,
    optimize_wkmedoids_params,
)


class _FakeClusterer:
    def __init__(self, n_clusters=2, **_kwargs):
        self.n_clusters = n_clusters
        self.inertia_ = 1.0
        self.labels_ = None

    def fit(self, dataframe):
        self.labels_ = np.arange(len(dataframe)) % self.n_clusters
        return self


class _FakeDBSCAN:
    def __init__(self, **_kwargs):
        self.labels_ = None

    def fit(self, dataframe):
        self.labels_ = np.arange(len(dataframe)) % 2
        return self


class OperationProgressTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"a": np.arange(12), "b": np.arange(12)[::-1]})
        self.metrics = {
            "silhouette": 0.2,
            "davies_bouldin": 1.0,
            "calinski_harabasz": 2.0,
        }

    def assert_complete_progress(self, events, expected_total):
        self.assertEqual(len(events), expected_total)
        self.assertEqual(events[-1][0], expected_total)
        self.assertEqual(events[-1][1], expected_total)
        self.assertTrue(all(event[2] for event in events))

    def test_wkmedoids_reports_each_candidate(self):
        events = []
        with patch("src.evaluation.parameter_optimizer.WKMedoids", _FakeClusterer), patch(
            "src.evaluation.parameter_optimizer._safe_metrics", return_value=self.metrics
        ):
            optimize_wkmedoids_params(
                self.data,
                weight_manager=object(),
                k_values=[2, 3, 4],
                on_progress=lambda *event: events.append(event),
            )
        self.assert_complete_progress(events, 3)

    def test_hierarchical_reports_every_linkage_and_k(self):
        events = []
        with patch(
            "src.evaluation.parameter_optimizer.WeightedHierarchicalClustering",
            _FakeClusterer,
        ), patch("src.evaluation.parameter_optimizer._safe_metrics", return_value=self.metrics):
            optimize_whierarchical_params(
                self.data,
                weight_manager=object(),
                k_values=[2, 3],
                linkages=["complete", "average"],
                on_progress=lambda *event: events.append(event),
            )
        self.assert_complete_progress(events, 4)

    def test_wdbscan_reports_each_eps_candidate(self):
        events = []
        with patch(
            "src.evaluation.parameter_optimizer._apply_weights",
            side_effect=lambda dataframe, _manager: dataframe,
        ), patch("src.evaluation.parameter_optimizer.WDBSCAN", _FakeDBSCAN), patch(
            "src.evaluation.parameter_optimizer._safe_metrics", return_value=self.metrics
        ):
            result = optimize_wdbscan_params(
                self.data,
                weight_manager=object(),
                sample_size=len(self.data),
                min_samples_values=[2],
                on_progress=lambda *event: events.append(event),
            )
        self.assert_complete_progress(events, len(result.rows))


if __name__ == "__main__":
    unittest.main()

