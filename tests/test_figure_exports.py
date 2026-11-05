"""Comprueba que las figuras exportables se preparen para descarga."""

import sys
import unittest
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.clustering_plots import (
    build_pca_projection_png,
    build_run_figures_zip,
)


class FigureExportsTest(unittest.TestCase):
    def test_figures_are_packaged_in_memory_as_a_zip(self):
        dataset = pd.DataFrame({
            "edad": [20, 21, 45, 46],
            "ingreso": [1.0, 1.2, 4.8, 5.0],
        })

        archive = build_run_figures_zip(dataset, np.array([0, 0, 1, 1]), "run_demo")

        self.assertIsInstance(archive, bytes)
        with ZipFile(BytesIO(archive)) as zip_file:
            self.assertEqual(
                set(zip_file.namelist()),
                {
                    "run_demo_pca_clusters.png",
                    "run_demo_distribution.png",
                    "run_demo_profiles.png",
                },
            )
            self.assertTrue(
                all(zip_file.read(name).startswith(b"\x89PNG") for name in zip_file.namelist())
            )

    def test_exported_figures_describe_their_axes_and_variables(self):
        source = (PROJECT_ROOT / "src" / "visualization" / "clustering_plots.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('f"{component} - " + ",\\n".join(drivers)', source)
        self.assertIn('_axis_endpoint_label(pc1_profile["low_values"])', source)
        self.assertIn('_axis_endpoint_label(pc1_profile["high_values"])', source)
        self.assertIn('_axis_endpoint_label(pc2_profile["low_values"])', source)
        self.assertIn('_axis_endpoint_label(pc2_profile["high_values"])', source)
        self.assertIn('"arrowstyle": "->"', source)
        self.assertIn('"linestyle": "--"', source)
        self.assertIn('"color": "#2563eb"', source)
        self.assertIn("valores próximos a 0 reflejan un perfil intermedio", source)
        self.assertIn("feature_display_name", source)
        self.assertIn("nombre legible (código entre paréntesis)", source)

    def test_pca_projection_records_interpretable_axis_profiles(self):
        dataset = pd.DataFrame({
            "q2": [0.0, 0.2, 0.8, 1.0],
            "wealth_index": [0.1, 0.2, 0.7, 0.9],
            "formal": [0.0, 1.0, 1.0, 0.0],
        })

        from src.visualization.clustering_plots import pca_projection

        projection = pca_projection(dataset, np.array([0, 0, 1, 1]))

        self.assertEqual(set(projection.attrs["axis_profiles"]), {"PC1", "PC2"})
        self.assertIn("Edad", projection.attrs["axis_profiles"]["PC1"]["drivers"])

    def test_pca_projection_can_be_downloaded_independently(self):
        dataset = pd.DataFrame({
            "edad": [20, 21, 45, 46],
            "ingreso": [1.0, 1.2, 4.8, 5.0],
        })

        image = build_pca_projection_png(dataset, np.array([0, 0, 1, 1]), "run_demo")

        self.assertTrue(image.startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()

