"""
Utilidades de visualizacion para resultados de clustering.
"""

from .clustering_plots import (
    cluster_distribution,
    cluster_profiles,
    pca_projection,
    build_pca_projection_png,
    build_run_figures_zip,
)
from .cluster_explainability import (
    EXPLANATORY_DIMENSIONS,
    cluster_dimension_scores,
    cluster_profile_deviation,
    cluster_profile_distance_matrix,
    top_distinctive_features,
)
from .profile_interpreter import (
    PROFILE_METADATA_COLUMNS,
    feature_display_name,
    interpret_feature_value,
)
from .high_dimensional import SemanticHeatmapData, semantic_profile_heatmap
from .semantic_profiles import (
    DISTRIBUTION_COLUMNS,
    SEMANTIC_PROFILE_EXPORT_SCHEMA,
    SUMMARY_COLUMNS,
    SemanticProfileTables,
    build_semantic_profile_export,
    cluster_semantic_profiles,
)

__all__ = [
    "DISTRIBUTION_COLUMNS",
    "EXPLANATORY_DIMENSIONS",
    "PROFILE_METADATA_COLUMNS",
    "SEMANTIC_PROFILE_EXPORT_SCHEMA",
    "SemanticHeatmapData",
    "SUMMARY_COLUMNS",
    "SemanticProfileTables",
    "build_semantic_profile_export",
    "build_pca_projection_png",
    "cluster_dimension_scores",
    "cluster_distribution",
    "cluster_profile_deviation",
    "cluster_profile_distance_matrix",
    "cluster_profiles",
    "cluster_semantic_profiles",
    "feature_display_name",
    "interpret_feature_value",
    "pca_projection",
    "build_run_figures_zip",
    "semantic_profile_heatmap",
    "top_distinctive_features",
]
