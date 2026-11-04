"""
Datos y figuras exportables para visualizacion de clustering.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from src.visualization.plotly_config import cluster_color_map


def cluster_distribution(labels) -> pd.DataFrame:
    """Retorna tamanos y porcentajes por cluster, incluyendo ruido."""
    labels = np.asarray(labels, dtype=int)
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    rows = []
    for label, count in zip(unique, counts):
        cluster = "noise" if int(label) == -1 else str(int(label))
        rows.append({
            "cluster": cluster,
            "cluster_id": int(label),
            "size": int(count),
            "percentage": round(float(count / total * 100), 2),
        })
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def cluster_profiles(
    df: pd.DataFrame,
    labels,
    exclude_noise: bool = True,
) -> pd.DataFrame:
    """Calcula perfiles promedio por cluster sobre el dataset procesado."""
    labels = np.asarray(labels, dtype=int)
    profile_df = df.copy()
    profile_df["cluster"] = labels
    if exclude_noise:
        profile_df = profile_df[profile_df["cluster"] != -1]
    if profile_df.empty:
        return pd.DataFrame()

    grouped = profile_df.groupby("cluster")
    means = grouped.mean(numeric_only=True).round(4)
    sizes = grouped.size().rename("size")
    percentages = (sizes / len(labels) * 100).round(2).rename("percentage")
    result = pd.concat([sizes, percentages, means], axis=1)
    return result.reset_index()


def pca_projection(
    df: pd.DataFrame,
    labels,
    *,
    return_metadata: bool = False,
):
    """Proyecta datos a 2 componentes PCA para visualizacion comparable.

    ``return_metadata`` conserva la compatibilidad con los consumidores que
    esperan únicamente el DataFrame y permite a la interfaz presentar la
    varianza explicada y los principales aportes de las dimensiones base.
    """
    labels = np.asarray(labels, dtype=int)
    pca = PCA(n_components=2, random_state=42)
    values = pca.fit_transform(df)
    projection = pd.DataFrame({
        "PC1": values[:, 0],
        "PC2": values[:, 1],
        "cluster": ["noise" if label == -1 else str(label) for label in labels],
        "cluster_id": labels,
    })

    if not return_metadata:
        return projection

    explained = pca.explained_variance_ratio_
    loadings = pca.components_
    top_contributions = {}
    for component_index, component_loadings in enumerate(loadings):
        squared = np.square(component_loadings)
        total = squared.sum()
        contributions = squared / total * 100 if total else np.zeros_like(squared)
        order = np.argsort(contributions)[::-1][:5]
        top_contributions[f"PC{component_index + 1}"] = [
            {
                "feature": str(df.columns[index]),
                "contribution": round(float(contributions[index]), 2),
                "loading": round(float(component_loadings[index]), 4),
            }
            for index in order
        ]

    metadata = {
        "explained_variance_ratio": {
            f"PC{index + 1}": round(float(value * 100), 2)
            for index, value in enumerate(explained)
        },
        "top_contributions": top_contributions,
    }
    return projection, metadata


def save_run_figures(
    df: pd.DataFrame,
    labels,
    run_id: str,
    output_dir: str | Path,
) -> dict[str, str]:
    """
    Guarda figuras estaticas del run para anexos/capitulo 8.

    Returns:
        Dict {nombre_figura: ruta}.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    projection, pca_metadata = pca_projection(df, labels, return_metadata=True)
    distribution = cluster_distribution(labels)
    profiles = cluster_profiles(df, labels)

    paths = {
        "pca_clusters": output_path / f"{run_id}_pca_clusters.png",
        "distribution": output_path / f"{run_id}_cluster_distribution.png",
    }

    plt.figure(figsize=(10, 7.5))
    projection_colors = cluster_color_map(projection["cluster"])
    sns.scatterplot(
        data=projection,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette=projection_colors,
        s=35,
        linewidth=0,
    )
    explained = pca_metadata["explained_variance_ratio"]
    plt.title(f"Clusters proyectados en PCA 2D - {run_id}")
    plt.xlabel(f"PC1 ({explained['PC1']:.2f}% varianza)")
    plt.ylabel(f"PC2 ({explained['PC2']:.2f}% varianza)")
    legend_lines = []
    for component in ("PC1", "PC2"):
        contributions = pca_metadata["top_contributions"][component]
        detail = " · ".join(
            f"{item['feature']}: {item['contribution']:.2f}%"
            for item in contributions
        )
        legend_lines.append(f"{component}: {detail}")
    plt.figtext(0.02, 0.01, "\n".join(legend_lines), ha="left", va="bottom", fontsize=7)
    plt.tight_layout(rect=(0, 0.15, 1, 1))
    plt.savefig(paths["pca_clusters"], dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()

    plt.figure(figsize=(8, 5))
    distribution_colors = cluster_color_map(distribution["cluster"])
    sns.barplot(
        data=distribution,
        x="cluster",
        y="size",
        hue="cluster",
        palette=distribution_colors,
        legend=False,
    )
    plt.title(f"Distribucion de clusters - {run_id}")
    plt.xlabel("Cluster")
    plt.ylabel("Registros")
    plt.tight_layout()
    plt.savefig(paths["distribution"], dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()

    if not profiles.empty and profiles.shape[1] > 3:
        heatmap_path = output_path / f"{run_id}_cluster_profiles.png"
        profile_values = profiles.set_index("cluster").drop(columns=["size", "percentage"])
        plt.figure(figsize=(12, max(5, len(profile_values) * 0.5)))
        sns.heatmap(profile_values, cmap="viridis", cbar=True)
        plt.title(f"Perfil promedio por cluster - {run_id}")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close()
        paths["profiles"] = heatmap_path

    return {name: str(path) for name, path in paths.items()}
