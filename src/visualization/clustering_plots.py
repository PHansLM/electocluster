"""
Datos y figuras exportables para visualizacion de clustering.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


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


def pca_projection(df: pd.DataFrame, labels) -> pd.DataFrame:
    """Proyecta datos a 2 componentes PCA para visualizacion comparable."""
    labels = np.asarray(labels, dtype=int)
    pca = PCA(n_components=2, random_state=42)
    values = pca.fit_transform(df)
    return pd.DataFrame({
        "PC1": values[:, 0],
        "PC2": values[:, 1],
        "cluster": ["noise" if label == -1 else str(label) for label in labels],
        "cluster_id": labels,
    })


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

    projection = pca_projection(df, labels)
    distribution = cluster_distribution(labels)
    profiles = cluster_profiles(df, labels)

    paths = {
        "pca_clusters": output_path / f"{run_id}_pca_clusters.png",
        "distribution": output_path / f"{run_id}_cluster_distribution.png",
    }

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=projection,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette="tab10",
        s=35,
        linewidth=0,
    )
    plt.title(f"Clusters proyectados en PCA 2D - {run_id}")
    plt.tight_layout()
    plt.savefig(paths["pca_clusters"], dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=distribution, x="cluster", y="size", color="#2563eb")
    plt.title(f"Distribucion de clusters - {run_id}")
    plt.xlabel("Cluster")
    plt.ylabel("Registros")
    plt.tight_layout()
    plt.savefig(paths["distribution"], dpi=220, bbox_inches="tight")
    plt.close()

    if not profiles.empty and profiles.shape[1] > 3:
        heatmap_path = output_path / f"{run_id}_cluster_profiles.png"
        profile_values = profiles.set_index("cluster").drop(columns=["size", "percentage"])
        plt.figure(figsize=(12, max(5, len(profile_values) * 0.5)))
        sns.heatmap(profile_values, cmap="viridis", cbar=True)
        plt.title(f"Perfil promedio por cluster - {run_id}")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=220, bbox_inches="tight")
        plt.close()
        paths["profiles"] = heatmap_path

    return {name: str(path) for name, path in paths.items()}
