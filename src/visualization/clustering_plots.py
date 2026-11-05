"""
Datos y figuras exportables para visualizacion de clustering.
"""

from io import BytesIO
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from src.visualization.plotly_config import cluster_color_map
from src.visualization.profile_interpreter import (
    feature_display_name,
    interpret_feature_value,
)
from src.utils.constants import TODAS_VARIABLES


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
    projection = pd.DataFrame({
        "PC1": values[:, 0],
        "PC2": values[:, 1],
        "cluster": ["noise" if label == -1 else str(label) for label in labels],
        "cluster_id": labels,
    })
    projection.attrs["explained_variance_ratio"] = pca.explained_variance_ratio_
    projection.attrs["axis_profiles"] = _pca_axis_profiles(df, pca, values)
    return projection


def _pca_axis_profiles(
    df: pd.DataFrame,
    pca: PCA,
    scores: np.ndarray,
) -> dict[str, dict[str, str | float]]:
    """Describe los extremos de cada eje con valores interpretables del dataset."""
    profiles = {}
    for component_index, component_name in enumerate(("PC1", "PC2")):
        loadings = pd.Series(pca.components_[component_index], index=df.columns)
        interpretable = [
            feature
            for feature in df.columns
            if TODAS_VARIABLES.get(feature, {}).get("tipo") != "categorico_nominal"
        ]
        driver_features = (
            loadings.loc[interpretable]
            .abs()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )
        component_scores = scores[:, component_index]
        low_mask = component_scores <= np.quantile(component_scores, 0.2)
        high_mask = component_scores >= np.quantile(component_scores, 0.8)

        def describe(mask: np.ndarray) -> str:
            descriptions = []
            for feature in driver_features:
                value = float(df.loc[mask, feature].mean())
                reading = interpret_feature_value(feature, value).text
                descriptions.append(f"{feature_display_name(feature)}: {reading}")
            return " · ".join(descriptions) or "Sin variables interpretables destacadas"

        def describe_briefly(mask: np.ndarray) -> str:
            descriptions = []
            for feature in driver_features:
                value = float(df.loc[mask, feature].mean())
                reading = _short_profile_reading(
                    interpret_feature_value(feature, value).text
                )
                name = TODAS_VARIABLES.get(feature, {}).get("nombre", feature)
                descriptions.append(f"{name}: {reading}")
            return " · ".join(descriptions) or "Perfil intermedio"

        def values_only(mask: np.ndarray) -> str:
            readings = []
            for feature in driver_features:
                reading = _short_profile_reading(
                    interpret_feature_value(feature, float(df.loc[mask, feature].mean())).text
                )
                if feature == "q2":
                    number = re.search(r"\d+(?:[.,]\d+)?", reading)
                    reading = number.group(0) if number else reading
                readings.append(reading)
            return " · ".join(readings) or "Sin lectura disponible"

        profiles[component_name] = {
            "variance": float(pca.explained_variance_ratio_[component_index] * 100),
            "drivers": ", ".join(
                feature_display_name(feature) for feature in driver_features
            ),
            "dimension": ", ".join(
                TODAS_VARIABLES.get(feature, {}).get("nombre", feature)
                for feature in driver_features
            ),
            "low": describe(low_mask),
            "high": describe(high_mask),
            "low_brief": describe_briefly(low_mask),
            "high_brief": describe_briefly(high_mask),
            "low_values": values_only(low_mask),
            "high_values": values_only(high_mask),
        }
    return profiles


def _short_profile_reading(reading: str) -> str:
    """Reduce una interpretación para que pueda caber como guía de un eje PCA."""
    cleaned = re.sub(r"^M[aá]s cercano a ", "", reading, flags=re.IGNORECASE)
    cleaned = cleaned.split(" que ")[0].split(" (")[0]
    return cleaned.replace("Aprox. ", "≈ ")


def _axis_dimension_label(component: str, profile: dict) -> str:
    """Forma una etiqueta breve y legible para el centro de un eje PCA."""
    drivers = profile["dimension"].split(", ")
    return f"{component} - " + ",\n".join(drivers)


def _axis_endpoint_label(values: str) -> str:
    """Apila las lecturas de un extremo sin nombrar su dirección."""
    return values.replace(" · ", "\n")


def _run_figure_pngs(df: pd.DataFrame, labels, run_id: str) -> dict[str, bytes]:
    """Crea las figuras del run en memoria, sin escribirlas en el proyecto."""
    projection = pca_projection(df, labels)
    distribution = cluster_distribution(labels)
    profiles = cluster_profiles(df, labels)

    figures = {}

    figure, axis = plt.subplots(figsize=(12, 8))
    projection_colors = cluster_color_map(projection["cluster"])
    sns.scatterplot(
        data=projection,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette=projection_colors,
        s=35,
        linewidth=0,
        ax=axis,
    )
    axis_profiles = projection.attrs["axis_profiles"]
    pc1_profile = axis_profiles["PC1"]
    pc2_profile = axis_profiles["PC2"]
    figure.suptitle(f"Clusters proyectados en PCA 2D - {run_id}", fontsize=14)
    axis.set_xlabel(
        _axis_dimension_label("PC1", pc1_profile),
        fontsize=11,
        labelpad=22,
    )
    axis.set_ylabel(
        _axis_dimension_label("PC2", pc2_profile),
        fontsize=11,
        labelpad=24,
    )
    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()
    endpoint_arrow = {
        "arrowstyle": "->",
        "lw": 0.8,
        "linestyle": "--",
        "color": "#2563eb",
    }
    axis.annotate(
        _axis_endpoint_label(pc1_profile["low_values"]),
        xy=(x_min, y_min),
        xytext=(0.20, 0.155),
        textcoords="figure fraction",
        ha="left",
        va="center",
        fontsize=10,
        arrowprops=endpoint_arrow,
        annotation_clip=False,
    )
    axis.annotate(
        _axis_endpoint_label(pc1_profile["high_values"]),
        xy=(x_max, y_min),
        xytext=(0.92, 0.155),
        textcoords="figure fraction",
        ha="right",
        va="center",
        fontsize=10,
        arrowprops=endpoint_arrow,
        annotation_clip=False,
    )
    axis.annotate(
        _axis_endpoint_label(pc2_profile["high_values"]),
        xy=(x_min, y_max),
        xytext=(0.14, 0.86),
        textcoords="figure fraction",
        ha="right",
        va="bottom",
        fontsize=10,
        arrowprops=endpoint_arrow,
        annotation_clip=False,
    )
    axis.annotate(
        _axis_endpoint_label(pc2_profile["low_values"]),
        xy=(x_min, y_min),
        xytext=(0.14, 0.22),
        textcoords="figure fraction",
        ha="right",
        va="top",
        fontsize=10,
        arrowprops=endpoint_arrow,
        annotation_clip=False,
    )
    plt.figtext(
        0.5,
        0.015,
        "Nota: valores próximos a 0 reflejan un perfil intermedio, sin rasgos claramente "
        "distintivos en estas dimensiones. Esta proyección 2D simplifica varias dimensiones "
        "y no sustituye los perfiles completos.",
        ha="center",
        va="bottom",
        fontsize=9,
        wrap=True,
    )
    figure.subplots_adjust(left=0.22, right=0.93, bottom=0.27, top=0.89)
    image = BytesIO()
    figure.savefig(image, format="png", dpi=220, facecolor="white")
    plt.close(figure)
    figures["pca_clusters"] = image.getvalue()

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
    image = BytesIO()
    plt.savefig(image, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()
    figures["distribution"] = image.getvalue()

    if not profiles.empty and profiles.shape[1] > 3:
        profile_values = profiles.set_index("cluster").drop(columns=["size", "percentage"])
        figure_width = max(16, min(30, len(profile_values.columns) * 0.95))
        plt.figure(figsize=(figure_width, max(6, len(profile_values) * 1.05)))
        axis = sns.heatmap(profile_values, cmap="viridis", cbar=True)
        axis.set_xticklabels(
            [feature_display_name(column) for column in profile_values.columns],
            rotation=60,
            ha="right",
            fontsize=8,
        )
        axis.set_xlabel(
            "Variables originales: nombre legible (código entre paréntesis)",
            labelpad=14,
        )
        axis.set_ylabel("Cluster")
        plt.title(f"Perfil promedio por cluster - {run_id}", pad=14)
        plt.figtext(
            0.5,
            0.012,
            "Cada celda representa el promedio procesado de la variable en el cluster; "
            "consulta la tabla de perfiles para su interpretación detallada.",
            ha="center",
            va="bottom",
            fontsize=8,
            wrap=True,
        )
        plt.tight_layout(rect=(0, 0.13, 1, 1))
        image = BytesIO()
        plt.savefig(image, format="png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close()
        figures["profiles"] = image.getvalue()

    return figures


def build_run_figures_zip(df: pd.DataFrame, labels, run_id: str) -> bytes:
    """Empaqueta las figuras de un run para descargar desde el navegador."""
    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        for name, image in _run_figure_pngs(df, labels, run_id).items():
            archive.writestr(f"{run_id}_{name}.png", image)
    return archive_buffer.getvalue()


def build_pca_projection_png(df: pd.DataFrame, labels, run_id: str) -> bytes:
    """Crea solo la proyección PCA 2D para una descarga independiente."""
    return _run_figure_pngs(df, labels, run_id)["pca_clusters"]


def save_run_figures(
    df: pd.DataFrame,
    labels,
    run_id: str,
    output_dir: str | Path,
) -> dict[str, str]:
    """Compatibilidad para usos internos que requieren archivos persistidos."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, image in _run_figure_pngs(df, labels, run_id).items():
        path = output_path / f"{run_id}_{name}.png"
        path.write_bytes(image)
        paths[name] = str(path)

    return paths
