"""Configuración de acciones visibles de Plotly para la interfaz."""


CLUSTER_COLOR_SEQUENCE = (
    "#2563EB",
    "#F97316",
    "#16A34A",
    "#9333EA",
    "#DC2626",
    "#0891B2",
    "#DB2777",
    "#65A30D",
    "#4F46E5",
    "#D97706",
    "#0D9488",
    "#E11D48",
    "#64748B",
)
NOISE_COLOR = "#6B7280"
ALGORITHM_COLOR_MAP = {
    "WKMedoids": "#2563EB",
    "W-Hierarchical Clustering": "#F97316",
    "W-DBSCAN": "#16A34A",
}


def cluster_color_map(cluster_values) -> dict[str, str]:
    """Asigna colores estables dentro de una vista y reserva gris para ruido."""
    labels = [str(value) for value in cluster_values]
    regular = sorted(
        {label for label in labels if label.lower() != "noise"},
        key=lambda label: (
            (0, int(label)) if label.lstrip("-").isdigit() else (1, label.casefold())
        ),
    )
    colors = {
        label: CLUSTER_COLOR_SEQUENCE[index % len(CLUSTER_COLOR_SEQUENCE)]
        for index, label in enumerate(regular)
    }
    if any(label.lower() == "noise" for label in labels):
        colors["noise"] = NOISE_COLOR
    return colors


def apply_plotly_theme(
    figure,
    theme_type: str,
    *,
    margin: dict | None = None,
    font_size: int = 14,
    polar: bool = False,
):
    """Integra una figura con el tema activo sin modificar datos ni escalas."""
    dark = theme_type == "dark"
    grid_color = "rgba(148, 163, 184, 0.22)" if dark else "rgba(71, 85, 105, 0.18)"
    figure.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            "size": font_size,
        },
        margin=margin,
    )
    if polar:
        figure.update_polars(
            bgcolor="rgba(0,0,0,0)",
            radialaxis={"gridcolor": grid_color, "linecolor": grid_color},
            angularaxis={"gridcolor": grid_color, "linecolor": grid_color},
        )
    else:
        figure.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color)
        figure.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color)
    return figure


PLOTLY_LOCALE_ES = {
    "dictionary": {
        "Download plot as a PNG": "Descargar gráfico como PNG",
        "Download plot as a png": "Descargar gráfico como PNG",
        "Edit in Chart Studio": "Editar en Chart Studio",
        "Share plot": "Compartir gráfico",
        "Zoom": "Acercar",
        "Pan": "Desplazar",
        "Zoom in": "Acercar",
        "Zoom out": "Alejar",
        "Autoscale": "Ajustar escala",
        "Reset axes": "Restablecer ejes",
        "Reset view": "Restablecer vista",
        "Reset views": "Restablecer vistas",
        "Select box": "Seleccionar área",
        "Select lasso": "Seleccionar con lazo",
        "Toggle Spike Lines": "Mostrar líneas de referencia",
        "Show closest data on hover": "Mostrar el dato más cercano al pasar el cursor",
        "Compare data on hover": "Comparar datos al pasar el cursor",
        "Taking snapshot - this may take a few seconds": "Preparando imagen…",
        "Snapshot succeeded": "Imagen descargada",
        "Sorry, there was a problem downloading your snapshot!": "No se pudo descargar la imagen",
    },
    "format": {"decimal": ",", "thousands": "."},
}


PLOT_CONFIG_ES = {
    "locale": "es",
    "locales": {"es": PLOTLY_LOCALE_ES},
    "displaylogo": False,
}


HEATMAP_PLOT_CONFIG_ES = {
    **PLOT_CONFIG_ES,
    "displayModeBar": True,
    "modeBarButtons": [["toImage"]],
    "scrollZoom": False,
    "doubleClick": False,
    "toImageButtonOptions": {"filename": "diferencias_por_grupo", "scale": 2},
}
