"""Configuración de acciones visibles de Plotly para la interfaz."""


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
