"""Diferencias legibles por tipo de variable, sin reinterpretar códigos nominales."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


PLOT_CONFIG_ES = {
    "locale": "es",
    "locales": {"es": {"dictionary": {
        "Download plot as a PNG": "Descargar gráfico como PNG",
        "Download plot as a png": "Descargar gráfico como PNG",
        "Taking snapshot - this may take a few seconds": "Preparando imagen…",
        "Snapshot succeeded": "Imagen descargada",
        "Sorry, there was a problem downloading your snapshot!": "No se pudo descargar la imagen",
    }, "format": {"decimal": ",", "thousands": "."}}},
    "displaylogo": False,
    "displayModeBar": True,
    "modeBarButtons": [["toImage"]],
    "scrollZoom": False,
    "doubleClick": False,
    "toImageButtonOptions": {"filename": "diferencias_por_grupo", "scale": 2},
}


def profile_difference_rows(profiles) -> pd.DataFrame:
    """Una fila por variable/grupo; nominales comparan la misma categoría.

    Se elige para cada nominal la categoría con mayor desviación absoluta
    observada. Los porcentajes se convierten a proporciones solo para el color.
    Las magnitudes se describen con su unidad original en el texto.
    """
    rows = []
    for code, summaries in profiles.summaries.groupby("feature_code", sort=False):
        kind = summaries.iloc[0]["feature_type"]
        name = summaries.iloc[0]["feature_name"]
        if kind == "categorico_nominal":
            distributions = profiles.distributions.query("feature_code == @code")
            if distributions.empty:
                continue
            category = distributions.loc[
                distributions["percentage_point_difference"].abs().idxmax(),
                "category_value",
            ]
            values = distributions[np.isclose(distributions["category_value"], category)]
            for _, row in values.iterrows():
                label = str(row["category_label"])
                rows.append(_row(code, name, row["cluster_id"],
                                 row["percentage"], row["global_percentage"],
                                 "porcentaje", f"Categoría: {label}"))
        else:
            for _, row in summaries.iterrows():
                statistic = "porcentaje" if kind == "binario" else (
                    "mediana" if kind == "categorico_ordinal" else "media"
                )
                detail = (f"Proporción: {row['representative_label']}" if kind == "binario"
                          else f"{statistic.capitalize()} en escala procesada")
                rows.append(_row(code, name, row["cluster_id"],
                                 row["representative_value"], row["global_reference"],
                                 statistic, detail))
    return pd.DataFrame(rows)


def _number(value, digits=1):
    return f"{float(value):.{digits}f}".replace(".", ",")


def _row(code, name, cluster, value, reference, statistic, detail):
    delta = float(value) - float(reference)
    percentage = statistic == "porcentaje"
    if not np.isfinite(delta):
        reading = "Sin datos suficientes"
    elif abs(delta) < 1e-9:
        reading = "Igual a la referencia"
    else:
        direction = "por encima" if delta > 0 else "por debajo"
        unit = "puntos porcentuales" if percentage else "unidades de escala procesada"
        reading = f"{_number(abs(delta), 1 if percentage else 3)} {unit} {direction}"
    return {
        "feature": code, "variable": f"{name} ({code})", "cluster_id": int(cluster),
        "group": f"Grupo {int(cluster)}", "detail": detail,
        "value": f"{_number(value, 1 if percentage else 3)}{' %' if percentage else ''}",
        "reference": f"{_number(reference, 1 if percentage else 3)}{' %' if percentage else ''}",
        "deviation": delta / 100 if percentage else delta,
        "reading": reading,
    }


def difference_heatmap(rows: pd.DataFrame, feature_codes: list[str]) -> go.Figure:
    selected = rows[rows.feature.isin(feature_codes)]
    groups = selected.sort_values("cluster_id")["group"].drop_duplicates().tolist()
    matrix = selected.pivot(index="feature", columns="group", values="deviation").reindex(
        index=feature_codes, columns=groups
    )
    labels = selected.drop_duplicates("feature").set_index("feature")["variable"]
    details = selected.set_index(["feature", "group"])
    custom = [[[details.loc[(code, group), field] for field in
                ("detail", "value", "reference", "reading")]
               for group in groups] for code in feature_codes]
    finite = matrix.to_numpy()[np.isfinite(matrix.to_numpy())]
    bound = max(float(np.abs(finite).max()), 0.01) if finite.size else 1.0
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=groups, y=[labels[code] for code in feature_codes],
        customdata=custom, colorscale="RdBu_r", zmin=-bound, zmax=bound, zmid=0,
        xgap=2, ygap=2,
        colorbar=dict(title="Diferencia", tickvals=[-bound, 0, bound],
                      ticktext=["Por debajo", "Igual", "Por encima"], thickness=16),
        hovertemplate=("<b>%{y}</b><br>%{x}<br>%{customdata[0]}<br>"
                       "Grupo: %{customdata[1]}<br>Referencia: %{customdata[2]}<br>"
                       "<b>%{customdata[3]}</b><extra></extra>"),
    ))
    fig.update_layout(
        height=max(440, len(feature_codes) * 34 + 110),
        margin=dict(l=300, r=125, t=25, b=65), dragmode=False,
        xaxis=dict(type="category", tickmode="array", tickvals=groups, title=None,
                   fixedrange=True, automargin=True),
        yaxis=dict(type="category", tickmode="array", tickvals=[labels[c] for c in feature_codes],
                   autorange="reversed", title=None, fixedrange=True, automargin=True),
        hoverlabel=dict(align="left"),
    )
    return fig
