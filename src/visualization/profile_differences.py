"""Diferencias legibles por tipo de variable, sin reinterpretar códigos nominales."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualization.profile_interpreter import interpret_feature_value

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
                summary = summaries[summaries["cluster_id"] == row["cluster_id"]].iloc[0]
                rows.append(_row(code, name, row["cluster_id"],
                                 row["percentage"], row["global_percentage"],
                                 "porcentaje", f"Categoría: {label}",
                                 feature_type=kind,
                                 practical_value=_profile_description(summary, kind, code),
                                 practical_reference=_reference_description(summary, kind, code)))
        else:
            for _, row in summaries.iterrows():
                statistic = "porcentaje" if kind == "binario" else (
                    "mediana" if kind == "categorico_ordinal" else "media"
                )
                detail = (f"Proporción: {row['representative_label']}" if kind == "binario"
                          else f"{statistic.capitalize()} en escala procesada")
                rows.append(_row(code, name, row["cluster_id"],
                                 row["representative_value"], row["global_reference"],
                                 statistic, detail,
                                 feature_type=kind,
                                 practical_value=_profile_description(row, kind, code),
                                 practical_reference=_reference_description(row, kind, code)))
    return pd.DataFrame(rows)


def _number(value, digits=1):
    return f"{float(value):.{digits}f}".replace(".", ",")


def _row(
    code,
    name,
    cluster,
    value,
    reference,
    statistic,
    detail,
    *,
    feature_type,
    practical_value,
    practical_reference,
):
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
        "feature_type": feature_type,
        "practical_value": practical_value,
        "practical_reference": practical_reference,
        "value": f"{_number(value, 1 if percentage else 3)}{' %' if percentage else ''}",
        "reference": f"{_number(reference, 1 if percentage else 3)}{' %' if percentage else ''}",
        "deviation": delta / 100 if percentage else delta,
        "reading": reading,
    }


def _profile_description(row, feature_type: str, code: str) -> str:
    if feature_type == "categorico_nominal":
        percentage = row.get("representative_percentage")
        suffix = f" ({_number(percentage)} %)" if pd.notna(percentage) else ""
        return f"Predomina {row.get('representative_label', 'Sin dato')}{suffix}"
    if feature_type == "categorico_ordinal":
        return f"Mediana: {row.get('representative_label', 'Sin dato')}"
    if feature_type == "binario":
        return (
            f"{row.get('representative_label', 'Categoría')}: "
            f"{_number(row.get('representative_percentage'))} %"
        )
    return interpret_feature_value(code, row.get("representative_value")).text


def _reference_description(row, feature_type: str, code: str) -> str:
    if feature_type == "categorico_nominal":
        percentage = row.get("global_reference_percentage")
        suffix = f" ({_number(percentage)} %)" if pd.notna(percentage) else ""
        return f"Predomina {row.get('global_reference_label', 'Sin dato')}{suffix}"
    if feature_type == "categorico_ordinal":
        return f"Mediana: {row.get('global_reference_label', 'Sin dato')}"
    if feature_type == "binario":
        return (
            f"{row.get('global_reference_label', 'Categoría')}: "
            f"{_number(row.get('global_reference_percentage'))} %"
        )
    return interpret_feature_value(code, row.get("global_reference")).text


def difference_heatmap(rows: pd.DataFrame, feature_codes: list[str]) -> go.Figure:
    selected = rows[rows.feature.isin(feature_codes)]
    groups = selected.sort_values("cluster_id")["group"].drop_duplicates().tolist()
    matrix = selected.pivot(index="feature", columns="group", values="deviation").reindex(
        index=feature_codes, columns=groups
    )
    labels = selected.drop_duplicates("feature").set_index("feature")["variable"]
    details = selected.set_index(["feature", "group"])
    custom = [[[details.loc[(code, group), field] for field in
                ("detail", "value", "reference", "reading",
                 "practical_value", "practical_reference")]
               for group in groups] for code in feature_codes]
    finite = matrix.to_numpy()[np.isfinite(matrix.to_numpy())]
    bound = max(float(np.abs(finite).max()), 0.01) if finite.size else 1.0
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=groups, y=[labels[code] for code in feature_codes],
        customdata=custom, colorscale="RdBu_r", zmin=-bound, zmax=bound, zmid=0,
        xgap=2, ygap=2,
        colorbar=dict(title="Diferencia", tickvals=[-bound, 0, bound],
                      ticktext=["Por debajo", "Igual", "Por encima"], thickness=14),
        hovertemplate=("<b>%{y}</b><br>%{x}<br>%{customdata[0]}<br>"
                       "Grupo: %{customdata[1]} · %{customdata[4]}<br>"
                       "Referencia: %{customdata[2]} · %{customdata[5]}<br>"
                       "<b>%{customdata[3]}</b><extra></extra>"),
    ))
    fig.update_layout(
        height=max(440, len(feature_codes) * 34 + 110),
        margin=dict(l=24, r=90, t=24, b=55), dragmode=False,
        xaxis=dict(type="category", tickmode="array", tickvals=groups, title=None,
                   fixedrange=True, automargin=True),
        yaxis=dict(type="category", tickmode="array", tickvals=[labels[c] for c in feature_codes],
                   autorange="reversed", title=None, fixedrange=True, automargin=True),
        hoverlabel=dict(align="left"),
    )
    return fig


def cluster_difference_summaries(rows: pd.DataFrame, top_n: int = 2) -> dict[int, str]:
    """Resume los rasgos con mayor desviacion de cada grupo para tooltips."""
    if rows.empty:
        return {}

    summaries = {}
    ranked = rows.assign(magnitude=rows["deviation"].abs()).sort_values(
        ["cluster_id", "magnitude"], ascending=[True, False], kind="stable"
    )
    for cluster_id, group_rows in ranked.groupby("cluster_id", sort=True):
        selected = group_rows.drop_duplicates("feature").head(top_n)
        details = [
            f"{row.variable}: {row.practical_value} ({str(row.reading).lower()})"
            for row in selected.itertuples()
        ]
        summaries[int(cluster_id)] = " · ".join(details) if details else "Sin diferencias destacadas"
    return summaries


def pair_difference_summaries(
    rows: pd.DataFrame, top_n: int = 2
) -> dict[tuple[int, int], str]:
    """Explica que variables separan mas cada pareja sin promediar codigos nominales."""
    if rows.empty:
        return {}

    clusters = sorted(int(value) for value in rows["cluster_id"].unique())
    indexed = rows.set_index(["feature", "cluster_id"])
    summaries = {}
    for position, first in enumerate(clusters):
        for second in clusters[position + 1:]:
            first_rows = rows[rows["cluster_id"] == first].set_index("feature")
            second_rows = rows[rows["cluster_id"] == second].set_index("feature")
            common = first_rows.index.intersection(second_rows.index)
            if common.empty:
                summaries[(first, second)] = "Sin variables comparables"
                continue
            contrasts = (
                first_rows.loc[common, "deviation"]
                .sub(second_rows.loc[common, "deviation"])
                .abs()
                .sort_values(ascending=False, kind="stable")
            )
            details = []
            for feature in contrasts.head(top_n).index:
                first_row = indexed.loc[(feature, first)]
                second_row = indexed.loc[(feature, second)]
                details.append(
                    f"<b>{first_row['variable']}</b>: Grupo {first}, "
                    f"{first_row['practical_value']}; Grupo {second}, "
                    f"{second_row['practical_value']}"
                )
            summaries[(first, second)] = "<br>• " + "<br>• ".join(details)
    return summaries


def pair_difference_details(
    rows: pd.DataFrame, first: int, second: int, top_n: int = 3
) -> pd.DataFrame:
    """Devuelve los contrastes semanticos principales de una pareja de grupos."""
    first_rows = rows[rows["cluster_id"] == int(first)].set_index("feature")
    second_rows = rows[rows["cluster_id"] == int(second)].set_index("feature")
    common = first_rows.index.intersection(second_rows.index)
    if common.empty:
        return pd.DataFrame()
    contrasts = (
        first_rows.loc[common, "deviation"]
        .sub(second_rows.loc[common, "deviation"])
        .abs()
        .sort_values(ascending=False, kind="stable")
    )
    records = []
    for feature in contrasts.head(top_n).index:
        left = first_rows.loc[feature]
        right = second_rows.loc[feature]
        records.append({
            "variable": left["variable"],
            f"Grupo {first}": left["practical_value"],
            f"Grupo {second}": right["practical_value"],
            "contraste_relativo": float(contrasts.loc[feature]),
        })
    result = pd.DataFrame(records)
    if not result.empty:
        maximum = float(result["contraste_relativo"].max())
        result["intensidad"] = (
            result["contraste_relativo"] / maximum * 100 if maximum else 0.0
        )
    return result
