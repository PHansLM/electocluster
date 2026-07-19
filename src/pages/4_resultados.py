"""
Pagina 4 - Resultados y reportes.

Lista ejecuciones persistidas, muestra metricas, perfiles y visualizaciones, y
genera reportes comparativos para el Capitulo 8.
"""

import sys
from html import escape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.evaluation.execution import load_processed_dataset
from src.evaluation.report_generator import ReportGenerator
from src.evaluation.results_manager import RESULTS_PATH, ResultsManager
from src.visualization import (
    PROFILE_METADATA_COLUMNS,
    cluster_dimension_scores,
    cluster_distribution,
    cluster_profile_deviation,
    cluster_profile_distance_matrix,
    cluster_profiles,
    feature_display_name,
    interpret_feature_value,
    pca_projection,
    save_run_figures,
    top_distinctive_features,
)


st.set_page_config(page_title="Resultados - ElectoCluster", layout="wide")

st.title("Resultados y Reportes")
st.markdown(
    "Explora ejecuciones guardadas, compara metricas y exporta evidencia "
    "para la redaccion del Capitulo 8."
)

rm = ResultsManager()
runs = rm.list_runs()

if not runs:
    st.info("Aun no hay ejecuciones guardadas. Ejecuta un algoritmo en la pagina de Ejecucion.")
    st.stop()

runs_df = pd.DataFrame(runs)
runs_df["timestamp"] = pd.to_datetime(runs_df["timestamp"], errors="coerce")


def _format_metric(value, precision: int):
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def _format_run_label(runs_df: pd.DataFrame, run_id: str) -> str:
    row = runs_df[runs_df["run_id"] == run_id].iloc[0]
    timestamp = row["timestamp"]
    stamp = timestamp.strftime("%d/%m/%Y %H:%M") if pd.notna(timestamp) else "sin fecha"
    return f"{row['algorithm']} - {stamp} - {run_id}"


METRIC_SPECS = {
    "silhouette": {
        "label": "Silhouette",
        "precision": 4,
        "direction": "mayor",
        "description": "Mayor es mejor. Rango teorico aproximado: -1 a 1.",
    },
    "davies_bouldin": {
        "label": "Davies-Bouldin",
        "precision": 4,
        "direction": "menor",
        "description": "Menor es mejor. Evalua separacion y compacidad.",
    },
    "calinski_harabasz": {
        "label": "Calinski-Harabasz",
        "precision": 2,
        "direction": "mayor",
        "description": "Mayor es mejor. Suele operar en una escala mucho mas amplia.",
    },
}


def _comparison_label(row: pd.Series) -> str:
    timestamp = row["timestamp"]
    stamp = timestamp.strftime("%d/%m %H:%M") if pd.notna(timestamp) else "sin fecha"
    return f"{row['algorithm']} | {stamp}"


def _format_profile_raw(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return f"{numeric:.4f}"


def _format_profile_metadata(column: str, value) -> str:
    if column == "percentage":
        return f"{float(value):.2f}%"
    if column in {"cluster", "size"}:
        return str(int(value))
    return str(value)


def _render_profiles_table(profiles: pd.DataFrame):
    columns = profiles.columns.tolist()
    header_cells = []
    for column in columns:
        label = column if column in PROFILE_METADATA_COLUMNS else feature_display_name(column)
        header_cells.append(f"<th>{escape(str(label))}</th>")

    body_rows = []
    for _, row in profiles.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if column in PROFILE_METADATA_COLUMNS:
                cells.append(f"<td class='profile-meta'>{escape(_format_profile_metadata(column, value))}</td>")
                continue

            interpretation = interpret_feature_value(column, value)
            raw_value = _format_profile_raw(interpretation.raw_value)
            cells.append(
                "<td>"
                f"<div class='profile-meaning'>{escape(interpretation.text)}</div>"
                f"<div class='profile-raw'>({escape(raw_value)})</div>"
                "</td>"
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    html = """
    <div class="profile-table-wrap">
      <table class="profile-table">
        <thead>
          <tr>{headers}</tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>
    <style>
      .profile-table-wrap {{
        overflow-x: auto;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 8px;
      }}
      .profile-table {{
        width: max-content;
        min-width: 100%;
        border-collapse: collapse;
        font-size: 0.84rem;
      }}
      .profile-table th {{
        position: sticky;
        top: 0;
        text-align: left;
        padding: 9px 10px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.25);
        background: rgba(148, 163, 184, 0.10);
        color: #cbd5e1;
        white-space: nowrap;
      }}
      .profile-table td {{
        min-width: 145px;
        max-width: 230px;
        vertical-align: top;
        padding: 9px 10px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.14);
        border-right: 1px solid rgba(148, 163, 184, 0.10);
      }}
      .profile-table tbody tr:last-child td {{
        border-bottom: none;
      }}
      .profile-meta {{
        min-width: 70px !important;
        font-weight: 800;
        color: #f8fafc;
      }}
      .profile-meaning {{
        color: #f8fafc;
        font-weight: 700;
        line-height: 1.25;
      }}
      .profile-raw {{
        margin-top: 0.25rem;
        color: rgba(203, 213, 225, 0.62);
        font-size: 0.78rem;
        line-height: 1.2;
      }}
    </style>
    """.format(headers="".join(header_cells), rows="".join(body_rows))
    st.markdown(html, unsafe_allow_html=True)


def _top_features_display(top_features: pd.DataFrame) -> pd.DataFrame:
    if top_features.empty:
        return top_features
    display = top_features[
        [
            "cluster",
            "rank",
            "feature_name",
            "cluster_interpretation",
            "global_interpretation",
            "deviation",
        ]
    ].copy()
    display = display.rename(columns={
        "cluster": "cluster",
        "rank": "orden",
        "feature_name": "variable",
        "cluster_interpretation": "perfil del cluster",
        "global_interpretation": "promedio global",
        "deviation": "diferencia",
    })
    display["diferencia"] = display["diferencia"].round(4)
    return display


def _dimension_scores_display(dimension_scores: pd.DataFrame) -> pd.DataFrame:
    if dimension_scores.empty:
        return dimension_scores
    display = dimension_scores[
        ["cluster", "dimension", "cluster_value", "global_value", "deviation", "features"]
    ].copy()
    display = display.rename(columns={
        "dimension": "dimension",
        "cluster_value": "valor cluster",
        "global_value": "valor global",
        "deviation": "diferencia",
        "features": "variables",
    })
    for column in ["valor cluster", "valor global", "diferencia"]:
        display[column] = display[column].round(4)
    return display


st.markdown("### Ejecuciones guardadas")
st.dataframe(
    runs_df.sort_values("timestamp", ascending=False),
    use_container_width=True,
    hide_index=True,
)

run_options = runs_df["run_id"].tolist()
default_run = st.session_state.get("last_run_id")
default_index = run_options.index(default_run) if default_run in run_options else 0

selected_run_id = st.selectbox(
    "Selecciona una ejecucion",
    options=run_options,
    index=default_index,
    format_func=lambda run_id: _format_run_label(runs_df, run_id),
)

run = rm.load_run(selected_run_id)
labels = rm.load_assignments(selected_run_id)
df = load_processed_dataset()

if len(labels) != len(df):
    st.error(
        "Las asignaciones no tienen el mismo numero de registros que el dataset procesado. "
        "Reejecuta el experimento con el dataset actual."
    )
    st.stop()

st.markdown("### Detalle del run")
col1, col2, col3, col4, col5 = st.columns(5)
metrics = run.get("metrics", {})
col1.metric("Algoritmo", run.get("algorithm"))
col2.metric("Clusters", run.get("n_clusters"))
col3.metric("Ruido", run.get("n_noise"))
col4.metric("Silhouette", _format_metric(metrics.get("silhouette"), 4))
col5.metric("Davies-Bouldin", _format_metric(metrics.get("davies_bouldin"), 4))

with st.expander("Parametros, pesos y metadata", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Parametros**")
        st.json(run.get("params", {}))
        st.markdown("**Metadata**")
        st.json(run.get("metadata", {}))
    with c2:
        st.markdown("**Snapshot de pesos**")
        st.json(run.get("weights_snapshot", {}))

distribution = cluster_distribution(labels)
profiles = cluster_profiles(df, labels)
projection = pca_projection(df, labels)
profile_deviations = cluster_profile_deviation(df, labels)
top_features = top_distinctive_features(df, labels, top_n=5)
profile_distances = cluster_profile_distance_matrix(df, labels)
dimension_scores = cluster_dimension_scores(df, labels)

tab_dist, tab_profiles, tab_deviation, tab_distances, tab_dimensions, tab_projection, tab_compare = st.tabs(
    [
        "Distribucion",
        "Perfiles",
        "Diferencias",
        "Distancias",
        "Dimensiones",
        "PCA 2D",
        "Comparacion",
    ]
)

with tab_dist:
    fig_dist = px.bar(
        distribution,
        x="cluster",
        y="size",
        text="percentage",
        labels={"cluster": "Cluster", "size": "Registros"},
        title="Distribucion de registros por cluster",
    )
    fig_dist.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_dist, use_container_width=True)
    st.dataframe(distribution, use_container_width=True, hide_index=True)

with tab_projection:
    st.markdown("#### Proyeccion de apoyo")
    st.caption(
        "Proyeccion de apoyo. En 28 variables, la interpretacion principal debe "
        "venir de perfiles, diferencias y distancias entre perfiles."
    )
    fig_projection = px.scatter(
        projection,
        x="PC1",
        y="PC2",
        color="cluster",
        hover_data=["cluster_id"],
        title="Proyeccion PCA 2D de clusters",
    )
    st.plotly_chart(fig_projection, use_container_width=True)

with tab_profiles:
    if profiles.empty:
        st.warning("No hay perfiles disponibles para este run.")
    else:
        _render_profiles_table(profiles)
        csv = profiles.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar perfiles CSV",
            data=csv,
            file_name=f"{selected_run_id}_perfiles.csv",
            mime="text/csv",
        )

    if st.button(
        "Guardar figuras exportables",
        icon=":material/save:",
        use_container_width=True,
    ):
        paths = save_run_figures(
            df=df,
            labels=labels,
            run_id=selected_run_id,
            output_dir=RESULTS_PATH / "figures",
        )
        st.success("Figuras guardadas en results/figures.")
        st.json(paths)

with tab_deviation:
    st.caption(
        "Mide cuanto se aleja cada cluster del promedio global en la escala procesada. "
        "Para variables nominales, interpreta el color junto con la tabla de perfiles."
    )
    if profile_deviations.empty:
        st.warning("No hay diferencias de perfil disponibles para este run.")
    else:
        feature_scores = (
            profile_deviations.groupby("feature_name")["abs_deviation"]
            .max()
            .sort_values(ascending=False)
        )
        max_features = len(feature_scores)
        default_features = min(12, max_features)
        n_features = st.slider(
            "Variables a mostrar en el heatmap",
            min_value=1,
            max_value=max_features,
            value=default_features,
        )
        selected_features = feature_scores.head(n_features).index.tolist()
        heatmap_data = profile_deviations[
            profile_deviations["feature_name"].isin(selected_features)
        ].pivot_table(
            index="feature_name",
            columns="cluster",
            values="deviation",
            aggfunc="mean",
        )
        st.markdown("#### Diferencia del perfil respecto al promedio global")
        fig_deviation = px.imshow(
            heatmap_data,
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            aspect="auto",
            labels={
                "x": "Cluster",
                "y": "Variable",
                "color": "Diferencia",
            },
            title="Diferencia del perfil respecto al promedio global",
        )
        st.plotly_chart(fig_deviation, use_container_width=True)

        st.markdown("#### Variables mas distintivas por cluster")
        st.dataframe(
            _top_features_display(top_features),
            use_container_width=True,
            hide_index=True,
        )

with tab_distances:
    st.caption(
        "Compara perfiles promedio entre clusters. Valores menores indican perfiles "
        "demograficos mas parecidos."
    )
    if profile_distances.empty:
        st.warning("No hay distancias entre perfiles disponibles para este run.")
    else:
        st.markdown("#### Distancia euclidiana entre perfiles de clusters")
        fig_distances = px.imshow(
            profile_distances,
            color_continuous_scale="Blues",
            aspect="auto",
            labels={
                "x": "Cluster",
                "y": "Cluster",
                "color": "Distancia",
            },
            title="Distancia euclidiana entre perfiles de clusters",
        )
        st.plotly_chart(fig_distances, use_container_width=True)
        st.dataframe(profile_distances.round(4), use_container_width=True)

with tab_dimensions:
    st.caption(
        "Agrupa variables en dimensiones conceptuales para lectura global. Es una "
        "vista explicativa, no una metrica de validacion."
    )
    if dimension_scores.empty:
        st.warning("No hay dimensiones agregadas disponibles para este run.")
    else:
        cluster_options = sorted(
            dimension_scores["cluster"].unique().tolist(),
            key=lambda value: int(value),
        )
        selected_clusters = st.multiselect(
            "Clusters a mostrar en radar",
            options=cluster_options,
            default=cluster_options[: min(4, len(cluster_options))],
        )
        if selected_clusters:
            radar_data = dimension_scores[
                dimension_scores["cluster"].isin(selected_clusters)
            ]
            st.markdown("#### Radar de dimensiones agregadas")
            fig_radar = px.line_polar(
                radar_data,
                r="cluster_value",
                theta="dimension",
                color="cluster",
                line_close=True,
                range_r=[0, 1],
                labels={
                    "cluster_value": "Valor agregado",
                    "dimension": "Dimension",
                    "cluster": "Cluster",
                },
                title="Radar de dimensiones agregadas",
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.warning("Selecciona al menos un cluster para el radar.", icon=":material/warning:")

        dimension_heatmap = dimension_scores.pivot_table(
            index="dimension",
            columns="cluster",
            values="deviation",
            aggfunc="mean",
        )
        st.markdown("#### Diferencia de dimensiones frente al promedio global")
        fig_dimensions = px.imshow(
            dimension_heatmap,
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            aspect="auto",
            labels={
                "x": "Cluster",
                "y": "Dimension",
                "color": "Diferencia",
            },
            title="Diferencia de dimensiones frente al promedio global",
        )
        st.plotly_chart(fig_dimensions, use_container_width=True)
        st.dataframe(
            _dimension_scores_display(dimension_scores),
            use_container_width=True,
            hide_index=True,
        )

with tab_compare:
    compare_ids = st.multiselect(
        "Selecciona ejecuciones para comparar",
        options=run_options,
        default=run_options[: min(3, len(run_options))],
        format_func=lambda run_id: _format_run_label(runs_df, run_id),
    )

    if compare_ids:
        comparison_df = runs_df[runs_df["run_id"].isin(compare_ids)].copy()
        comparison_df["run_label"] = comparison_df.apply(_comparison_label, axis=1)

        metric_tabs = st.tabs([spec["label"] for spec in METRIC_SPECS.values()])
        for tab, (metric_key, spec) in zip(metric_tabs, METRIC_SPECS.items()):
            with tab:
                metric_df = comparison_df[
                    ["run_id", "algorithm", "run_label", metric_key]
                ].copy()
                metric_df = metric_df.dropna(subset=[metric_key])

                if metric_df.empty:
                    st.warning(
                        f"No hay valores validos para {spec['label']} en los runs seleccionados.",
                        icon=":material/warning:",
                    )
                    continue

                ascending = spec["direction"] == "menor"
                metric_df = metric_df.sort_values(metric_key, ascending=ascending)
                best_row = metric_df.iloc[0]

                col_best, col_note = st.columns([1, 2])
                col_best.metric(
                    f"Mejor {spec['label']}",
                    _format_metric(best_row[metric_key], spec["precision"]),
                    best_row["algorithm"],
                )
                col_note.caption(spec["description"])

                fig_metric = px.bar(
                    metric_df,
                    x="run_label",
                    y=metric_key,
                    color="algorithm",
                    hover_data=["run_id"],
                    text=metric_key,
                    labels={
                        "run_label": "Ejecucion",
                        metric_key: spec["label"],
                        "algorithm": "Algoritmo",
                    },
                    title=f"Comparacion individual: {spec['label']}",
                )
                fig_metric.update_traces(
                    texttemplate=f"%{{text:.{spec['precision']}f}}",
                    textposition="outside",
                )
                fig_metric.update_layout(xaxis_tickangle=-25)
                st.plotly_chart(fig_metric, use_container_width=True)

        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    if len(compare_ids) >= 2:
        if st.button(
            "Generar reporte comparativo",
            type="primary",
            icon=":material/description:",
        ):
            report = ReportGenerator(rm).compare_runs(compare_ids)
            st.success(
                f"Reporte generado: {report['report_name']}",
                icon=":material/check_circle:",
            )
            st.json(report)
    else:
        st.caption("Selecciona al menos dos ejecuciones para generar un reporte comparativo.")
