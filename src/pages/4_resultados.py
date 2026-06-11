"""
Pagina 4 - Resultados y reportes.

Lista ejecuciones persistidas, muestra metricas, perfiles y visualizaciones, y
genera reportes comparativos para el Capitulo 8.
"""

import sys
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
    cluster_distribution,
    cluster_profiles,
    pca_projection,
    save_run_figures,
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

tab_dist, tab_projection, tab_profiles, tab_compare = st.tabs(
    ["Distribucion", "PCA 2D", "Perfiles", "Comparacion"]
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
        st.dataframe(profiles, use_container_width=True, hide_index=True)
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
