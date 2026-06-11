"""
Pagina 3 - Ejecucion de clustering.

Ejecuta el algoritmo configurado, calcula metricas y persiste el run para
analisis posterior en la pagina de Resultados.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.evaluation.execution import (
    canonical_experiments,
    default_params,
    load_processed_dataset,
    run_clustering_experiment,
)
from src.visualization import cluster_distribution
from src.weighting.weight_manager import WeightManager


st.set_page_config(page_title="Ejecucion - ElectoCluster", layout="wide")

st.title("Ejecucion de Clustering")
st.markdown(
    "Ejecuta el algoritmo ponderado seleccionado, calcula metricas de validacion "
    "interna y guarda el resultado para comparacion posterior."
)


@st.cache_data(ttl=10)
def get_dataset_preview():
    df = load_processed_dataset()
    return df.shape, df.head(10), df.columns.tolist()


try:
    shape, preview, columns = get_dataset_preview()
    st.success(f"Dataset procesado disponible: {shape[0]:,} registros x {shape[1]} variables.")
except Exception as exc:
    st.error(str(exc))
    st.stop()

algorithm = st.session_state.get("algoritmo", "WKMedoids")
default = default_params(algorithm)
session_params = st.session_state.get("params", {})
params = {**default, **{k: v for k, v in session_params.items() if k in default}}

st.markdown("### Configuracion activa")
col_alg, col_params, col_data = st.columns([1.2, 1.4, 1])
with col_alg:
    st.metric("Algoritmo", algorithm)
with col_params:
    st.json(params)
with col_data:
    st.metric("Variables", len(columns))
    st.metric("Registros", f"{shape[0]:,}")

with st.expander("Vista previa del dataset procesado", expanded=False):
    st.dataframe(preview, use_container_width=True, hide_index=True)

st.markdown("---")


def _format_metric(value, precision: int):
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def _render_result_summary(metrics: dict, labels, metadata: dict):
    st.markdown("### Resultado de la ejecucion")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Silhouette", _format_metric(metrics.get("silhouette"), 4))
    col2.metric("Davies-Bouldin", _format_metric(metrics.get("davies_bouldin"), 4))
    col3.metric("Calinski-Harabasz", _format_metric(metrics.get("calinski_harabasz"), 2))
    col4.metric("Evaluados", metadata.get("n_evaluated", len(labels)))

    if metadata.get("excluded_noise"):
        st.caption(
            f"Metricas calculadas sin ruido. Ruido: {metadata.get('n_noise', 0)} "
            f"({metadata.get('noise_percentage', 0):.1f}%)."
        )

    st.dataframe(
        cluster_distribution(labels),
        use_container_width=True,
        hide_index=True,
    )


col_run, col_suite = st.columns([1, 1])

with col_run:
    run_clicked = st.button(
        "Ejecutar configuracion activa",
        type="primary",
        icon=":material/play_arrow:",
        use_container_width=True,
    )

with col_suite:
    suite_clicked = st.button(
        "Ejecutar configuraciones canonicas",
        icon=":material/view_list:",
        use_container_width=True,
        help="Ejecuta WKMedoids, W-Hierarchical y W-DBSCAN con los parametros finales de iteracion 4.",
    )

if run_clicked:
    with st.spinner("Ejecutando clustering y guardando resultados..."):
        wm = WeightManager()
        result = run_clustering_experiment(
            algorithm=algorithm,
            params=params,
            weight_manager=wm,
            persist=True,
        )
        st.session_state["last_run_id"] = result.run_id

    st.success(f"Run guardado: {result.run_id}")
    _render_result_summary(result.metrics, result.labels, result.metadata)

if suite_clicked:
    rows = []
    progress = st.progress(0)
    experiments = canonical_experiments()
    for i, (suite_algorithm, suite_params) in enumerate(experiments, start=1):
        with st.spinner(f"Ejecutando {suite_algorithm}..."):
            result = run_clustering_experiment(
                algorithm=suite_algorithm,
                params=suite_params,
                weight_manager=WeightManager(),
                persist=True,
            )
        rows.append({
            "run_id": result.run_id,
            "algorithm": suite_algorithm,
            "silhouette": result.metrics.get("silhouette"),
            "davies_bouldin": result.metrics.get("davies_bouldin"),
            "calinski_harabasz": result.metrics.get("calinski_harabasz"),
            "n_noise": int((result.labels == -1).sum()),
        })
        progress.progress(i / len(experiments))

    st.success("Suite canonica completada.")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
