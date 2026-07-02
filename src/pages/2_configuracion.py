"""
Pagina 2 - Configuracion de algoritmo.

Seleccion de algoritmo, ajuste manual de parametros y busqueda automatica
controlada para la iteracion 5.
"""

import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.evaluation.execution import default_params, load_processed_dataset
from src.evaluation.parameter_optimizer import (
    optimize_wdbscan_params,
    optimize_whierarchical_params,
    optimize_wkmedoids_params,
)
from src.weighting.weight_manager import WeightManager


st.set_page_config(page_title="Configuración · ElectoCluster", layout="wide")

ALGORITHMS = ["WKMedoids", "W-Hierarchical Clustering", "W-DBSCAN"]

st.title("Configuración de Algoritmos")
st.markdown(
    "Edita y guarda la configuración activa de cada algoritmo. La ejecución se "
    "realiza después desde la página de Ejecución, donde puedes correr una o varias "
    "configuraciones."
)


def _params_store() -> dict:
    store = st.session_state.get("params_by_algorithm")
    if not isinstance(store, dict):
        store = {}
    for algo in ALGORITHMS:
        defaults = default_params(algo)
        saved = store.get(algo, {})
        store[algo] = {**defaults, **{key: value for key, value in saved.items() if key in defaults}}
    st.session_state["params_by_algorithm"] = store
    return store


def _current_params_for(algorithm: str) -> dict:
    defaults = default_params(algorithm)
    session_params = _params_store().get(algorithm, {})
    return {**defaults, **{key: value for key, value in session_params.items() if key in defaults}}


def _save_active_params(algorithm: str, params: dict):
    store = _params_store()
    defaults = default_params(algorithm)
    clean_params = {**defaults, **{key: value for key, value in params.items() if key in defaults}}
    store[algorithm] = clean_params
    st.session_state["params_by_algorithm"] = store
    st.session_state["params"] = clean_params


def _display_search_result(result):
    result_dict = asdict(result)
    st.session_state["last_param_search"] = result_dict
    st.session_state["last_param_search_algorithm"] = result.algorithm
    _save_active_params(result.algorithm, result.best_params)


def _format_results_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ["silhouette", "davies_bouldin", "calinski_harabasz", "eps", "elbow_eps"]:
        if col in df.columns:
            df[col] = df[col].astype(float).round(4)
    for col in ["pca_explained_variance", "noise_percentage"]:
        if col in df.columns:
            df[col] = df[col].astype(float).round(3)
    return df


st.markdown("### Algoritmo a configurar")

algorithm = st.radio(
    label="Selecciona la configuración que quieres editar:",
    options=ALGORITHMS,
    index=st.session_state.get("algoritmo_idx", 0),
    horizontal=True,
)
st.session_state["algoritmo"] = algorithm
st.session_state["algoritmo_idx"] = ALGORITHMS.index(algorithm)

st.markdown("---")
st.markdown("### Parámetros manuales")

params = _current_params_for(algorithm)

if algorithm == "WKMedoids":
    st.markdown(
        "**WKMedoids** usa PAM con distancia euclidiana ponderada precomputada. "
        "El parámetro principal es el número de clusters `k`."
    )
    col1, col2 = st.columns(2)
    with col1:
        params["n_clusters"] = st.number_input(
            "Número de clusters (k)",
            min_value=2,
            max_value=20,
            value=int(params.get("n_clusters", 13)),
        )
    with col2:
        params["random_state"] = st.number_input(
            "Semilla aleatoria (random_state)",
            min_value=0,
            max_value=999,
            value=int(params.get("random_state", 42)),
        )
    st.info(
        "El k óptimo bajo ponderación puede diferir del obtenido con el algoritmo "
        "tradicional. Usa el cálculo automático como orientación y valida el run final.",
        icon=":material/info:",
    )

elif algorithm == "W-Hierarchical Clustering":
    st.markdown(
        "**W-Hierarchical Clustering** construye un dendrograma en el espacio ponderado. "
        "El linkage `ward` no se usa porque no es compatible con matrices de distancia "
        "precomputadas en scikit-learn."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        params["n_clusters"] = st.number_input(
            "Número de clusters",
            min_value=2,
            max_value=15,
            value=int(params.get("n_clusters", 2)),
        )
    with col2:
        linkages = ["complete", "average", "single"]
        params["linkage"] = st.selectbox(
            "Método de linkage",
            options=linkages,
            index=linkages.index(params.get("linkage", "complete")),
        )
    with col3:
        st.metric("Métrica de distancia", "Euclidiana ponderada")
    st.info(
        "La configuración canónica actual es `complete` con k=2. "
        "El cálculo automático permite contrastarla contra otros cortes.",
        icon=":material/info:",
    )

else:
    st.markdown(
        "**W-DBSCAN** determina clusters por densidad sobre datos ponderados y reducidos "
        "por PCA. Sus parámetros críticos son `eps`, `min_samples` y componentes PCA."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        params["eps"] = st.number_input(
            "Radio de vecindad (eps)",
            min_value=0.1,
            max_value=5.0,
            step=0.05,
            value=float(params.get("eps", 0.606)),
            format="%.3f",
        )
    with col2:
        params["min_samples"] = st.number_input(
            "Mínimo de puntos núcleo",
            min_value=2,
            max_value=100,
            value=int(params.get("min_samples", 34)),
        )
    with col3:
        params["pca_components"] = st.number_input(
            "Componentes PCA previos",
            min_value=2,
            max_value=28,
            value=int(params.get("pca_components", 17)),
        )
    st.warning(
        "W-DBSCAN es sensible a dimensionalidad y ruido. La búsqueda automática aplica "
        "pesos antes de PCA y calcula métricas excluyendo puntos de ruido.",
        icon=":material/warning:",
    )

_save_active_params(algorithm, params)

st.markdown("---")
st.markdown("### Asistente de parámetros")
st.caption(
    "La configuración manual queda guardada de inmediato. Abre el asistente solo si "
    "quieres calcular una recomendación automática para este algoritmo."
)

show_param_search = st.session_state.get("show_param_search", False)
toggle_label = (
    "Ocultar cálculo automático"
    if show_param_search
    else "Abrir cálculo automático"
)
toggle_icon = ":material/close:" if show_param_search else ":material/tune:"
if st.button(toggle_label, type="secondary", icon=toggle_icon):
    st.session_state["show_param_search"] = not show_param_search
    st.rerun()

if st.session_state.get("show_param_search", False):
    with st.container(border=True):
        panel_intro, panel_dataset = st.columns([1.6, 1])
        with panel_intro:
            st.markdown("#### Recomendación automática")
            st.caption(
                "Explora parámetros sobre una muestra reproducible del dataset procesado. "
                "Úsalo como apoyo para guardar una configuración activa, no como ejecución final."
            )

        try:
            df_processed = load_processed_dataset()
            dataset_ready = True
            dataset_error = ""
        except Exception as exc:
            df_processed = None
            dataset_ready = False
            dataset_error = str(exc)

        with panel_dataset:
            if dataset_ready:
                st.metric("Registros", f"{len(df_processed):,}")
                st.metric("Variables", len(df_processed.columns))
            else:
                st.error(dataset_error, icon=":material/error:")

        st.markdown("##### Alcance de búsqueda")
        search_cols = st.columns([1, 1, 1])
        with search_cols[0]:
            sample_default = 500 if algorithm == "W-DBSCAN" else 300
            sample_size = st.number_input(
                "Tamaño de muestra",
                min_value=100,
                max_value=1000,
                value=sample_default,
                step=50,
                help="Usa una muestra menor si el equipo tiene poca memoria disponible.",
            )
        with search_cols[1]:
            random_state = st.number_input(
                "Semilla de búsqueda",
                min_value=0,
                max_value=999,
                value=int(params.get("random_state", 42)),
            )

        if algorithm in ["WKMedoids", "W-Hierarchical Clustering"]:
            with search_cols[2]:
                k_min = st.number_input("k mínimo", min_value=2, max_value=20, value=2)
            k_max = st.slider("k máximo a explorar", min_value=int(k_min), max_value=20, value=8)
            k_values = list(range(int(k_min), int(k_max) + 1))
            variance_target = None
        else:
            with search_cols[2]:
                variance_target = st.slider(
                    "Varianza PCA objetivo",
                    min_value=0.60,
                    max_value=0.95,
                    value=0.85,
                    step=0.05,
                )
            k_values = []

        if algorithm == "W-Hierarchical Clustering":
            linkages_to_search = st.multiselect(
                "Linkages a explorar",
                options=["complete", "average", "single"],
                default=["complete", "average"],
            )
        else:
            linkages_to_search = []

        run_search = st.button(
            "Calcular recomendación",
            type="primary",
            icon=":material/play_arrow:",
            disabled=not dataset_ready,
        )

        if run_search and dataset_ready:
            with st.spinner("Calculando recomendación de parámetros..."):
                wm = WeightManager()
                if algorithm == "WKMedoids":
                    search_result = optimize_wkmedoids_params(
                        df_processed,
                        weight_manager=wm,
                        k_values=k_values,
                        sample_size=int(sample_size),
                        random_state=int(random_state),
                    )
                elif algorithm == "W-Hierarchical Clustering":
                    search_result = optimize_whierarchical_params(
                        df_processed,
                        weight_manager=wm,
                        k_values=k_values,
                        linkages=linkages_to_search or ["complete"],
                        sample_size=int(sample_size),
                        random_state=int(random_state),
                    )
                else:
                    search_result = optimize_wdbscan_params(
                        df_processed,
                        weight_manager=wm,
                        sample_size=int(sample_size),
                        random_state=int(random_state),
                        variance_target=float(variance_target),
                    )

            _display_search_result(search_result)
            st.rerun()

        last_result = st.session_state.get("last_param_search")
        if last_result and st.session_state.get("last_param_search_algorithm") == algorithm:
            st.markdown("##### Última recomendación")
            col_best, col_criterion, col_sample = st.columns([1.2, 1.4, 1])
            col_best.json(last_result["best_params"])
            col_criterion.metric("Criterio", last_result["criterion"])
            col_sample.metric("Muestra evaluada", last_result["sample_size"])

            for note in last_result.get("notes", []):
                st.caption(note)

            result_table = _format_results_table(last_result.get("rows", []))
            if result_table.empty:
                st.warning(
                    "La búsqueda no produjo métricas válidas. Ajusta muestra o rangos.",
                    icon=":material/warning:",
                )
            else:
                st.dataframe(result_table, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### Resumen de configuración activa")
col_alg, col_params = st.columns(2)
with col_alg:
    st.metric("Configuración en edición", algorithm)
with col_params:
    st.json(st.session_state["params"])

if st.button("Guardar configuración activa", type="primary", icon=":material/check:"):
    st.success(
        f"Configuración guardada: **{algorithm}** con parámetros `{st.session_state['params']}`. "
        "Navega a **Ejecución** para correr una o varias configuraciones.",
        icon=":material/check_circle:",
    )
