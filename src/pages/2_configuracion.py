"""
Pagina 2 - Configuracion de algoritmo.

Seleccion de algoritmo, ajuste manual de parametros y busqueda automatica
controlada para la iteracion 5.
"""

import sys
from dataclasses import asdict
from math import isfinite
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
from src.ui import apply_app_shell
from src.ui_feedback import (
    handle_configuration_navigation,
    mark_parameter_change,
    queue_feedback,
    show_feedback_dialog,
)
from src.weighting.weight_manager import WeightManager


st.set_page_config(page_title="Configuración · ElectoCluster", layout="wide")
apply_app_shell("Configuración")

ALGORITHMS = ["WKMedoids", "W-Hierarchical Clustering", "W-DBSCAN"]

st.title("Configuración de algoritmos")
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


def _current_draft_params_for(algorithm: str) -> dict:
    drafts = st.session_state.get("params_drafts_by_algorithm", {})
    active = _current_params_for(algorithm)
    draft = drafts.get(algorithm, {})
    return {**active, **{key: value for key, value in draft.items() if key in active}}


def _save_draft_params(algorithm: str, params: dict, *, reset_widgets: bool = False):
    """Conserva la edicion sin habilitarla para la ejecucion."""
    drafts = st.session_state.get("params_drafts_by_algorithm", {})
    defaults = default_params(algorithm)
    drafts[algorithm] = {
        **defaults,
        **{key: value for key, value in params.items() if key in defaults},
    }
    st.session_state["params_drafts_by_algorithm"] = drafts
    if reset_widgets:
        versions = st.session_state.get("params_draft_versions", {})
        versions[algorithm] = versions.get(algorithm, 0) + 1
        st.session_state["params_draft_versions"] = versions


def _apply_assistant_draft(algorithm: str, params: dict, title: str, message: str):
    """Evita que una recomendación fuera de los controles rompa el siguiente rerun."""
    ranges = {
        "WKMedoids": {"n_clusters": (2, 20), "random_state": (0, 999)},
        "W-Hierarchical Clustering": {"n_clusters": (2, 15)},
        "W-DBSCAN": {"eps": (0.1, 5.0), "min_samples": (2, 100), "pca_components": (2, 28)},
    }
    try:
        for key, (minimum, maximum) in ranges[algorithm].items():
            if key in params and not minimum <= params[key] <= maximum:
                raise ValueError(f"{key}={params[key]} está fuera del rango {minimum}–{maximum}.")
        _save_draft_params(algorithm, params, reset_widgets=True)
        queue_feedback("success", title, message)
    except Exception as exc:
        queue_feedback(
            "error", "No se pudieron aplicar los parámetros",
            "Revisa el detalle y ajusta el rango de búsqueda si es necesario. "
            "Los valores en edición y la configuración activa se conservan.",
            details=f"{type(exc).__name__}: {exc}",
        )
    st.rerun()


def _display_search_result(result):
    result_dict = asdict(result)
    st.session_state["last_param_search"] = result_dict
    st.session_state["last_param_search_algorithm"] = result.algorithm


def _parameter_comparison(recommended: dict, canonical: dict) -> pd.DataFrame:
    """Compara una recomendacion exploratoria con la referencia canonica."""
    rows = []
    for key in canonical:
        recommended_value = recommended.get(key)
        canonical_value = canonical[key]
        rows.append({
            "Parámetro": key,
            "Recomendación": str(recommended_value),
            "Canónica": str(canonical_value),
            "Estado": "Coincide" if recommended_value == canonical_value else "Difiere",
        })
    return pd.DataFrame(rows)


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

confirmed_algorithm = st.session_state.pop("confirmed_configuration_algorithm", None)
if confirmed_algorithm:
    st.session_state["config_algorithm_selector"] = confirmed_algorithm
st.session_state.setdefault(
    "config_algorithm_selector", st.session_state.get("algoritmo", "WKMedoids")
)


def _request_algorithm_change():
    current = st.session_state.get("algoritmo", "WKMedoids")
    requested = st.session_state["config_algorithm_selector"]
    if current != requested:
        st.session_state["configuration_navigation"] = {
            "kind": "algorithm", "value": requested,
        }
        st.session_state["config_algorithm_selector"] = current


algorithm = st.radio(
    label="Selecciona la configuración que quieres editar:",
    options=ALGORITHMS,
    key="config_algorithm_selector",
    on_change=_request_algorithm_change,
    horizontal=True,
)
st.session_state["algoritmo"] = algorithm
st.session_state["algoritmo_idx"] = ALGORITHMS.index(algorithm)
st.session_state["params"] = _current_params_for(algorithm)

st.markdown("### Parámetros manuales")
st.caption(
    "Los cambios quedan pendientes hasta que pulses Guardar configuración activa. "
    "Cada algoritmo conserva por separado sus valores en edición y los guardados."
)

params = _current_draft_params_for(algorithm)
active_params = _current_params_for(algorithm)
draft_version = st.session_state.get("params_draft_versions", {}).get(algorithm, 0)
widget_prefix = f"params_draft_{algorithm}_{draft_version}"

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
            key=f"{widget_prefix}_n_clusters",
        )
        mark_parameter_change("n_clusters", params["n_clusters"], active_params)
    with col2:
        params["random_state"] = st.number_input(
            "Semilla aleatoria (random_state)",
            min_value=0,
            max_value=999,
            value=int(params.get("random_state", 42)),
            key=f"{widget_prefix}_random_state",
        )
        mark_parameter_change("random_state", params["random_state"], active_params)
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
            key=f"{widget_prefix}_n_clusters",
        )
        mark_parameter_change("n_clusters", params["n_clusters"], active_params)
    with col2:
        linkages = ["complete", "average", "single"]
        params["linkage"] = st.selectbox(
            "Método de linkage",
            options=linkages,
            index=linkages.index(params.get("linkage", "complete")),
            key=f"{widget_prefix}_linkage",
        )
        mark_parameter_change("linkage", params["linkage"], active_params)
    with col3:
        st.markdown("**Métrica de distancia**")
        st.write("Euclidiana ponderada")
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
            key=f"{widget_prefix}_eps",
        )
        mark_parameter_change("eps", params["eps"], active_params)
    with col2:
        params["min_samples"] = st.number_input(
            "Mínimo de puntos núcleo",
            min_value=2,
            max_value=100,
            value=int(params.get("min_samples", 34)),
            key=f"{widget_prefix}_min_samples",
        )
        mark_parameter_change("min_samples", params["min_samples"], active_params)
    with col3:
        params["pca_components"] = st.number_input(
            "Componentes PCA previos",
            min_value=2,
            max_value=28,
            value=int(params.get("pca_components", 17)),
            key=f"{widget_prefix}_pca_components",
        )
        mark_parameter_change("pca_components", params["pca_components"], active_params)
    st.warning(
        "W-DBSCAN es sensible a dimensionalidad y ruido. La búsqueda automática aplica "
        "pesos antes de PCA y calcula métricas excluyendo puntos de ruido.",
        icon=":material/warning:",
    )

_save_draft_params(algorithm, params)

st.divider()

st.markdown("### Asistente de parámetros")
st.caption(
    "Abre el asistente si quieres calcular una recomendación exploratoria. "
    "Aplicarla actualiza los valores en edición; después debes guardar la configuración activa."
)

show_param_search = st.session_state.get("show_param_search", False)
toggle_label = (
    "Ocultar exploración rápida"
    if show_param_search
    else "Abrir exploración rápida"
)
toggle_icon = ":material/close:" if show_param_search else ":material/tune:"
if st.button(toggle_label, type="secondary", icon=toggle_icon):
    st.session_state["show_param_search"] = not show_param_search
    if not show_param_search:
        st.session_state.pop("assistant_dataset_status", None)
    st.rerun()

if st.session_state.get("show_param_search", False):
    with st.container(border=True, gap="small"):
        st.markdown("#### Exploración rápida")
        st.caption(
            "Explora parámetros sobre una muestra reproducible del dataset procesado. "
            "El resultado es orientativo y no reproduce la búsqueda exhaustiva con la que "
            "se definió la batería canónica."
        )

        canonical_params = default_params(algorithm)
        canonical_col, canonical_action = st.columns([1.5, 1])
        with canonical_col:
            st.markdown("**Referencia canónica validada**")
            st.json(canonical_params)
        with canonical_action:
            st.caption(
                "Restaura estos valores en la edición y guarda la configuración activa "
                "si quieres utilizarlos como fuente Activa en Ejecución."
            )
            if st.button(
                "Restaurar configuración canónica",
                icon=":material/restore:",
                key=f"restore_canonical_{algorithm}",
            ):
                _apply_assistant_draft(algorithm, canonical_params, "Valores canónicos restaurados",
                    f"Valores canónicos restaurados en la edición de {algorithm}. "
                    "Pulsa Guardar configuración activa para confirmarlos."
                )

        st.info(
            "Calcular una recomendación no cambiará la configuración activa. Podrás "
            "compararla, aplicarla a la edición y guardarla explícitamente después.",
            icon=":material/info:",
        )

        try:
            df_processed = load_processed_dataset()
            dataset_ready = True
            dataset_error = ""
        except Exception as exc:
            df_processed = None
            dataset_ready = False
            dataset_error = str(exc)

        dataset_status = "ready" if dataset_ready else dataset_error
        if not dataset_ready and st.session_state.get("assistant_dataset_status") != dataset_status:
            queue_feedback(
                "error", "Asistente no disponible",
                "No se pudo cargar el dataset procesado. Revisa Preprocesamiento antes de calcular una recomendación.",
                details=dataset_error,
            )
        st.session_state["assistant_dataset_status"] = dataset_status

        if dataset_ready:
            st.caption(
                f":material/database: Dataset disponible · "
                f"{len(df_processed):,} registros · {len(df_processed.columns)} variables"
            )
        else:
            st.error(dataset_error, icon=":material/error:")

        st.markdown("##### Alcance de búsqueda")
        search_cols = st.columns([1, 1, 1])
        with search_cols[0]:
            sample_default = 500 if algorithm == "W-DBSCAN" else 300
            sample_max = max(100, len(df_processed)) if dataset_ready else 1000
            sample_size = st.number_input(
                "Tamaño de muestra",
                min_value=100,
                max_value=sample_max,
                value=min(sample_default, sample_max),
                step=50,
                help=(
                    "Usa una muestra menor si el equipo tiene poca memoria. El dataset "
                    "completo ofrece resultados más comparables, pero tarda más."
                ),
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
                k_min = st.number_input("k mínimo", min_value=2, max_value=24, value=2)
            canonical_k = int(canonical_params["n_clusters"])
            k_max = st.slider(
                "k máximo a explorar",
                min_value=int(k_min),
                max_value=24,
                value=max(int(k_min), 8, canonical_k),
                help="El rango inicial incluye el k canónico para evitar excluirlo por omisión.",
            )
            k_values = list(range(int(k_min), int(k_max) + 1))
            variance_target = None
        else:
            with search_cols[2]:
                variance_target = st.slider(
                    "Varianza PCA objetivo",
                    min_value=0.60,
                    max_value=0.95,
                    value=0.90,
                    step=0.05,
                    help="La batería canónica se estableció con un objetivo de 0.90.",
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
            try:
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
                has_valid_metrics = any(
                    row.get("silhouette") is not None and isfinite(float(row["silhouette"]))
                    for row in search_result.rows
                )
                st.session_state["last_param_search"]["valid_recommendation"] = has_valid_metrics
                if has_valid_metrics:
                    queue_feedback(
                        "success", "Recomendación calculada",
                        f"La exploración de {algorithm} terminó sobre {search_result.sample_size} registros. "
                        "Revisa el resultado antes de aplicarlo a la edición y guardar la configuración activa.",
                    )
                else:
                    queue_feedback(
                        "error", "Sin recomendación válida",
                        "La búsqueda terminó sin métricas válidas para seleccionar parámetros. "
                        "Ajusta la muestra o los rangos y vuelve a calcular.",
                    )
            except Exception as exc:
                st.session_state.pop("last_param_search", None)
                st.session_state.pop("last_param_search_algorithm", None)
                queue_feedback(
                    "error", "No se pudo calcular la recomendación",
                    "El asistente encontró un error. Revisa el detalle y vuelve a intentarlo. "
                    "Los valores en edición y la configuración activa se conservan.",
                    details=f"{type(exc).__name__}: {exc}",
                )
            st.rerun()

        last_result = st.session_state.get("last_param_search")
        if last_result and st.session_state.get("last_param_search_algorithm") == algorithm:
            valid_recommendation = last_result.get("valid_recommendation", True)
            st.markdown("##### Última recomendación" if valid_recommendation else "##### Última búsqueda sin recomendación válida")
            col_best, col_criterion, col_sample = st.columns([1.2, 1.4, 1])
            if not valid_recommendation:
                col_best.caption("Valores de respaldo del cálculo; no constituyen una recomendación.")
            col_best.json(last_result["best_params"])
            with col_criterion:
                st.markdown("**Criterio**")
                st.write(last_result["criterion"])
            col_sample.metric("Muestra evaluada", last_result["sample_size"])

            comparison = _parameter_comparison(
                last_result["best_params"],
                canonical_params,
            )
            if not valid_recommendation:
                st.error("No hubo métricas válidas para recomendar parámetros. Ajusta la muestra o los rangos y vuelve a calcular.")
            elif (comparison["Estado"] == "Coincide").all():
                st.success(
                    "La recomendación coincide con la configuración canónica.",
                    icon=":material/check_circle:",
                )
            else:
                st.warning(
                    "La recomendación difiere de la referencia canónica. Esto es esperable "
                    "en una exploración por muestra; aplícala solo si quieres probar una "
                    "configuración alternativa.",
                    icon=":material/warning:",
                )
            if valid_recommendation:
                st.dataframe(comparison, width="stretch", hide_index=True)

            if st.button(
                "Aplicar esta recomendación",
                icon=":material/check:",
                key=f"apply_recommendation_{algorithm}",
                disabled=not valid_recommendation,
            ):
                _apply_assistant_draft(algorithm, last_result["best_params"], "Recomendación aplicada a la edición",
                    f"Recomendación aplicada a la edición de {algorithm}. "
                    "Pulsa Guardar configuración activa para confirmarla."
                )

            for note in last_result.get("notes", []):
                st.caption(note)

            result_table = _format_results_table(last_result.get("rows", []))
            if result_table.empty:
                st.warning(
                    "La búsqueda no produjo métricas válidas. Ajusta muestra o rangos.",
                    icon=":material/warning:",
                )
            else:
                st.dataframe(result_table, width="stretch", hide_index=True)

st.markdown("### Resumen de configuración")
active_params = _current_params_for(algorithm)
col_draft, col_active = st.columns(2)
with col_draft:
    st.markdown("**Valores en edición**")
    st.json(params)
with col_active:
    st.markdown("**Configuración activa guardada**")
    st.json(active_params)

if params != active_params:
    st.warning(
        "Hay cambios sin guardar. Ejecución utilizará la configuración activa guardada "
        "hasta que confirmes estos valores.",
        icon=":material/pending:",
    )

if st.button("Guardar configuración activa", type="primary", icon=":material/check:"):
    try:
        _save_active_params(algorithm, params)
        queue_feedback("success", "Configuración activa guardada",
            f"Configuración activa guardada: {algorithm} con parámetros {params}. "
            "Ya está disponible en Ejecución."
        )
    except Exception as exc:
        queue_feedback(
            "error", "No se pudo guardar la configuración",
            "No se pudo confirmar la configuración activa. Revisa el detalle y vuelve a intentarlo.",
            details=f"{type(exc).__name__}: {exc}",
        )
    st.rerun()

if not handle_configuration_navigation():
    show_feedback_dialog()
