"""
Pagina 3 - Ejecucion de clustering.

Ejecuta una o varias configuraciones de clustering, calcula metricas y persiste
los runs para analisis posterior en la pagina de Resultados.
"""

import sys
from html import escape
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

SOURCE_ACTIVE = "Activa"
SOURCE_CANONICAL = "Canonica"
SOURCE_MODE_ACTIVE = "Activas"
SOURCE_MODE_CANONICAL = "Canonicas"
SOURCE_MODE_CUSTOM = "Personalizado"
VIEW_ALL = "Ver todas"

PARAM_ORDER = [
    "n_clusters",
    "random_state",
    "linkage",
    "eps",
    "min_samples",
    "pca_components",
]
PARAM_LABELS = {
    "n_clusters": "Clusters",
    "random_state": "Semilla",
    "linkage": "Linkage",
    "eps": "Eps",
    "min_samples": "Min. muestras",
    "pca_components": "PCA",
}

st.title("Ejecucion de Clustering")
st.markdown(
    "Arma un plan con una o varias configuraciones ponderadas, calcula metricas "
    "de validacion interna y guarda cada run para comparacion posterior."
)


@st.cache_data(ttl=10)
def get_dataset_preview():
    df = load_processed_dataset()
    return df.shape, df.head(10), df.columns.tolist()


try:
    shape, preview, columns = get_dataset_preview()
except Exception as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()

algorithm = st.session_state.get("algoritmo", "WKMedoids")


def _active_configs(selected_algorithm: str) -> dict:
    stored = st.session_state.get("params_by_algorithm", {})
    if not isinstance(stored, dict):
        stored = {}

    configs = {}
    for config_algorithm, _ in canonical_experiments():
        defaults = default_params(config_algorithm)
        saved = stored.get(config_algorithm, {})

        if config_algorithm == selected_algorithm:
            legacy_params = st.session_state.get("params", {})
            saved = {**saved, **legacy_params}

        configs[config_algorithm] = {
            **defaults,
            **{key: value for key, value in saved.items() if key in defaults},
        }

    st.session_state["params_by_algorithm"] = configs
    st.session_state["params"] = configs[selected_algorithm]
    return configs


active_configs = _active_configs(algorithm)


def _format_metric(value, precision: int):
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def _format_param_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _state_key(algorithm_name: str) -> str:
    return (
        algorithm_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _ordered_param_keys(*param_sets: dict) -> list[str]:
    available = set()
    for params in param_sets:
        available.update(params.keys())

    ordered = [key for key in PARAM_ORDER if key in available]
    ordered.extend(sorted(available.difference(ordered)))
    return ordered


def _params_as_columns(params: dict) -> dict:
    return {
        PARAM_LABELS.get(key, key): _format_param_value(params.get(key))
        for key in PARAM_ORDER
    }


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


def _canonical_configs() -> dict:
    return {
        config_algorithm: config_params
        for config_algorithm, config_params in canonical_experiments()
    }


def _config_comparison_table(configs: dict, selected_view: str) -> pd.DataFrame:
    rows = []
    canonical_configs = _canonical_configs()
    algorithms = (
        list(canonical_configs.keys())
        if selected_view == VIEW_ALL
        else [selected_view]
    )

    for config_algorithm in algorithms:
        canonical_params = canonical_configs[config_algorithm]
        active_params = configs[config_algorithm]
        for key in _ordered_param_keys(active_params, canonical_params):
            active_value = active_params.get(key)
            canonical_value = canonical_params.get(key)
            rows.append({
                "algoritmo": config_algorithm,
                "parametro": PARAM_LABELS.get(key, key),
                "activa": _format_param_value(active_value),
                "canonica": _format_param_value(canonical_value),
                "estado": "Igual" if active_value == canonical_value else "Difiere",
            })
    return pd.DataFrame(rows)


def _source_mode(sources: dict, algorithms: list[str]) -> str:
    selected_sources = [sources[algorithm_name] for algorithm_name in algorithms]
    if all(source == SOURCE_ACTIVE for source in selected_sources):
        return SOURCE_MODE_ACTIVE
    if all(source == SOURCE_CANONICAL for source in selected_sources):
        return SOURCE_MODE_CANONICAL
    return SOURCE_MODE_CUSTOM


def _execution_plan_table(selected_algorithms: list[str], sources: dict, configs: dict) -> pd.DataFrame:
    canonical_configs = _canonical_configs()
    rows = []
    for selected in selected_algorithms:
        source = sources[selected]
        selected_params = (
            configs[selected]
            if source == SOURCE_ACTIVE
            else canonical_configs[selected]
        )
        row = {
            "algoritmo": selected,
            "fuente": source,
        }
        row.update(_params_as_columns(selected_params))
        rows.append(row)
    return pd.DataFrame(rows)


def _status_badge(status: str) -> str:
    if status == "Igual":
        colors = {
            "background": "rgba(34, 197, 94, 0.14)",
            "border": "rgba(34, 197, 94, 0.72)",
            "text": "#86efac",
        }
    else:
        colors = {
            "background": "rgba(234, 179, 8, 0.15)",
            "border": "rgba(234, 179, 8, 0.78)",
            "text": "#fde68a",
        }

    return (
        "<span style='"
        "display:inline-flex;align-items:center;justify-content:center;"
        "min-width:72px;padding:2px 10px;border-radius:999px;"
        f"background:{colors['background']};"
        f"border:1px solid {colors['border']};"
        f"color:{colors['text']};"
        "font-weight:700;font-size:0.78rem;"
        "'>"
        f"{escape(status)}"
        "</span>"
    )


def _source_mode_badge(mode: str) -> str:
    palettes = {
        SOURCE_MODE_ACTIVE: {
            "background": "rgba(234, 179, 8, 0.15)",
            "border": "rgba(234, 179, 8, 0.78)",
            "text": "#fde68a",
        },
        SOURCE_MODE_CANONICAL: {
            "background": "rgba(34, 197, 94, 0.14)",
            "border": "rgba(34, 197, 94, 0.72)",
            "text": "#86efac",
        },
        SOURCE_MODE_CUSTOM: {
            "background": "rgba(59, 130, 246, 0.15)",
            "border": "rgba(59, 130, 246, 0.78)",
            "text": "#93c5fd",
        },
    }
    colors = palettes.get(mode, palettes[SOURCE_MODE_CUSTOM])
    return (
        "<span style='"
        "display:inline-flex;align-items:center;justify-content:center;"
        "min-width:120px;padding:5px 14px;border-radius:999px;"
        f"background:{colors['background']};"
        f"border:1px solid {colors['border']};"
        f"color:{colors['text']};"
        "font-weight:800;font-size:1rem;"
        "'>"
        f"{escape(mode)}"
        "</span>"
    )


def _render_config_comparison_table(comparison_table: pd.DataFrame):
    grouped_rows = []
    for algorithm_name, group in comparison_table.groupby("algoritmo", sort=False):
        group_rows = group.to_dict("records")
        for index, row in enumerate(group_rows):
            grouped_rows.append((algorithm_name, index, len(group_rows), row))

    html_rows = []
    for algorithm_name, index, group_size, row in grouped_rows:
        cells = []
        if index == 0:
            cells.append(
                "<td rowspan='{rows}' style='"
                "vertical-align:top;font-weight:700;color:#f8fafc;"
                "border-right:1px solid rgba(148, 163, 184, 0.22);"
                "'>{algorithm}</td>".format(
                    rows=group_size,
                    algorithm=escape(str(algorithm_name)),
                )
            )
        cells.extend([
            f"<td>{escape(str(row['parametro']))}</td>",
            f"<td>{escape(str(row['activa']))}</td>",
            f"<td>{escape(str(row['canonica']))}</td>",
            f"<td>{_status_badge(str(row['estado']))}</td>",
        ])
        html_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = """
    <div class="electo-comparison-table" style="overflow-x:auto;border:1px solid rgba(148, 163, 184, 0.22);border-radius:8px;">
      <table style="width:100%;border-collapse:collapse;font-size:0.88rem;">
        <thead>
          <tr style="background:rgba(148, 163, 184, 0.10);color:#cbd5e1;">
            <th style="text-align:left;padding:9px 10px;border-bottom:1px solid rgba(148, 163, 184, 0.24);">algoritmo</th>
            <th style="text-align:left;padding:9px 10px;border-bottom:1px solid rgba(148, 163, 184, 0.24);">parametro</th>
            <th style="text-align:left;padding:9px 10px;border-bottom:1px solid rgba(148, 163, 184, 0.24);">activa</th>
            <th style="text-align:left;padding:9px 10px;border-bottom:1px solid rgba(148, 163, 184, 0.24);">canonica</th>
            <th style="text-align:left;padding:9px 10px;border-bottom:1px solid rgba(148, 163, 184, 0.24);">estado</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>
    <style>
      .electo-comparison-table td {{
        padding: 9px 10px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.14);
      }}
      .electo-comparison-table tbody tr:last-child td {{
        border-bottom: none;
      }}
    </style>
    """.format(rows="".join(html_rows))
    st.markdown(table_html, unsafe_allow_html=True)


st.markdown("### Dataset activo")
col_rows, col_columns, col_configs = st.columns(3)
with col_rows:
    st.metric("Registros", f"{shape[0]:,}")
with col_columns:
    st.metric("Variables", len(columns))
with col_configs:
    st.metric("Configuraciones activas", len(active_configs))

with st.expander("Vista previa del dataset procesado", expanded=False):
    st.dataframe(preview, use_container_width=True, hide_index=True)

st.markdown("---")

algorithm_options = [config_algorithm for config_algorithm, _ in canonical_experiments()]

st.markdown("### Configuraciones activas vs canonicas")
st.caption(
    "Revisa una configuracion concreta o todas. Esta seleccion solo controla la "
    "vista de comparacion; no modifica el plan de ejecucion."
)
comparison_view = st.radio(
    "Configuracion a revisar",
    options=[VIEW_ALL, *algorithm_options],
    horizontal=True,
)
comparison_table = _config_comparison_table(active_configs, comparison_view)
diff_count = int((comparison_table["estado"] == "Difiere").sum())
same_count = int((comparison_table["estado"] == "Igual").sum())

metric_diff, metric_same = st.columns(2)
with metric_diff:
    st.metric("Diferencias", diff_count)
with metric_same:
    st.metric("Coincidencias", same_count)

_render_config_comparison_table(comparison_table)

st.markdown("---")
st.markdown("### Plan de ejecucion")
st.caption(
    "Activa usa lo guardado desde Configuracion. Canonica usa los parametros finales "
    "del documento. Si mezclas fuentes, el estado pasa a Personalizado."
)

for option in algorithm_options:
    slug = _state_key(option)
    st.session_state.setdefault(f"execute_{slug}", True)
    st.session_state.setdefault(f"source_{slug}", SOURCE_ACTIVE)

bulk_active, bulk_canonical, _ = st.columns([1, 1, 2])
if bulk_active.button("Usar activas en todos", icon=":material/tune:"):
    for option in algorithm_options:
        st.session_state[f"source_{_state_key(option)}"] = SOURCE_ACTIVE
    st.rerun()

if bulk_canonical.button("Usar canonicas en todos", icon=":material/rule:"):
    for option in algorithm_options:
        st.session_state[f"source_{_state_key(option)}"] = SOURCE_CANONICAL
    st.rerun()

header_exec, header_algo, header_source = st.columns([0.8, 1.5, 1.7])
header_exec.caption("Ejecutar")
header_algo.caption("Algoritmo")
header_source.caption("Fuente de parametros")

sources = {}
selected_algorithms = []
for option in algorithm_options:
    slug = _state_key(option)
    col_exec, col_algo, col_source = st.columns([0.8, 1.5, 1.7])
    with col_exec:
        execute = st.toggle(
            "Ejecutar",
            key=f"execute_{slug}",
            label_visibility="collapsed",
        )
    with col_algo:
        st.markdown(f"**{option}**")
    with col_source:
        selected_source = st.radio(
            "Fuente de parametros",
            options=[SOURCE_ACTIVE, SOURCE_CANONICAL],
            key=f"source_{slug}",
            horizontal=True,
            label_visibility="collapsed",
        )

    sources[option] = selected_source
    if execute:
        selected_algorithms.append(option)

mode_col, count_col = st.columns(2)
with mode_col:
    source_mode = _source_mode(sources, algorithm_options)
    st.caption("Estado de fuentes")
    st.markdown(_source_mode_badge(source_mode), unsafe_allow_html=True)
with count_col:
    st.metric("Algoritmos seleccionados", len(selected_algorithms))

if selected_algorithms:
    st.markdown("#### Plan resultante")
    st.dataframe(
        _execution_plan_table(selected_algorithms, sources, active_configs),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning("Selecciona al menos un algoritmo para ejecutar.", icon=":material/warning:")

run_clicked = st.button(
    "Ejecutar seleccion",
    type="primary",
    icon=":material/play_arrow:",
    use_container_width=True,
    disabled=not selected_algorithms,
)

if run_clicked:
    rows = []
    run_ids = []
    last_result = None
    canonical_configs = _canonical_configs()
    progress = st.progress(0)

    for i, selected_algorithm in enumerate(selected_algorithms, start=1):
        selected_source = sources[selected_algorithm]
        selected_params = (
            active_configs[selected_algorithm]
            if selected_source == SOURCE_ACTIVE
            else canonical_configs[selected_algorithm]
        )
        with st.spinner(f"Ejecutando {selected_algorithm}..."):
            result = run_clustering_experiment(
                algorithm=selected_algorithm,
                params=selected_params,
                weight_manager=WeightManager(),
                persist=True,
            )
        last_result = result
        run_ids.append(result.run_id)
        rows.append({
            "run_id": result.run_id,
            "algorithm": selected_algorithm,
            "fuente": selected_source,
            "silhouette": result.metrics.get("silhouette"),
            "davies_bouldin": result.metrics.get("davies_bouldin"),
            "calinski_harabasz": result.metrics.get("calinski_harabasz"),
            "n_noise": int((result.labels == -1).sum()),
        })
        progress.progress(i / len(selected_algorithms))

    st.session_state["last_suite_run_ids"] = run_ids
    st.session_state["last_run_id"] = run_ids[-1] if run_ids else None
    st.success("Ejecucion completada.")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if len(run_ids) == 1 and last_result is not None:
        _render_result_summary(last_result.metrics, last_result.labels, last_result.metadata)
