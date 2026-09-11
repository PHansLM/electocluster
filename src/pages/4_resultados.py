"""
Pagina 4 - Resultados y reportes.

Lista ejecuciones persistidas, muestra metricas, perfiles y visualizaciones, y
genera reportes comparativos para el Capitulo 8.
"""

import json
import sys
from html import escape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.evaluation.execution import load_processed_dataset
from src.evaluation.comparability import assess_run_comparability, run_evaluation_summary
from src.evaluation.provenance import sha256_file, stored_dataset_sha256
from src.evaluation.report_generator import ReportGenerator
from src.evaluation.results_manager import RESULTS_PATH, ResultsManager
from src.utils.constants import PROCESSED_DATA_PATH
from src.visualization.plotly_config import HEATMAP_PLOT_CONFIG_ES, PLOT_CONFIG_ES
from src.visualization.profile_differences import difference_heatmap, profile_difference_rows
from src.visualization import (
    PROFILE_METADATA_COLUMNS,
    build_semantic_profile_export,
    cluster_dimension_scores,
    cluster_distribution,
    cluster_profile_distance_matrix,
    cluster_profiles,
    cluster_semantic_profiles,
    feature_display_name,
    interpret_feature_value,
    pca_projection,
    save_run_figures,
    semantic_profile_heatmap,
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

SEMANTIC_TYPE_LABELS = {
    "numerico": "Numericas e indices",
    "categorico_ordinal": "Ordinales",
    "categorico_nominal": "Nominales",
    "binario": "Binarias",
}

SEMANTIC_TYPE_ORDER = [
    "numerico",
    "categorico_nominal",
    "categorico_ordinal",
    "binario",
]


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


def _semantic_summary_display(
    summaries: pd.DataFrame,
    feature_type: str,
) -> pd.DataFrame:
    if summaries.empty:
        return summaries

    common = ["feature_name", "valid_count", "missing_count"]
    type_columns = {
        "numerico": [
            "mean",
            "median",
            "std",
            "q1",
            "q3",
            "minimum",
            "maximum",
            "global_reference",
            "difference_from_global",
        ],
        "categorico_ordinal": [
            "representative_label",
            "median",
            "q1",
            "q3",
            "global_reference_label",
            "difference_from_global",
        ],
        "categorico_nominal": [
            "representative_label",
            "representative_percentage",
            "global_reference_label",
            "global_reference_percentage",
            "category_mapping_status",
        ],
        "binario": [
            "representative_label",
            "representative_percentage",
            "global_reference_percentage",
            "difference_from_global",
            "category_mapping_status",
        ],
    }
    selected_columns = common + type_columns[feature_type]
    display = summaries[selected_columns].copy()
    display = display.rename(
        columns={
            "feature_name": "variable",
            "valid_count": "datos validos",
            "missing_count": "datos faltantes",
            "representative_label": "valor representativo",
            "representative_percentage": "porcentaje representativo",
            "mean": "media",
            "median": "mediana",
            "std": "desviacion estandar",
            "q1": "Q1",
            "q3": "Q3",
            "minimum": "minimo",
            "maximum": "maximo",
            "global_reference": "referencia total",
            "global_reference_label": "referencia total",
            "global_reference_percentage": "porcentaje total",
            "difference_from_global": "diferencia frente al total",
            "category_mapping_status": "trazabilidad de categorias",
        }
    )
    numeric_columns = display.select_dtypes(include="number").columns
    display[numeric_columns] = display[numeric_columns].round(4)
    for percentage_column in (
        "porcentaje representativo",
        "porcentaje total",
    ):
        if percentage_column in display:
            display[percentage_column] = display[percentage_column].map(
                lambda value: f"{value:.2f}%" if pd.notna(value) else "N/A"
            )
    if "trazabilidad de categorias" in display:
        display["trazabilidad de categorias"] = display[
            "trazabilidad de categorias"
        ].map(
            {
                "configured_exact": "categorias verificadas",
                "processed_only": "solo valor procesado",
                "not_applicable": "no aplica",
            }
        )
    return display


def _semantic_distribution_display(distribution: pd.DataFrame) -> pd.DataFrame:
    display = distribution[
        [
            "category_label",
            "count",
            "percentage",
            "global_percentage",
            "percentage_point_difference",
        ]
    ].copy()
    display = display.rename(
        columns={
            "category_label": "categoria",
            "count": "registros",
            "percentage": "porcentaje del cluster",
            "global_percentage": "porcentaje total",
            "percentage_point_difference": "diferencia en puntos porcentuales",
        }
    )
    numeric_columns = display.select_dtypes(include="number").columns
    display[numeric_columns] = display[numeric_columns].round(2)
    for percentage_column in (
        "porcentaje del cluster",
        "porcentaje total",
    ):
        display[percentage_column] = display[percentage_column].map(
            lambda value: f"{value:.2f}%" if pd.notna(value) else "N/A"
        )
    display["diferencia en puntos porcentuales"] = display[
        "diferencia en puntos porcentuales"
    ].map(lambda value: f"{value:+.2f} pp" if pd.notna(value) else "N/A")
    return display


st.markdown("### Ejecuciones guardadas")
st.dataframe(
    runs_df.sort_values("timestamp", ascending=False),
    use_container_width=True,
    hide_index=True,
)

suites = rm.list_suites()
with st.expander("Evidencia integrada de la bateria canonica", expanded=False):
    if not suites:
        st.caption(
            "Aun no existe una bateria integrada. En Ejecucion, selecciona los tres "
            "algoritmos y usa sus configuraciones canonicas."
        )
    else:
        suite_options = [suite["suite_id"] for suite in suites]
        default_suite = st.session_state.get("last_canonical_suite_id")
        suite_index = suite_options.index(default_suite) if default_suite in suite_options else 0
        selected_suite_id = st.selectbox(
            "Bateria canonica consolidada",
            options=suite_options,
            index=suite_index,
        )
        suite = rm.load_suite(selected_suite_id)
        regression = suite.get("regression", {})
        comparability = suite.get("comparability", {})
        state_col, comparison_col = st.columns(2)
        state_col.metric(
            "Control historico",
            "Conforme" if regression.get("all_passed") else "Revisar",
        )
        comparison_col.metric(
            "Comparabilidad", comparability.get("status", "no registrada")
        )
        if not comparability.get("directly_comparable"):
            st.info(
                "La bateria conserva cobertura y ruido por algoritmo, pero no genera un "
                "ranking global cuando los espacios o poblaciones de evaluacion difieren.",
                icon=":material/info:",
            )
        suite_table = pd.DataFrame([
            {
                "algoritmo": item.get("algorithm"),
                "run_id": item.get("run_id"),
                "clusters": item.get("n_clusters"),
                "evaluadas": item.get("evaluation", {}).get("n_evaluated"),
                "cobertura (%)": item.get("evaluation", {}).get("coverage_percentage"),
                "ruido": item.get("n_noise"),
                "silhouette": item.get("metrics", {}).get("silhouette"),
                "tiempo (s)": item.get("elapsed_seconds"),
                "regresion": item.get("regression", {}).get("status"),
            }
            for item in suite.get("runs", [])
        ])
        st.dataframe(suite_table, use_container_width=True, hide_index=True)

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
stored_dataset_hash = stored_dataset_sha256(run)
current_dataset_hash = sha256_file(PROCESSED_DATA_PATH)

if stored_dataset_hash is None:
    st.warning(
        "Ejecucion historica sin hash de dataset. La coincidencia solo puede "
        "comprobarse por cantidad de registros.",
        icon=":material/history:",
    )
elif stored_dataset_hash != current_dataset_hash:
    st.error(
        "El dataset procesado actual no coincide con el que produjo esta ejecucion. "
        "Carga el dataset correcto o selecciona otro run.",
        icon=":material/fingerprint_off:",
    )
    st.stop()
else:
    st.caption(f"Dataset verificado por SHA-256: `{current_dataset_hash[:12]}...`")

if len(labels) != len(df):
    st.error(
        "Las asignaciones no tienen el mismo numero de registros que el dataset procesado. "
        "Reejecuta el experimento con el dataset actual."
    )
    st.stop()

st.markdown("### Detalle del run")
evaluation_summary = run_evaluation_summary(run)
st.caption(f"Algoritmo: **{run.get('algorithm')}**")
col1, col2, col3, col4, col5, col6 = st.columns(6)
metrics = run.get("metrics", {})
coverage_percentage = evaluation_summary["coverage_percentage"]
coverage_label = "N/A" if coverage_percentage is None else f"{coverage_percentage:.1f}%"
col1.metric("Clusters", run.get("n_clusters"))
col2.metric("Evaluadas", f"{evaluation_summary['n_evaluated']}/{evaluation_summary['n_total']}")
col3.metric("Cobertura", coverage_label)
col4.metric("Ruido", run.get("n_noise"))
col5.metric("Silhouette", _format_metric(metrics.get("silhouette"), 4))
col6.metric("Davies-Bouldin", _format_metric(metrics.get("davies_bouldin"), 4))

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
        st.download_button(
            "Descargar registro JSON",
            data=json.dumps(run, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name=f"{selected_run_id}.json",
            mime="application/json",
        )

distribution = cluster_distribution(labels)
profiles = cluster_profiles(df, labels)
projection = pca_projection(df, labels)
profile_distances = cluster_profile_distance_matrix(df, labels)
dimension_scores = cluster_dimension_scores(df, labels)
semantic_profiles = cluster_semantic_profiles(df, labels)

(
    tab_dist,
    tab_profile_analysis,
    tab_deviation,
    tab_distances,
    tab_dimensions,
    tab_heatmap,
    tab_projection,
    tab_compare,
) = st.tabs(
    [
        "Distribucion",
        "Analisis de perfiles",
        "Diferencias",
        "Distancias",
        "Dimensiones",
        "Heatmap semántico",
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
    st.plotly_chart(fig_dist, use_container_width=True, config=PLOT_CONFIG_ES)
    st.markdown("#### Perfiles descriptivos por cluster")
    st.caption(
        "La tabla integra tamaño, porcentaje e interpretacion de variables, por lo que "
        "sustituye el resumen tabular simple de distribucion."
    )
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
    st.plotly_chart(fig_projection, use_container_width=True, config=PLOT_CONFIG_ES)

with tab_heatmap:
    st.markdown("#### ¿En qué características destaca cada grupo?")
    st.write(
        "Este mapa de colores permite reconocer el perfil de cada grupo de un vistazo. "
        "Cada fila es una característica y cada columna es un grupo. La referencia es "
        "el conjunto de personas asignadas a grupos; las observaciones de ruido quedan fuera."
    )
    st.info(
        "Rojo: por encima de la referencia · Azul: por debajo · Blanco: cerca de ella. "
        "Lee una fila de izquierda a derecha para comparar los grupos. "
        "Estos colores no significan mejor o peor."
    )
    heatmap = semantic_profile_heatmap(semantic_profiles, top_n=10)
    if heatmap.matrix.empty:
        st.warning("No hay suficientes perfiles semanticamente comparables para el heatmap.")
    else:
        st.write(
            "**Atención a la intensidad:** se ajusta por separado en cada fila para destacar "
            "sus diferencias. Dos celdas de rojo intenso en filas distintas no implican "
            "diferencias del mismo tamaño. Para ver cuánto cambia realmente, pasa el cursor "
            "por una celda o consulta la explicación debajo."
        )
        semantic_codes = heatmap.features["feature_code"].tolist()
        semantic_rows = profile_difference_rows(semantic_profiles)
        semantic_rows = semantic_rows[semantic_rows.feature.isin(semantic_codes)].copy()
        semantic_rows["deviation"] = [
            heatmap.matrix.loc[f"Cluster {int(row.cluster_id)}", row.feature]
            for row in semantic_rows.itertuples()
        ]
        fig_heatmap = difference_heatmap(semantic_rows, semantic_codes)
        fig_heatmap.update_traces(zmin=-1, zmax=1, colorbar_title="Por variable",
                                  colorbar_tickvals=[-1, 0, 1])
        fig_heatmap.update_layout(font_size=15)
        st.plotly_chart(fig_heatmap, width="stretch", config=HEATMAP_PLOT_CONFIG_ES,
                        key=f"semantic_heatmap_{selected_run_id}")
        st.markdown("##### Leer una característica paso a paso")
        semantic_feature = st.selectbox(
            "Característica para explicar", options=semantic_codes,
            format_func=feature_display_name, key="semantic_map_feature",
        )
        for row in semantic_rows[semantic_rows.feature == semantic_feature].itertuples():
            st.write(
                f"**{row.group}:** {row.reading.lower()}. "
                f"{row.detail}: {row.value}, frente a {row.reference} en la referencia."
            )
        with st.expander("Ejemplo de lectura y detalles del cálculo"):
            st.write(
                "Ejemplo ilustrativo: si una característica aparece en el 60 % de un grupo "
                "y en el 40 % de la referencia, la diferencia es de 20 puntos porcentuales "
                "y la celda será roja. No significa que el grupo sea un 20 % mejor."
            )
            st.write(
                "Se muestran hasta diez variables, seleccionadas por la dispersión de sus "
                "diferencias entre grupos. Se usan medias para variables numéricas, medianas "
                "para categorías ordenadas y porcentajes para respuestas binarias. "
                "Las medias y medianas están en la escala procesada, no en años o bolivianos. "
                "En cada fila, las diferencias se dividen por la mayor diferencia absoluta "
                "de esa variable: el resultado va de −1 a 1. No mide significación estadística."
            )
            if heatmap.excluded_nominal_features:
                st.write(
                    "Aquí se omiten categorías sin orden, como religión u ocupación, porque "
                    "sus códigos no indican más o menos. Sus porcentajes pueden consultarse "
                    "en Análisis de perfiles y Diferencias. Variables omitidas: "
                    + ", ".join(feature_display_name(code)
                                for code in heatmap.excluded_nominal_features) + "."
                )

with tab_profile_analysis:
    st.markdown("#### Estadisticas segun el tipo de variable")
    st.caption(
        "Vista aditiva calculada sobre los mismos datos procesados y etiquetas del run. "
        "No recalcula el clustering ni reemplaza los perfiles promedio historicos."
    )

    summaries = semantic_profiles.summaries
    semantic_distributions = semantic_profiles.distributions
    if summaries.empty:
        st.warning("No hay perfiles semanticos disponibles para este run.")
    else:
        semantic_clusters = sorted(summaries["cluster_id"].unique().tolist())
        available_types = [
            feature_type
            for feature_type in SEMANTIC_TYPE_ORDER
            if feature_type in set(summaries["feature_type"])
        ]
        semantic_col_cluster, semantic_col_type = st.columns(2)
        with semantic_col_cluster:
            selected_semantic_cluster = st.selectbox(
                "Cluster del perfil semantico",
                options=semantic_clusters,
                format_func=lambda cluster_id: f"Cluster {cluster_id}",
            )
        with semantic_col_type:
            selected_semantic_type = st.selectbox(
                "Tipo de variable del perfil semantico",
                options=available_types,
                format_func=lambda feature_type: SEMANTIC_TYPE_LABELS[feature_type],
            )

        selected_summaries = summaries[
            (summaries["cluster_id"] == selected_semantic_cluster)
            & (summaries["feature_type"] == selected_semantic_type)
        ].sort_values("feature_name")

        cluster_size = int(selected_summaries["cluster_size"].iloc[0])
        semantic_metric_cluster, semantic_metric_type, semantic_metric_features = st.columns(3)
        semantic_metric_cluster.metric("Cluster", selected_semantic_cluster)
        semantic_metric_type.metric(
            "Tipo",
            SEMANTIC_TYPE_LABELS[selected_semantic_type],
        )
        semantic_metric_features.metric("Variables", len(selected_summaries))
        st.caption(
            f"Estadisticas calculadas sobre {cluster_size} registros asignados al cluster."
        )

        ambiguous_features = selected_summaries[
            selected_summaries["category_mapping_status"] == "processed_only"
        ]["feature_code"].tolist()
        if ambiguous_features:
            st.warning(
                "No existe una correspondencia verificable entre todos los valores "
                "procesados y las categorias configuradas para: "
                + ", ".join(f"`{feature}`" for feature in ambiguous_features)
                + ". Se muestran los valores procesados sin inferir etiquetas.",
                icon=":material/info:",
            )

        st.dataframe(
            _semantic_summary_display(selected_summaries, selected_semantic_type),
            use_container_width=True,
            hide_index=True,
        )

        if selected_semantic_type == "numerico":
            st.caption(
                "Las diferencias se expresan en la escala procesada: valor positivo "
                "indica una media superior al total analizado."
            )
            difference_data = selected_summaries[
                ["feature_name", "difference_from_global"]
            ].copy()
            difference_data["direccion"] = difference_data[
                "difference_from_global"
            ].map(lambda value: "Sobre el total" if value >= 0 else "Bajo el total")
            fig_semantic_numeric = px.bar(
                difference_data,
                x="feature_name",
                y="difference_from_global",
                color="direccion",
                color_discrete_map={
                    "Sobre el total": "#2563eb",
                    "Bajo el total": "#f97316",
                },
                labels={
                    "feature_name": "Variable",
                    "difference_from_global": "Diferencia frente al total",
                    "direccion": "Direccion",
                },
                title="Diferencia de medias frente al total analizado",
            )
            st.plotly_chart(fig_semantic_numeric, use_container_width=True, config=PLOT_CONFIG_ES)
        else:
            feature_options = selected_summaries["feature_code"].tolist()
            selected_semantic_feature = st.selectbox(
                "Variable para inspeccionar su distribucion",
                options=feature_options,
                format_func=lambda feature_code: feature_display_name(feature_code),
            )
            selected_distribution = semantic_distributions[
                (semantic_distributions["cluster_id"] == selected_semantic_cluster)
                & (semantic_distributions["feature_code"] == selected_semantic_feature)
            ].sort_values("category_value")

            if selected_distribution.empty:
                st.warning("No hay distribucion disponible para esta variable.")
            else:
                distribution_chart = selected_distribution[
                    ["category_label", "percentage", "global_percentage"]
                ].melt(
                    id_vars="category_label",
                    value_vars=["percentage", "global_percentage"],
                    var_name="population",
                    value_name="percentage_value",
                )
                distribution_chart["population"] = distribution_chart["population"].map(
                    {
                        "percentage": f"Cluster {selected_semantic_cluster}",
                        "global_percentage": "Total analizado",
                    }
                )
                fig_semantic_distribution = px.bar(
                    distribution_chart,
                    x="category_label",
                    y="percentage_value",
                    color="population",
                    barmode="group",
                    labels={
                        "category_label": "Categoria",
                        "percentage_value": "Porcentaje",
                        "population": "Poblacion",
                    },
                    title=f"Distribucion de {feature_display_name(selected_semantic_feature)}",
                )
                st.plotly_chart(fig_semantic_distribution, use_container_width=True,
                                config=PLOT_CONFIG_ES)
                st.dataframe(
                    _semantic_distribution_display(selected_distribution),
                    use_container_width=True,
                    hide_index=True,
                )

        provenance = run.get("metadata", {}).get("provenance", {})
        execution_sha256 = provenance.get("configuration", {}).get(
            "execution_sha256"
        )
        semantic_export = build_semantic_profile_export(
            semantic_profiles,
            run_id=selected_run_id,
            algorithm=run.get("algorithm"),
            run_timestamp=run.get("timestamp"),
            dataset_sha256=current_dataset_hash,
            execution_sha256=execution_sha256,
            exclude_noise=True,
        )
        semantic_json = json.dumps(
            semantic_export,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

        st.markdown("#### Exportar evidencia completa")
        st.caption(
            "Las descargas incluyen todos los clusters y variables. Los CSV conservan "
            "porcentajes numericos; el JSON agrega trazabilidad del run y del dataset."
        )
        export_summary, export_distributions, export_json = st.columns(3)
        with export_summary:
            st.download_button(
                "Resumen semantico CSV",
                data=summaries.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_run_id}_perfiles_semanticos_resumen.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with export_distributions:
            st.download_button(
                "Distribuciones semanticas CSV",
                data=semantic_distributions.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_run_id}_perfiles_semanticos_distribuciones.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with export_json:
            st.download_button(
                "Perfiles semanticos JSON",
                data=semantic_json,
                file_name=f"{selected_run_id}_perfiles_semanticos.json",
                mime="application/json",
                use_container_width=True,
            )

with tab_deviation:
    st.markdown("#### ¿Qué distingue a cada grupo?")
    st.caption(
        "Compara cada grupo con el conjunto de observaciones agrupadas. "
        "Los puntos de ruido quedan fuera de esta referencia."
    )
    st.info(
        "**Rojo: por encima de la referencia · Azul: por debajo · Blanco: cerca de ella.** "
        "Un color más intenso indica una diferencia mayor; no significa un resultado mejor o peor. "
        "Pasa el cursor por una celda para ver el valor del grupo, la referencia y una explicación."
    )
    differences = profile_difference_rows(semantic_profiles)
    if differences.empty:
        st.warning("No hay diferencias de perfil disponibles para este run.")
    else:
        feature_scores = (
            differences.assign(magnitude=differences.deviation.abs())
            .groupby("feature", sort=False)["magnitude"]
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
        ) if max_features > 1 else 1
        selected_features = feature_scores.head(n_features).index.tolist()
        fig_deviation = difference_heatmap(differences, selected_features)
        st.plotly_chart(fig_deviation, width="stretch", config=HEATMAP_PLOT_CONFIG_ES,
                        key=f"profile_differences_{selected_run_id}")
        with st.expander("Cómo se calculan las diferencias"):
            st.markdown(
                "- **Numéricas:** media del grupo frente a la media de referencia.\n"
                "- **Ordinales:** mediana del grupo frente a la mediana de referencia.\n"
                "- **Binarias:** porcentaje de la categoría indicada.\n"
                "- **Nominales:** porcentaje de una categoría, indicada al pasar el cursor. "
                "Se elige la categoría que más cambia entre grupos y se mantiene la misma en toda la fila.\n\n"
                "Ejemplo: **60 % frente a 40 % = 20 puntos porcentuales por encima**, "
                "no un aumento del 20 %. Para el color, esa diferencia se representa como 0,20. "
                "Las medias y medianas se expresan en la escala procesada; no en años ni bolivianos. "
                "Compara principalmente los grupos dentro de una misma fila."
            )
        st.markdown("#### Diferencias explicadas por grupo")
        group = st.selectbox("Grupo para interpretar", differences.sort_values("cluster_id").group.unique())
        explained = differences[differences.group == group].copy()
        explained["magnitude"] = explained.deviation.abs()
        explained = explained.sort_values("magnitude", ascending=False)
        for _, item in explained.head(3).iterrows():
            st.markdown(
                f"**{item['variable']}** — {item['reading']}. "
                f"{item['detail']}: **{item['value']}** en el grupo frente a "
                f"**{item['reference']}** en la referencia."
            )
        st.dataframe(
            explained[["variable", "reading", "detail", "value", "reference"]].rename(columns={
                "variable": "Variable", "detail": "Qué se compara", "value": "En el grupo",
                "reference": "Referencia", "reading": "Cómo interpretarlo",
            }),
            use_container_width=True,
            hide_index=True,
        )

with tab_distances:
    st.markdown("#### ¿Qué grupos tienen perfiles promedio más parecidos?")
    st.write(
        "Esta vista resume en un solo número la diferencia entre dos perfiles promedio. "
        "Busca un grupo en la fila y otro en la columna: la celda donde se cruzan "
        "muestra su distancia."
    )
    st.info(
        "Azul claro y número menor: perfiles más parecidos. Azul oscuro y número mayor: "
        "perfiles más diferentes. La diagonal vale cero porque compara cada grupo consigo "
        "mismo; las dos mitades del mapa repiten las mismas comparaciones."
    )
    if profile_distances.empty:
        st.warning("No hay distancias entre perfiles disponibles para este run.")
    else:
        distance_display = profile_distances.rename(
            index=lambda value: f"Grupo {value}", columns=lambda value: f"Grupo {value}"
        )
        pairs = [
            {"Primer grupo": distance_display.index[i],
             "Segundo grupo": distance_display.columns[j],
             "Distancia": float(distance_display.iloc[i, j])}
            for i in range(len(distance_display)) for j in range(i + 1, len(distance_display))
        ]
        pair_table = pd.DataFrame(pairs)
        if not pair_table.empty:
            pair_table = pair_table.sort_values("Distancia", kind="stable")
            nearest, farthest = pair_table.iloc[0], pair_table.iloc[-1]
            nearest_value = f"{nearest['Distancia']:.3f}".replace(".", ",")
            farthest_value = f"{farthest['Distancia']:.3f}".replace(".", ",")
            st.write(
                f"**Una pareja con la menor distancia:** {nearest['Primer grupo']} y "
                f"{nearest['Segundo grupo']} ({nearest_value}). "
                f"**Una pareja con la mayor distancia:** {farthest['Primer grupo']} y "
                f"{farthest['Segundo grupo']} ({farthest_value})."
            )
            if len(pair_table) == 1:
                st.caption("Solo hay dos grupos: existe una única pareja para comparar.")
        else:
            st.caption("Se necesitan al menos dos grupos para comparar perfiles diferentes.")
        fig_distances = px.imshow(
            distance_display,
            color_continuous_scale="Blues",
            zmin=0,
            aspect="auto",
            text_auto=".2f",
            labels={
                "x": "",
                "y": "",
                "color": "Distancia",
            },
        )
        fig_distances.update_layout(height=max(460, len(distance_display) * 42 + 120),
                                    dragmode=False, font_size=15)
        fig_distances.update_xaxes(type="category", fixedrange=True, tickmode="array",
                                  tickvals=distance_display.columns.tolist())
        fig_distances.update_yaxes(type="category", fixedrange=True, tickmode="array",
                                  tickvals=distance_display.index.tolist())
        fig_distances.update_traces(
            hovertemplate="%{y} y %{x}<br>Distancia: %{z:.3f}<extra></extra>"
        )
        st.plotly_chart(fig_distances, width="stretch", config=HEATMAP_PLOT_CONFIG_ES,
                        key=f"profile_distances_{selected_run_id}")
        st.write(
            "**¿Qué puedo concluir?** Una distancia menor indica mayor semejanza entre "
            "los promedios en esta ejecución. No significa que todas las personas de ambos "
            "grupos sean iguales, ni que los grupos deban unirse. Consulta Diferencias "
            "y Análisis de perfiles para entender qué características los separan."
        )
        with st.expander("Ver todas las parejas y cómo se calcula la distancia"):
            st.write(
                "Se calcula la distancia euclidiana entre las medias de las variables "
                "procesadas, sin aplicar aquí los pesos del algoritmo. No es una distancia "
                "geográfica, un porcentaje ni necesariamente la distancia usada para formar "
                "los grupos. No existe un umbral universal de cerca o lejos."
            )
            st.write(
                "Ejemplo ilustrativo: 0,2 indica más semejanza que 0,8 dentro de una misma "
                "ejecución, pero no equivale a un 80 % de semejanza. Esta vista incluye "
                "promedios de códigos de categorías sin orden; por eso es un resumen "
                "exploratorio y debe acompañarse de sus distribuciones por categoría. "
                "Las observaciones de ruido no se incluyen."
            )
            if not pair_table.empty:
                st.dataframe(pair_table, hide_index=True, width="stretch")

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
            st.plotly_chart(fig_radar, use_container_width=True, config=PLOT_CONFIG_ES)
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
        st.plotly_chart(fig_dimensions, use_container_width=True, config=PLOT_CONFIG_ES)
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
        selected_runs = [rm.load_run(run_id) for run_id in compare_ids]
        comparability = assess_run_comparability(selected_runs)
        if comparability["directly_comparable"]:
            st.success(
                "Comparacion directa habilitada: dataset, espacio y poblacion coinciden.",
                icon=":material/verified:",
            )
        else:
            st.warning(
                "Ranking deshabilitado. " + " ".join(comparability["reasons"]),
                icon=":material/compare_arrows:",
            )

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
                if comparability["directly_comparable"]:
                    best_row = metric_df.iloc[0]
                    col_best, col_note = st.columns([1, 2])
                    col_best.metric(
                        f"Mejor {spec['label']}",
                        _format_metric(best_row[metric_key], spec["precision"]),
                        best_row["algorithm"],
                    )
                    col_note.caption(spec["description"])
                else:
                    st.caption(spec["description"] + " Se muestra sin declarar un ganador.")

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
                st.plotly_chart(fig_metric, use_container_width=True, config=PLOT_CONFIG_ES)

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
