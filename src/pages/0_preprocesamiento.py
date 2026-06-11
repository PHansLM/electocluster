"""
Página 0 — Preprocesamiento de datos
Permite configurar y ejecutar el pipeline de preprocesamiento
y visualizar el reporte resultante.
"""

import sys
import json
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.preprocessing.preprocessor_configured import (
    run_pipeline_configured,
    PreprocessingConfig,
)
from src.preprocessing.dataset_inspector import inspect_expected_variables
from src.utils.constants import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    RESULTS_PATH,
    TODAS_VARIABLES,
)


UPLOADS_PATH = PROJECT_ROOT / "data" / "uploads"


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _save_uploaded_dataset(uploaded_file) -> Path:
    UPLOADS_PATH.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(uploaded_file.name)
    signature = f"{safe_name}:{uploaded_file.size}"

    if st.session_state.get("uploaded_dataset_signature") == signature:
        saved = st.session_state.get("uploaded_dataset_path")
        if saved and Path(saved).exists():
            return Path(saved)

    saved_path = UPLOADS_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
    saved_path.write_bytes(uploaded_file.getbuffer())
    st.session_state["uploaded_dataset_signature"] = signature
    st.session_state["uploaded_dataset_path"] = str(saved_path)
    return saved_path


def _render_variable_diagnostic(dataset_path: str | Path):
    try:
        inspection = inspect_expected_variables(dataset_path)
    except Exception as exc:
        st.error(f"No se pudo inspeccionar el dataset: {exc}", icon=":material/error:")
        return None

    coverage_pct = inspection["coverage"] * 100
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros", f"{inspection['n_rows']:,}")
    col2.metric("Variables totales", inspection["n_columns"])
    col3.metric(
        "Variables contempladas",
        f"{len(inspection['present_base'])}/{len(inspection['base_expected'])}",
    )
    col4.metric("Cobertura", f"{coverage_pct:.1f}%")

    if inspection["missing_base"]:
        st.warning(
            "El dataset no contiene todas las variables base contempladas. "
            "El pipeline puede ejecutarse con las variables disponibles, pero los resultados "
            "no seran comparables directamente con LAPOP 2023 si faltan variables relevantes.",
            icon=":material/warning:",
        )
    else:
        st.success(
            "El dataset contiene todas las variables base contempladas por el proyecto.",
            icon=":material/check_circle:",
        )

    status_df = pd.DataFrame(inspection["variable_rows"])
    tab_present, tab_missing, tab_extra = st.tabs([
        "Presentes",
        "Faltantes",
        "Columnas extra",
    ])

    with tab_present:
        st.dataframe(
            status_df[status_df["estado"] == "Presente"],
            use_container_width=True,
            hide_index=True,
        )

    with tab_missing:
        missing_df = status_df[status_df["estado"] == "Faltante"]
        if missing_df.empty:
            st.info("No hay variables base faltantes.", icon=":material/info:")
        else:
            st.dataframe(missing_df, use_container_width=True, hide_index=True)

    with tab_extra:
        extras = inspection["extra_columns"]
        if extras:
            st.dataframe(
                pd.DataFrame({"columna": extras}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No hay columnas extra fuera del contrato del proyecto.", icon=":material/info:")

    return inspection


st.set_page_config(page_title="Preprocesamiento · ElectoCluster", layout="wide")

st.title("Preprocesamiento de Datos")
st.markdown(
    "Configura y ejecuta el pipeline de preprocesamiento sobre el dataset electoral. "
    "El dataset procesado se guarda en `data/processed/` y es consumido por las "
    "páginas de configuración y ejecución."
)

# ── Estado del dataset procesado ─────────────────────────────────────────────
st.markdown("### Estado actual del dataset")

processed_path = Path(PROCESSED_DATA_PATH)
col_status, col_info = st.columns([1, 2])

with col_status:
    if processed_path.exists():
        mod_time = datetime.fromtimestamp(processed_path.stat().st_mtime)
        st.success("Dataset procesado disponible", icon=":material/check_circle:")
        st.caption(f"Última ejecución: {mod_time.strftime('%d/%m/%Y %H:%M')}")
        try:
            df_existing = pd.read_csv(processed_path)
            st.metric("Registros", f"{len(df_existing):,}")
            st.metric("Variables", len(df_existing.columns))
            st.metric("Missings", df_existing.isnull().sum().sum())
        except Exception:
            st.warning("No se pudo leer el dataset procesado.")
    else:
        st.warning("No existe dataset procesado. Ejecuta el pipeline.", icon=":material/warning:")

with col_info:
    raw_path = Path(RAW_DATA_PATH)
    if raw_path.exists():
        st.info(
            f"Dataset crudo: `{RAW_DATA_PATH}`  \n"
            f"Formato: Stata (.dta) · LAPOP Bolivia 2023  \n"
            f"Registros esperados: 1,706 · Variables originales: 208",
            icon=":material/folder_open:",
        )
    else:
        st.error(
            f"Dataset crudo no encontrado en `{RAW_DATA_PATH}`.  \n"
            "Verifica que el archivo .dta esté en `data/raw/`.",
            icon=":material/error:",
        )

st.markdown("---")

# ── Fuente de datos ──────────────────────────────────────────────────────────
st.markdown("### Fuente de datos")
st.caption(
    "Puedes usar el dataset LAPOP base o cargar un dataset sintetico/externo "
    "con estructura compatible. Al cargarlo se revisa que variables del contrato "
    "original estan presentes."
)

source_mode = st.radio(
    "Dataset a preprocesar",
    options=["LAPOP base", "Dataset cargado"],
    horizontal=True,
)

selected_dataset_path = Path(RAW_DATA_PATH)
selected_source_label = "lapop_base"

if source_mode == "Dataset cargado":
    uploaded_dataset = st.file_uploader(
        "Carga un archivo .dta o .csv",
        type=["dta", "csv"],
        accept_multiple_files=False,
    )
    if uploaded_dataset is None:
        st.info("Carga un archivo para inspeccionar sus variables.", icon=":material/upload_file:")
        selected_dataset_path = None
    else:
        selected_dataset_path = _save_uploaded_dataset(uploaded_dataset)
        selected_source_label = Path(selected_dataset_path).stem
        st.success(
            f"Dataset cargado en `{selected_dataset_path}`.",
            icon=":material/check_circle:",
        )
else:
    if selected_dataset_path.exists():
        st.caption(f"Usando dataset base: `{selected_dataset_path}`")

if selected_dataset_path and Path(selected_dataset_path).exists():
    _render_variable_diagnostic(selected_dataset_path)
else:
    st.warning("No hay una fuente de datos disponible para ejecutar el pipeline.", icon=":material/warning:")

st.markdown("---")

# ── Información de variables seleccionadas ────────────────────────────────────
with st.expander("Variables seleccionadas del dataset (40 base → 28 finales)", expanded=False):
    var_rows = []
    for code, info in TODAS_VARIABLES.items():
        if code in ['wealth_index', 'civic_index']:
            continue
        var_rows.append({
            "Código": code,
            "Nombre": info.get('nombre', ''),
            "Tipo": info.get('tipo', ''),
            "Rango": str(info.get('rango', '')),
            "Peso": info.get('peso', ''),
            "Justificación": info.get('justificacion', ''),
        })

    # Índices compuestos
    var_rows.append({
        "Código": "wealth_index",
        "Nombre": "Índice de riqueza material",
        "Tipo": "compuesto (10 vars R)",
        "Rango": "[0, 10]",
        "Peso": TODAS_VARIABLES.get('wealth_index', {}).get('peso', ''),
        "Justificación": "Suma de bienes del hogar. Reemplaza r3-r27.",
    })
    var_rows.append({
        "Código": "civic_index",
        "Nombre": "Índice de participación cívica",
        "Tipo": "compuesto (4 vars CP)",
        "Rango": "[4, 16]",
        "Peso": TODAS_VARIABLES.get('civic_index', {}).get('peso', ''),
        "Justificación": "Suma invertida de asistencia a organizaciones. Reemplaza cp6-cp13.",
    })

    df_vars = pd.DataFrame(var_rows)
    st.dataframe(df_vars, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Configuración del pipeline ────────────────────────────────────────────────
st.markdown("### Configuración del pipeline")
st.caption(
    "Los valores por defecto replican el comportamiento base validado en la iteración 1. "
    "Modifícalos solo si experimentas con variantes del preprocesamiento."
)

tab_limpieza, tab_engineering, tab_transformacion = st.tabs(
    ["Limpieza", "Feature Engineering", "Transformación"]
)

with tab_limpieza:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Validación y missings**")
        validate_ranges = st.toggle(
            "Validar rangos de variables",
            value=True,
            help="Convierte a NaN valores fuera del rango documentado en el cuestionario LAPOP.",
        )
        handle_missing = st.toggle(
            "Imputar valores faltantes",
            value=True,
            help="Mediana para numéricas, moda para categóricas y binarias.",
        )
        remove_incomplete = st.toggle(
            "Eliminar registros incompletos",
            value=False,
            help="Elimina registros con más del umbral% de missings. Desactivado por defecto: se preservan los 1,706 registros.",
        )
        if remove_incomplete:
            incomplete_threshold = st.slider(
                "Umbral de incompletitud (%)",
                min_value=10, max_value=90, value=50, step=5,
                help="Registros con más de este % de missings son eliminados.",
            ) / 100
        else:
            incomplete_threshold = 0.5

    with col2:
        st.markdown("**Detección de outliers**")
        detect_outliers = st.toggle(
            "Detectar outliers",
            value=True,
            help="Solo detecta y reporta. No elimina outliers (son plausibles en contexto boliviano).",
        )
        if detect_outliers:
            outlier_method = st.selectbox(
                "Método de detección",
                options=["iqr", "zscore"],
                index=0,
                help="IQR: Rango intercuartílico (robusto ante distribuciones asimétricas). Z-score: distancia a la media.",
            )
            outlier_threshold = st.number_input(
                "Umbral IQR / Z-score",
                min_value=1.0, max_value=5.0, value=1.5, step=0.1,
                help="IQR: 1.5 es el estándar. Z-score: 3.0 es el estándar.",
            )
        else:
            outlier_method = 'iqr'
            outlier_threshold = 1.5

with tab_engineering:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Índice de riqueza material (variables R)**")
        build_wealth = st.toggle("Construir wealth_index", value=True)
        if build_wealth:
            wealth_method = st.radio(
                "Método de construcción",
                options=["sum", "pca"],
                index=0,
                format_func=lambda x: {
                    "sum": "Suma simple (0-10, interpretación directa)",
                    "pca": "Primer componente PCA (pesos diferenciados)",
                }[x],
                help="Suma simple: cada bien tiene igual peso. PCA: pesos según varianza explicada.",
            )
        else:
            wealth_method = 'sum'

    with col2:
        st.markdown("**Índice de participación cívica (variables CP)**")
        build_civic = st.toggle("Construir civic_index", value=True)
        st.caption(
            "Suma invertida de frecuencia de asistencia a 4 organizaciones "
            "(cp6 religiosa, cp7 padres, cp8 comunitaria, cp13 política). "
            "Escala: 4=nunca participa, 16=participa semanalmente en todo."
        )

with tab_transformacion:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Codificación de categóricas**")
        encode = st.toggle("Codificar variables categóricas", value=True)
        st.caption(
            "Ordinales: recodificación a [0, N] + MinMaxScaler.  \n"
            "Nominales: LabelEncoder + MinMaxScaler (compromiso de implementación documentado).  \n"
            "Binarias: mapeo a 0/1."
        )

    with col2:
        st.markdown("**Normalización de numéricas**")
        normalize = st.toggle("Normalizar variables numéricas", value=True)
        if normalize:
            normalization_method = st.radio(
                "Método de normalización",
                options=["minmax", "standard"],
                index=0,
                format_func=lambda x: {
                    "minmax": "MinMaxScaler → rango [0, 1]",
                    "standard": "StandardScaler → media 0, desv. 1",
                }[x],
                help="MinMaxScaler recomendado para algoritmos de clustering basados en distancias.",
            )
        else:
            normalization_method = 'minmax'

st.markdown("---")

# ── Ejecución ─────────────────────────────────────────────────────────────────
col_run, col_note = st.columns([1, 3])

with col_run:
    run_button = st.button(
        "Ejecutar pipeline",
        type="primary",
        icon=":material/play_arrow:",
        use_container_width=True,
        disabled=not (selected_dataset_path and Path(selected_dataset_path).exists()),
    )

with col_note:
    st.caption(
        "La ejecución completa tarda aproximadamente 10-20 segundos. "
        "El dataset procesado activo se guardará en `data/processed/lapop_bolivia_2023_processed.csv` "
        "y estará disponible inmediatamente para las páginas de Configuración y Ejecución. "
        "Si cargas un dataset externo, reemplazará temporalmente el dataset procesado activo."
    )

if run_button:
    config = PreprocessingConfig(
        validate_ranges=validate_ranges,
        handle_missing=handle_missing,
        remove_incomplete=remove_incomplete,
        incomplete_threshold=incomplete_threshold,
        detect_outliers=detect_outliers,
        outlier_method=outlier_method,
        outlier_threshold=outlier_threshold,
        build_wealth=build_wealth,
        wealth_method=wealth_method,
        build_civic=build_civic,
        drop_originals=True,
        normalize=normalize,
        encode=encode,
        normalization_method=normalization_method,
    )

    with st.spinner("Ejecutando pipeline de preprocesamiento..."):
        log_container = st.empty()
        try:
            df_result, report = run_pipeline_configured(
                str(selected_dataset_path),
                config=config,
                save_report_path=str(Path(RESULTS_PATH) / "preprocessing" / "reporte_real.json"),
                report_metadata={
                    "source_dataset": str(selected_dataset_path),
                    "source_label": selected_source_label,
                },
            )
            st.session_state['preprocessing_report'] = report
            st.session_state['preprocessing_config'] = config
            st.session_state['active_dataset_source'] = str(selected_dataset_path)
            st.success(
                f"Pipeline completado. Dataset procesado: "
                f"**{len(df_result):,} registros × {len(df_result.columns)} variables** · "
                f"0 missings.",
                icon=":material/check_circle:",
            )
            st.rerun()
        except Exception as e:
            st.error(f"Error en el pipeline: {e}", icon=":material/error:")
            st.exception(e)

st.markdown("---")

# ── Reporte de la última ejecución ────────────────────────────────────────────
st.markdown("### Reporte de la última ejecución")

report = st.session_state.get('preprocessing_report')

# Si no hay reporte en sesión, intentar leer el JSON guardado
if report is None:
    report_path = Path(RESULTS_PATH) / "preprocessing" / "reporte_real.json"
    if report_path.exists():
        try:
            with open(report_path, encoding='utf-8') as f:
                report = json.load(f)
        except json.JSONDecodeError:
            st.warning("El reporte guardado no es un JSON valido. Ejecuta nuevamente el pipeline.")

if report is None:
    st.info("Ejecuta el pipeline para ver el reporte.")
else:
    sections = report.get('sections', report)

    # ── Carga ──
    loading = sections.get('loading', {})
    if loading:
        with st.expander("Carga de datos", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Registros cargados", f"{loading.get('n_registros', 0):,}")
            col2.metric("Variables originales", loading.get('n_variables_total', 0))
            col3.metric("Variables seleccionadas", loading.get('n_variables_seleccionadas', 0))

            missings = loading.get('missings', {})
            miss_count = missings.get('count', {})
            miss_pct = missings.get('percentage', {})

            if miss_count:
                miss_rows = [
                    {
                        "Variable": var,
                        "Missings": cnt,
                        "% del total": f"{miss_pct.get(var, 0):.1f}%",
                        "Nombre": TODAS_VARIABLES.get(var, {}).get('nombre', ''),
                    }
                    for var, cnt in miss_count.items()
                    if cnt > 0
                ]
                if miss_rows:
                    st.markdown("**Variables con valores faltantes antes de imputación:**")
                    st.dataframe(
                        pd.DataFrame(miss_rows).sort_values("Missings", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )

    # ── Limpieza ──
    cleaning = sections.get('cleaning', {})
    if cleaning:
        with st.expander("Limpieza", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Registros originales", f"{cleaning.get('registros_originales', 0):,}")
            col2.metric("Registros finales", f"{cleaning.get('registros_finales', 0):,}")
            col3.metric("Registros eliminados", cleaning.get('registros_eliminados', 0))

            vars_imputadas = cleaning.get('variables_imputadas', [])
            st.markdown(f"**Variables imputadas:** {len(vars_imputadas)}")
            st.caption(", ".join(vars_imputadas) if vars_imputadas else "Ninguna")

            outliers = cleaning.get('outliers', {})
            if outliers:
                st.markdown("**Outliers detectados (no eliminados):**")
                out_rows = [
                    {
                        "Variable": var,
                        "Outliers": info['count'],
                        "% del total": f"{info['percentage']:.1f}%",
                        "Nombre": TODAS_VARIABLES.get(var, {}).get('nombre', ''),
                    }
                    for var, info in outliers.items()
                ]
                st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)
                st.caption(
                    "Los outliers se reportan pero no se eliminan. "
                    "En el contexto boliviano, hogares numerosos y edades extremas son plausibles."
                )

    # ── Feature Engineering ──
    engineering = sections.get('feature_engineering', {})
    if engineering:
        with st.expander("Feature Engineering", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Índices creados", engineering.get('n_variables_creadas', 0))
            col2.metric("Variables consolidadas", engineering.get('n_variables_eliminadas', 0))
            dims = engineering.get('dimensiones_finales', [])
            col3.metric("Dimensiones finales", f"{dims[0]} × {dims[1]}" if dims else "—")

            for idx in engineering.get('variables_creadas', []):
                st.markdown(
                    f"**{idx['nombre']}**: {idx['interpretacion']}  \n"
                    f"Fuente: {', '.join(idx['variables_fuente'])} · "
                    f"Rango: {idx['rango_teorico']} · Método: {idx['metodo']}"
                )

            if engineering.get('advertencias'):
                for adv in engineering['advertencias']:
                    st.warning(adv)

    # ── Transformación ──
    transformation = sections.get('transformation', {})
    if transformation:
        with st.expander("Transformación", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Método de normalización",
                transformation.get('metodo_normalizacion', '—').upper()
            )
            col2.metric(
                "Variables normalizadas",
                len(transformation.get('variables_normalizadas', []))
            )
            col3.metric(
                "Variables codificadas",
                len(transformation.get('variables_codificadas', []))
            )

            vars_norm = transformation.get('variables_normalizadas', [])
            if vars_norm:
                st.markdown(f"**Normalizadas:** {', '.join(vars_norm)}")

            recodificadas = transformation.get('variables_recodificadas', [])
            if recodificadas:
                st.markdown("**Recodificaciones aplicadas:**")
                rec_rows = [
                    {
                        "Variable": r['variable'],
                        "Rango original": r['rango_original'],
                        "Rango nuevo": r['rango_nuevo'],
                    }
                    for r in recodificadas
                ]
                st.dataframe(
                    pd.DataFrame(rec_rows),
                    use_container_width=True,
                    hide_index=True,
                )
