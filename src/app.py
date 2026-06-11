"""
ElectoCluster — Interfaz de experimentación
Prototipo de segmentación electoral mediante clustering ponderado
LAPOP Bolivia 2023

Ejecución: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="ElectoCluster",
    page_icon="EC",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos globales ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Tipografía y color base */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    code, .stCode {
        font-family: 'IBM Plex Mono', monospace !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f1117;
        border-right: 1px solid #2a2d3a;
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Tarjetas de métricas */
    [data-testid="metric-container"] {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 8px;
        padding: 12px;
    }

    /* Botón principal */
    .stButton > button[kind="primary"] {
        background: #2563eb;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ElectoCluster")
    st.markdown("**Segmentación electoral ponderada**")
    st.markdown("---")
    st.markdown(
        "Dataset: *Barómetro de las Américas 2023*  \n"
        "LAPOP — Bolivia  \n"
        "n = 1,706 registros · 28 variables"
    )
    st.markdown("---")
    st.caption("Trabajo de Grado · UMSS 2026")
    st.caption("Pablo Limachi Martínez")

# ── Página de inicio ──────────────────────────────────────────────────────────
st.title("ElectoCluster")
st.subheader("Prototipo de segmentación electoral mediante clustering con ponderación demográfica")

st.markdown("""
Este prototipo permite explorar la segmentación de votantes del Barómetro de las Américas 2023 (LAPOP Bolivia)
mediante algoritmos de clustering con ponderación diferenciada de características demográficas.

Utiliza el menú lateral para navegar entre las secciones del sistema:
""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    #### Ponderación
    Configura el esquema de pesos de características demográficas.
    Los pesos se persisten en el JSON de configuración del módulo WeightManager.
    """)

with col2:
    st.markdown("""
    #### Configuración
    Selecciona el algoritmo de clustering (WKMedoids, W-Hierarchical, W-DBSCAN)
    y ajusta sus parámetros de ejecución.
    """)

with col3:
    st.markdown("""
    #### Ejecución
    Ejecuta la segmentación sobre el dataset electoral preprocesado.
    Calcula métricas de validación interna y guarda los resultados.
    """)

with col4:
    st.markdown("""
    #### Resultados
    Visualiza los clusters identificados, perfiles de votantes
    y comparativas entre ejecuciones.
    """)

st.markdown("---")
st.info(
    "**Flujo recomendado**: Ponderación → Configuración → Ejecución → Resultados. "
    "Los pesos y resultados se persisten entre sesiones.",
    icon=":material/info:"
)
