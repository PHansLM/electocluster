"""
ElectoCluster — Interfaz de experimentación
Prototipo de segmentación electoral mediante clustering ponderado
LAPOP Bolivia 2023

Ejecución: streamlit run app.py
"""

import streamlit as st

from src.ui import apply_app_shell

st.set_page_config(
    page_title="ElectoCluster",
    page_icon="EC",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_shell("Inicio")

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
