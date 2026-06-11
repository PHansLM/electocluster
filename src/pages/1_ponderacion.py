"""
Página 1 — Esquema de Ponderación
Permite ajustar los pesos de cada característica demográfica
mediante sliders y persistirlos via WeightManager.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.weighting.weight_manager import WeightManager

st.set_page_config(page_title="Ponderación · ElectoCluster", layout="wide")

st.title("Esquema de Ponderación")
st.markdown(
    "Ajusta el peso de cada característica demográfica. "
    "Los cambios se guardan automáticamente en el esquema JSON activo."
)

# ── Cargar WeightManager ──────────────────────────────────────────────────────
@st.cache_resource
def get_weight_manager():
    return WeightManager()

wm = get_weight_manager()
weights = wm.get_weights()

# ── Grupos de características ─────────────────────────────────────────────────
GRUPOS = {
    "Relevancia Alta": {
        "edre":        "Nivel educativo",
        "q10inc":      "Ingreso familiar",
        "ocupoit":     "Clase ocupacional",
        "estratopri":  "Región geográfica",
        "ur":          "Urbano / Rural",
        "wealth_index":"Índice de bienes del hogar",
    },
    "Relevancia Contextual": {
        "etid":        "Identidad étnica",
        "boletidnew":  "Pertenencia indígena",
        "boletidnewb": "Pueblo indígena específico",
        "q3cn":        "Religión",
        "q5b":         "Importancia de la religión",
        "leng1":       "Lengua materna",
        "leng4":       "Idioma de padres",
        "wf1":         "Ayuda gubernamental",
        "bolcct1a":    "Renta Dignidad",
        "bolcct1b":    "Bono Juancito Pinto",
        "bolcct1c":    "Bono Juana Azurduy",
    },
    "Relevancia Moderada": {
        "q2":          "Edad",
        "q1tc_r":      "Género",
        "q11n":        "Estado civil",
        "q12cn":       "Tamaño del hogar",
        "q12bn":       "Niños menores de 13 años",
        "estratosec":  "Tamaño de municipio",
        "formal":      "Formalidad laboral",
    },
    "Relevancia Emergente / Contextual": {
        "gi0n":        "Consumo de medios",
        "smedia3n":    "Info política en redes",
        "civic_index": "Índice de participación cívica",
        "q10e":        "Cambio en ingreso",
    },
}

# ── Sliders por grupo ─────────────────────────────────────────────────────────
nuevos_pesos = {}

for grupo, variables in GRUPOS.items():
    st.markdown(f"### {grupo}")
    cols = st.columns(3)
    col_idx = 0
    for var, nombre in variables.items():
        peso_actual = weights.get(var, 0.3)
        with cols[col_idx % 3]:
            nuevo = st.slider(
                label=f"{nombre}  `{var}`",
                min_value=0.0,
                max_value=1.0,
                value=float(peso_actual),
                step=0.05,
                key=f"slider_{var}",
            )
            nuevos_pesos[var] = nuevo
        col_idx += 1
    st.markdown("---")

# ── Botones de acción ─────────────────────────────────────────────────────────
col_save, col_reset, col_export = st.columns([1, 1, 2])

with col_save:
    if st.button(
        "Guardar esquema",
        type="primary",
        icon=":material/save:",
        use_container_width=True,
    ):
        try:
            wm.set_weights(nuevos_pesos)
            st.success("Esquema guardado correctamente.")
            st.cache_resource.clear()
        except Exception as e:
            st.error(f"Error al guardar: {e}")

with col_reset:
    if st.button(
        "Restaurar valores por defecto",
        icon=":material/refresh:",
        use_container_width=True,
    ):
        wm.reset_weights_values()
        st.success("Pesos restaurados desde constants.py.")
        st.cache_resource.clear()
        st.rerun()

# ── Vista previa del esquema actual ──────────────────────────────────────────
with st.expander("Ver esquema completo actual (JSON)"):
    st.json(weights)
