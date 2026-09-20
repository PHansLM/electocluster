"""
Página 1 — Esquema de Ponderación
Permite ajustar los pesos de cada característica demográfica
mediante sliders y persistirlos via WeightManager.
"""

import sys
from html import escape
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.ui import apply_app_shell
from src.weighting.weight_manager import WeightManager

st.set_page_config(page_title="Ponderación · ElectoCluster", layout="wide")
apply_app_shell("Ponderación")

st.title("Esquema de Ponderación")
st.markdown(
    "Ajusta el peso de cada característica demográfica. "
    "Los cambios quedan pendientes hasta que guardes el esquema activo."
)

st.markdown(
    """
    <style>
      .weight-change-note {
        min-height: 1.15rem;
        margin: -0.35rem 0 0.55rem;
        color: #facc15;
        font-size: 0.78rem;
        line-height: 1.15rem;
      }
      .weight-unsaved-summary {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.5rem 0.8rem;
        border: 1px solid rgba(234, 179, 8, 0.72);
        border-radius: 999px;
        background: rgba(234, 179, 8, 0.13);
        color: #fde68a;
        font-weight: 800;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cargar WeightManager ──────────────────────────────────────────────────────
@st.cache_resource
def get_weight_manager():
    return WeightManager()

wm = get_weight_manager()
weights = wm.get_weights()

flash_message = st.session_state.pop("weights_flash_message", None)
if flash_message:
    st.success(flash_message, icon=":material/check_circle:")

if "weights_form_version" not in st.session_state:
    st.session_state["weights_form_version"] = 0


def _format_weight(value: float) -> str:
    return f"{float(value):.2f}"


def _is_modified(original: float, current: float) -> bool:
    return abs(float(original) - float(current)) > 1e-9


def _render_weight_change(original: float, current: float):
    original_label = escape(_format_weight(original))
    current_label = escape(_format_weight(current))
    st.markdown(
        f"""
        <div class="weight-change-note">
          Sin guardar · {original_label} → <strong>{current_label}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
pesos_modificados = []
form_version = st.session_state["weights_form_version"]

for grupo, variables in GRUPOS.items():
    st.markdown(f"### {grupo}")
    variable_items = list(variables.items())
    for row_start in range(0, len(variable_items), 3):
        row_cols = st.columns(3)
        row_items = variable_items[row_start : row_start + 3]
        for col, (var, nombre) in zip(row_cols, row_items):
            peso_actual = weights.get(var, 0.3)
            with col:
                nuevo = st.slider(
                    label=f"{nombre}  `{var}`",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(peso_actual),
                    step=0.05,
                    key=f"slider_{form_version}_{var}",
                )
                nuevos_pesos[var] = nuevo
                if _is_modified(peso_actual, nuevo):
                    pesos_modificados.append((var, nombre, peso_actual, nuevo))
                    _render_weight_change(peso_actual, nuevo)
    st.markdown("---")

if pesos_modificados:
    st.markdown(
        f"""
        <div class="weight-unsaved-summary">
          Cambios sin guardar: {len(pesos_modificados)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Ver detalle de cambios pendientes", expanded=False):
        st.dataframe(
            [
                {
                    "variable": var,
                    "descripcion": nombre,
                    "guardado": _format_weight(original),
                    "actual": _format_weight(current),
                    "delta": _format_weight(current - original),
                }
                for var, nombre, original, current in pesos_modificados
            ],
            use_container_width=True,
            hide_index=True,
        )

# ── Botones de acción ─────────────────────────────────────────────────────────
col_save, col_reset, col_export = st.columns([1, 1.4, 1.6])

with col_save:
    if st.button(
        "Guardar esquema",
        type="primary",
        icon=":material/save:",
        use_container_width=True,
    ):
        try:
            wm.set_weights(nuevos_pesos)
            st.cache_resource.clear()
            st.session_state["weights_form_version"] += 1
            st.session_state["weights_flash_message"] = "Esquema guardado correctamente."
            st.rerun()
        except Exception as e:
            st.error(f"Error al guardar: {e}")

with col_reset:
    if st.button(
        "Restaurar valores por defecto",
        icon=":material/refresh:",
        use_container_width=True,
    ):
        wm.reset_weights_values()
        st.cache_resource.clear()
        st.session_state["weights_form_version"] += 1
        st.session_state["weights_flash_message"] = "Pesos restaurados a los valores predeterminados (constants.py)."
        st.rerun()

# ── Vista previa del esquema actual ──────────────────────────────────────────
with st.expander("Ver esquema completo actual (JSON)"):
    st.json(weights)
