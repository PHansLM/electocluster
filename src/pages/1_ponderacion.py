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
from src.weighting.weight_manager import WeightManager

st.set_page_config(page_title="Ponderación · ElectoCluster", layout="wide")

st.title("Esquema de Ponderación")
st.markdown(
    "Ajusta el peso de cada característica demográfica. "
    "Los cambios quedan pendientes hasta que guardes el esquema activo."
)

st.markdown(
    """
    <style>
      .weight-change-note {
        margin: -0.25rem 0 0.85rem;
        padding: 0.55rem 0.65rem;
        border: 1px solid rgba(234, 179, 8, 0.62);
        border-radius: 8px;
        background: rgba(234, 179, 8, 0.10);
      }
      .weight-change-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.4rem;
      }
      .weight-change-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 2px 10px;
        border: 1px solid rgba(234, 179, 8, 0.78);
        border-radius: 999px;
        background: rgba(234, 179, 8, 0.15);
        color: #fde68a;
        font-size: 0.72rem;
        font-weight: 800;
      }
      .weight-change-values {
        display: flex;
        justify-content: space-between;
        gap: 0.7rem;
        color: #cbd5e1;
        font-size: 0.76rem;
      }
      .weight-trail {
        position: relative;
        height: 8px;
        margin: 0.45rem 0;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.18);
      }
      .weight-trail-fill {
        position: absolute;
        top: 0;
        height: 8px;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(234, 179, 8, 0.18), rgba(234, 179, 8, 0.70));
      }
      .weight-point {
        position: absolute;
        top: 50%;
        width: 12px;
        height: 12px;
        border-radius: 999px;
        transform: translate(-50%, -50%);
      }
      .weight-point-original {
        border: 2px solid rgba(203, 213, 225, 0.9);
        background: #111827;
      }
      .weight-point-current {
        border: 2px solid rgba(234, 179, 8, 0.95);
        background: #facc15;
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
    original = float(original)
    current = float(current)
    left = min(original, current) * 100
    width = abs(current - original) * 100
    original_pos = original * 100
    current_pos = current * 100
    st.markdown(
        f"""
        <div class="weight-change-note">
          <div class="weight-change-header">
            <span class="weight-change-badge">Modificada</span>
          </div>
          <div class="weight-trail">
            <span class="weight-trail-fill" style="left:{left:.2f}%;width:{width:.2f}%;"></span>
            <span class="weight-point weight-point-original" title="Valor guardado" style="left:{original_pos:.2f}%;"></span>
            <span class="weight-point weight-point-current" title="Valor actual" style="left:{current_pos:.2f}%;"></span>
          </div>
          <div class="weight-change-values">
            <span>Guardado: <strong>{escape(_format_weight(original))}</strong></span>
            <span>Actual: <strong>{escape(_format_weight(current))}</strong></span>
          </div>
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
                key=f"slider_{form_version}_{var}",
            )
            nuevos_pesos[var] = nuevo
            if _is_modified(peso_actual, nuevo):
                pesos_modificados.append((var, nombre, peso_actual, nuevo))
                _render_weight_change(peso_actual, nuevo)
        col_idx += 1
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
