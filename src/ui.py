"""Base visual compartida por las páginas de ElectoCluster."""

from html import escape

import streamlit as st

from src.ui_feedback import configuration_changes, handle_configuration_navigation


_PAGES = {
    "Inicio": ("app.py", ":material/home:"),
    "Preprocesamiento": ("pages/0_preprocesamiento.py", ":material/table_view:"),
    "Ponderación": ("pages/1_ponderacion.py", ":material/balance:"),
    "Configuración": ("pages/2_configuracion.py", ":material/tune:"),
    "Ejecución": ("pages/3_ejecucion.py", ":material/play_arrow:"),
    "Resultados": ("pages/4_resultados.py", ":material/analytics:"),
}


def _request_page(section: str) -> None:
    st.session_state["configuration_navigation"] = {
        "kind": "page", "value": _PAGES[section][0], "section": section,
    }


_SHARED_STYLES = """
<style>
  html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }

  code, pre, .stCode {
    font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace !important;
  }

  h1 {
    font-size: 2rem !important;
    font-weight: 650 !important;
    letter-spacing: -0.025em;
  }

  h2 {
    font-size: 1.5rem !important;
    font-weight: 650 !important;
  }

  h3 {
    font-size: 1.3rem !important;
    font-weight: 620 !important;
  }

  h4 {
    font-size: 1.05rem !important;
    font-weight: 620 !important;
  }

  [data-testid="stMarkdownContainer"] p,
  [data-testid="stWidgetLabel"] p {
    font-size: 1rem;
    line-height: 1.55;
  }

  [data-testid="stMetric"],
  [data-testid="metric-container"] {
    background: var(--secondary-background-color);
    border: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
    border-radius: 0.5rem;
    box-shadow: none;
    padding: 0.75rem;
  }

  .stButton > button,
  .stDownloadButton > button {
    border-radius: 0.45rem;
    font-weight: 550;
  }

  button:focus-visible,
  input:focus-visible,
  textarea:focus-visible,
  [role="radio"]:focus-visible,
  [role="slider"]:focus-visible,
  [role="tab"]:focus-visible {
    outline: 3px solid color-mix(in srgb, var(--primary-color) 72%, white);
    outline-offset: 2px;
  }

  .ec-sidebar-identity {
    margin: 0.35rem 0 0.75rem;
    padding: 0.25rem 0 0.8rem;
    border-bottom: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
  }

  .ec-sidebar-identity strong,
  .ec-sidebar-identity span {
    display: block;
  }

  .ec-sidebar-identity strong {
    font-size: 1rem;
    letter-spacing: 0.01em;
  }

  .ec-sidebar-identity span {
    margin-top: 0.15rem;
    color: color-mix(in srgb, var(--text-color) 68%, transparent);
    font-size: 0.78rem;
  }
</style>
"""


def apply_app_shell(section: str) -> None:
    """Aplica la base visual y añade una identificación lateral breve."""
    previous_section = st.session_state.get("ui_current_section")
    target = st.session_state.get("configuration_navigation")
    if section != "Configuración":
        if target and target["kind"] == "page" and target["section"] != section:
            st.session_state.pop("configuration_navigation", None)
            st.switch_page(target["value"])
        # También protege la navegación directa, por ejemplo con Atrás del navegador.
        if previous_section == "Configuración" and configuration_changes():
            _request_page(section)
            handle_configuration_navigation(already_on_target=True)
            st.stop()
        st.session_state.pop("configuration_navigation", None)
    st.session_state["ui_current_section"] = section
    st.markdown(_SHARED_STYLES, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(
            f"""
            <div class="ec-sidebar-identity">
              <strong>ElectoCluster</strong>
              <span>{escape(section)} · LAPOP Bolivia 2023</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for name, (path, icon) in _PAGES.items():
            st.button(
                name, icon=icon, width="stretch", key=f"navigate_{path}",
                type="primary" if name == section else "secondary",
                disabled=name == section, on_click=_request_page, args=(name,),
            )
