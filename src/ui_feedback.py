"""Diálogos de resultado y confirmación de cambios pendientes de la sesión."""

import streamlit as st

from src.evaluation.execution import default_params


PARAMETER_LABELS = {
    "n_clusters": "Número de clusters",
    "random_state": "Semilla aleatoria",
    "linkage": "Método de linkage",
    "eps": "Radio de vecindad (eps)",
    "min_samples": "Mínimo de puntos núcleo",
    "pca_components": "Componentes PCA previos",
}


def configuration_changes(algorithm: str | None = None) -> dict:
    """Compara borradores con valores activos sin modificar ninguno."""
    drafts = st.session_state.get("params_drafts_by_algorithm", {})
    saved = st.session_state.get("params_by_algorithm", {})
    changes = {}
    for name, draft in drafts.items():
        if algorithm is not None and name != algorithm:
            continue
        active = {**default_params(name), **saved.get(name, {})}
        fields = {
            key: (active[key], value)
            for key, value in draft.items()
            if key in active and active[key] != value
        }
        if fields:
            changes[name] = fields
    return changes


def mark_parameter_change(key: str, value, active: dict) -> None:
    if value != active.get(key):
        st.caption(
            f":orange[:material/edit: **Sin guardar** · "
            f"Guardado: {active.get(key)} → En edición: {value}]"
        )


def queue_feedback(kind: str, title: str, message: str, *, details: str = "") -> None:
    st.session_state["operation_feedback"] = {
        "kind": kind, "title": title, "message": message, "details": details,
        "section": st.session_state.get("ui_current_section"),
    }


def _dismiss_feedback() -> None:
    st.session_state.pop("operation_feedback", None)


def show_feedback_dialog() -> None:
    feedback = st.session_state.get("operation_feedback")
    if not feedback:
        return
    if feedback["section"] != st.session_state.get("ui_current_section"):
        _dismiss_feedback()
        return
    icons = {
        "success": ":material/check_circle:",
        "error": ":material/error:",
        "warning": ":material/warning:",
    }

    @st.dialog(
        feedback["title"], width="medium", icon=icons[feedback["kind"]],
        on_dismiss=_dismiss_feedback,
    )
    def render_feedback():
        getattr(st, feedback["kind"])(feedback["message"])
        if feedback["details"]:
            with st.expander("Detalle del error"):
                st.code(feedback["details"], language=None)
        if st.button("Entendido", type="primary", key="feedback_acknowledge"):
            _dismiss_feedback()
            st.rerun()

    render_feedback()


def _continue_navigation(target: dict, *, already_on_target: bool) -> None:
    st.session_state.pop("configuration_navigation", None)
    _dismiss_feedback()
    if target["kind"] == "algorithm":
        # La radio se actualiza antes de renderizarla en el siguiente rerun completo.
        st.session_state["confirmed_configuration_algorithm"] = target["value"]
        st.rerun()
    elif already_on_target:
        st.session_state["ui_current_section"] = target["section"]
        st.rerun()
    else:
        st.session_state["ui_current_section"] = target["section"]
        st.switch_page(target["value"])


def handle_configuration_navigation(*, already_on_target: bool = False) -> bool:
    """Resuelve un cambio de algoritmo o página; devuelve si abrió un diálogo."""
    target = st.session_state.get("configuration_navigation")
    if not target:
        return False
    changes = configuration_changes(
        st.session_state.get("algoritmo", "WKMedoids")
        if target["kind"] == "algorithm" else None
    )
    if not changes:
        _continue_navigation(target, already_on_target=already_on_target)
        return False

    @st.dialog("Cambios sin guardar", width="medium", dismissible=False,
               icon=":material/pending:")
    def confirm_navigation():
        destination = target.get("section", target["value"])
        st.warning(f"Hay cambios sin guardar antes de pasar a **{destination}**.")
        for name, fields in changes.items():
            st.markdown(f"**{name}**")
            for key, (old, new) in fields.items():
                st.write(f"• {PARAMETER_LABELS.get(key, key)}: {old} → {new}")
        st.caption(
            "Si continúas, los valores en edición se conservarán en esta sesión. "
            "Ejecución seguirá utilizando la configuración activa guardada."
        )
        with st.container(horizontal=True):
            if st.button("Seguir editando", type="primary", key="config_navigation_stay"):
                st.session_state.pop("configuration_navigation", None)
                if already_on_target:
                    st.switch_page("pages/2_configuracion.py")
                st.rerun()
            if st.button("Continuar sin guardar", key="config_navigation_continue"):
                _continue_navigation(target, already_on_target=already_on_target)

    confirm_navigation()
    return True
