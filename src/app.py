"""
ElectoCluster — Interfaz de experimentación
Prototipo de segmentación electoral mediante clustering ponderado
LAPOP Bolivia 2023

Ejecución: streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from src.ui import apply_app_shell

st.set_page_config(
    page_title="ElectoCluster",
    page_icon="EC",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_shell("Inicio")
HERO_IMAGE_PATH = Path(__file__).resolve().parent / "assets" / "electocluster_flow.png"

# ── Página de inicio ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      .ec-home-kicker {
        margin-bottom: 0.6rem;
        color: var(--primary-color);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
      }
      .ec-home-lead {
        max-width: 42rem;
        margin: 0.4rem 0 1rem;
        color: color-mix(in srgb, var(--text-color) 78%, transparent);
        font-size: 1.08rem;
        line-height: 1.65;
      }
      .ec-home-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0 0 1.5rem;
      }
      .ec-home-meta span {
        padding: 0.3rem 0.65rem;
        border: 1px solid rgba(148, 163, 184, 0.32);
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.08);
        font-size: 0.78rem;
      }
      .ec-flow {
        width: 100%;
      }
      .ec-flow-step {
        display: grid;
        grid-template-columns: 2.5rem 1fr;
        gap: 0.75rem;
        align-items: center;
        padding: 0.8rem 0;
      }
      .ec-flow-step + .ec-flow-step {
        border-top: 1px solid rgba(148, 163, 184, 0.24);
      }
      .ec-flow-number {
        display: block;
        width: 2rem;
        color: var(--primary-color, #60a5fa);
        font-size: 0.74rem;
        font-weight: 750;
        text-align: center;
      }
      .ec-flow-copy strong,
      .ec-flow-copy span {
        display: block;
      }
      .ec-flow-copy strong {
        font-size: 0.94rem;
      }
      .ec-flow-copy span {
        margin-top: 0.08rem;
        color: color-mix(in srgb, var(--text-color) 66%, transparent);
        font-size: 0.82rem;
        line-height: 1.35;
      }
      .ec-home-divider {
        width: 1px;
        min-height: 35rem;
        margin: 0 auto;
        background: linear-gradient(
          to bottom,
          transparent,
          rgba(148, 163, 184, 0.42) 8%,
          rgba(148, 163, 184, 0.42) 92%,
          transparent
        );
      }
      .st-key-home_hero_image {
        position: fixed !important;
        top: 4rem;
        right: clamp(2rem, 5vw, 6rem);
        width: min(32vw, 34rem);
        z-index: 1;
      }
      .st-key-home_hero_image [data-testid="stImage"] img {
        width: 100% !important;
        max-height: calc(100vh - 6rem);
        object-fit: contain;
      }
      @media (max-width: 900px) {
        .ec-home-divider {
          width: 100%;
          height: 1px;
          min-height: 1px;
          margin: 0.25rem 0;
          background: linear-gradient(
            to right,
            transparent,
            rgba(148, 163, 184, 0.42) 8%,
            rgba(148, 163, 184, 0.42) 92%,
            transparent
          );
        }
        .st-key-home_hero_image {
          position: relative !important;
          top: auto;
          right: auto;
          width: 100%;
        }
        .st-key-home_hero_image [data-testid="stImage"] img {
          max-height: none;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

content_col, divider_col, image_col = st.columns(
    [1.15, 0.035, 0.85], gap="medium", vertical_alignment="center"
)

with content_col:
    st.markdown('<div class="ec-home-kicker">Análisis electoral · LAPOP Bolivia 2023</div>',
                unsafe_allow_html=True)
    st.title("ElectoCluster")
    st.markdown(
        """
        <div class="ec-home-lead">
          Un entorno de experimentación para construir, comparar e interpretar
          segmentaciones electorales con ponderación demográfica.
        </div>
        <div class="ec-home-meta">
          <span>1.706 registros</span>
          <span>28 variables</span>
          <span>3 algoritmos</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Del dato al perfil electoral")
    st.caption("El flujo conserva cada configuración y resultado para facilitar su revisión.")
    st.markdown(
        """
        <div class="ec-flow">
          <div class="ec-flow-step">
            <span class="ec-flow-number">01</span>
            <div class="ec-flow-copy"><strong>Preprocesamiento</strong><span>Prepara y valida el conjunto de datos.</span></div>
          </div>
          <div class="ec-flow-step">
            <span class="ec-flow-number">02</span>
            <div class="ec-flow-copy"><strong>Ponderación</strong><span>Define la importancia relativa de las variables.</span></div>
          </div>
          <div class="ec-flow-step">
            <span class="ec-flow-number">03</span>
            <div class="ec-flow-copy"><strong>Configuración</strong><span>Selecciona el algoritmo y ajusta sus parámetros.</span></div>
          </div>
          <div class="ec-flow-step">
            <span class="ec-flow-number">04</span>
            <div class="ec-flow-copy"><strong>Ejecución</strong><span>Genera y guarda la segmentación electoral.</span></div>
          </div>
          <div class="ec-flow-step">
            <span class="ec-flow-number">05</span>
            <div class="ec-flow-copy"><strong>Resultados</strong><span>Interpreta perfiles, compara y exporta evidencia.</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with divider_col:
    st.markdown('<div class="ec-home-divider" aria-hidden="true"></div>',
                unsafe_allow_html=True)

with image_col:
    with st.container(key="home_hero_image"):
        st.image(
            str(HERO_IMAGE_PATH),
            width="stretch",
            caption=None,
        )

st.caption("Trabajo de Grado · UMSS 2026 · Pablo Limachi")
