from __future__ import annotations

import streamlit as st


def apply_fa_style() -> None:
    """Apply Persian fonts, RTL layout, and correct slider direction."""
    st.markdown(
        """
        <style>
        :root {
            --fa-body-font: "B Nazanin", "BNazanin", "Nazanin", Tahoma, Arial, sans-serif;
            --fa-title-font: "B Titr", "BTitr", "Titr", Tahoma, Arial, sans-serif;
        }

        html, body, [class*="st-"],
        [data-testid="stAppViewContainer"],
        [data-testid="stSidebar"],
        input, textarea, button, select {
            direction: rtl;
            text-align: right;
            font-family: var(--fa-body-font) !important;
        }

        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
        [data-testid="stHeadingWithActionElements"] {
            font-family: var(--fa-title-font) !important;
            font-weight: 700 !important;
        }

        .block-container {
            padding-top: 1.2rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid #e6e6e6;
            padding: .6rem;
            border-radius: .6rem;
        }

        /* RTL must not reverse the visual order of slider endpoints. */
        div[data-testid="stSlider"] div[data-baseweb="slider"],
        div[data-testid="stSelectSlider"] div[data-baseweb="slider"] {
            direction: ltr !important;
        }

        div[data-testid="stSlider"] label,
        div[data-testid="stSelectSlider"] label {
            direction: rtl !important;
            text-align: right !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
