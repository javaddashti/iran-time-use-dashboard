from __future__ import annotations

from html import escape

import streamlit as st


def plotly_rtl(text: object) -> str:
    """Isolate Persian/number text for Plotly without HTML spans.

    Unicode RLI/PDI keeps mixed Persian text and multi-digit numbers in the
    intended order, while avoiding the SVG width-calculation problems that
    HTML ``<span>`` tags can create in legends and axis labels.
    """
    return "\u2067" + str(text) + "\u2069"


def apply_fa_style() -> None:
    """Apply the shared Persian visual system used across dashboard pages."""
    st.markdown(
        """
        <style>
        @import url("https://fonts.googleapis.com/css2?family=Lalezar&family=Vazirmatn:wght@300;400;500;600;700;800&display=swap");

        :root {
            --fa-body-font: "Vazirmatn", Tahoma, Arial, sans-serif;
            --fa-title-font: "Lalezar", "Vazirmatn", Tahoma, Arial, sans-serif;
            --ink: #23283A;
            --muted: #6B7280;
            --line: #E5E7EB;
            --surface: #FFFFFF;
            --surface-soft: #F5F7FB;
            --accent: #315B7D;
        }

        html, body, [class*="st-"],
        [data-testid="stAppViewContainer"],
        [data-testid="stSidebar"],
        input, textarea, button, select {
            direction: rtl;
            text-align: right;
            font-family: var(--fa-body-font) !important;
        }

        [data-testid="stAppViewContainer"] {
            background: #F7F8FB;
        }

        [data-testid="stSidebar"] {
            background: #F1F3F7;
            border-left: 1px solid #E2E6ED;

            width: 380px !important;
            min-width: 380px !important;
            max-width: 380px !important;
        }

        /* محتوای داخلی سایدبار هم دقیقاً همان عرض را بگیرد */
        [data-testid="stSidebar"] > div:first-child {
            width: 380px !important;
            min-width: 380px !important;
        }

        /* برای نسخه‌های جدیدتر Streamlit */
        [data-testid="stSidebarContent"] {
            width: 380px !important;
        }

        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
        [data-testid="stHeadingWithActionElements"] {
            font-family: var(--fa-title-font) !important;
            font-weight: 700 !important;
            color: var(--ink) !important;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        .dashboard-hero {
            position: relative;
            overflow: hidden;
            padding: 1.35rem 1.55rem 1.45rem;
            margin-bottom: .85rem;
            border: 1px solid #DCE4EC;
            border-radius: 1.15rem;
            background:
                radial-gradient(circle at 10% 20%, rgba(121, 198, 234, .30), transparent 30%),
                linear-gradient(135deg, #FFFFFF 0%, #EEF4F8 55%, #F8F2F6 100%);
            box-shadow: 0 10px 30px rgba(35, 40, 58, .06);
        }

        .hero-eyebrow {
            color: #315B7D;
            font-size: 1rem;
            margin-bottom: .2rem;
        }

        .hero-title {
            font-family: var(--fa-title-font) !important;
            color: var(--ink);
            font-size: clamp(2rem, 4vw, 3.5rem);
            line-height: 1.25;
            margin-bottom: .25rem;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.9;
        }

        .filter-summary {
            display: flex;
            flex-wrap: wrap;
            gap: .45rem;
            margin: .15rem 0 1rem;
        }

        .filter-chip {
            display: inline-block;
            padding: .28rem .65rem;
            border-radius: 999px;
            border: 1px solid #DDE4EB;
            background: rgba(255,255,255,.85);
            color: #4B5563;
            font-size: .92rem;
        }

        .metric-card {
            min-height: 155px;
            padding: .9rem .85rem .85rem;
            border: 1px solid #E1E6EC;
            border-radius: 1rem;
            background: var(--surface);
            box-shadow: 0 8px 24px rgba(35, 40, 58, .055);
            transition: transform .15s ease, box-shadow .15s ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 28px rgba(35, 40, 58, .085);
        }

        .metric-icon {
            font-size: 1.45rem;
            margin-bottom: .15rem;
        }

        .metric-title {
            color: #6B7280;
            font-size: .95rem;
            min-height: 2.6rem;
        }

        .metric-value {
            color: var(--ink);
            font-family: var(--fa-title-font) !important;
            font-size: 1.55rem;
            line-height: 1.55;
            margin-top: .1rem;
        }

        .metric-subtitle {
            color: #9CA3AF;
            font-size: .83rem;
        }

        .section-title {
            font-family: var(--fa-title-font) !important;
            color: var(--ink);
            font-size: 1.65rem;
            margin: 1.55rem 0 .15rem;
        }

        .section-title.small {
            font-size: 1.45rem;
            margin-top: 1.3rem;
        }

        .insight-box {
            margin: 1.15rem 0 .25rem;
            padding: 1rem 1.2rem;
            border-radius: 1rem;
            border-right: 5px solid #315B7D;
            background: linear-gradient(90deg, #F9FBFD 0%, #EEF4F8 100%);
            box-shadow: 0 6px 20px rgba(35, 40, 58, .045);
        }

        .insight-title {
            font-family: var(--fa-title-font) !important;
            color: #315B7D;
            font-size: 1.25rem;
            margin-bottom: .25rem;
        }

        .insight-text {
            color: #374151;
            font-size: 1.05rem;
            line-height: 2;
        }

        div[data-testid="stMetric"] {
            border: 1px solid #E1E6EC;
            padding: .7rem;
            border-radius: .8rem;
            background: #FFFFFF;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stPlotlyChart"] {
            border-radius: .8rem;
        }

        button[kind="primary"] {
            border-radius: .65rem !important;
        }

        /* Keep numerical slider endpoints in their natural left-to-right order. */
        div[data-testid="stSlider"] div[data-baseweb="slider"],
        div[data-testid="stSelectSlider"] div[data-baseweb="slider"] {
            direction: ltr !important;
        }

        div[data-testid="stSlider"] label,
        div[data-testid="stSelectSlider"] label {
            direction: rtl !important;
            text-align: right !important;
        }

        @media (max-width: 900px) {
            .metric-card { min-height: 130px; }
            .hero-title { font-size: 2.2rem; }
            .block-container { padding-left: .8rem; padding-right: .8rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
