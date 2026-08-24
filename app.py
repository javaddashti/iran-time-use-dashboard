from __future__ import annotations

import streamlit as st

from ui_style import apply_fa_style

st.set_page_config(
    page_title="داشبورد زندگی روزانه",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_fa_style()

overview_page = st.Page(
    "dashboard_main.py",
    title="نمای کلی",
    icon="📊",
    default=True,
)
daily_page = st.Page(
    "pages/2_الگوی_روزانه.py",
    title="الگوی روزانه",
    icon="🕓",
)
time_series_page = st.Page(
    "pages/3_روند_زمانی_فعالیت‌ها.py",
    title="روند زمانی فعالیت‌ها",
    icon="📈",
)
geo_page = st.Page(
    "pages/4_توزیع_جغرافیایی.py",
    title="توزیع جغرافیایی",
    icon="🗺️",
)

navigation = st.navigation(
    [overview_page, daily_page, time_series_page, geo_page],
    position="sidebar",
)
navigation.run()
