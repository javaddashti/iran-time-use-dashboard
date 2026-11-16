from __future__ import annotations

import pandas as pd
import streamlit as st


def activity_picker(catalog: pd.DataFrame, *, id_prefix: str) -> tuple[list[int], dict[int, str]]:
    """Two-step Persian activity picker: small group selector + code popover.

    Returns selected integer codes and the display-name lookup for the visible
    group. Selection remains a normal Streamlit widget; it has no impact on
    time-use calculations or weighting.
    """
    st.subheader("انتخاب فعالیت")
    groups = sorted(catalog["activity_main_label"].dropna().unique().tolist())
    group_col, codes_col = st.columns([1.2, 1], gap="medium")
    with group_col:
        selected_group = st.selectbox(
            "گروه اصلی فعالیت",
            options=["همه گروه‌ها", *groups],
            key=f"{id_prefix}_group",
            help="برای محدودکردن فهرست فعالیت‌ها، یک گروه انتخاب کنید.",
        )

    filtered = catalog if selected_group == "همه گروه‌ها" else catalog.loc[
        catalog["activity_main_label"].eq(selected_group)
    ]
    lookup = dict(zip(filtered["activity_code"].astype(int), filtered["activity_display"]))
    options = sorted(lookup)
    if not options:
        st.info("در این گروه هیچ فعالیتی در داده‌های انتخاب‌شده ثبت نشده است.")
        return [], lookup

    default_codes = [c for c in [741, 742, 749] if c in options]
    if not default_codes:
        default_codes = options[:1]

    with codes_col:
        st.markdown("<div style='height: 1.85rem'></div>", unsafe_allow_html=True)
        with st.popover("انتخاب کدهای فعالیت", use_container_width=True):
            st.caption("می‌توانید نام یا کد فعالیت را جست‌وجو کنید و چند مورد را انتخاب کنید.")
            selected = st.multiselect(
                "کد یا کدهای فعالیت",
                options=options,
                default=default_codes,
                format_func=lambda code: lookup.get(code, str(code)),
                key=f"{id_prefix}_codes_{selected_group}",
                help="در حالت تجمیعی، زمان کدهای منتخب برای هر فرد جمع می‌شود.",
            )
    # Avoid a long forest of tags in the main content area.
    preview = "، ".join(str(code) for code in selected[:6])
    if len(selected) > 6:
        preview += "، ..."
    st.markdown(
        f'<div class="activity-selection-summary">'
        f'فعالیت‌های انتخاب‌شده: {len(selected)}'
        + (f' | کدها: {preview}' if preview else '')
        + '</div>',
        unsafe_allow_html=True,
    )
    return list(selected), lookup
