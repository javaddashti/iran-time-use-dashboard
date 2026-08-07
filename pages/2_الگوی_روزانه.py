from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_style import apply_fa_style, plotly_rtl

DATA_FILE = Path("data/processed/timeuse_diary_wide.parquet")
FALLBACK_FILE = Path("data/processed/timeuse_diary_wide.pkl.gz")
BROAD_COLUMNS = [f"broad{i}" for i in range(1, 97)]
DIARY_START_HOUR = 4

GENDER_OPTIONS = {
    "همه": None,
    "مرد": 1,
    "زن": 2,
}
EMPLOYMENT_OPTIONS = {
    "همه": None,
    "شاغل": 1,
    "غیرشاغل": 0,
}
RELATION_OPTIONS = {
    "همه اعضای خانوار": None,
    "سرپرست": {1},
    "همسر سرپرست": {2},
    "فرزند": {3},
    "سرپرست یا همسر": {1, 2},
}

CATEGORY_SPECS_8 = [
    ("paid", "کار با مزد", "#3A8FBD"),
    ("home", "خدمات خانگی", "#35AD8C"),
    ("care", "مراقبت", "#E07B2D"),
    ("learning", "آموزش", "#E8A7C8"),
    ("communication", "ارتباطات/مذهب", "#79C6EA"),
    ("leisure", "فرهنگ و فراغت", "#F1B433"),
    ("selfcare", "مراقبت شخصی", "#CF8EB4"),
    ("other", "سایر", "#9B9B9B"),
]
CATEGORY_SPECS_6 = [
    ("paid", "کار با مزد", "#3A8FBD"),
    ("home", "خدمات خانگی", "#35AD8C"),
    ("communication", "ارتباطات/مذهب", "#79C6EA"),
    ("leisure", "فرهنگ و فراغت", "#F1B433"),
    ("selfcare", "مراقبت شخصی", "#CF8EB4"),
    ("other", "سایر", "#9B9B9B"),
]


@dataclass(frozen=True)
class GroupDefinition:
    title: str
    gender: int | None
    employment: int | None
    age_min: int
    age_max: int
    relation_codes: set[int] | None


apply_fa_style()


@st.cache_data(show_spinner="در حال بارگذاری دفترچه‌های زمانی...")
def load_data() -> pd.DataFrame:
    if DATA_FILE.exists():
        return pd.read_parquet(DATA_FILE)
    if FALLBACK_FILE.exists():
        return pd.read_pickle(FALLBACK_FILE, compression="gzip")
    raise FileNotFoundError(
        "فایل داده الگوی روزانه پیدا نشد. "
        "ابتدا prepare_daily_profile_data.py را اجرا کنید."
    )


def make_group_mask(data: pd.DataFrame, group: GroupDefinition) -> pd.Series:
    mask = data["age"].between(group.age_min, group.age_max)

    if group.gender is not None:
        mask &= data["gender"].eq(group.gender)
    if group.employment is not None:
        mask &= data["employed"].eq(group.employment)
    if group.relation_codes is not None:
        mask &= data["relation_head"].isin(group.relation_codes)

    return mask


def calculate_weighted_profile(
    data: pd.DataFrame,
    weight_column: str | None,
    category_system: str,
) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    activities = data[BROAD_COLUMNS].to_numpy(dtype=np.uint8, copy=False)

    if weight_column is None:
        weights = np.ones(len(data), dtype=np.float64)
    else:
        weights = pd.to_numeric(
            data[weight_column], errors="coerce"
        ).to_numpy(dtype=np.float64)

    valid_weight = np.isfinite(weights) & (weights > 0)
    activities = activities[valid_weight]
    weights = weights[valid_weight]

    if len(weights) == 0 or weights.sum() <= 0:
        return pd.DataFrame()

    shares = np.zeros((9, 96), dtype=np.float64)
    total_weight = weights.sum()

    for slot_index in range(96):
        weighted_counts = np.bincount(
            activities[:, slot_index],
            weights=weights,
            minlength=10,
        )
        shares[:, slot_index] = 100 * weighted_counts[1:10] / total_weight

    slots = np.arange(1, 97)
    hours = np.mod((slots - 1) / 4 + DIARY_START_HOUR, 24)

    result = pd.DataFrame(
        {
            "slot": slots,
            "hour": hours,
            "paid": shares[0],
            "own_goods": shares[1],
            "home": shares[2],
            "care": shares[3],
            "voluntary": shares[4],
            "learning": shares[5],
            "communication": shares[6],
            "leisure": shares[7],
            "selfcare": shares[8],
        }
    )

    if category_system == "۸ گروه":
        result["other"] = result["own_goods"] + result["voluntary"]
    else:
        result["other"] = (
            result["own_goods"]
            + result["care"]
            + result["voluntary"]
            + result["learning"]
        )

    return result.sort_values("hour").reset_index(drop=True)


def build_area_figure(
    profile: pd.DataFrame,
    title: str,
    category_system: str,
    show_legend: bool,
) -> go.Figure:
    specs = CATEGORY_SPECS_8 if category_system == "۸ گروه" else CATEGORY_SPECS_6

    figure = go.Figure()
    for key, label, color in specs:
        figure.add_trace(
            go.Scatter(
                x=profile["hour"],
                y=profile[key],
                name=plotly_rtl(label),
                mode="lines",
                stackgroup="one",
                groupnorm=None,
                line=dict(color="white", width=1.2),
                fillcolor=color,
                hovertemplate=(
                    f"{label}<br>ساعت: %{{x:.2f}}<br>سهم: %{{y:.1f}}٪<extra></extra>"
                ),
                showlegend=show_legend,
            )
        )

    figure.update_layout(
        title=dict(
            text=plotly_rtl(title),
            x=0.5,
            xanchor="center",
            font=dict(family="Lalezar, Vazirmatn, Tahoma, Arial", size=22),
        ),
        height=500,
        margin=dict(l=85, r=225, t=75, b=70),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Vazirmatn, Tahoma, Arial", size=14),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            traceorder="normal",
            font=dict(size=12),
            itemsizing="constant",
            bgcolor="rgba(255,255,255,0.88)",
        ),
        xaxis=dict(
            title=plotly_rtl("ساعت شبانه‌روز"),
            title_standoff=12,
            range=[0, 24],
            tickmode="array",
            tickvals=list(range(0, 25, 3)),
            showgrid=False,
            zeroline=False,
            automargin=True,
        ),
        yaxis=dict(
            title=plotly_rtl("سهم پاسخ‌گویان (درصد)"),
            title_standoff=16,
            range=[0, 100],
            tickmode="array",
            tickvals=[0, 20, 40, 60, 80, 100],
            showgrid=False,
            zeroline=False,
            ticklabelposition="outside",
            automargin=True,
        ),
    )
    return figure


def automatic_title(group: GroupDefinition) -> str:
    parts: list[str] = []
    if group.gender == 1:
        parts.append("مردان")
    elif group.gender == 2:
        parts.append("زنان")
    else:
        parts.append("همه افراد")

    if group.employment == 1:
        parts.append("شاغل")
    elif group.employment == 0:
        parts.append("غیرشاغل")

    parts.append(f"{group.age_min} تا {group.age_max} سال")
    return "، ".join(parts)


st.title("الگوی فعالیت در طول شبانه‌روز")
st.caption(
    "در هر بازه ۱۵ دقیقه‌ای، ارتفاع هر ناحیه سهم وزنی افرادی را نشان می‌دهد "
    "که فعالیت اصلی آن‌ها در آن گروه قرار گرفته است. مجموع نواحی در هر ساعت ۱۰۰ درصد است."
)

try:
    df = load_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.code("python prepare_daily_profile_data.py\npython -m streamlit run app.py")
    st.stop()

all_years = sorted(int(x) for x in df["survey_year"].dropna().unique())
all_quarters = sorted(int(x) for x in df["survey_quarter"].dropna().unique())
min_age = int(df["age"].min())
max_age = int(df["age"].max())

with st.sidebar:
    st.subheader("تنظیمات مشترک")
    selected_years = st.multiselect(
        "سال آمارگیری",
        options=all_years,
        default=all_years,
    )
    selected_quarters = st.multiselect(
        "فصل آمارگیری",
        options=all_quarters,
        default=all_quarters,
        format_func=lambda value: {
            1: "بهار",
            2: "تابستان",
            3: "پاییز",
            4: "زمستان",
        }.get(value, str(value)),
    )
    category_system = st.radio(
        "طبقه‌بندی فعالیت‌ها",
        options=["۸ گروه", "۶ گروه"],
        index=0,
    )
    weighting_label = st.radio(
        "روش وزن‌دهی",
        options=["وزن طرح", "بدون وزن"],
        index=0,
    )
    number_of_groups = st.select_slider(
        "تعداد گروه‌های مقایسه",
        options=[1, 2, 3, 4, 5, 6],
        value=2,
    )

weight_column = {
    "وزن طرح": "weight_person",
    "بدون وزن": None,
}[weighting_label]

base = df.loc[
    df["survey_year"].isin(selected_years)
    & df["survey_quarter"].isin(selected_quarters)
].copy()

if base.empty:
    st.warning("برای سال‌ها و فصل‌های انتخاب‌شده مشاهده‌ای وجود ندارد.")
    st.stop()

st.subheader("تعریف گروه‌ها")
st.write(
    "برای هر گروه می‌توان جنسیت، وضعیت اشتغال، دامنه سنی و نسبت با سرپرست خانوار را جداگانه تعیین کرد."
)

defaults = [
    ("مردان", "مرد", "همه"),
    ("زنان", "زن", "همه"),
    ("مردان شاغل", "مرد", "شاغل"),
    ("زنان شاغل", "زن", "شاغل"),
    ("مردان غیرشاغل", "مرد", "غیرشاغل"),
    ("زنان غیرشاغل", "زن", "غیرشاغل"),
]

groups: list[GroupDefinition] = []
with st.form("group_builder"):
    group_columns = st.columns(2)

    for index in range(number_of_groups):
        with group_columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"**گروه {index + 1}**")
                default_title, default_gender, default_employment = defaults[index]

                title = st.text_input(
                    "عنوان اختیاری",
                    value=default_title,
                    key=f"title_{index}",
                )
                gender_label = st.selectbox(
                    "جنسیت",
                    options=list(GENDER_OPTIONS),
                    index=list(GENDER_OPTIONS).index(default_gender),
                    key=f"gender_{index}",
                )
                employment_label = st.selectbox(
                    "وضعیت اشتغال",
                    options=list(EMPLOYMENT_OPTIONS),
                    index=list(EMPLOYMENT_OPTIONS).index(default_employment),
                    key=f"employment_{index}",
                )
                age_range = st.slider(
                    "دامنه سنی",
                    min_value=min_age,
                    max_value=max_age,
                    value=(15, min(64, max_age)),
                    key=f"age_{index}",
                )
                relation_label = st.selectbox(
                    "نسبت با سرپرست خانوار",
                    options=list(RELATION_OPTIONS),
                    index=0,
                    key=f"relation_{index}",
                )

                group = GroupDefinition(
                    title=title.strip(),
                    gender=GENDER_OPTIONS[gender_label],
                    employment=EMPLOYMENT_OPTIONS[employment_label],
                    age_min=age_range[0],
                    age_max=age_range[1],
                    relation_codes=RELATION_OPTIONS[relation_label],
                )
                groups.append(group)

    submitted = st.form_submit_button(
        "رسم نمودارها",
        use_container_width=True,
        type="primary",
    )

if not submitted:
    st.info("پس از تنظیم گروه‌ها، دکمه «رسم نمودارها» را بزنید.")
    st.stop()

profiles: list[pd.DataFrame] = []
chart_columns = st.columns(2)

for index, group in enumerate(groups):
    mask = make_group_mask(base, group)
    group_data = base.loc[mask]
    title = group.title or automatic_title(group)

    with chart_columns[index % 2]:
        if group_data.empty:
            st.warning(f"برای گروه «{title}» مشاهده‌ای وجود ندارد.")
            continue

        profile = calculate_weighted_profile(
            group_data,
            weight_column=weight_column,
            category_system=category_system,
        )
        if profile.empty:
            st.warning(f"محاسبه نمودار برای گروه «{title}» ممکن نبود.")
            continue

        profile["group"] = title
        profiles.append(profile)

        sample_size = group_data["pid"].nunique()
        weighted_size = (
            group_data[weight_column].sum()
            if weight_column is not None
            else float(sample_size)
        )

        metric_1, metric_2 = st.columns(2)
        metric_1.metric("تعداد نمونه", f"{sample_size:,}")
        metric_2.metric("جمع وزن مورد استفاده", f"{weighted_size:,.3f}")

        figure = build_area_figure(
            profile,
            title=title,
            category_system=category_system,
            show_legend=True,
        )
        st.plotly_chart(figure, width="stretch")

        csv_bytes = profile.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "دانلود داده این نمودار",
            data=csv_bytes,
            file_name=f"daily_profile_group_{index + 1}.csv",
            mime="text/csv",
            key=f"download_group_{index}",
            use_container_width=True,
        )

if profiles:
    combined = pd.concat(profiles, ignore_index=True)
    st.subheader("دانلود خروجی همه گروه‌ها")
    st.download_button(
        "دانلود CSV ترکیبی",
        data=combined.to_csv(index=False).encode("utf-8-sig"),
        file_name="daily_profiles_selected_groups.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.info(
    "این صفحه از فعالیت اصلی Q2 استفاده می‌کند؛ بنابراین سهم گروه‌های فعالیت در هر بازه زمانی "
    "باید جمعاً برابر ۱۰۰ درصد باشد. فعالیت‌های هم‌زمان Q3 باید در صفحه‌ای جداگانه تحلیل شوند."
)
