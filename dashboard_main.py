from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ui_style import apply_fa_style

DATA_FILE = Path("data/processed/timeuse_person_activity.parquet")
FALLBACK_FILE = Path("data/processed/timeuse_person_activity.pkl.gz")
LABEL_FILE = Path("data/processed/activity_labels.csv")

GENDER_LABELS = {1: "مرد", 2: "زن"}
METRIC_LABELS = {
    "mean_minutes": "متوسط سرانه زمان (دقیقه)",
    "participation_rate": "نرخ مشارکت (درصد)",
    "participant_mean": "متوسط زمان مشارکت‌کنندگان (دقیقه)",
}

apply_fa_style()


@st.cache_data(show_spinner="در حال بارگذاری داده‌ها...")
def load_data() -> pd.DataFrame:
    if DATA_FILE.exists():
        data = pd.read_parquet(DATA_FILE)
    elif FALLBACK_FILE.exists():
        data = pd.read_pickle(FALLBACK_FILE, compression="gzip")
    else:
        raise FileNotFoundError(
            "فایل داده داشبورد پیدا نشد. ابتدا prepare_data.py را اجرا کنید."
        )

    data["activity_code"] = pd.to_numeric(
        data["activity_code"], errors="coerce"
    ).astype("Int32")

    if LABEL_FILE.exists():
        labels = pd.read_csv(LABEL_FILE)
        labels["activity_code"] = pd.to_numeric(
            labels["activity_code"], errors="coerce"
        ).astype("Int32")
        keep = [
            column
            for column in [
                "activity_code",
                "activity_label",
                "activity_main_code",
                "activity_main_label",
                "activity_display",
            ]
            if column in labels.columns
        ]
        data = data.merge(labels[keep], on="activity_code", how="left")

    fallback_label = data["activity_code"].map(
        lambda value: f"فعالیت با کد {int(value)}" if pd.notna(value) else "نامشخص"
    )

    if "activity_label" not in data.columns:
        data["activity_label"] = fallback_label
    else:
        data["activity_label"] = data["activity_label"].fillna(fallback_label)

    if "activity_display" not in data.columns:
        data["activity_display"] = (
            data["activity_code"].astype("Int64").astype(str)
            + " — "
            + data["activity_label"]
        )
    else:
        missing_display = data["activity_display"].isna()
        data.loc[missing_display, "activity_display"] = (
            data.loc[missing_display, "activity_code"].astype("Int64").astype(str)
            + " — "
            + data.loc[missing_display, "activity_label"]
        )

    if "activity_main_label" not in data.columns:
        data["activity_main_label"] = ""

    return data


def calculate_statistics(data: pd.DataFrame) -> pd.DataFrame:
    persons = data[["pid", "weight_person"]].drop_duplicates("pid")
    total_weight = persons["weight_person"].sum()
    if total_weight <= 0:
        return pd.DataFrame()

    work = data.copy()
    work["weighted_minutes"] = work["minutes"] * work["weight_person"]

    result = (
        work.groupby(
            [
                "activity_code",
                "activity_label",
                "activity_display",
                "activity_main_label",
            ],
            dropna=False,
            observed=True,
        )
        .agg(
            weighted_minutes=("weighted_minutes", "sum"),
            participant_weight=("weight_person", "sum"),
            sample_participants=("pid", "nunique"),
        )
        .reset_index()
    )

    result["mean_minutes"] = result["weighted_minutes"] / total_weight
    result["participation_rate"] = 100 * result["participant_weight"] / total_weight
    result["participant_mean"] = (
        result["weighted_minutes"] / result["participant_weight"]
    )
    return result


st.title("داشبورد تعاملی گذران وقت ایران")
st.caption(
    "فیلتر بر اساس سال، جنسیت و سن؛ نمایش کد و عنوان رسمی فعالیت‌های ICATUS."
)

try:
    df = load_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.code("python prepare_data.py\npython -m streamlit run app.py")
    st.stop()

all_years = sorted(int(x) for x in df["survey_year"].dropna().unique())
all_genders = sorted(int(x) for x in df["gender"].dropna().unique())
min_age = int(df["age"].min())
max_age = int(df["age"].max())

with st.sidebar.form("filter_form"):
    st.subheader("فیلترهای جامعه مورد بررسی")

    selected_years = st.multiselect(
        "سال آمارگیری",
        options=all_years,
        default=[max(all_years)],
    )
    selected_genders = st.multiselect(
        "جنسیت",
        options=all_genders,
        default=all_genders,
        format_func=lambda value: GENDER_LABELS.get(value, str(value)),
    )
    age_range = st.slider(
        "دامنه سنی",
        min_value=min_age,
        max_value=max_age,
        value=(15, min(65, max_age)),
    )
    selected_metric = st.radio(
        "شاخص نمودار",
        options=list(METRIC_LABELS),
        format_func=lambda value: METRIC_LABELS[value],
    )
    top_n = st.slider(
        "تعداد فعالیت‌های نمودار",
        min_value=5,
        max_value=30,
        value=15,
    )
    st.form_submit_button("اعمال فیلترها", use_container_width=True)

filtered = df.loc[
    df["survey_year"].isin(selected_years)
    & df["gender"].isin(selected_genders)
    & df["age"].between(age_range[0], age_range[1])
].copy()

if filtered.empty:
    st.warning("برای فیلترهای انتخاب‌شده مشاهده‌ای وجود ندارد.")
    st.stop()

stats = calculate_statistics(filtered)
if stats.empty:
    st.warning("محاسبه شاخص‌ها برای این انتخاب ممکن نبود.")
    st.stop()

person_data = filtered[["pid", "weight_person"]].drop_duplicates("pid")
col1, col2, col3 = st.columns(3)
col1.metric("تعداد افراد نمونه", f"{len(person_data):,}")
col2.metric("جمع وزن‌های فردی", f"{person_data['weight_person'].sum():,.0f}")
col3.metric("تعداد کدهای فعالیت", f"{stats['activity_code'].nunique():,}")

plot_data = stats.nlargest(top_n, selected_metric).sort_values(selected_metric)
fig = px.bar(
    plot_data,
    x=selected_metric,
    y="activity_display",
    orientation="h",
    labels={
        selected_metric: METRIC_LABELS[selected_metric],
        "activity_display": "کد و عنوان فعالیت",
    },
    hover_data={
        "activity_code": True,
        "activity_label": True,
        "activity_main_label": True,
        "sample_participants": True,
        selected_metric: ":.2f",
    },
)
fig.update_layout(
    height=max(500, 36 * top_n),
    yaxis_title=None,
    xaxis_title=METRIC_LABELS[selected_metric],
    margin=dict(l=30, r=20, t=30, b=20),
    font=dict(family="B Nazanin, BNazanin, Nazanin, Tahoma, Arial", size=14),
    yaxis=dict(automargin=True),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("جدول نتایج")
display_columns = [
    "activity_code",
    "activity_label",
    "activity_main_label",
    "mean_minutes",
    "participation_rate",
    "participant_mean",
    "sample_participants",
]
result_table = stats[display_columns].sort_values("mean_minutes", ascending=False)
st.dataframe(
    result_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "activity_code": "کد فعالیت",
        "activity_label": "عنوان فعالیت",
        "activity_main_label": "گروه اصلی",
        "mean_minutes": st.column_config.NumberColumn(
            "متوسط سرانه زمان (دقیقه)", format="%.2f"
        ),
        "participation_rate": st.column_config.NumberColumn(
            "نرخ مشارکت (درصد)", format="%.2f"
        ),
        "participant_mean": st.column_config.NumberColumn(
            "متوسط زمان مشارکت‌کنندگان (دقیقه)", format="%.2f"
        ),
        "sample_participants": "تعداد مشارکت‌کنندگان نمونه",
    },
)

csv_data = result_table.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "دانلود جدول CSV",
    data=csv_data,
    file_name="timeuse_dashboard_results.csv",
    mime="text/csv",
)

st.info(
    "عنوان فعالیت‌ها از جدول رسمی طبقه‌بندی ICATUS مرکز آمار ایران گرفته شده است. "
    "اگر بیش از یک سال را انتخاب کنید، نتایج به‌صورت تجمیعی نمایش داده می‌شوند."
)
