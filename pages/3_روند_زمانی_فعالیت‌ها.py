from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui_style import apply_fa_style

DATA_FILE = Path("data/processed/timeuse_person_activity.parquet")
FALLBACK_FILE = Path("data/processed/timeuse_person_activity.pkl.gz")
LABEL_FILE = Path("data/processed/activity_labels.csv")

# فقط ستون‌های لازم خوانده می‌شوند تا مصرف حافظه پایین بماند.
DATA_COLUMNS = [
    "pid",
    "survey_year",
    "survey_quarter",
    "gender",
    "age",
    "activity_status",
    "weight_person",
    "activity_code",
    "minutes",
]

GENDER_OPTIONS = {"همه": None, "مرد": 1, "زن": 2}
EMPLOYMENT_OPTIONS = {"همه": None, "شاغل": 1, "غیرشاغل": 0}
QUARTER_LABELS = {1: "بهار", 2: "تابستان", 3: "پاییز", 4: "زمستان"}
SURVEY_MONTH_LABELS = {1: "خرداد", 2: "شهریور", 3: "آذر", 4: "اسفند"}
METRIC_LABELS = {
    "mean_all": "متوسط زمان در کل جامعه",
    "mean_participants": "متوسط زمان در میان مشارکت‌کنندگان",
    "participation_rate": "نرخ مشارکت",
}

apply_fa_style()


def to_persian_digits(value: object) -> str:
    return str(value).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


def period_label(year: int, quarter: int) -> str:
    month = SURVEY_MONTH_LABELS.get(int(quarter), str(quarter))
    return f"{month} {to_persian_digits(year)}"


@st.cache_resource(show_spinner="در حال بارگذاری داده‌های سری زمانی...")
def load_activity_data() -> pd.DataFrame:
    """Load a compact, numeric, read-only dataframe once for all reruns."""
    if DATA_FILE.exists():
        data = pd.read_parquet(DATA_FILE, columns=DATA_COLUMNS)
    elif FALLBACK_FILE.exists():
        data = pd.read_pickle(FALLBACK_FILE, compression="gzip")
        missing = [column for column in DATA_COLUMNS if column not in data.columns]
        if missing:
            raise KeyError(f"ستون‌های لازم در فایل داده وجود ندارند: {missing}")
        data = data[DATA_COLUMNS]
    else:
        raise FileNotFoundError(
            "فایل timeuse_person_activity پیدا نشد. ابتدا prepare_data.py را اجرا کنید."
        )

    # شناسه فرد در داده اصلی رشته‌ای است (برای نمونه شامل خط زیر و underscore).
    # بنابراین نباید آن را با pd.to_numeric تبدیل کرد؛ این کار همه شناسه‌ها را NaN
    # می‌کند. ابتدا شناسه‌های معتبر را نگه می‌داریم و بعد برای کاهش حافظه،
    # هر شناسه یکتا را به یک کد عددی فشرده تبدیل می‌کنیم.
    valid_pid = data["pid"].notna() & data["pid"].astype("string").str.len().gt(0)
    data = data.loc[valid_pid]
    data["pid"] = pd.factorize(data["pid"], sort=False)[0].astype("int32")

    # سایر ستون‌ها واقعاً عددی‌اند و به صورت فشرده تبدیل می‌شوند.
    for column in [
        "survey_year",
        "survey_quarter",
        "gender",
        "age",
        "activity_status",
        "weight_person",
        "activity_code",
        "minutes",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(
        subset=[
            "survey_year",
            "survey_quarter",
            "gender",
            "age",
            "weight_person",
            "activity_code",
            "minutes",
        ]
    )
    data = data.loc[data["weight_person"].gt(0)]

    if data.empty:
        raise ValueError(
            "پس از پاک‌سازی داده، هیچ مشاهده معتبری باقی نماند. "
            "ساختار ستون‌های فایل پردازش‌شده را بررسی کنید."
        )

    data["survey_year"] = data["survey_year"].astype("int16")
    data["survey_quarter"] = data["survey_quarter"].astype("int8")
    data["gender"] = data["gender"].astype("int8")
    data["age"] = data["age"].astype("int16")
    data["activity_status"] = data["activity_status"].astype("Int16")
    data["activity_code"] = data["activity_code"].astype("int16")
    data["minutes"] = data["minutes"].astype("int16")
    data["weight_person"] = data["weight_person"].astype("float64")

    return data


@st.cache_data(show_spinner=False)
def load_activity_catalog() -> pd.DataFrame:
    if not LABEL_FILE.exists():
        return pd.DataFrame(
            columns=["activity_code", "activity_label", "activity_main_label", "activity_display"]
        )

    labels = pd.read_csv(LABEL_FILE)
    labels["activity_code"] = pd.to_numeric(
        labels["activity_code"], errors="coerce"
    ).astype("Int16")
    labels = labels.dropna(subset=["activity_code"]).copy()
    labels["activity_code"] = labels["activity_code"].astype("int16")

    if "activity_label" not in labels.columns:
        labels["activity_label"] = labels["activity_code"].map(
            lambda code: f"فعالیت با کد {int(code)}"
        )
    else:
        labels["activity_label"] = labels["activity_label"].fillna(
            labels["activity_code"].map(lambda code: f"فعالیت با کد {int(code)}")
        )

    if "activity_main_label" not in labels.columns:
        labels["activity_main_label"] = "سایر/نامشخص"
    else:
        labels["activity_main_label"] = labels["activity_main_label"].fillna(
            "سایر/نامشخص"
        )

    if "activity_display" not in labels.columns:
        labels["activity_display"] = (
            labels["activity_code"].astype(str) + " — " + labels["activity_label"]
        )
    else:
        missing = labels["activity_display"].isna()
        labels.loc[missing, "activity_display"] = (
            labels.loc[missing, "activity_code"].astype(str)
            + " — "
            + labels.loc[missing, "activity_label"]
        )

    return labels[
        ["activity_code", "activity_label", "activity_main_label", "activity_display"]
    ].drop_duplicates("activity_code")


def build_catalog(data: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    present = pd.DataFrame(
        {"activity_code": np.sort(data["activity_code"].dropna().unique())}
    )
    catalog = present.merge(labels, on="activity_code", how="left")
    fallback = catalog["activity_code"].map(
        lambda code: f"فعالیت با کد {int(code)}"
    )
    catalog["activity_label"] = catalog["activity_label"].fillna(fallback)
    catalog["activity_main_label"] = catalog["activity_main_label"].fillna(
        "سایر/نامشخص"
    )
    catalog["activity_display"] = catalog["activity_display"].fillna(
        catalog["activity_code"].astype(str) + " — " + catalog["activity_label"]
    )
    return catalog.sort_values("activity_code").reset_index(drop=True)


def person_denominators(base: pd.DataFrame, use_weights: bool) -> pd.DataFrame:
    persons = base[
        ["pid", "survey_year", "survey_quarter", "weight_person"]
    ].drop_duplicates(["pid", "survey_year", "survey_quarter"])
    if use_weights:
        analysis_weight = persons["weight_person"].to_numpy(dtype="float64")
    else:
        analysis_weight = np.ones(len(persons), dtype="float64")
    persons = persons.assign(
        analysis_weight=analysis_weight,
        sample_person=np.ones(len(persons), dtype="int8"),
    )
    return (
        persons.groupby(["survey_year", "survey_quarter"], observed=True)
        .agg(
            population_weight=("analysis_weight", "sum"),
            sample_size=("sample_person", "sum"),
        )
        .reset_index()
    )


def add_period_columns(result: pd.DataFrame) -> pd.DataFrame:
    output = result.copy()
    output["survey_year"] = output["survey_year"].astype(int)
    output["survey_quarter"] = output["survey_quarter"].astype(int)
    output["period_order"] = output["survey_year"] * 10 + output["survey_quarter"]
    output["period_label"] = [
        period_label(year, quarter)
        for year, quarter in zip(output["survey_year"], output["survey_quarter"])
    ]
    return output.sort_values("period_order").reset_index(drop=True)


def calculate_combined_series(
    base: pd.DataFrame,
    selected_codes: list[int],
    use_weights: bool,
) -> pd.DataFrame:
    denominator = person_denominators(base, use_weights)

    selected = base.loc[
        base["activity_code"].isin(selected_codes),
        ["pid", "survey_year", "survey_quarter", "weight_person", "minutes"],
    ]
    person_minutes = (
        selected.groupby(
            ["pid", "survey_year", "survey_quarter", "weight_person"],
            observed=True,
        )["minutes"]
        .sum()
        .reset_index()
    )

    persons = base[
        ["pid", "survey_year", "survey_quarter", "weight_person"]
    ].drop_duplicates(["pid", "survey_year", "survey_quarter"])
    persons = persons.merge(
        person_minutes,
        on=["pid", "survey_year", "survey_quarter", "weight_person"],
        how="left",
    )
    persons["minutes"] = persons["minutes"].fillna(0.0)
    persons["participant"] = persons["minutes"].gt(0).astype("int8")
    persons["analysis_weight"] = (
        persons["weight_person"] if use_weights else 1.0
    )
    persons["weighted_minutes"] = (
        persons["analysis_weight"] * persons["minutes"]
    )
    persons["participant_weight"] = (
        persons["analysis_weight"] * persons["participant"]
    )

    numerators = (
        persons.groupby(["survey_year", "survey_quarter"], observed=True)
        .agg(
            weighted_minutes=("weighted_minutes", "sum"),
            participant_weight=("participant_weight", "sum"),
            participant_sample=("participant", "sum"),
        )
        .reset_index()
    )

    result = denominator.merge(
        numerators, on=["survey_year", "survey_quarter"], how="left"
    )
    for column in ["weighted_minutes", "participant_weight", "participant_sample"]:
        result[column] = result[column].fillna(0.0)

    result["mean_all_minutes"] = (
        result["weighted_minutes"] / result["population_weight"]
    )
    result["participation_rate"] = np.where(
        result["population_weight"].gt(0),
        100 * result["participant_weight"] / result["population_weight"],
        np.nan,
    )
    result["mean_participants_minutes"] = np.where(
        result["participant_weight"].gt(0),
        result["weighted_minutes"] / result["participant_weight"],
        np.nan,
    )
    return add_period_columns(result)


def calculate_separate_series(
    base: pd.DataFrame,
    selected_codes: list[int],
    use_weights: bool,
    code_info: pd.DataFrame,
) -> pd.DataFrame:
    denominator = person_denominators(base, use_weights)
    selected = base.loc[
        base["activity_code"].isin(selected_codes),
        [
            "pid",
            "activity_code",
            "survey_year",
            "survey_quarter",
            "weight_person",
            "minutes",
        ],
    ].copy()
    selected["analysis_weight"] = selected["weight_person"] if use_weights else 1.0
    selected["weighted_minutes"] = selected["analysis_weight"] * selected["minutes"]

    numerators = (
        selected.groupby(
            ["activity_code", "survey_year", "survey_quarter"],
            observed=True,
        )
        .agg(
            weighted_minutes=("weighted_minutes", "sum"),
            participant_weight=("analysis_weight", "sum"),
            participant_sample=("pid", "nunique"),
        )
        .reset_index()
    )

    periods = denominator[["survey_year", "survey_quarter"]].drop_duplicates()
    grid = code_info[["activity_code", "activity_display"]].merge(
        periods, how="cross"
    )
    result = (
        grid.merge(
            numerators,
            on=["activity_code", "survey_year", "survey_quarter"],
            how="left",
        )
        .merge(denominator, on=["survey_year", "survey_quarter"], how="left")
    )

    for column in ["weighted_minutes", "participant_weight", "participant_sample"]:
        result[column] = result[column].fillna(0.0)

    result["mean_all_minutes"] = (
        result["weighted_minutes"] / result["population_weight"]
    )
    result["participation_rate"] = np.where(
        result["population_weight"].gt(0),
        100 * result["participant_weight"] / result["population_weight"],
        np.nan,
    )
    result["mean_participants_minutes"] = np.where(
        result["participant_weight"].gt(0),
        result["weighted_minutes"] / result["participant_weight"],
        np.nan,
    )
    return add_period_columns(result)


def convert_time_unit(result: pd.DataFrame, unit: str) -> pd.DataFrame:
    output = result.copy()
    divisor = 60.0 if unit == "ساعت" else 1.0
    output["mean_all"] = output["mean_all_minutes"] / divisor
    output["mean_participants"] = output["mean_participants_minutes"] / divisor
    return output


def combined_chart(
    result: pd.DataFrame,
    show_mean_all: bool,
    show_mean_participants: bool,
    show_participation_rate: bool,
    time_unit: str,
    chart_title: str,
) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])

    if show_mean_all:
        figure.add_trace(
            go.Bar(
                x=result["period_label"],
                y=result["mean_all"],
                name=f"متوسط زمان در کل جامعه ({time_unit})",
                marker_color="#3548F4",
                customdata=np.column_stack(
                    [result["sample_size"], result["participant_sample"]]
                ),
                hovertemplate=(
                    "%{x}<br>متوسط کل جامعه: %{y:.3f} "
                    + time_unit
                    + "<br>حجم نمونه: %{customdata[0]:,.0f}"
                    + "<br>مشارکت‌کنندگان نمونه: %{customdata[1]:,.0f}<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    if show_mean_participants:
        figure.add_trace(
            go.Bar(
                x=result["period_label"],
                y=result["mean_participants"],
                name=f"متوسط زمان مشارکت‌کنندگان ({time_unit})",
                marker_color="#2E9B32",
                customdata=np.column_stack(
                    [result["sample_size"], result["participant_sample"]]
                ),
                hovertemplate=(
                    "%{x}<br>متوسط مشارکت‌کنندگان: %{y:.3f} "
                    + time_unit
                    + "<br>مشارکت‌کنندگان نمونه: %{customdata[1]:,.0f}<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    if show_participation_rate:
        figure.add_trace(
            go.Scatter(
                x=result["period_label"],
                y=result["participation_rate"],
                name="نرخ مشارکت (درصد)",
                mode="lines+markers",
                line=dict(color="#E31A1C", width=3),
                marker=dict(size=7),
                hovertemplate="%{x}<br>نرخ مشارکت: %{y:.3f} درصد<extra></extra>",
            ),
            secondary_y=True,
        )

    figure.update_layout(
        title=chart_title or None,
        barmode="group",
        height=620,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=30, r=30, t=65 if chart_title else 30, b=90),
        font=dict(family="B Nazanin, BNazanin, Nazanin, Tahoma, Arial", size=15),
        legend=dict(
            orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5
        ),
        xaxis=dict(
            title=None,
            categoryorder="array",
            categoryarray=result["period_label"].tolist(),
            tickangle=-35,
            showgrid=False,
        ),
    )
    figure.update_yaxes(
        title_text=f"متوسط زمان ({time_unit})",
        rangemode="tozero",
        showgrid=False,
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text="نرخ مشارکت (درصد)",
        rangemode="tozero",
        showgrid=False,
        secondary_y=True,
    )
    return figure


def separate_chart(
    result: pd.DataFrame,
    metric: str,
    time_unit: str,
    chart_title: str,
) -> go.Figure:
    y_title = (
        "نرخ مشارکت (درصد)"
        if metric == "participation_rate"
        else f"{METRIC_LABELS[metric]} ({time_unit})"
    )
    figure = px.line(
        result,
        x="period_label",
        y=metric,
        color="activity_display",
        markers=True,
        labels={
            "period_label": "دوره",
            metric: y_title,
            "activity_display": "فعالیت",
        },
        title=chart_title or None,
        category_orders={
            "period_label": (
                result[["period_order", "period_label"]]
                .drop_duplicates()
                .sort_values("period_order")["period_label"]
                .tolist()
            )
        },
        hover_data={
            "sample_size": ":,.0f",
            "participant_sample": ":,.0f",
            "period_order": False,
        },
    )
    figure.update_traces(line=dict(width=2.5), marker=dict(size=7))
    figure.update_layout(
        height=620,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=30, r=20, t=65 if chart_title else 30, b=90),
        font=dict(family="B Nazanin, BNazanin, Nazanin, Tahoma, Arial", size=15),
        legend=dict(
            orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5
        ),
        xaxis=dict(title=None, tickangle=-35, showgrid=False),
        yaxis=dict(title=y_title, rangemode="tozero", showgrid=False),
    )
    return figure


st.title("روند زمانی فعالیت‌ها")
st.caption(
    "کد یا مجموعه‌ای از کدهای فعالیت را انتخاب کنید و تغییرات متوسط زمان و نرخ مشارکت را در دوره‌های آمارگیری مشاهده کنید."
)

try:
    df = load_activity_data()
    labels = load_activity_catalog()
except (FileNotFoundError, KeyError) as exc:
    st.error(str(exc))
    st.code("python prepare_data.py\npython -m streamlit run app.py")
    st.stop()

all_years = sorted(int(value) for value in df["survey_year"].dropna().unique())
all_quarters = sorted(int(value) for value in df["survey_quarter"].dropna().unique())
age_values = df["age"].dropna()
if age_values.empty:
    st.error("ستون سن پس از پاک‌سازی فاقد مقدار معتبر است.")
    st.stop()
min_age = int(age_values.min())
max_age = int(age_values.max())

with st.sidebar.form("time_series_filters"):
    st.subheader("فیلترهای جامعه مورد بررسی")
    selected_years = st.multiselect(
        "سال آمارگیری", options=all_years, default=all_years
    )
    selected_quarters = st.multiselect(
        "فصل آمارگیری",
        options=all_quarters,
        default=all_quarters,
        format_func=lambda value: QUARTER_LABELS.get(value, str(value)),
    )
    gender_label = st.selectbox("جنسیت", options=list(GENDER_OPTIONS))
    employment_label = st.selectbox(
        "وضعیت اشتغال", options=list(EMPLOYMENT_OPTIONS)
    )
    age_range = st.slider(
        "دامنه سنی",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
    )
    weighting_label = st.radio(
        "روش وزن‌دهی", options=["وزن طرح", "بدون وزن"], index=0
    )
    time_unit = st.radio(
        "واحد زمان", options=["ساعت", "دقیقه"], index=0, horizontal=True
    )
    st.form_submit_button("اعمال فیلترها", width="stretch")

# همه شروط در یک ماسک ساخته می‌شوند تا فقط یک بار از داده اصلی کپی گرفته شود.
mask = (
    df["survey_year"].isin(selected_years)
    & df["survey_quarter"].isin(selected_quarters)
    & df["age"].between(age_range[0], age_range[1])
)
selected_gender = GENDER_OPTIONS[gender_label]
if selected_gender is not None:
    mask &= df["gender"].eq(selected_gender)

selected_employment = EMPLOYMENT_OPTIONS[employment_label]
if selected_employment == 1:
    mask &= df["activity_status"].isin([1, 2])
elif selected_employment == 0:
    mask &= df["activity_status"].notna() & ~df["activity_status"].isin([1, 2])

base = df.loc[mask, DATA_COLUMNS]
if base.empty:
    st.warning("برای فیلترهای انتخاب‌شده مشاهده‌ای وجود ندارد.")
    st.stop()

activity_catalog = build_catalog(base, labels)

st.subheader("انتخاب فعالیت")
main_groups = sorted(activity_catalog["activity_main_label"].dropna().unique())
selected_main_groups = st.multiselect(
    "محدودکردن فهرست بر اساس گروه اصلی",
    options=main_groups,
    default=main_groups,
)
code_catalog = activity_catalog.loc[
    activity_catalog["activity_main_label"].isin(selected_main_groups)
]
code_to_display = dict(
    zip(code_catalog["activity_code"].astype(int), code_catalog["activity_display"])
)
code_options = sorted(code_to_display)
religious_defaults = [code for code in [741, 742, 749] if code in code_options]
default_codes = religious_defaults or code_options[:1]

selected_codes = st.multiselect(
    "کد یا کدهای فعالیت",
    options=code_options,
    default=default_codes,
    format_func=lambda code: code_to_display.get(code, str(code)),
    help="در حالت تجمیعی، زمان همه کدهای انتخاب‌شده برای هر فرد با هم جمع می‌شود.",
)
if not selected_codes:
    st.info("حداقل یک کد فعالیت انتخاب کنید.")
    st.stop()

selection_mode = st.radio(
    "نحوه نمایش چند کد",
    options=[
        "تجمیع کدهای انتخاب‌شده به‌عنوان یک فعالیت",
        "نمایش جداگانه هر کد",
    ],
    index=0,
    horizontal=True,
)

selected_activity_table = activity_catalog.loc[
    activity_catalog["activity_code"].isin(selected_codes),
    ["activity_code", "activity_display", "activity_main_label"],
].sort_values("activity_code")

with st.expander("مشاهده فهرست فعالیت‌های انتخاب‌شده", expanded=False):
    st.dataframe(
        selected_activity_table,
        width="stretch",
        hide_index=True,
        column_config={
            "activity_code": "کد فعالیت",
            "activity_display": "کد و عنوان فعالیت",
            "activity_main_label": "گروه اصلی",
        },
    )

use_weights = weighting_label == "وزن طرح"

if selection_mode.startswith("تجمیع"):
    result = calculate_combined_series(base, selected_codes, use_weights)
    result = convert_time_unit(result, time_unit)

    st.subheader("تنظیم نمودار")
    option_col1, option_col2, option_col3 = st.columns(3)
    show_mean_all = option_col1.checkbox("متوسط زمان در کل جامعه", value=True)
    show_mean_participants = option_col2.checkbox(
        "متوسط زمان مشارکت‌کنندگان", value=True
    )
    show_participation_rate = option_col3.checkbox("نرخ مشارکت", value=True)

    if not any([show_mean_all, show_mean_participants, show_participation_rate]):
        st.warning("حداقل یکی از شاخص‌های نمودار را انتخاب کنید.")
        st.stop()

    default_title = (
        code_to_display[selected_codes[0]]
        if len(selected_codes) == 1
        else "روند زمانی مجموعه فعالیت‌های انتخاب‌شده"
    )
    chart_title = st.text_input("عنوان نمودار", value=default_title)
    figure = combined_chart(
        result,
        show_mean_all,
        show_mean_participants,
        show_participation_rate,
        time_unit,
        chart_title,
    )
    st.plotly_chart(figure, width="stretch")

    table = result[
        [
            "period_label",
            "mean_all",
            "mean_participants",
            "participation_rate",
            "sample_size",
            "participant_sample",
            "population_weight",
            "participant_weight",
        ]
    ].copy()
    st.subheader("جدول داده‌های نمودار")
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "period_label": "دوره آمارگیری",
            "mean_all": st.column_config.NumberColumn(
                f"متوسط زمان در کل جامعه ({time_unit})", format="%.3f"
            ),
            "mean_participants": st.column_config.NumberColumn(
                f"متوسط زمان مشارکت‌کنندگان ({time_unit})", format="%.3f"
            ),
            "participation_rate": st.column_config.NumberColumn(
                "نرخ مشارکت (درصد)", format="%.3f"
            ),
            "sample_size": "تعداد افراد نمونه",
            "participant_sample": "تعداد مشارکت‌کنندگان نمونه",
            "population_weight": st.column_config.NumberColumn(
                "جمع وزن جامعه", format="%.0f"
            ),
            "participant_weight": st.column_config.NumberColumn(
                "جمع وزن مشارکت‌کنندگان", format="%.0f"
            ),
        },
    )
    export_table = table.copy()
    export_table.insert(0, "selected_activity_codes", ",".join(map(str, selected_codes)))
    file_name = "timeuse_activity_time_series_combined.csv"

else:
    if len(selected_codes) > 12:
        st.warning(
            "برای خوانایی نمودار جداگانه، حداکثر ۱۲ کد را انتخاب کنید یا حالت تجمیعی را به کار ببرید."
        )
        st.stop()

    selected_code_info = activity_catalog.loc[
        activity_catalog["activity_code"].isin(selected_codes),
        ["activity_code", "activity_display"],
    ]
    result = calculate_separate_series(
        base, selected_codes, use_weights, selected_code_info
    )
    result = convert_time_unit(result, time_unit)

    selected_metric = st.radio(
        "شاخص نمودار",
        options=list(METRIC_LABELS),
        format_func=lambda value: (
            METRIC_LABELS[value]
            if value == "participation_rate"
            else f"{METRIC_LABELS[value]} ({time_unit})"
        ),
        horizontal=True,
    )
    chart_title = st.text_input(
        "عنوان نمودار", value="مقایسه روند زمانی فعالیت‌های انتخاب‌شده"
    )
    figure = separate_chart(result, selected_metric, time_unit, chart_title)
    st.plotly_chart(figure, width="stretch")

    table = result[
        [
            "activity_code",
            "activity_display",
            "period_label",
            "mean_all",
            "mean_participants",
            "participation_rate",
            "sample_size",
            "participant_sample",
        ]
    ].copy()
    st.subheader("جدول داده‌های نمودار")
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "activity_code": "کد فعالیت",
            "activity_display": "کد و عنوان فعالیت",
            "period_label": "دوره آمارگیری",
            "mean_all": st.column_config.NumberColumn(
                f"متوسط زمان در کل جامعه ({time_unit})", format="%.3f"
            ),
            "mean_participants": st.column_config.NumberColumn(
                f"متوسط زمان مشارکت‌کنندگان ({time_unit})", format="%.3f"
            ),
            "participation_rate": st.column_config.NumberColumn(
                "نرخ مشارکت (درصد)", format="%.3f"
            ),
            "sample_size": "تعداد افراد نمونه",
            "participant_sample": "تعداد مشارکت‌کنندگان نمونه",
        },
    )
    export_table = table.copy()
    file_name = "timeuse_activity_time_series_separate.csv"

st.download_button(
    "دانلود داده‌های نمودار",
    data=export_table.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
    file_name=file_name,
    mime="text/csv",
    width="stretch",
)
