from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_style import apply_fa_style

DATA_FILE = Path("data/processed/timeuse_person_activity.parquet")
FALLBACK_FILE = Path("data/processed/timeuse_person_activity.pkl.gz")
LABEL_FILE = Path("data/processed/activity_labels.csv")

REQUIRED_COLUMNS = [
    "pid",
    "wave",
    "survey_year",
    "survey_quarter",
    "gender",
    "age",
    "education_level",
    "marital_status",
    "activity_status",
    "weight_person",
    "activity_code",
    "minutes",
]

GENDER_LABELS = {1: "مرد", 2: "زن"}
WAVE_LABELS = {8788: "۱۳۸۷–۱۳۸۸", 9394: "۱۳۹۳–۱۳۹۴", 9899: "۱۳۹۸–۱۳۹۹"}
QUARTER_LABELS = {1: "بهار", 2: "تابستان", 3: "پاییز", 4: "زمستان"}
MARITAL_LABELS = {1: "دارای همسر", 2: "بی‌همسر بر اثر فوت", 3: "بی‌همسر بر اثر طلاق", 4: "هرگز ازدواج‌نکرده"}
EDUCATION_LABELS = {
    1: "بی‌سواد",
    2: "ابتدایی",
    3: "راهنمایی / متوسطه اول",
    4: "متوسطه دوم",
    5: "دیپلم",
    6: "فوق‌دیپلم",
    7: "کارشناسی",
    8: "کارشناسی ارشد",
    9: "دکتری",
}

# The eight-category system matches the daily-profile page. Broad ICATUS groups
# 2 and 5 are combined into "other" so the full day still sums to 1,440 minutes.
CATEGORY_SPECS = [
    ("paid", "کار با مزد", (1,), "#3A8FBD"),
    ("home", "خدمات خانگی", (3,), "#35AD8C"),
    ("care", "مراقبت بدون مزد", (4,), "#E07B2D"),
    ("learning", "یادگیری", (6,), "#E8A7C8"),
    ("communication", "ارتباطات و مذهب", (7,), "#79C6EA"),
    ("leisure", "فرهنگ و فراغت", (8,), "#F1B433"),
    ("selfcare", "رسیدگی و خودمراقبتی", (9,), "#CF8EB4"),
    ("other", "سایر", (2, 5), "#9B9B9B"),
]
CATEGORY_BY_KEY = {key: (label, codes, color) for key, label, codes, color in CATEGORY_SPECS}
CATEGORY_KEY_BY_BROAD = {
    broad: key
    for key, _label, codes, _color in CATEGORY_SPECS
    for broad in codes
}

TOP_METRICS = {
    "mean_minutes": "متوسط زمان در کل جامعه",
    "participation_rate": "نرخ مشارکت",
    "participant_mean": "متوسط زمان مشارکت‌کنندگان",
}

apply_fa_style()


@st.cache_resource(show_spinner="در حال بارگذاری داده‌های داشبورد...")
def load_data() -> pd.DataFrame:
    """Load only the columns needed by the overview page and reduce memory use."""
    if DATA_FILE.exists():
        data = pd.read_parquet(DATA_FILE, columns=REQUIRED_COLUMNS)
    elif FALLBACK_FILE.exists():
        data = pd.read_pickle(FALLBACK_FILE, compression="gzip")
        missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"ستون‌های ضروری در فایل داده وجود ندارند: {missing}")
        data = data[REQUIRED_COLUMNS]
    else:
        raise FileNotFoundError(
            "فایل داده داشبورد پیدا نشد. ابتدا prepare_data.py را اجرا کنید."
        )

    # Keep a compact numeric ID rather than 1.2 million repeated strings.
    person_codes, _ = pd.factorize(data.pop("pid"), sort=False)
    data["person_id"] = person_codes.astype("int32", copy=False)

    numeric_types = {
        "wave": "int16",
        "survey_year": "int16",
        "survey_quarter": "int8",
        "gender": "int8",
        "age": "int8",
        "weight_person": "float32",
        "activity_code": "int16",
        "minutes": "int16",
    }
    for column, dtype in numeric_types.items():
        data[column] = pd.to_numeric(data[column], errors="coerce").astype(dtype)

    data["education_level"] = pd.to_numeric(
        data["education_level"], errors="coerce"
    ).astype("float32")
    data["marital_status"] = pd.to_numeric(
        data["marital_status"], errors="coerce"
    ).astype("float32")
    activity_status = pd.to_numeric(data["activity_status"], errors="coerce")
    data["employed"] = activity_status.isin([1, 2]).astype("int8")

    codes = data["activity_code"].to_numpy(dtype=np.int32, copy=False)
    broad = np.where(codes >= 100_000, codes // 100_000, codes // 100)
    data["broad_group"] = broad.astype("int8", copy=False)

    return data


@st.cache_resource(show_spinner=False)
def load_labels() -> tuple[dict[int, str], dict[int, str]]:
    if not LABEL_FILE.exists():
        return {}, {}

    labels = pd.read_csv(LABEL_FILE)
    labels["activity_code"] = pd.to_numeric(
        labels["activity_code"], errors="coerce"
    )
    labels = labels.dropna(subset=["activity_code"])
    labels["activity_code"] = labels["activity_code"].astype(int)

    activity_labels = dict(
        zip(labels["activity_code"], labels.get("activity_label", ""))
    )
    main_labels = dict(
        zip(labels["activity_code"], labels.get("activity_main_label", ""))
    )
    return activity_labels, main_labels


def to_fa_digits(value: object) -> str:
    translation = str.maketrans("0123456789.,-%", "۰۱۲۳۴۵۶۷۸۹٫،−٪")
    return str(value).translate(translation)


def format_number(value: float | int, decimals: int = 0) -> str:
    if decimals == 0:
        text = f"{value:,.0f}"
    else:
        text = f"{value:,.{decimals}f}"
    return to_fa_digits(text)


def format_duration(minutes: float) -> str:
    if not np.isfinite(minutes):
        return "—"
    rounded = max(0, int(round(minutes)))
    hours, remaining = divmod(rounded, 60)
    if hours and remaining:
        return f"{to_fa_digits(hours)} ساعت و {to_fa_digits(remaining)} دقیقه"
    if hours:
        return f"{to_fa_digits(hours)} ساعت"
    return f"{to_fa_digits(remaining)} دقیقه"


def hex_to_rgba(hex_color: str, alpha: float = 0.14) -> str:
    """Convert a six-digit hex color to Plotly-compatible rgba()."""
    value = str(hex_color).strip().lstrip("#")
    if len(value) != 6:
        return f"rgba(58,143,189,{alpha})"
    try:
        red = int(value[0:2], 16)
        green = int(value[2:4], 16)
        blue = int(value[4:6], 16)
    except ValueError:
        return f"rgba(58,143,189,{alpha})"
    return f"rgba({red},{green},{blue},{alpha})"


def effective_person_weight(persons: pd.DataFrame, weighted: bool) -> pd.Series:
    if weighted:
        return pd.to_numeric(persons["weight_person"], errors="coerce").fillna(0)
    return pd.Series(np.ones(len(persons), dtype=np.float64), index=persons.index)


def total_population_weight(data: pd.DataFrame, weighted: bool) -> float:
    persons = data[["person_id", "weight_person"]].drop_duplicates("person_id")
    return float(effective_person_weight(persons, weighted).sum())


def calculate_activity_statistics(
    data: pd.DataFrame,
    weighted: bool,
    activity_labels: dict[int, str],
    main_labels: dict[int, str],
) -> pd.DataFrame:
    denominator = total_population_weight(data, weighted)
    if denominator <= 0:
        return pd.DataFrame()

    if weighted:
        row_weight = data["weight_person"].to_numpy(dtype=np.float64, copy=False)
    else:
        row_weight = np.ones(len(data), dtype=np.float64)

    weighted_minutes = data["minutes"].to_numpy(dtype=np.float64, copy=False) * row_weight
    numerator = pd.Series(weighted_minutes).groupby(
        data["activity_code"].to_numpy(), sort=False
    ).sum()
    participant_weight = pd.Series(row_weight).groupby(
        data["activity_code"].to_numpy(), sort=False
    ).sum()
    sample_participants = data.groupby("activity_code", sort=False).size()

    result = pd.DataFrame(
        {
            "activity_code": numerator.index.astype(int),
            "weighted_minutes": numerator.to_numpy(),
            "participant_weight": participant_weight.reindex(numerator.index).to_numpy(),
            "sample_participants": sample_participants.reindex(numerator.index).to_numpy(),
        }
    )
    result["mean_minutes"] = result["weighted_minutes"] / denominator
    result["participation_rate"] = (
        100 * result["participant_weight"] / denominator
    )
    result["participant_mean"] = (
        result["weighted_minutes"] / result["participant_weight"]
    )
    result["activity_label"] = result["activity_code"].map(activity_labels)
    result["activity_label"] = result["activity_label"].fillna(
        result["activity_code"].map(lambda value: f"فعالیت با کد {value}")
    )
    result["activity_main_label"] = result["activity_code"].map(main_labels).fillna("")
    result["activity_display"] = (
        result["activity_code"].astype(str) + " — " + result["activity_label"]
    )
    return result


def calculate_composition(data: pd.DataFrame, weighted: bool) -> pd.DataFrame:
    denominator = total_population_weight(data, weighted)
    if denominator <= 0:
        return pd.DataFrame()

    if weighted:
        row_weight = data["weight_person"].to_numpy(dtype=np.float64, copy=False)
    else:
        row_weight = np.ones(len(data), dtype=np.float64)
    weighted_minutes = data["minutes"].to_numpy(dtype=np.float64, copy=False) * row_weight
    by_broad = pd.Series(weighted_minutes).groupby(
        data["broad_group"].to_numpy(), sort=False
    ).sum()

    rows: list[dict[str, object]] = []
    for key, label, broad_codes, color in CATEGORY_SPECS:
        total = float(sum(by_broad.get(code, 0.0) for code in broad_codes))
        mean_minutes = total / denominator
        rows.append(
            {
                "category_key": key,
                "category_label": label,
                "mean_minutes": mean_minutes,
                "share_day": 100 * mean_minutes / 1440,
                "color": color,
            }
        )
    return pd.DataFrame(rows)


def calculate_group_participation(
    data: pd.DataFrame, broad_codes: Iterable[int], weighted: bool
) -> float:
    denominator = total_population_weight(data, weighted)
    if denominator <= 0:
        return np.nan
    participants = data.loc[
        data["broad_group"].isin(tuple(broad_codes)), ["person_id", "weight_person"]
    ].drop_duplicates("person_id")
    return 100 * float(effective_person_weight(participants, weighted).sum()) / denominator


def calculate_gender_activity_gap(
    data: pd.DataFrame,
    weighted: bool,
    activity_labels: dict[int, str],
    top_n: int = 10,
) -> pd.DataFrame:
    data = data.loc[data["gender"].isin([1, 2])]
    if data.empty:
        return pd.DataFrame()

    persons = data[["person_id", "gender", "weight_person"]].drop_duplicates("person_id")
    if weighted:
        persons = persons.assign(_w=persons["weight_person"].astype(float))
        row_weight = data["weight_person"].to_numpy(dtype=np.float64, copy=False)
    else:
        persons = persons.assign(_w=1.0)
        row_weight = np.ones(len(data), dtype=np.float64)

    denominators = persons.groupby("gender", observed=True)["_w"].sum()
    weighted_minutes = data["minutes"].to_numpy(dtype=np.float64, copy=False) * row_weight
    grouped = (
        pd.DataFrame(
            {
                "gender": data["gender"].to_numpy(),
                "activity_code": data["activity_code"].to_numpy(),
                "weighted_minutes": weighted_minutes,
            }
        )
        .groupby(["gender", "activity_code"], observed=True, sort=False)["weighted_minutes"]
        .sum()
        .reset_index()
    )
    grouped["mean_minutes"] = grouped.apply(
        lambda row: row["weighted_minutes"] / denominators.get(row["gender"], np.nan),
        axis=1,
    )
    pivot = grouped.pivot(
        index="activity_code", columns="gender", values="mean_minutes"
    ).fillna(0)
    if 1 not in pivot.columns or 2 not in pivot.columns:
        return pd.DataFrame()

    pivot = pivot.rename(columns={1: "men", 2: "women"})
    pivot["gap"] = pivot["women"] - pivot["men"]
    pivot["abs_gap"] = pivot["gap"].abs()
    result = pivot.nlargest(top_n, "abs_gap").reset_index()
    result["activity_code"] = result["activity_code"].astype(int)
    result["activity_label"] = result["activity_code"].map(activity_labels)
    result["activity_label"] = result["activity_label"].fillna(
        result["activity_code"].map(lambda value: f"فعالیت با کد {value}")
    )
    result["activity_display"] = (
        result["activity_code"].astype(str) + " — " + result["activity_label"]
    )
    return result.sort_values("gap")


def calculate_gender_broad_means(data: pd.DataFrame, weighted: bool) -> dict[int, dict[int, float]]:
    output: dict[int, dict[int, float]] = {}
    for gender in [1, 2]:
        subset = data.loc[data["gender"].eq(gender)]
        denominator = total_population_weight(subset, weighted)
        if denominator <= 0:
            continue
        if weighted:
            row_weight = subset["weight_person"].to_numpy(dtype=np.float64, copy=False)
        else:
            row_weight = np.ones(len(subset), dtype=np.float64)
        weighted_minutes = subset["minutes"].to_numpy(dtype=np.float64, copy=False) * row_weight
        by_broad = pd.Series(weighted_minutes).groupby(
            subset["broad_group"].to_numpy(), sort=False
        ).sum()
        output[gender] = {
            broad: float(by_broad.get(broad, 0.0)) / denominator for broad in range(1, 10)
        }
    return output


def calculate_trend(
    data: pd.DataFrame,
    broad_codes: Iterable[int],
    weighted: bool,
) -> pd.DataFrame:
    periods = (
        data[["survey_year", "survey_quarter"]]
        .drop_duplicates()
        .sort_values(["survey_year", "survey_quarter"])
    )
    persons = data[
        ["person_id", "survey_year", "survey_quarter", "weight_person"]
    ].drop_duplicates("person_id")
    if weighted:
        persons = persons.assign(_w=persons["weight_person"].astype(float))
    else:
        persons = persons.assign(_w=1.0)
    denominator = (
        persons.groupby(["survey_year", "survey_quarter"], observed=True)["_w"]
        .sum()
        .rename("denominator")
    )

    selected = data.loc[data["broad_group"].isin(tuple(broad_codes))]
    if weighted:
        row_weight = selected["weight_person"].to_numpy(dtype=np.float64, copy=False)
    else:
        row_weight = np.ones(len(selected), dtype=np.float64)
    weighted_minutes = selected["minutes"].to_numpy(dtype=np.float64, copy=False) * row_weight
    numerator = (
        pd.DataFrame(
            {
                "survey_year": selected["survey_year"].to_numpy(),
                "survey_quarter": selected["survey_quarter"].to_numpy(),
                "weighted_minutes": weighted_minutes,
            }
        )
        .groupby(["survey_year", "survey_quarter"], observed=True)["weighted_minutes"]
        .sum()
        .rename("numerator")
    )

    result = periods.merge(
        denominator.reset_index(), on=["survey_year", "survey_quarter"], how="left"
    ).merge(
        numerator.reset_index(), on=["survey_year", "survey_quarter"], how="left"
    )
    result["numerator"] = result["numerator"].fillna(0)
    result["mean_minutes"] = result["numerator"] / result["denominator"]
    result["period_order"] = result["survey_year"] * 10 + result["survey_quarter"]
    result["period_label"] = result.apply(
        lambda row: f"{QUARTER_LABELS.get(int(row['survey_quarter']), '')} {to_fa_digits(int(row['survey_year']))}",
        axis=1,
    )
    return result.sort_values("period_order")


def metric_card(icon: str, title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{escape(icon)}</div>
            <div class="metric-title">{escape(title)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-subtitle">{escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def filter_chip(text: str) -> str:
    return f'<span class="filter-chip">{escape(text)}</span>'


def signed_gap_sentence(gap: float, positive_subject: str, negative_subject: str, activity: str) -> str:
    if not np.isfinite(gap) or abs(gap) < 0.5:
        return f"زمان {activity} زنان و مردان تقریباً برابر است"
    if gap > 0:
        return f"{positive_subject} روزانه حدود {format_duration(abs(gap))} بیشتر صرف {activity} می‌کنند"
    return f"{negative_subject} روزانه حدود {format_duration(abs(gap))} بیشتر صرف {activity} می‌کنند"


# -----------------------------------------------------------------------------
# Header and data loading
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="dashboard-hero">
        <div class="hero-eyebrow">طرح گذران وقت ایران</div>
        <div class="hero-title">داشبورد زندگی روزانه</div>
        <div class="hero-subtitle">مروری سریع بر ترکیب شبانه‌روز، فعالیت‌های اصلی، شکاف جنسیتی و تغییرات میان موج‌های آمارگیری</div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    df = load_data()
    activity_labels, main_labels = load_labels()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.code("python prepare_data.py\npython -m streamlit run app.py")
    st.stop()

all_waves = sorted(int(value) for value in df["wave"].dropna().unique())
all_genders = sorted(int(value) for value in df["gender"].dropna().unique())
min_age = int(df["age"].min())
max_age = int(df["age"].max())
all_marital = sorted(int(value) for value in df["marital_status"].dropna().unique())
all_education = sorted(int(value) for value in df["education_level"].dropna().unique())

if st.sidebar.button("بازنشانی فیلترها", width="stretch"):
    for key in [
        "overview_preset",
        "overview_waves",
        "overview_genders",
        "overview_age",
        "overview_employment",
        "overview_marital",
        "overview_education",
        "overview_weighting",
        "overview_top_metric",
        "overview_top_n",
        "overview_trend_group",
    ]:
        st.session_state.pop(key, None)
    st.rerun()

st.sidebar.markdown("### انتخاب سریع جامعه")
preset = st.sidebar.selectbox(
    "الگوی آماده",
    options=["سن کار (۱۵ تا ۶۴ سال)", "کل جمعیت", "شاغلان سن کار", "تنظیم دستی"],
    key="overview_preset",
)

with st.sidebar.form("overview_filters"):
    st.markdown("### فیلترهای جامعه مورد بررسی")
    selected_waves = st.multiselect(
        "موج آمارگیری",
        options=all_waves,
        default=[max(all_waves)],
        format_func=lambda value: WAVE_LABELS.get(value, to_fa_digits(value)),
        key="overview_waves",
    )
    selected_genders = st.multiselect(
        "جنسیت",
        options=all_genders,
        default=all_genders,
        format_func=lambda value: GENDER_LABELS.get(value, str(value)),
        key="overview_genders",
    )

    if preset == "کل جمعیت":
        age_range = (min_age, max_age)
        employment_filter = "همه"
        st.caption(f"سن: {to_fa_digits(min_age)} تا {to_fa_digits(max_age)} سال | وضعیت اشتغال: همه")
    elif preset == "شاغلان سن کار":
        age_range = (max(15, min_age), min(64, max_age))
        employment_filter = "شاغل"
        st.caption("سن: ۱۵ تا ۶۴ سال | فقط شاغلان")
    elif preset == "تنظیم دستی":
        age_range = st.slider(
            "دامنه سنی",
            min_value=min_age,
            max_value=max_age,
            value=(max(15, min_age), min(64, max_age)),
            key="overview_age",
        )
        employment_filter = st.selectbox(
            "وضعیت اشتغال",
            options=["همه", "شاغل", "غیرشاغل"],
            key="overview_employment",
        )
    else:
        age_range = (max(15, min_age), min(64, max_age))
        employment_filter = "همه"
        st.caption("سن: ۱۵ تا ۶۴ سال | وضعیت اشتغال: همه")

    selected_marital = st.multiselect(
        "وضعیت تأهل (خالی = همه)",
        options=all_marital,
        default=[],
        format_func=lambda value: MARITAL_LABELS.get(value, f"کد {value}"),
        key="overview_marital",
    )
    selected_education = st.multiselect(
        "سطح تحصیلات (خالی = همه)",
        options=all_education,
        default=[],
        format_func=lambda value: EDUCATION_LABELS.get(value, f"کد {value}"),
        key="overview_education",
    )
    weighting_label = st.radio(
        "نوع برآورد",
        options=["وزن طرح", "بدون وزن"],
        horizontal=True,
        key="overview_weighting",
    )

    with st.expander("تنظیمات نمایش", expanded=False):
        top_metric = st.selectbox(
            "شاخص نمودار فعالیت‌ها",
            options=list(TOP_METRICS),
            format_func=lambda value: TOP_METRICS[value],
            key="overview_top_metric",
        )
        top_n = st.slider(
            "تعداد فعالیت‌های اصلی",
            min_value=6,
            max_value=15,
            value=10,
            key="overview_top_n",
        )
        trend_key = st.selectbox(
            "گروه فعالیت برای روند کوتاه",
            options=[spec[0] for spec in CATEGORY_SPECS],
            format_func=lambda value: CATEGORY_BY_KEY[value][0],
            key="overview_trend_group",
        )

    submitted = st.form_submit_button("اعمال فیلترها", width="stretch")

if not selected_waves or not selected_genders:
    st.warning("حداقل یک موج آمارگیری و یک جنسیت را انتخاب کنید.")
    st.stop()

weighted = weighting_label == "وزن طرح"
base_mask = (
    df["wave"].isin(selected_waves)
    & df["age"].between(age_range[0], age_range[1])
)
if employment_filter == "شاغل":
    base_mask &= df["employed"].eq(1)
elif employment_filter == "غیرشاغل":
    base_mask &= df["employed"].eq(0)
if selected_marital:
    base_mask &= df["marital_status"].isin(selected_marital)
if selected_education:
    base_mask &= df["education_level"].isin(selected_education)

comparison_base = df.loc[base_mask]
filtered = comparison_base.loc[comparison_base["gender"].isin(selected_genders)]

if filtered.empty:
    st.warning("برای فیلترهای انتخاب‌شده مشاهده‌ای وجود ندارد.")
    st.stop()

# Filter summary chips
wave_text = "، ".join(WAVE_LABELS.get(value, str(value)) for value in selected_waves)
gender_text = " و ".join(GENDER_LABELS.get(value, str(value)) for value in selected_genders)
chips = [
    filter_chip(f"موج: {wave_text}"),
    filter_chip(f"جنسیت: {gender_text}"),
    filter_chip(f"سن: {to_fa_digits(age_range[0])} تا {to_fa_digits(age_range[1])}"),
    filter_chip(f"اشتغال: {employment_filter}"),
    filter_chip(weighting_label),
]
if selected_marital:
    chips.append(filter_chip("تأهل: " + "، ".join(MARITAL_LABELS.get(v, str(v)) for v in selected_marital)))
if selected_education:
    chips.append(filter_chip("تحصیلات: " + "، ".join(EDUCATION_LABELS.get(v, str(v)) for v in selected_education)))
st.markdown('<div class="filter-summary">' + "".join(chips) + "</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Core statistics and cards
# -----------------------------------------------------------------------------
stats = calculate_activity_statistics(
    filtered, weighted, activity_labels, main_labels
)
composition = calculate_composition(filtered, weighted)
if stats.empty or composition.empty:
    st.warning("محاسبه شاخص‌ها برای این انتخاب ممکن نبود.")
    st.stop()

persons = filtered[["person_id", "weight_person"]].drop_duplicates("person_id")
sample_size = len(persons)
composition_by_key = composition.set_index("category_key")["mean_minutes"].to_dict()
paid_minutes = composition_by_key.get("paid", 0.0)
homecare_minutes = composition_by_key.get("home", 0.0) + composition_by_key.get("care", 0.0)
leisure_minutes = composition_by_key.get("leisure", 0.0)
paid_participation = calculate_group_participation(filtered, (1,), weighted)

card_columns = st.columns(5)
with card_columns[0]:
    metric_card("👥", "تعداد افراد نمونه", format_number(sample_size), "دفترچه کامل")
with card_columns[1]:
    metric_card("💼", "کار با مزد", format_duration(paid_minutes), "متوسط روزانه")
with card_columns[2]:
    metric_card("🏠", "کار خانگی و مراقبت", format_duration(homecare_minutes), "متوسط روزانه")
with card_columns[3]:
    metric_card("🎭", "فرهنگ و فراغت", format_duration(leisure_minutes), "متوسط روزانه")
with card_columns[4]:
    metric_card("📌", "مشارکت در کار با مزد", f"{format_number(paid_participation, 1)}٪", "حداقل ۱۵ دقیقه")

# -----------------------------------------------------------------------------
# 24-hour composition
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">ترکیب متوسط ۲۴ ساعت</div>', unsafe_allow_html=True)
st.caption("متوسط زمان روزانه هر فرد در جامعه انتخاب‌شده؛ مجموع اجزای نمودار برابر با کل شبانه‌روز است.")

composition_fig = go.Figure()
for row in composition.itertuples(index=False):
    text = format_duration(row.mean_minutes) if row.mean_minutes >= 35 else ""
    composition_fig.add_trace(
        go.Bar(
            y=["شبانه‌روز"],
            x=[row.mean_minutes],
            name=row.category_label,
            orientation="h",
            marker_color=row.color,
            text=[text],
            textposition="inside",
            insidetextanchor="middle",
            customdata=[[row.category_label, row.share_day, format_duration(row.mean_minutes)]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "زمان: %{customdata[2]}<br>"
                "سهم از شبانه‌روز: %{customdata[1]:.1f}٪<extra></extra>"
            ),
        )
    )
composition_fig.update_layout(
    barmode="stack",
    height=230,
    margin=dict(l=10, r=10, t=10, b=70),
    xaxis=dict(range=[0, 1440], tickvals=[0, 240, 480, 720, 960, 1200, 1440], ticktext=["۰", "۴", "۸", "۱۲", "۱۶", "۲۰", "۲۴"], title="ساعت"),
    yaxis=dict(showticklabels=False, title=None),
    legend=dict(orientation="h", yanchor="top", y=-0.32, xanchor="center", x=0.5),
    font=dict(family="B Nazanin, BNazanin, Nazanin, Tahoma, Arial", size=14),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(composition_fig, width="stretch", config={"displayModeBar": False})

# -----------------------------------------------------------------------------
# Top activities and gender gap
# -----------------------------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-title small">فعالیت‌های برجسته</div>', unsafe_allow_html=True)
    plot_data = stats.nlargest(top_n, top_metric).sort_values(top_metric)
    metric_is_percent = top_metric == "participation_rate"
    values = plot_data[top_metric]
    bar_fig = go.Figure(
        go.Bar(
            x=values,
            y=plot_data["activity_display"],
            orientation="h",
            marker_color="#3A8FBD",
            customdata=np.column_stack(
                [
                    plot_data["activity_label"],
                    plot_data["mean_minutes"],
                    plot_data["participation_rate"],
                    plot_data["participant_mean"],
                ]
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "متوسط کل جامعه: %{customdata[1]:.1f} دقیقه<br>"
                "نرخ مشارکت: %{customdata[2]:.1f}٪<br>"
                "متوسط مشارکت‌کنندگان: %{customdata[3]:.1f} دقیقه<extra></extra>"
            ),
        )
    )
    bar_fig.update_layout(
        height=max(420, 37 * top_n),
        margin=dict(l=15, r=15, t=10, b=35),
        xaxis_title="درصد" if metric_is_percent else "دقیقه در روز",
        yaxis_title=None,
        yaxis=dict(automargin=True),
        font=dict(family="B Nazanin, BNazanin, Nazanin, Tahoma, Arial", size=13),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(bar_fig, width="stretch", config={"displayModeBar": False})

with right:
    st.markdown('<div class="section-title small">شکاف زمانی زنان و مردان</div>', unsafe_allow_html=True)
    st.caption("مقادیر مثبت نشان‌دهنده زمان بیشتر زنان و مقادیر منفی نشان‌دهنده زمان بیشتر مردان است؛ فیلتر جنسیت در این نمودار اعمال نمی‌شود.")
    gap_data = calculate_gender_activity_gap(
        comparison_base, weighted, activity_labels, top_n=top_n
    )
    if gap_data.empty:
        st.info("برای محاسبه شکاف جنسیتی، مشاهده کافی از هر دو جنس وجود ندارد.")
    else:
        colors = np.where(gap_data["gap"] >= 0, "#C75B8B", "#3A8FBD")
        gap_fig = go.Figure(
            go.Bar(
                x=gap_data["gap"],
                y=gap_data["activity_display"],
                orientation="h",
                marker_color=colors,
                customdata=np.column_stack(
                    [gap_data["activity_label"], gap_data["women"], gap_data["men"]]
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "زنان: %{customdata[1]:.1f} دقیقه<br>"
                    "مردان: %{customdata[2]:.1f} دقیقه<br>"
                    "شکاف زنان منهای مردان: %{x:.1f} دقیقه<extra></extra>"
                ),
            )
        )
        gap_fig.add_vline(x=0, line_width=1, line_color="#6B7280")
        gap_fig.update_layout(
            height=max(420, 37 * top_n),
            margin=dict(l=15, r=15, t=10, b=35),
            xaxis_title="دقیقه در روز؛ زنان منهای مردان",
            yaxis_title=None,
            yaxis=dict(automargin=True),
            font=dict(family="B Nazanin, BNazanin, Nazanin, Tahoma, Arial", size=13),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(gap_fig, width="stretch", config={"displayModeBar": False})

# -----------------------------------------------------------------------------
# Automated descriptive insight
# -----------------------------------------------------------------------------
top_activity = stats.nlargest(1, "mean_minutes").iloc[0]
gender_broad = calculate_gender_broad_means(comparison_base, weighted)
insight_sentences = [
    f"در جامعه انتخاب‌شده، «{escape(str(top_activity['activity_label']))}» با متوسط {format_duration(float(top_activity['mean_minutes']))} در روز، بیشترین زمان را به خود اختصاص می‌دهد."
]
if 1 in gender_broad and 2 in gender_broad:
    women_homecare = gender_broad[2].get(3, 0) + gender_broad[2].get(4, 0)
    men_homecare = gender_broad[1].get(3, 0) + gender_broad[1].get(4, 0)
    paid_gap = gender_broad[2].get(1, 0) - gender_broad[1].get(1, 0)
    homecare_gap = women_homecare - men_homecare
    insight_sentences.append(
        signed_gap_sentence(homecare_gap, "زنان", "مردان", "خدمات خانگی و مراقبت") + "."
    )
    insight_sentences.append(
        signed_gap_sentence(paid_gap, "زنان", "مردان", "کار با مزد") + "."
    )

st.markdown(
    """
    <div class="insight-box">
        <div class="insight-title">برداشت سریع از داده‌ها</div>
        <div class="insight-text">{}</div>
    </div>
    """.format(" ".join(insight_sentences)),
    unsafe_allow_html=True,
)
st.caption("این متن صرفاً توصیفی است و نباید به‌عنوان تفسیر علّی خوانده شود.")

# -----------------------------------------------------------------------------
# Short trend
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">تغییرات میان دوره‌های آمارگیری</div>', unsafe_allow_html=True)
trend_label, trend_codes, trend_color = CATEGORY_BY_KEY[trend_key]
trend = calculate_trend(filtered, trend_codes, weighted)
trend_fig = go.Figure(
    go.Scatter(
        x=trend["period_label"],
        y=trend["mean_minutes"],
        mode="lines+markers",
        line=dict(color=trend_color, width=3),
        marker=dict(size=9),
        fill="tozeroy",
        fillcolor=hex_to_rgba(trend_color, 0.14),
        customdata=trend[["survey_year", "survey_quarter"]],
        hovertemplate=(
            "<b>%{x}</b><br>"
            + escape(trend_label)
            + ": %{y:.1f} دقیقه در روز<extra></extra>"
        ),
    )
)
trend_fig.update_layout(
    height=360,
    margin=dict(l=20, r=20, t=15, b=80),
    xaxis=dict(title=None, tickangle=-35),
    yaxis=dict(title="متوسط دقیقه در روز", rangemode="tozero", gridcolor="#E5E7EB"),
    font=dict(family="B Nazanin, BNazanin, Nazanin, Tahoma, Arial", size=14),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(trend_fig, width="stretch", config={"displayModeBar": False})
try:
    st.page_link(
        "pages/3_روند_زمانی_فعالیت‌ها.py",
        label="مشاهده تحلیل کامل روند زمانی فعالیت‌ها",
        icon="📈",
    )
except Exception:
    st.caption("برای جزئیات بیشتر، صفحه «روند زمانی فعالیت‌ها» را از منوی کناری باز کنید.")

# -----------------------------------------------------------------------------
# Detailed table and download
# -----------------------------------------------------------------------------
with st.expander("جدول تفصیلی و دریافت داده", expanded=False):
    result_table = stats[
        [
            "activity_code",
            "activity_label",
            "activity_main_label",
            "mean_minutes",
            "participation_rate",
            "participant_mean",
            "sample_participants",
        ]
    ].sort_values("mean_minutes", ascending=False)
    st.dataframe(
        result_table,
        width="stretch",
        hide_index=True,
        column_config={
            "activity_code": "کد فعالیت",
            "activity_label": "عنوان فعالیت",
            "activity_main_label": "گروه اصلی",
            "mean_minutes": st.column_config.NumberColumn(
                "متوسط کل جامعه (دقیقه)", format="%.2f"
            ),
            "participation_rate": st.column_config.NumberColumn(
                "نرخ مشارکت (درصد)", format="%.2f"
            ),
            "participant_mean": st.column_config.NumberColumn(
                "متوسط مشارکت‌کنندگان (دقیقه)", format="%.2f"
            ),
            "sample_participants": "مشارکت‌کنندگان نمونه",
        },
    )
    csv_data = result_table.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "دریافت جدول CSV",
        data=csv_data,
        file_name="timeuse_overview_results.csv",
        mime="text/csv",
        width="stretch",
    )
