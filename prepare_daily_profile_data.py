"""Build the four-wave, diary-level time-of-day dashboard file.

Source: the harmonized Stata diary dataset, selected via TIMEUSE_DTA.
Destination: data/processed/timeuse_diary_wide.parquet

Compared with the previous script, this version preserves missing survey_month
(and other nullable demographic fields) instead of coercing them to non-nullable
NumPy integers. It validates the output and replaces the previous Parquet file
only after the new file has been written and checked successfully.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW_FILE = Path(os.environ.get("TIMEUSE_DTA", "data/raw/TimeUse_Diary_v2_2.dta"))
OUTPUT_FILE = Path("data/processed/timeuse_diary_wide.parquet")
CHUNK_SIZE = 5_000

BASE_COLUMNS = [
    "pid",
    "wave",
    "survey_year",
    "survey_quarter",
    "survey_month",
    "gender",
    "age",
    "education_level",
    "marital_status",
    "activity_status",
    "employment_status",
    "relation_head",
    "weight_person",
    "complete_primary_diary",
]
Q2_COLUMNS = [f"Q2Code{i}" for i in range(1, 97)]
BROAD_COLUMNS = [f"broad{i}" for i in range(1, 97)]


def to_broad_activity(values: pd.Series) -> np.ndarray:
    """Map 3/6-digit ICATUS codes to broad activity categories 1..9."""
    codes = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    broad = np.full(codes.shape, np.nan, dtype="float64")
    three_digit = (codes >= 100) & (codes <= 999)
    six_digit = (codes >= 100_000) & (codes <= 999_999)
    broad[three_digit] = np.floor(codes[three_digit] / 100)
    broad[six_digit] = np.floor(codes[six_digit] / 100_000)
    return broad


def prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Retain complete diaries and compute 96 broad activities per person."""
    chunk = chunk.loc[
        chunk["complete_primary_diary"].eq(1)
        & chunk["weight_person"].notna()
        & chunk["weight_person"].gt(0)
        & chunk["gender"].isin([1, 2])
    ].copy()
    if chunk.empty:
        return pd.DataFrame()

    valid_diary = np.ones(len(chunk), dtype=bool)
    for slot, source_col in enumerate(Q2_COLUMNS, start=1):
        broad = to_broad_activity(chunk[source_col])
        valid_slot = np.isfinite(broad) & (broad >= 1) & (broad <= 9)
        valid_diary &= valid_slot
        chunk[f"broad{slot}"] = np.where(valid_slot, broad, 0).astype("uint8")

    chunk = chunk.loc[valid_diary].copy()
    if chunk.empty:
        return pd.DataFrame()

    activity_status = pd.to_numeric(chunk["activity_status"], errors="coerce")
    chunk["employed"] = np.where(
        activity_status.isna(),
        np.nan,
        activity_status.isin([1, 2]).astype("float32"),
    )
    chunk["adult_sample"] = chunk["age"].between(15, 64).astype("uint8")
    chunk["sibling_sample"] = (
        chunk["relation_head"].eq(3) & chunk["age"].between(6, 24)
    ).astype("uint8")

    keep_columns = [
        c for c in BASE_COLUMNS if c != "complete_primary_diary"
    ] + ["employed", "adult_sample", "sibling_sample"] + BROAD_COLUMNS
    return chunk[keep_columns]


def numeric_nullable(series: pd.Series, dtype: str) -> pd.Series:
    """Retain NA instead of crashing with NumPy int8/int16 conversions."""
    parsed = pd.to_numeric(series, errors="coerce")
    parsed = parsed.replace([np.inf, -np.inf], np.nan)
    return parsed.astype(dtype)


def main() -> None:
    if not RAW_FILE.is_file():
        raise FileNotFoundError(
            f"Source .dta not found: {RAW_FILE.resolve()}\n"
            "Set TIMEUSE_DTA to TimeUse_Diary_v2_2.dta before running."
        )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Reading:", RAW_FILE.resolve())
    reader = pd.read_stata(
        RAW_FILE,
        columns=BASE_COLUMNS + Q2_COLUMNS,
        convert_categoricals=False,
        chunksize=CHUNK_SIZE,
    )

    parts: list[pd.DataFrame] = []
    read_total = 0
    for number, chunk in enumerate(reader, start=1):
        part = prepare_chunk(chunk)
        if not part.empty:
            parts.append(part)
        read_total += len(chunk)
        print(
            f"Chunk {number:02d}: read {len(chunk):,}; "
            f"retained {len(part):,}; total read {read_total:,}"
        )

    if not parts:
        raise RuntimeError("No valid, complete activity diaries found.")

    output = pd.concat(parts, ignore_index=True)
    if not output["pid"].is_unique:
        raise AssertionError("Duplicate pid values in prepared diary file")

    # Two identifiers must be nonmissing, other columns may genuinely be NA.
    for col in ("wave", "survey_year"):
        nums = pd.to_numeric(output[col], errors="coerce")
        if nums.isna().any():
            raise AssertionError(f"Required field {col} contains missing values")
        output[col] = nums.astype("int16")

    # CRITICAL FIX: pandas nullable integer types allow NA/NaN in survey_month
    # and age. Do not fabricate a survey month or convert an unknown into zero.
    for col in ("survey_quarter", "survey_month", "gender"):
        output[col] = numeric_nullable(output[col], "Int8")
    output["age"] = numeric_nullable(output["age"], "Int16")

    # A negative age or age above 110 is not a valid age for dashboard filters.
    invalid_age = output["age"].notna() & ~output["age"].between(0, 110)
    n_invalid_age = int(invalid_age.sum())
    if n_invalid_age:
        output.loc[invalid_age, "age"] = pd.NA
        print(f"NOTE: {n_invalid_age} invalid ages set to NA in the dashboard copy only")

    if not set(output["wave"].unique()).issubset({8788, 9394, 9899, 304}):
        raise AssertionError("Unexpected survey-wave code")
    if 304 not in set(output["wave"].unique()):
        raise AssertionError("Wave 304 missing before export; output not overwritten")
    if not output[BROAD_COLUMNS].isin(range(1, 10)).all().all():
        raise AssertionError("Activity category outside 1..9; output not overwritten")

    print("\nRetained diaries by wave:")
    print(output.groupby("wave", observed=True).size().to_string())
    print("\nMissing calendar and demographic fields (preserved as NA):")
    for col in ("survey_quarter", "survey_month", "gender", "age"):
        print(f"  {col}: {int(output[col].isna().sum()):,}")
    print("\nMissing survey_month by wave:")
    print(output.groupby("wave", observed=True)["survey_month"].apply(lambda x: int(x.isna().sum())).to_string())

    # Atomic replacement: a failed conversion or write cannot destroy the
    # existing prepared file or make Streamlit read a half-written Parquet.
    temporary = OUTPUT_FILE.with_name("_timeuse_diary_wide.pending.parquet")
    try:
        output.to_parquet(temporary, index=False, compression="snappy")
        audit = pd.read_parquet(temporary, columns=["pid", "wave"])
        if len(audit) != len(output) or not audit["pid"].is_unique:
            raise AssertionError("Parquet verification failed")
        if 304 not in set(audit["wave"].unique()):
            raise AssertionError("Wave 304 absent from newly written Parquet")
        os.replace(temporary, OUTPUT_FILE)
    finally:
        if temporary.exists():
            temporary.unlink()

    print("\nDaily-profile preparation completed successfully.")
    print(f"Source diaries read: {read_total:,}")
    print(f"Valid diaries saved: {len(output):,}")
    print(f"Output: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
