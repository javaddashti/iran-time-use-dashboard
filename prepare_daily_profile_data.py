from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_FILE = Path("data/raw/TimeUse_Cleaned_AllWaves(2).dta")
OUTPUT_FILE = Path("data/processed/timeuse_diary_wide.parquet")
FALLBACK_FILE = Path("data/processed/timeuse_diary_wide.pkl.gz")
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
    """Convert harmonized 3- or 6-digit activity codes to broad groups 1,...,9."""
    codes = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    broad = np.full(codes.shape, np.nan, dtype="float64")

    three_digit = (codes >= 100) & (codes <= 999)
    six_digit = (codes >= 100_000) & (codes <= 999_999)

    broad[three_digit] = np.floor(codes[three_digit] / 100)
    broad[six_digit] = np.floor(codes[six_digit] / 100_000)
    return broad


def prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.loc[
        chunk["complete_primary_diary"].eq(1)
        & chunk["weight_person"].notna()
        & chunk["weight_person"].gt(0)
        & chunk["gender"].isin([1, 2])
    ].copy()

    if chunk.empty:
        return pd.DataFrame()

    valid_diary = np.ones(len(chunk), dtype=bool)

    for slot, q2_column in enumerate(Q2_COLUMNS, start=1):
        broad = to_broad_activity(chunk[q2_column])
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


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {RAW_FILE.resolve()}\n"
            "Copy TimeUse_Cleaned_AllWaves(2).dta into data/raw."
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    reader = pd.read_stata(
        RAW_FILE,
        columns=BASE_COLUMNS + Q2_COLUMNS,
        convert_categoricals=False,
        chunksize=CHUNK_SIZE,
    )

    parts: list[pd.DataFrame] = []
    rows_read = 0

    for number, chunk in enumerate(reader, start=1):
        part = prepare_chunk(chunk)
        if not part.empty:
            parts.append(part)
        rows_read += len(chunk)
        print(
            f"Chunk {number:02d}: read {len(chunk):,} rows; "
            f"retained {len(part):,}; total read {rows_read:,}"
        )

    if not parts:
        raise RuntimeError("No complete and valid primary diaries were retained.")

    output = pd.concat(parts, ignore_index=True)

    for column in ["survey_year", "wave"]:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("int16")
    for column in ["survey_quarter", "survey_month", "gender", "age"]:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("int8")

    try:
        output.to_parquet(OUTPUT_FILE, index=False, compression="snappy")
        saved_file = OUTPUT_FILE
    except ImportError:
        output.to_pickle(FALLBACK_FILE, compression="gzip")
        saved_file = FALLBACK_FILE
        print("pyarrow was not available; a compressed pickle file was created instead.")

    print("\nDaily-profile data preparation completed.")
    print(f"Retained diaries: {len(output):,}")
    print(f"Columns: {len(output.columns):,}")
    print(f"Output: {saved_file.resolve()}")


if __name__ == "__main__":
    main()
