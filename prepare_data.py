from __future__ import annotations

from pathlib import Path
import pandas as pd

RAW_FILE = Path("data/raw/TimeUse_Cleaned_AllWaves(2).dta")
OUTPUT_FILE = Path("data/processed/timeuse_person_activity.parquet")
FALLBACK_FILE = Path("data/processed/timeuse_person_activity.pkl.gz")
CHUNK_SIZE = 5_000

ID_COLUMNS = [
    "pid",
    "wave",
    "survey_year",
    "survey_quarter",
    "ostan",
    "gender",
    "age",
    "education_level",
    "marital_status",
    "activity_status",
    "employment_status",
    "weight_person",
    "complete_primary_diary",
]
ACTIVITY_COLUMNS = [f"Q2Code{i}" for i in range(1, 97)]


def clean_ostan(series: pd.Series) -> pd.Series:
    """
    Keep genuine province codes and preserve missing province values as missing.

    IMPORTANT:
    Province code "00" is a real code (Markazi). Therefore an empty string must
    never be padded with zfill(2), because "" -> "00" would incorrectly assign
    every observation with missing geography to Markazi.
    """
    out = series.astype("string").str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.mask(out.isna() | out.eq("") | out.isin([".", "nan", "None", "<NA>"]))
    out = out.str.zfill(2)
    return out


def prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.loc[chunk["complete_primary_diary"].eq(1)].copy()
    if chunk.empty:
        return pd.DataFrame()

    chunk["ostan"] = clean_ostan(chunk["ostan"])

    long = chunk.melt(
        id_vars=ID_COLUMNS,
        value_vars=ACTIVITY_COLUMNS,
        var_name="time_slot",
        value_name="activity_code",
    )
    long = long.dropna(subset=["activity_code"])
    long["activity_code"] = pd.to_numeric(
        long["activity_code"], errors="coerce"
    ).astype("Int32")
    long = long.dropna(subset=["activity_code"])

    # Each diary slot represents 15 minutes.
    grouped = (
        long.groupby(
            [c for c in ID_COLUMNS if c != "complete_primary_diary"]
            + ["activity_code"],
            dropna=False,
            observed=True,
        )
        .size()
        .mul(15)
        .rename("minutes")
        .reset_index()
    )
    grouped["minutes"] = grouped["minutes"].astype("int16")
    return grouped


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {RAW_FILE.resolve()}\n"
            "Copy the .dta file into data/raw and keep the expected filename."
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    reader = pd.read_stata(
        RAW_FILE,
        columns=ID_COLUMNS + ACTIVITY_COLUMNS,
        convert_categoricals=False,
        chunksize=CHUNK_SIZE,
    )

    parts: list[pd.DataFrame] = []
    processed_rows = 0
    for number, chunk in enumerate(reader, start=1):
        part = prepare_chunk(chunk)
        if not part.empty:
            parts.append(part)
        processed_rows += len(chunk)
        print(
            f"Chunk {number:02d}: read {len(chunk):,} diary rows; "
            f"total {processed_rows:,}"
        )

    if not parts:
        raise RuntimeError("No complete diary observations were found.")

    output = pd.concat(parts, ignore_index=True)

    # QA: report how many genuine province codes are available by survey year.
    province_qa = (
        output.loc[output["ostan"].notna(), ["survey_year", "ostan"]]
        .drop_duplicates()
        .groupby("survey_year", observed=True)["ostan"]
        .nunique()
    )
    print("\nDistinct non-missing province codes by survey year:")
    if province_qa.empty:
        print("  No valid province codes were found.")
    else:
        for year, nprov in province_qa.items():
            print(f"  {int(year)}: {int(nprov)} province codes")

    try:
        output.to_parquet(OUTPUT_FILE, index=False, compression="snappy")
        saved_file = OUTPUT_FILE
    except ImportError:
        output.to_pickle(FALLBACK_FILE, compression="gzip")
        saved_file = FALLBACK_FILE
        print("pyarrow was not available; a compressed pickle file was created instead.")

    print("\nData preparation completed.")
    print(f"Rows in dashboard file: {len(output):,}")
    print(f"Output: {saved_file.resolve()}")


if __name__ == "__main__":
    main()
