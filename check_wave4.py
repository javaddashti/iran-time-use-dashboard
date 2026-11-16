from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")

def load(name):
    parquet = DATA_DIR / f"{name}.parquet"
    fallback = DATA_DIR / f"{name}.pkl.gz"
    if parquet.exists():
        print("Read", parquet)
        return pd.read_parquet(parquet)
    if fallback.exists():
        print("Read", fallback)
        return pd.read_pickle(fallback, compression="gzip")
    raise FileNotFoundError(f"Neither {parquet} nor {fallback} was found")

long = load("timeuse_person_activity")
wide = load("timeuse_diary_wide")

assert not long.empty and not wide.empty, "Prepared data is empty"
assert long["pid"].notna().all() and wide["pid"].notna().all()
assert wide["pid"].is_unique, "Wide file must have one record per diary"
assert (pd.to_numeric(long["minutes"],errors="coerce") > 0).all()
assert long["wave"].isin([8788,9394,9899,304]).all(), "Unexpected wave code"
assert (long["activity_code"].between(100,999)).all(), "Unexpected activity code"
assert len([c for c in wide.columns if c.startswith("broad") and c[5:].isdigit()]) == 96
assert (wide[[f"broad{i}" for i in range(1,97)]].isin(range(1,10))).all().all(), "Invalid broad group"

person_minutes = long.groupby("pid",observed=True)["minutes"].sum()
print("\nNumber of unique diaries (long):",person_minutes.size)
print("Number of diaries (wide):",len(wide))
print("\nLong wave counts by unique pid:")
print(long[["pid","wave"]].drop_duplicates().groupby("wave",observed=True).size().to_string())
print("\nWide wave counts:")
print(wide.groupby("wave",observed=True).size().to_string())
print("\nYear and quarter in new wave (long):")
print(long.loc[long.wave.eq(304),["survey_year","survey_quarter","pid"]].drop_duplicates("pid").groupby(["survey_year","survey_quarter"],observed=True).size().to_string())

bad = person_minutes.loc[person_minutes.ne(1440)]
if len(bad):
    print("FAIL: persons with minutes != 1440:",len(bad),"example:",bad.head().to_dict())
    raise AssertionError("Not every diary sums to 1440 minutes")
else:
    print("PASS: every long diary sums to 1440 minutes")

assert 304 in set(long["wave"]), "New wave 304 missing in long file"
assert 304 in set(wide["wave"]), "New wave 304 missing in wide file"
assert set(long["pid"]) == set(wide["pid"]), "Long and wide person samples differ"
print("PASS: wave 304 present, samples agree, both files ready for local UI test")
