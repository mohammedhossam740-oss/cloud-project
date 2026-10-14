"""Streaming Backblaze ZIP ingestion and conservative seven-day label creation."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SMART_IDS = (5, 9, 187, 188, 194, 197, 198)
SMART = [f"smart_{i}_raw" for i in SMART_IDS]
BASE_COLUMNS = ["date", "serial_number", "model", "capacity_bytes", "failure"]
COLUMNS = BASE_COLUMNS + SMART
DAY_FILE = re.compile(r"(?:^|/)(\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE)


def ingest_zip(archive: str | Path, models: list[str] | None = None,
               max_days: int | None = None) -> pd.DataFrame:
    """Read only the needed columns, only selected HDD models, day by day.

    A quarter is large; defaults target a small set of HDD models, not a
    downsample of healthy observations (test prevalence remains natural).
    """
    archive = Path(archive)
    if not archive.exists():
        raise FileNotFoundError(f"Archive missing: {archive}. Use `python -m src.cli download`.")
    parts = []
    with zipfile.ZipFile(archive) as zf:
        files = sorted((n for n in zf.namelist() if DAY_FILE.search(n) and not n.startswith("__MACOSX/") and not Path(n).name.startswith("._")),
                       key=lambda n: DAY_FILE.search(n).group(1))
        if not files:
            raise ValueError("No daily YYYY-MM-DD.csv files found in the archive.")
        if max_days:
            files = files[:max_days]
        for idx, name in enumerate(files, 1):
            with zf.open(name) as stream:
                for chunk in pd.read_csv(stream, usecols=lambda c: c in COLUMNS,
                                         chunksize=110_000, low_memory=False):
                    required = {"serial_number", "model", "failure"}
                    if not required.issubset(chunk.columns):
                        raise ValueError(f"Missing required columns in {name}: {required - set(chunk)}")
                    if models:
                        chunk = chunk[chunk["model"].isin(models)]
                    # If no explicit models, apply a conservative HDD-focused convention;
                    # Backblaze has separate SSD/boot drive records in recent quarters.
                    else:
                        chunk = chunk[chunk["model"].fillna("").str.match(
                            r"^(ST\d|HGST|Hitachi|WDC|TOSHIBA|MG0)", case=False)]
                    if chunk.empty:
                        continue
                    if "date" not in chunk.columns:
                        chunk["date"] = DAY_FILE.search(name).group(1)
                    for col in COLUMNS:
                        if col not in chunk:
                            chunk[col] = np.nan
                    chunk = chunk[COLUMNS].copy()
                    parts.append(chunk)
            if idx % 15 == 0:
                print(f"Read {idx}/{len(files)} daily files, {sum(len(p) for p in parts):,} selected rows")
    if not parts:
        raise ValueError("No rows found for the selected model(s); choose models present in this quarter.")
    frame = pd.concat(parts, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["failure"] = pd.to_numeric(frame["failure"], errors="coerce")
    frame = frame.dropna(subset=["date", "serial_number", "model", "failure"])
    for col in SMART + ["capacity_bytes"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")
    frame["failure"] = frame["failure"].astype("int8")
    frame = frame[frame["failure"].isin((0, 1))]
    frame = frame.sort_values(["serial_number", "date", "failure"])
    duplicates = frame.duplicated(["serial_number", "date"], keep=False).sum()
    if duplicates:
        print(f"WARNING: {duplicates:,} duplicate serial/date observations; keeping failure record if present.")
    frame = frame.drop_duplicates(["serial_number", "date"], keep="last")
    return frame.reset_index(drop=True)


def label_seven_days(raw: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """Label t using a *confirmed* failure on (t, t+h], else require h daily followups.

    Negatives with unknown outcome or a gap in daily followup are dropped;
    post-failure and failure-day observations are never prediction examples.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["failure"] = pd.to_numeric(df["failure"])
    df = df.sort_values(["serial_number", "date", "failure"])
    df = df.drop_duplicates(["serial_number", "date"], keep="last").reset_index(drop=True)
    fail_dates = (df.loc[df["failure"].eq(1)]
                  .groupby("serial_number")["date"].min().rename("failure_date"))
    df = df.join(fail_dates, on="serial_number")
    # No post-failure measurements can be used.
    df = df[df["failure_date"].isna() | df["date"].le(df["failure_date"])].copy()
    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)
    grp = df.groupby("serial_number", sort=False)["date"]
    last_followup = grp.shift(-horizon)
    # If observation h steps away is exactly h days away, all h intervening
    # daily followups exist, as there cannot be duplicate dates per drive.
    complete_followup = (last_followup - df["date"]).eq(pd.Timedelta(days=horizon))
    until_failure = (df["failure_date"] - df["date"]).dt.days
    positive = until_failure.between(1, horizon, inclusive="both")
    known = (positive | complete_followup) & df["failure"].eq(0)
    df = df[known].copy()
    df["target"] = positive[known].astype("int8")
    assert not df["failure"].eq(1).any()
    return df.drop(columns=["failure_date", "failure"]).reset_index(drop=True)


def engineer_features(raw_labeled: pd.DataFrame) -> pd.DataFrame:
    """Past-and-present-only SMART differences / rolling means, with gap checks.

    Call on the *whole* chronologically ordered dataset, before time splitting,
    so test-time history that predates the split is available (as in deployment).
    """
    df = raw_labeled.sort_values(["serial_number", "date"]).copy()
    grp = df.groupby("serial_number", sort=False)
    df["observed_days"] = grp.cumcount().astype("float32")
    for col in ["smart_5_raw", "smart_187_raw", "smart_197_raw", "smart_198_raw"]:
        prev = grp[col].shift(3)
        past_date = grp["date"].shift(3)
        continuous = (df["date"] - past_date).eq(pd.Timedelta(days=3))
        df[f"{col}_delta3"] = (df[col] - prev).where(continuous).astype("float32")
    for col in ["smart_5_raw", "smart_197_raw", "smart_198_raw"]:
        rolling = grp[col].transform(lambda s: s.rolling(3, min_periods=3).mean())
        prev_date = grp["date"].shift(2)
        continuous = (df["date"] - prev_date).eq(pd.Timedelta(days=2))
        df[f"{col}_mean3"] = rolling.where(continuous).astype("float32")
    return df.sort_values(["date", "serial_number"]).reset_index(drop=True)


def split_temporally(df: pd.DataFrame, horizon: int, train_fraction: float,
                     val_fraction: float) -> dict[str, pd.DataFrame]:
    """Chronological disjoint prediction windows with h-day label embargoes.

    All splits have unobserved-label boundary rows removed, ensuring that the
    seven-day outcome for a split does not reach into the following split.
    """
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    if len(dates) < 3 * (horizon + 2):
        raise ValueError("Not enough distinct dates for 3 chronological splits and embargoes.")
    a = int(len(dates) * train_fraction)
    b = int(len(dates) * (train_fraction + val_fraction))
    if not horizon < a < b - horizon < len(dates) - horizon:
        raise ValueError("Fractions leave insufficient time for seven-day label embargoes.")
    # train has [start, a) predictions and observation horizon entirely < a
    mask = {
        "train": (df["date"] < dates[a] - pd.Timedelta(days=horizon)),
        "validation": ((df["date"] >= dates[a]) &
                       (df["date"] < dates[b] - pd.Timedelta(days=horizon))),
        "test": ((df["date"] >= dates[b]) &
                 (df["date"] <= dates[-1] - pd.Timedelta(days=horizon)))
    }
    return {key: df.loc[value].copy() for key, value in mask.items()}
