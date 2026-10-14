import numpy as np
import pandas as pd
import pytest

from src.data import label_seven_days, engineer_features, split_temporally
from src.evaluation import metrics, threshold_for_recall


def toy(n_days=70, drives=10):
    dates = pd.date_range("2023-01-01", periods=n_days)
    rows = []
    for d in range(drives):
        fail_day = (20 + (d % 3) * 18) if d % 2 == 0 else -1
        for j, date in enumerate(dates):
            if fail_day > 0 and j > fail_day:
                break
            rows.append({"date": date, "serial_number": f"D{d}",
                         "model": "ST12000NM0008" if d < 6 else "ST14000NM001G",
                         "capacity_bytes": 12000000000000, "failure": int(j == fail_day),
                         "smart_5_raw": float(max(0, j - fail_day + 9)) if fail_day > 0 else 0,
                         "smart_9_raw": j * 24, "smart_187_raw": 0.,
                         "smart_188_raw": 0., "smart_194_raw": 32.,
                         "smart_197_raw": float(int(fail_day > 0 and j >= fail_day - 6)),
                         "smart_198_raw": 0.})
    return pd.DataFrame(rows)


def test_future_warning_window_excludes_failure_day():
    raw = toy(40, 2)
    labeled = label_seven_days(raw, horizon=7)
    drive = labeled[labeled.serial_number.eq("D0")].set_index("date")
    assert drive.loc[pd.Timestamp("2023-01-14"), "target"] == 1  # day 13 (7 days prior)
    assert drive.loc[pd.Timestamp("2023-01-13"), "target"] == 0  # day 12
    assert pd.Timestamp("2023-01-21") not in drive.index  # failure day
    assert drive.target.sum() == 7


def test_missing_followup_is_not_assumed_negative():
    df = toy(18, 2)
    df = df[~((df.serial_number == "D1") & (df.date == pd.Timestamp("2023-01-05")))]
    labeled = label_seven_days(df, horizon=7)
    assert not ((labeled.serial_number == "D1") &
                (labeled.date == pd.Timestamp("2023-01-01"))).any()


def test_temporal_features_only_use_past():
    raw = toy(55, 5)
    a = engineer_features(raw)
    later = raw.copy()
    later.loc[later.date >= pd.Timestamp("2023-02-10"), "smart_5_raw"] = 999999
    b = engineer_features(later)
    key = a.date < pd.Timestamp("2023-02-10")
    pd.testing.assert_series_equal(a.loc[key, "smart_5_raw_delta3"].reset_index(drop=True),
                                   b.loc[key, "smart_5_raw_delta3"].reset_index(drop=True))


def test_temporal_splits_embargo_horizon():
    labeled = label_seven_days(engineer_features(toy(80, 8)), 7)
    splits = split_temporally(labeled, 7, 0.55, 0.22)
    assert splits["train"].date.max() + pd.Timedelta(days=7) < splits["validation"].date.min()
    assert splits["validation"].date.max() + pd.Timedelta(days=7) < splits["test"].date.min()


def test_threshold_and_false_alarms():
    y = [0, 0, 0, 1, 1]
    s = [.01, .2, .3, .8, .9]
    threshold, valrecall = threshold_for_recall(y, s, .8)
    assert threshold == .8 and valrecall == 1.0
    stats = metrics(y, s, threshold)
    assert stats["fp"] == 0 and stats["tp"] == 2 and stats["fpr"] == 0


def test_right_censoring():
    out = label_seven_days(toy(28, 2), 7)
    healthy = out[out.serial_number == "D1"]
    assert healthy.date.max() <= pd.Timestamp("2023-01-21")
