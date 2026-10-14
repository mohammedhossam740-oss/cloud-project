"""Synthetic ONLY FOR TESTING the pipeline; NEVER cite as Backblaze results."""
from pathlib import Path
import numpy as np
import pandas as pd

from src.data import label_seven_days, engineer_features


def main():
    rng = np.random.default_rng(2026)
    models = ("ST12000NM0008", "ST14000NM001G")
    rows = []
    dates = pd.date_range("2023-01-01", periods=104)
    for i in range(120):
        model = models[i % 2]
        fail = (25, 70, 96)[(i // 6) % 3] if i % 6 == 0 else -1
        for d, date in enumerate(dates):
            if fail >= 0 and d > fail:
                break
            warning = max(0, 9 - (fail - d)) if fail >= 0 else 0
            rows.append(dict(date=date, serial_number=f"SIM{i:03d}", model=model,
                             capacity_bytes=12e12,
                             failure=int(d == fail),
                             smart_5_raw=warning * 3 + rng.integers(0, 3),
                             smart_9_raw=d * 24, smart_187_raw=warning,
                             smart_188_raw=0., smart_194_raw=30 + rng.integers(-3, 4),
                             smart_197_raw=warning, smart_198_raw=warning * .5))
    df = label_seven_days(engineer_features(pd.DataFrame(rows)))
    out = Path("data/processed/SMOKE_ONLY_NOT_BACKBLAZE.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)
    print("TEST ONLY synthetic output:", out, "rows", len(df), "positives", df.target.sum())


if __name__ == "__main__":
    main()
