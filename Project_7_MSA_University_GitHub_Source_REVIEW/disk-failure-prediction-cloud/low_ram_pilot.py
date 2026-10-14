#!/usr/bin/env python3
"""Create a real-data, memory-friendly Backblaze PILOT ZIP on a small EC2.

IMPORTANT: This preserves all observed failing drives from configured HDD models,
while sampling healthy drives deterministically. Therefore it deliberately changes
natural prevalence, and results must NOT be presented as final population metrics.

Run from project root: python low_ram_pilot.py
Then: python -m src.cli prepare --config config/pilot.yaml --archive data/raw/pilot_Q1_2023.zip --parquet data/processed/pilot_features.parquet
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import yaml

SOURCE = Path("data/raw/data_Q1_2023.zip")
DEST = Path("data/raw/pilot_Q1_2023.zip")
CFG = Path("config/config.yaml")
PILOT_CFG = Path("config/pilot.yaml")
STATS = Path("data/processed/pilot_selection.json")
SAMPLE_PCT = 5
CHUNK_ROWS = 20000
COLUMNS = [
    "date", "serial_number", "model", "capacity_bytes", "failure",
    "smart_5_raw", "smart_9_raw", "smart_187_raw", "smart_188_raw",
    "smart_194_raw", "smart_197_raw", "smart_198_raw"
]

def is_sampled(s: str) -> bool:
    """Choose approximately SAMPLE_PCT of other drives by stable ID hash."""
    n = int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")
    return n % 100 < SAMPLE_PCT

def archive_files(z: zipfile.ZipFile) -> list[str]:
    files = [n for n in z.namelist()
             if re.search(r"(?:^|/)\d{4}-\d{2}-\d{2}\.csv$", n, re.I)
             and not n.startswith("__MACOSX/")
             and not Path(n).name.startswith("._")]
    return sorted(files, key=lambda s: Path(s).name)

def main():
    cfg = yaml.safe_load(CFG.read_text())
    models = set(cfg["models"])
    if not SOURCE.exists():
        raise SystemExit(f"Missing {SOURCE}; run download first")
    print("PILOT ONLY: selection is positive-enriched; not final test prevalence.", flush=True)
    failing_drives = set()
    with zipfile.ZipFile(SOURCE) as zin:
        files = archive_files(zin)
        if len(files) < 30:
            raise SystemExit(f"Only {len(files)} daily files found: insufficient days")
        print(f"Pass 1/2: finding known failing drives among {len(files)} days", flush=True)
        for j, name in enumerate(files, 1):
            with zin.open(name) as fileobj:
                for chunk in pd.read_csv(fileobj, usecols=["serial_number", "model", "failure"],
                                         chunksize=CHUNK_ROWS, dtype={"serial_number": str, "model": str},
                                         low_memory=False):
                    hit = chunk.loc[chunk["model"].isin(models) & (chunk["failure"] == 1),
                                    ["model", "serial_number"]]
                    failing_drives.update(zip(hit["model"], hit["serial_number"]))
            if j % 30 == 0: print(f"  Scanned {j}/{len(files)} days", flush=True)
    print(f"Failure-associated drives in selected models: {len(failing_drives)}", flush=True)

    # Each daily file retains ALL selected failure-drive records and a stable 5%
    # sample of other drives, not a random sample of individual drive-days.
    # This avoids losing days in a given drive's 7-day observation history.
    healthy_cache: dict[tuple[str, str], bool] = {}
    counts = {"input_daily_files": len(files), "sample_percent_of_other_drives": SAMPLE_PCT,
              "failing_drive_count": len(failing_drives), "selected_models": sorted(models),
              "rows_kept": 0, "failure_day_rows_kept": 0, "purpose": "PILOT_ONLY",
              "warning": "Pilot prevalence is artificially higher than natural prevalence; do not cite pilot precision, alert burden or failure prevalence as population results."}
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE) as zin, zipfile.ZipFile(DEST, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as zout:
        print("Pass 2/2: writing memory-bounded pilot archive", flush=True)
        for j, name in enumerate(files, 1):
            date_name = Path(name).name
            outname = f"pilot_q1_2023/{date_name}"
            with zin.open(name) as source_file, zout.open(outname, "w") as outbytes:
                with io.TextIOWrapper(outbytes, encoding="utf-8") as out:
                    wrote_header = False
                    for chunk in pd.read_csv(source_file, usecols=lambda c: c in COLUMNS,
                                             chunksize=CHUNK_ROWS, low_memory=False,
                                             dtype={"serial_number": str, "model": str}):
                        chunk = chunk.loc[chunk["model"].isin(models)].copy()
                        if not len(chunk):
                            continue
                        unique_keys = set(zip(chunk["model"], chunk["serial_number"]))
                        for key in unique_keys:
                            if key not in healthy_cache:
                                healthy_cache[key] = is_sampled(key[0] + ":" + key[1])
                        keep = [key in failing_drives or healthy_cache[key]
                                for key in zip(chunk["model"], chunk["serial_number"])]
                        chunk = chunk.loc[keep]
                        if chunk.empty:
                            continue
                        if "date" not in chunk: chunk["date"] = date_name[:10]
                        for col in COLUMNS:
                            if col not in chunk: chunk[col] = pd.NA
                        chunk = chunk[COLUMNS]
                        counts["rows_kept"] += len(chunk)
                        counts["failure_day_rows_kept"] += int(chunk["failure"].eq(1).sum())
                        chunk.to_csv(out, index=False, header=not wrote_header)
                        wrote_header = True
                    if not wrote_header:
                        pd.DataFrame(columns=COLUMNS).to_csv(out, index=False)
            if j % 15 == 0: print(f"  Wrote {j}/{len(files)} days, {counts['rows_kept']:,} rows", flush=True)

    PILOT_CFG.parent.mkdir(exist_ok=True)
    pilot_cfg = dict(cfg)
    pilot_cfg["max_training_rows"] = min(int(cfg.get("max_training_rows", 350000)), 60000)
    PILOT_CFG.write_text(yaml.safe_dump(pilot_cfg, sort_keys=False))
    STATS.parent.mkdir(exist_ok=True, parents=True)
    counts["pilot_zip"] = str(DEST)
    counts["pilot_bytes"] = DEST.stat().st_size
    STATS.write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2), flush=True)
    print("SUCCESS: PILOT ZIP prepared. Final natural-prevalence experiments remain pending.", flush=True)

if __name__ == "__main__":
    main()
