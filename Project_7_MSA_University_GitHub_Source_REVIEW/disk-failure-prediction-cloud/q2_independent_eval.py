#!/usr/bin/env python3
"""Memory-conscious, natural-prevalence Q2 evaluation of saved Q1 models.

Usage (run from project root):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u q2_independent_eval.py

Only selected *models* are retained; every eligible drive belonging to those
models is evaluated. Never select drives according to their failure outcome.
No models are retrained and the Q1 validation thresholds are reused unchanged.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

SMART = [f"smart_{n}_raw" for n in (5, 9, 187, 188, 194, 197, 198)]
DELTA = [f"smart_{n}_raw" for n in (5, 187, 197, 198)]
MEAN = [f"smart_{n}_raw" for n in (5, 197, 198)]
FIELDS = {"serial_number", "model", "failure", "capacity_bytes", *SMART}
MODEL_NAMES = (
    "logistic_baseline", "lightgbm_static_weighted",
    "lightgbm_temporal_unweighted", "lightgbm_temporal_weighted",
)
DAY_RE = re.compile(r"(?:^|/)(\d{4}-\d{2}-\d{2})\.csv$", re.I)


def list_days(z):
    """Ignore macOS resource fork files, require one real CSV per date."""
    result = {}
    for path in z.namelist():
        hit = DAY_RE.search(path)
        if not hit or path.startswith("__MACOSX/") or Path(path).name.startswith("._"):
            continue
        date = pd.Timestamp(hit.group(1)).date()
        if date in result:
            raise ValueError(f"Ambiguous duplicate daily dataset file for {date}")
        result[date] = path
    if len(result) < 15:
        raise ValueError(f"Found only {len(result)} daily CSV files; need at least 15")
    return result


def load_day(z, member, selected_models):
    """Read one day in chunks, retaining all drives of the selected HDD models."""
    pieces = []
    with z.open(member) as stream:
        for chunk in pd.read_csv(stream, usecols=lambda c: c in FIELDS,
                                 chunksize=20_000, low_memory=False,
                                 dtype={"serial_number": "string", "model": "string"}):
            if not {"model", "serial_number", "failure"}.issubset(chunk.columns):
                raise ValueError(f"Missing required columns in {member}")
            part = chunk[chunk["model"].isin(selected_models)].copy()
            if part.empty:
                continue
            for key in SMART + ["capacity_bytes"]:
                if key not in part:
                    part[key] = np.nan
                part[key] = pd.to_numeric(part[key], errors="coerce").astype("float32")
            part["failure"] = pd.to_numeric(part["failure"], errors="coerce").fillna(0).astype("int8")
            pieces.append(part[["serial_number", "model", "failure", "capacity_bytes", *SMART]])
    if not pieces:
        return pd.DataFrame({"serial_number": pd.Series(dtype="string"),
                             "model": pd.Series(dtype="string"),
                             "failure": pd.Series(dtype="int8"),
                             **{k: pd.Series(dtype="float32") for k in ["capacity_bytes", *SMART]}})
    day = pd.concat(pieces, ignore_index=True)
    day = day.dropna(subset=["serial_number", "model"])
    # Keep confirmed failure if duplicate same-day serials exist.
    day = (day.sort_values("failure").drop_duplicates("serial_number", keep="last")
           .reset_index(drop=True))
    return day


def prev_column(cache, date, offset, col, serials):
    prev = cache.get(date - timedelta(days=offset))
    if prev is None or prev.empty:
        return pd.Series(np.nan, index=serials.index, dtype="float32")
    values = prev.set_index("serial_number")[col]
    return serials.map(values).astype("float32")


def construct_features(anchor, cache, date, seen):
    result = anchor.copy()
    serial = result["serial_number"]
    result["observed_days"] = serial.map(seen).fillna(0).astype("float32")
    # Observed-days means days seen in the Q2 evaluation quarter, not drive age.
    for col in DELTA:
        prev3 = prev_column(cache, date, 3, col, serial)
        result[f"{col}_delta3"] = (result[col] - prev3).astype("float32")
    for col in MEAN:
        prev1 = prev_column(cache, date, 1, col, serial)
        prev2 = prev_column(cache, date, 2, col, serial)
        # Mean is only available if all three consecutive days report values.
        result[f"{col}_mean3"] = ((result[col] + prev1 + prev2) / 3).astype("float32")
    return result


def label_day(anchor, cache, date):
    """Positive iff known failure on d+1..d+7; negative iff full 7-day followup.

    Excludes failure-day observations and censored/unknown negatives.
    """
    serial = anchor["serial_number"]
    followup = pd.Series(True, index=anchor.index)
    positive = pd.Series(False, index=anchor.index)
    for k in range(1, 8):
        fut = cache[date + timedelta(days=k)]
        followup &= serial.isin(fut["serial_number"])
        confirmed_failure = fut.loc[fut["failure"].eq(1), "serial_number"]
        positive |= serial.isin(confirmed_failure)
    eligible = anchor["failure"].eq(0) & (positive | followup)
    return eligible, positive.loc[eligible].astype("int8")


def measure(y, scores, threshold):
    labels = (scores >= threshold).astype("uint8")
    tn, fp, fn, tp = confusion_matrix(y, labels, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "observations": int(len(y)), "positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "precision": float(precision_score(y, labels, zero_division=0)),
        "recall": float(recall_score(y, labels, zero_division=0)),
        "f1": float(f1_score(y, labels, zero_division=0)),
        "average_precision": float(average_precision_score(y, scores)) if both else None,
        "roc_auc": float(roc_auc_score(y, scores)) if both else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "false_alarms_per_1000_eligible_drive_days": float(fp / len(y) * 1000) if len(y) else None,
        "threshold_reused_from_q1_validation": threshold,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def run(archive, models_dir, output, selected_models, limit_days=None):
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name in MODEL_NAMES:
        file = models_dir / f"{name}.joblib"
        if not file.exists():
            raise FileNotFoundError(f"Missing trained model: {file}")
        artifact = joblib.load(file)
        if not all(k in artifact for k in ("pipeline", "features", "threshold")):
            raise ValueError(f"Invalid model artifact {file}")
        artifacts[name] = artifact

    predictions = output / "q2_predictions.csv"
    if predictions.exists():
        raise FileExistsError(f"Refusing to overwrite {predictions}; move/delete it to rerun")
    seen = defaultdict(int)
    # Track unique failing drives with eligible positive prediction dates.
    events = {name: {"positive": set(), "detected": set()} for name in MODEL_NAMES}
    by_model_counts = {name: defaultdict(Counter) for name in MODEL_NAMES}
    evaluated = 0
    positives = 0
    first_date, last_date = None, None
    with zipfile.ZipFile(archive) as z:
        day_files = list_days(z)
        dates = sorted(day_files)
        if limit_days is not None:
            dates = dates[:limit_days]
        # A real 7-day outcome needs 7 complete future daily files.
        dates_set = set(dates)
        cache = {}
        total = len(dates) - 7
        if total < 1:
            raise ValueError("Not enough days for seven-day labels")
        for index, date in enumerate(dates[:total], 1):
            required = [date + timedelta(days=i) for i in range(1, 8)]
            if not all(d in dates_set for d in required):
                print(f"Skipping {date}: missing full next-seven-day archive coverage", flush=True)
                continue
            for offset in range(-3, 8):
                d = date + timedelta(days=offset)
                if d in dates_set and d not in cache:
                    cache[d] = load_day(z, day_files[d], selected_models)
            # Anchor rows on this day, minus failure-day and unknown-label rows.
            anchor = cache[date]
            if anchor.empty:
                continue
            engineered = construct_features(anchor, cache, date, seen)
            for sn in anchor["serial_number"]:
                seen[str(sn)] += 1
            eligible, label = label_day(anchor, cache, date)
            engineered = engineered.loc[eligible].copy()
            if not len(engineered):
                continue
            engineered["target"] = label.to_numpy(dtype="int8")
            scores_by_name = {}
            for name, artifact in artifacts.items():
                features = artifact["features"]
                for col in features:
                    if col not in engineered:
                        engineered[col] = "UNKNOWN" if col == "model" else np.nan
                chunks = []
                for start in range(0, len(engineered), 5000):
                    end = start + 5000
                    scores = artifact["pipeline"].predict_proba(engineered[features].iloc[start:end])[:, 1]
                    chunks.append(np.asarray(scores, dtype="float32"))
                arr = np.concatenate(chunks)
                scores_by_name[name] = arr
                alert = arr >= float(artifact["threshold"])
                positive_mask = engineered["target"].eq(1).to_numpy()
                for model in selected_models:
                    model_mask = engineered["model"].eq(model).to_numpy(dtype=bool)
                    y_part = positive_mask[model_mask]
                    a_part = alert[model_mask]
                    c = by_model_counts[name][model]
                    c["tp"] += int(np.sum(y_part & a_part))
                    c["fp"] += int(np.sum(~y_part & a_part))
                    c["tn"] += int(np.sum(~y_part & ~a_part))
                    c["fn"] += int(np.sum(y_part & ~a_part))
                for serial, model in engineered.loc[engineered["target"].eq(1),
                                                    ["serial_number", "model"]].itertuples(index=False):
                    events[name]["positive"].add((str(model), str(serial)))
                for serial, model in engineered.loc[positive_mask & alert,
                                                    ["serial_number", "model"]].itertuples(index=False):
                    events[name]["detected"].add((str(model), str(serial)))
            prediction = engineered[["serial_number", "model", "target"]].copy()
            prediction.insert(0, "date", date.isoformat())
            for name, scores in scores_by_name.items():
                prediction[f"score_{name}"] = scores
            prediction.to_csv(predictions, mode="a", index=False, header=(evaluated == 0))
            evaluated += len(engineered)
            positives += int(engineered["target"].sum())
            first_date = min(first_date, date) if first_date else date
            last_date = date
            # Retain only past 3 days for lookback and future 7 for labeling.
            old = [d for d in cache if d < date - timedelta(days=2)]
            for d in old:
                del cache[d]
            if index % 10 == 0 or index == total:
                print(f"Q2: processed {index}/{total} eligible prediction dates, "
                      f"{evaluated:,} labeled drive-days, {positives:,} positives", flush=True)
    if evaluated == 0:
        raise ValueError("No Q2 labeled drive-days were produced")
    cache.clear()
    gc.collect()

    summary = {"dataset": str(archive), "selected_models": selected_models,
               "source_days": len(dates), "prediction_date_range": [str(first_date), str(last_date)],
               "eligible_drive_days": evaluated, "positive_drive_days": positives,
               "natural_prevalence": positives / evaluated,
               "evaluation": "Independent Q2 test, all eligible selected-model drives; no failure-based sampling",
               "thresholds": "All thresholds fixed from Q1 validation; no Q2 threshold tuning",
               "feature_caveat": "observed_days resets at Q2 start; no pre-quarter historical SMART records",
               "models": {}}
    rows = []
    for name, artifact in artifacts.items():
        col = f"score_{name}"
        data = pd.read_csv(predictions, usecols=["target", col],
                           dtype={"target": "uint8", col: "float32"})
        y = data["target"].to_numpy(copy=False)
        score = data[col].to_numpy(copy=False)
        result = measure(y, score, float(artifact["threshold"]))
        good = events[name]["positive"]
        hit = events[name]["detected"]
        result.update({"unique_failing_drives_with_eligible_positives": len(good),
                       "unique_failing_drives_detected": len(hit),
                       "drive_event_recall": len(hit) / len(good) if good else None})
        result["by_drive_model"] = {}
        for model, counts in by_model_counts[name].items():
            tp, fp, tn, fn = (counts[k] for k in ("tp", "fp", "tn", "fn"))
            result["by_drive_model"][model] = {
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else 0,
                "recall": tp / (tp + fn) if tp + fn else 0,
                "false_positive_rate": fp / (tn + fp) if tn + fp else None,
            }
        summary["models"][name] = result
        rows.append({"model": name,
                     **{k: v for k, v in result.items() if not isinstance(v, dict)}})
        del data, score, y
        gc.collect()
        print(f"{name}: precision={result['precision']:.5f}, recall={result['recall']:.5f}, "
              f"FPR={result['false_positive_rate']:.5f}", flush=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    print(f"SUCCESS: independent Q2 results saved in {output}", flush=True)
    print("Important: Q1 pilot model training used outcome-enriched sampling; this Q2 evaluation does not.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, default=Path("data/raw/data_Q2_2023.zip"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--output", type=Path, default=Path("results/q2_independent"))
    p.add_argument("--models", nargs="+", default=["ST12000NM0008", "ST14000NM001G"])
    p.add_argument("--max-days", type=int, help="For quick tests only; NEVER claim this is full Q2")
    a = p.parse_args()
    run(a.archive, a.models_dir, a.output, a.models, a.max_days)


if __name__ == "__main__":
    main()
