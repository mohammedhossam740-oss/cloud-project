"""Entry point: python -m src.cli {download,prepare,train,predict}."""
from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from .data import ingest_zip, label_seven_days, engineer_features
from .model import run_experiments


def config_load(path):
    return yaml.safe_load(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description="Backblaze 7-day failure prediction pipeline")
    p.add_argument("command", choices=["download", "prepare", "train", "predict", "eda"])
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--archive", default=None)
    p.add_argument("--parquet", default="data/processed/drive_features.parquet")
    p.add_argument("--max-days", type=int, default=None)
    p.add_argument("--model-path", default="models/lightgbm_temporal_weighted.joblib")
    p.add_argument("--input-json", default=None, help="Path to one model/S.M.A.R.T. row JSON")
    p.add_argument("--holdout-model", default=None)
    args = p.parse_args()
    cfg = config_load(args.config)
    archive = Path(args.archive or f'data/raw/{cfg["archive_name"]}')
    if args.command == "download":
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            print("Already downloaded:", archive)
            return
        # Official large archive is downloaded once, then stored in S3.
        with urllib.request.urlopen(cfg["dataset_url"], timeout=90) as src, archive.open("wb") as out:
            shutil.copyfileobj(src, out, length=1024 * 1024)
        print("Downloaded", archive, archive.stat().st_size, "bytes")
    elif args.command == "prepare":
        df = ingest_zip(archive, models=cfg.get("models"), max_days=args.max_days)
        selected_rows = len(df)
        failures = int(df["failure"].sum())
        enriched = engineer_features(df)  # past-only features BEFORE label-based filtering
        frame = label_seven_days(enriched, int(cfg["horizon_days"]))
        output = Path(args.parquet)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False)
        overview = {"archive": str(archive), "models": cfg.get("models"),
                    "source_rows": selected_rows, "confirmed_failure_day_rows": failures,
                    "unique_drives": int(df["serial_number"].nunique()),
                    "dates": [str(df["date"].min().date()), str(df["date"].max().date())],
                    "labeled_rows": len(frame),
                    "positive_drive_day_labels": int(frame["target"].sum()),
                    "positive_drives": int(frame.loc[frame["target"].eq(1), "serial_number"].nunique()),
                    "observed_models": frame["model"].value_counts().to_dict(),
                    "target_prevalence": float(frame["target"].mean()),
                    "missing_fraction": df.isna().mean().sort_values(ascending=False).to_dict()}
        desc = output.parent / "dataset_summary.json"
        desc.write_text(json.dumps(overview, indent=2, default=str))
        print(json.dumps({k: v for k, v in overview.items() if k != "missing_fraction"}, indent=2))
    elif args.command == "train":
        comparison = run_experiments(
            args.parquet, output_dir=(f"holdout_runs/{args.holdout_model}" if args.holdout_model else "."),
            horizon=int(cfg["horizon_days"]),
            train_fraction=float(cfg["train_fraction"]),
            val_fraction=float(cfg["val_fraction"]),
            recall_target=float(cfg["recall_target"]),
            max_training_rows=int(cfg["max_training_rows"]),
            seed=int(cfg["random_seed"]), holdout_model=args.holdout_model)
        print(comparison.to_string(index=False))
    elif args.command == "eda":
        from .eda import generate_eda
        generate_eda(args.parquet, "results/figures")
    elif args.command == "predict":
        if not args.input_json:
            p.error("predict requires --input-json JSON_FILE")
        artifact = joblib.load(args.model_path)
        row = pd.DataFrame([json.loads(Path(args.input_json).read_text())])
        for col in artifact["features"]:
            if col not in row:
                row[col] = "UNKNOWN" if col == "model" else np.nan
        score = float(artifact["pipeline"].predict_proba(row[artifact["features"]])[:, 1][0])
        print(json.dumps({"uncalibrated_risk_score": score,
                          "warning": bool(score >= artifact["threshold"]),
                          "decision_threshold": float(artifact["threshold"]),
                          "horizon_days": artifact["prediction_horizon_days"]}, indent=2))


if __name__ == "__main__":
    main()
