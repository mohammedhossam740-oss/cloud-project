"""Train four controlled comparisons on the same chronological dataset."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

from .data import SMART, split_temporally
from .evaluation import by_drive_model, make_plots, metrics, threshold_for_recall

BASE_FEATURES = SMART + ["capacity_bytes", "model"]
TEMPORAL_FEATURES = ([f"{n}_delta3" for n in
                      ("smart_5_raw", "smart_187_raw", "smart_197_raw", "smart_198_raw")]
                     + [f"{n}_mean3" for n in
                        ("smart_5_raw", "smart_197_raw", "smart_198_raw")]
                     + ["observed_days"])


def limit_training_set(train: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Memory cap applies to training only. Keep all failure warnings."""
    if cap <= 0 or len(train) <= cap:
        return train
    positives = train[train["target"] == 1]
    negatives = train[train["target"] == 0]
    keep = cap - len(positives)
    if keep < 1:
        raise ValueError("Training cap smaller than number of positive examples.")
    negative_sample = negatives.sample(n=min(keep, len(negatives)), random_state=seed)
    return pd.concat([positives, negative_sample]).sample(frac=1, random_state=seed)


def make_preprocessor(features):
    numeric = [f for f in features if f != "model"]
    return ColumnTransformer([
        ("numbers", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", MaxAbsScaler())
        ]), numeric),
        ("model", OneHotEncoder(handle_unknown="ignore"), ["model"])
    ], sparse_threshold=0.3)


def validate_splits(splits):
    for name, df in splits.items():
        labels = df["target"].value_counts().to_dict()
        if len(labels) < 2:
            raise ValueError(f"{name} requires both classes; has {labels}. "
                             "Use a longer period, more models, or different quarter(s).")


def fit_one(name, features, clf, splits, target_recall, artifacts, metrics_dir):
    train, val, test = (splits[x] for x in ("train", "validation", "test"))
    transform = make_preprocessor(features)
    pipeline = Pipeline([("transform", transform), ("classifier", clf)])
    start = time.perf_counter()
    pipeline.fit(train[features], train["target"])
    duration = time.perf_counter() - start
    val_scores = pipeline.predict_proba(val[features])[:, 1]
    threshold, selected_val_recall = threshold_for_recall(val["target"], val_scores, target_recall)
    start_predict = time.perf_counter()
    test_scores = pipeline.predict_proba(test[features])[:, 1]
    inference = time.perf_counter() - start_predict
    val_scores_dict = metrics(val["target"], val_scores, threshold)
    test_scores_dict = metrics(test["target"], test_scores, threshold)
    # Drive-level event recall (distinct failing drive IDs, not replicated days).
    failure_drives = set(test.loc[test["target"].eq(1), "serial_number"])
    alert_drives = set(test.loc[(test_scores >= threshold) & test["target"].eq(1), "serial_number"])
    test_scores_dict["failing_drives_with_eligible_positive_days"] = len(failure_drives)
    test_scores_dict["failing_drives_detected"] = len(alert_drives)
    test_scores_dict["drive_event_recall"] = (len(alert_drives) / len(failure_drives)
                                             if failure_drives else None)
    result = {
        "model": name, "train_seconds": round(duration, 3),
        "test_inference_seconds": round(inference, 3),
        "features": list(features),
        "train_rows": int(len(train)), "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "validation_target_recall": target_recall,
        "validation_observed_recall_at_threshold": selected_val_recall,
        "validation": val_scores_dict, "test": test_scores_dict,
        "test_by_drive_model": by_drive_model(test, test_scores, threshold),
    }
    joblib.dump({"pipeline": pipeline, "features": features, "threshold": threshold,
                 "prediction_horizon_days": 7}, artifacts / f"{name}.joblib", compress=3)
    (metrics_dir / f"{name}.json").write_text(json.dumps(result, indent=2, default=str))
    return result, {"target": test["target"].to_numpy(), "score": test_scores}


def run_experiments(features_parquet, output_dir=".", horizon=7, train_fraction=.58,
                    val_fraction=.21, recall_target=.8, max_training_rows=350_000,
                    seed=42, holdout_model=None):
    """Natural-prevalence validation/test; optional unseen-model evaluation.

    With holdout_model, the temporal training/validation windows exclude that
    model. The test window contains ONLY the holdout model. Metrics from this
    mode are separate from ordinary model performance and are NOT directly
    comparable as a same-population A/B test.
    """
    workdir = Path(output_dir)
    metrics_dir = workdir / "results" / "metrics"
    figures_dir = workdir / "results" / "figures"
    artifacts = workdir / "models"
    for directory in (metrics_dir, figures_dir, artifacts, workdir / "results" / "predictions"):
        directory.mkdir(parents=True, exist_ok=True)
    path = Path(features_parquet)
    df = pd.read_pickle(path) if path.suffix == ".pkl" else pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    splits = split_temporally(df, horizon, train_fraction, val_fraction)
    if holdout_model:
        splits["train"] = splits["train"][splits["train"]["model"] != holdout_model].copy()
        splits["validation"] = splits["validation"][splits["validation"]["model"] != holdout_model].copy()
        splits["test"] = splits["test"][splits["test"]["model"] == holdout_model].copy()
    validate_splits(splits)
    pre_cap_train_rows = len(splits["train"])
    splits["train"] = limit_training_set(splits["train"], max_training_rows, seed)
    y_train = splits["train"]["target"]
    if y_train.nunique() != 2:
        raise ValueError("Both failure classes are required in training data")
    weight = min(200.0, float((y_train == 0).sum() / max(1, (y_train == 1).sum())))
    base_lgb = dict(n_estimators=140, num_leaves=31, learning_rate=.06,
                    min_child_samples=30, max_depth=-1, random_state=seed,
                    n_jobs=4, verbosity=-1)
    experiments = [
        ("logistic_baseline", BASE_FEATURES,
         LogisticRegression(solver="liblinear", max_iter=280, class_weight="balanced", random_state=seed)),
        ("lightgbm_static_weighted", BASE_FEATURES,
         lgb.LGBMClassifier(**base_lgb, scale_pos_weight=weight)),
        ("lightgbm_temporal_unweighted", BASE_FEATURES + TEMPORAL_FEATURES,
         lgb.LGBMClassifier(**base_lgb)),
        ("lightgbm_temporal_weighted", BASE_FEATURES + TEMPORAL_FEATURES,
         lgb.LGBMClassifier(**base_lgb, scale_pos_weight=weight)),
    ]
    records, preds = [], {}
    print("Split populations (uncapped training rows):", pre_cap_train_rows,
          "trained on:", len(splits["train"]),
          "validation:", len(splits["validation"]), "test:", len(splits["test"]))
    for name, feature_list, clf in experiments:
        print("Training", name)
        record, prediction = fit_one(name, feature_list, clf, splits, recall_target,
                                     artifacts, metrics_dir)
        records.append(record)
        preds[name] = prediction
        print(name, record["test"])
    make_plots(preds, figures_dir)
    # Confusion matrices and model feature importance from the *actual* test set.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for entry in records:
        t = entry["test"]
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (r, c), val in np.ndenumerate(cm):
            ax.text(c, r, str(val), ha="center", va="center", color="black", fontsize=12)
        ax.set_xticks([0, 1], ["Healthy", "Failure"])
        ax.set_yticks([0, 1], ["Healthy", "Failure"])
        ax.set(xlabel="Predicted", ylabel="Actual", title=f"Test confusion matrix — {entry['model']}")
        fig.tight_layout()
        fig.savefig(figures_dir / f"confusion_{entry['model']}.png", dpi=170)
        plt.close(fig)
    best_artifact = joblib.load(artifacts / "lightgbm_temporal_weighted.joblib")
    clf = best_artifact["pipeline"].named_steps["classifier"]
    feature_labels = best_artifact["pipeline"].named_steps["transform"].get_feature_names_out()
    importance = pd.Series(clf.feature_importances_, index=feature_labels).nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    importance.plot.barh(ax=ax)
    ax.set(xlabel="LightGBM split importance", title="Top 15 learned features")
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_importance.png", dpi=170)
    plt.close(fig)
    overall = []
    for entry in records:
        overall.append({"model": entry["model"], "train_seconds": entry["train_seconds"],
                        **entry["test"]})
    comparison = pd.DataFrame(overall)
    comparison.to_csv(metrics_dir / "comparison.csv", index=False)
    pd.DataFrame({"date": splits["test"]["date"].to_numpy(),
                  "serial_number": splits["test"]["serial_number"].to_numpy(),
                  "model": splits["test"]["model"].to_numpy(),
                  "actual": splits["test"]["target"].to_numpy(),
                  **{f"score_{name}": pred["score"] for name, pred in preds.items()}}).to_csv(
                      workdir / "results" / "predictions" / "test_predictions.csv", index=False)
    meta = {"train_date_range": [str(splits["train"]["date"].min().date()),
                                  str(splits["train"]["date"].max().date())],
            "validation_date_range": [str(splits["validation"]["date"].min().date()),
                                      str(splits["validation"]["date"].max().date())],
            "test_date_range": [str(splits["test"]["date"].min().date()),
                                str(splits["test"]["date"].max().date())],
            "pre_cap_train_rows": pre_cap_train_rows,
            "train_cap": max_training_rows,
            "train_label_balance": splits["train"]["target"].value_counts().to_dict(),
            "validation_label_balance": splits["validation"]["target"].value_counts().to_dict(),
            "test_label_balance": splits["test"]["target"].value_counts().to_dict(),
            "mode_holdout": holdout_model or "none"}
    (metrics_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return comparison
