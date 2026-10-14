"""Real-data EDA figures and a text summary; no fabricated results."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_eda(parquet_path: str, out_dir: str):
    df = pd.read_parquet(parquet_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = df["target"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    bars = ax.bar(["No failure next 7 days", "Fails next 7 days"], target.to_numpy())
    ax.bar_label(bars, labels=[f"{n:,}" for n in target.to_numpy()])
    ax.set(ylabel="Eligible drive-days", title="Real-data class imbalance")
    fig.tight_layout()
    fig.savefig(out / "class_imbalance.png", dpi=180)
    plt.close(fig)
    counts = df.groupby("model")["target"].agg(["sum", "count"]).sort_values("sum")
    fig, ax = plt.subplots(figsize=(8.4, max(3, .55 * len(counts) + 1.2)))
    counts["sum"].plot.barh(ax=ax)
    ax.set(xlabel="Positive seven-day labels", title="Failure-warning observations by HDD model")
    fig.tight_layout()
    fig.savefig(out / "failure_labels_by_model.png", dpi=180)
    plt.close(fig)
    numeric = [col for col in ("smart_5_raw", "smart_9_raw", "smart_187_raw",
                "smart_194_raw", "smart_197_raw", "smart_198_raw") if col in df]
    if numeric:
        sampled = df[numeric].sample(min(15000, len(df)), random_state=42)
        corr = sampled.corr(numeric_only=True).fillna(0)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        fig.colorbar(im, ax=ax, fraction=.045)
        ax.set_xticks(np.arange(len(numeric)), numeric, rotation=60, ha="right")
        ax.set_yticks(np.arange(len(numeric)), numeric)
        ax.set(title="SMART attribute correlations (sample of eligible rows)")
        fig.tight_layout()
        fig.savefig(out / "smart_correlation.png", dpi=170)
        plt.close(fig)
        for col in ("smart_5_raw", "smart_197_raw", "smart_194_raw"):
            if col not in df:
                continue
            fig, ax = plt.subplots(figsize=(7, 4.4))
            # SMART errors may have long tails; log1p is meaningful only on nonnegative values.
            for label, name in [(0, "Healthy"), (1, "Failure in seven days")]:
                x = df.loc[df["target"] == label, col].dropna().clip(lower=0)
                if len(x) > 30000:
                    x = x.sample(30000, random_state=42)
                ax.hist(np.log1p(x), bins=32, density=True, alpha=.5, label=name)
            ax.set(xlabel=f"log(1 + {col})", ylabel="Density", title=f"Distribution of {col}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / f"dist_{col}.png", dpi=170)
            plt.close(fig)
    print("Generated real-data EDA figures:", out)
