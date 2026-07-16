"""Summarise and plot imputation slope sensitivity results."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2] if SCRIPT_PATH.parents[1].name == "scripts" else SCRIPT_PATH.parents[1]
INPUT_DIR = ROOT / "outputs" / "slope_sensitivity"
OUTPUT_DIR = INPUT_DIR / "figures"

SLOPE_ORDER = [-2.50, -2.00, -1.25, -0.75, -0.50]
DATASET_ORDER = [204, 2000, 5000]
MECHANISM_ORDER = ["MAR", "MNAR"]
RATE_ORDER = [0.10, 0.20, 0.30, 0.40, 0.50]
METHOD_ORDER = ["median", "knn", "kpod", "mice_pmm", "random_forest"]
METHOD_LABELS = {
    "median": "Median",
    "knn": "KNN",
    "kpod": "K-POD",
    "mice_pmm": "MICE-PMM",
    "random_forest": "Random forest",
}
METHOD_COLORS = {
    "median": "#009E73",
    "knn": "#0072B2",
    "kpod": "#E69F00",
    "mice_pmm": "#CC79A7",
    "random_forest": "#D55E00",
}
METHOD_MARKERS = {
    "median": "^",
    "knn": "o",
    "kpod": "s",
    "mice_pmm": "D",
    "random_forest": "v",
}


def load_results() -> pd.DataFrame:
    frames = []
    for path in sorted(INPUT_DIR.glob("*_summary_by_seed_n*.csv")):
        match = re.search(r"n(\d+)", path.name)
        if not match:
            continue
        frame = pd.read_csv(path)
        frame["dataset_n"] = int(match.group(1))
        frame["source_file"] = path.name
        frame["source_mtime"] = path.stat().st_mtime
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No sensitivity summary CSVs found in {INPUT_DIR}")
    df = pd.concat(frames, ignore_index=True)
    dedupe_cols = [
        "dataset_n",
        "base_random_state",
        "method",
        "mechanism",
        "target_missing_rate",
        "missingness_logit_slope",
    ]
    return (
        df.sort_values("source_mtime")
        .drop_duplicates(subset=dedupe_cols, keep="last")
        .reset_index(drop=True)
    )


def methods_present(df: pd.DataFrame) -> list[str]:
    present = set(df["method"].dropna())
    ordered = [method for method in METHOD_ORDER if method in present]
    return ordered + sorted(present - set(ordered))


def aggregate_method_trends(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(
            ["dataset_n", "method", "mechanism", "target_missing_rate", "missingness_logit_slope"],
            as_index=False,
        )
        .agg(
            mean_ari=("mean_ari", "mean"),
            sd_ari=("mean_ari", "std"),
            min_seed_ari=("mean_ari", "min"),
            max_seed_ari=("mean_ari", "max"),
            mean_centroid_error=("mean_centroid_error", "mean"),
            mean_cluster_size_error=("mean_cluster_size_error", "mean"),
            mean_silhouette=("mean_silhouette", "mean"),
        )
        .sort_values(
            ["dataset_n", "method", "mechanism", "target_missing_rate", "missingness_logit_slope"]
        )
    )
    summary["previous_slope_ari"] = summary.groupby(
        ["dataset_n", "method", "mechanism", "target_missing_rate"]
    )["mean_ari"].shift(1)
    summary["ari_adjacent_slope_jump"] = summary["mean_ari"] - summary["previous_slope_ari"]
    return summary


def save_summary_tables(summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "method_ari_self_trend_summary.csv", index=False)

    top_jumps = (
        summary.dropna(subset=["ari_adjacent_slope_jump"])
        .sort_values("ari_adjacent_slope_jump", ascending=False)
        .head(20)
    )
    top_jumps.to_csv(OUTPUT_DIR / "top_method_ari_adjacent_jumps.csv", index=False)


def plot_method_ari_by_slope(df: pd.DataFrame) -> None:
    methods = methods_present(df)
    fig, axes = plt.subplots(
        len(DATASET_ORDER),
        len(MECHANISM_ORDER) * len(RATE_ORDER),
        figsize=(22, 8.5),
        sharex=True,
        sharey=True,
    )

    for row_idx, dataset_n in enumerate(DATASET_ORDER):
        for col_idx, (mechanism, rate) in enumerate(
            (mechanism, rate)
            for mechanism in MECHANISM_ORDER
            for rate in RATE_ORDER
        ):
            ax = axes[row_idx, col_idx]
            subset = df[
                (df["dataset_n"] == dataset_n)
                & (df["mechanism"] == mechanism)
                & np.isclose(df["target_missing_rate"], rate)
            ].copy()
            for method in methods:
                method_subset = subset[subset["method"] == method]
                for _, seed_frame in method_subset.groupby("base_random_state"):
                    seed_frame = seed_frame.sort_values("missingness_logit_slope")
                    ax.plot(
                        seed_frame["missingness_logit_slope"],
                        seed_frame["mean_ari"],
                        color=METHOD_COLORS[method],
                        alpha=0.14,
                        linewidth=0.9,
                    )
                mean_frame = (
                    method_subset.groupby("missingness_logit_slope", as_index=False)["mean_ari"]
                    .mean()
                    .sort_values("missingness_logit_slope")
                )
                ax.plot(
                    mean_frame["missingness_logit_slope"],
                    mean_frame["mean_ari"],
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    linewidth=2,
                    markersize=4,
                        label=METHOD_LABELS.get(method, method)
                        if row_idx == 0 and col_idx == 0
                        else None,
                )
            ax.set_title(f"{mechanism}, missing={rate:.0%}", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"n={dataset_n}\nMean ARI")
            ax.grid(True, color="#dddddd", linewidth=0.6)
            ax.set_xticks(SLOPE_ORDER)
            ax.tick_params(axis="x", rotation=45)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(methods)), frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("ARI across missingness-logit slopes: each method's trend", y=0.998)
    fig.supxlabel("Missingness-logit slope")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT_DIR / "method_ari_by_slope_seed_lines.png", dpi=220)
    plt.close(fig)


def plot_ari_fixed_slope(df: pd.DataFrame, slope: float) -> None:
    methods = methods_present(df)
    subset = df[np.isclose(df["missingness_logit_slope"], slope)].copy()
    if subset.empty:
        return
    summary = (
        subset.groupby(["dataset_n", "mechanism", "target_missing_rate", "method"], as_index=False)
        .agg(mean_ari=("mean_ari", "mean"), sd_ari=("mean_ari", "std"))
        .sort_values(["dataset_n", "mechanism", "method", "target_missing_rate"])
    )

    fig, axes = plt.subplots(
        len(DATASET_ORDER),
        len(MECHANISM_ORDER),
        figsize=(8.2, 8.2),
        sharex=True,
        sharey=True,
    )

    for row_idx, dataset_n in enumerate(DATASET_ORDER):
        for col_idx, mechanism in enumerate(MECHANISM_ORDER):
            ax = axes[row_idx, col_idx]
            panel = summary[
                (summary["dataset_n"] == dataset_n) & (summary["mechanism"] == mechanism)
            ]
            for method in methods:
                method_panel = panel[panel["method"] == method]
                if method_panel.empty:
                    continue
                x = method_panel["target_missing_rate"].to_numpy()
                y = method_panel["mean_ari"].to_numpy()
                sd = method_panel["sd_ari"].fillna(0).to_numpy()
                ax.plot(
                    x,
                    y,
                    marker=METHOD_MARKERS[method],
                    linewidth=2,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS.get(method, method),
                )
                ax.fill_between(x, y - sd, y + sd, color=METHOD_COLORS[method], alpha=0.12)

            ax.set_title(f"n={dataset_n}, {mechanism}", fontsize=10)
            ax.set_xticks(RATE_ORDER, [f"{rate:.0%}" for rate in RATE_ORDER])
            ax.set_ylim(0, 1)
            ax.grid(True, color="#dddddd", linewidth=0.6)
            if col_idx == 0:
                ax.set_ylabel("Mean ARI across seeds")
            if row_idx == len(DATASET_ORDER) - 1:
                ax.set_xlabel("Target missing rate")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(methods)), frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle(f"ARI performance at fixed missingness-logit slope = {slope:g}", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    slope_label = str(slope).replace("-", "neg").replace(".", "p")
    fig.savefig(OUTPUT_DIR / f"ari_by_missing_rate_fixed_slope_{slope_label}.png", dpi=220)
    plt.close(fig)


def plot_ari_fixed_slope_overview(df: pd.DataFrame) -> None:
    methods = methods_present(df)
    summary = (
        df.groupby(
            [
                "dataset_n",
                "mechanism",
                "target_missing_rate",
                "missingness_logit_slope",
                "method",
            ],
            as_index=False,
        )["mean_ari"]
        .mean()
        .sort_values(
            [
                "dataset_n",
                "mechanism",
                "missingness_logit_slope",
                "method",
                "target_missing_rate",
            ]
        )
    )

    fig, axes = plt.subplots(
        len(SLOPE_ORDER),
        len(MECHANISM_ORDER),
        figsize=(8.5, 12),
        sharex=True,
        sharey=True,
    )
    dataset_styles = {
        204: {"linestyle": ":", "alpha": 0.95},
        2000: {"linestyle": "--", "alpha": 0.95},
        5000: {"linestyle": "-", "alpha": 0.95},
    }

    for row_idx, slope in enumerate(SLOPE_ORDER):
        for col_idx, mechanism in enumerate(MECHANISM_ORDER):
            ax = axes[row_idx, col_idx]
            panel = summary[
                (summary["mechanism"] == mechanism)
                & np.isclose(summary["missingness_logit_slope"], slope)
            ]
            for dataset_n in DATASET_ORDER:
                for method in methods:
                    method_panel = panel[
                        (panel["dataset_n"] == dataset_n) & (panel["method"] == method)
                    ]
                    if method_panel.empty:
                        continue
                    label = f"n={dataset_n} {METHOD_LABELS.get(method, method)}"
                    ax.plot(
                        method_panel["target_missing_rate"],
                        method_panel["mean_ari"],
                        color=METHOD_COLORS[method],
                        marker=METHOD_MARKERS[method],
                        linewidth=1.8,
                        markersize=4,
                        label=label if row_idx == 0 and col_idx == 0 else None,
                        **dataset_styles[dataset_n],
                    )
            ax.set_title(f"{mechanism}, slope={slope:g}", fontsize=9)
            ax.set_xticks(RATE_ORDER, [f"{rate:.0%}" for rate in RATE_ORDER])
            ax.set_ylim(0, 1)
            ax.grid(True, color="#dddddd", linewidth=0.6)
            if col_idx == 0:
                ax.set_ylabel("Mean ARI")
            if row_idx == len(SLOPE_ORDER) - 1:
                ax.set_xlabel("Target missing rate")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("ARI by missing rate within each fixed slope", y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUTPUT_DIR / "ari_by_missing_rate_fixed_slope_overview.png", dpi=220)
    plt.close(fig)


def main() -> None:
    df = load_results()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = aggregate_method_trends(df)

    save_summary_tables(summary)
    plot_method_ari_by_slope(df)
    for slope in SLOPE_ORDER:
        plot_ari_fixed_slope(df, slope)
    plot_ari_fixed_slope_overview(df)

    print(f"Saved summary tables and figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
