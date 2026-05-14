from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data_raw" / "COPE_Final_Indicators.csv"
OUT_DIR = ROOT / "outputs" / "cope_distributions"

PALETTE = [
    "#12355b",
    "#1f5f99",
    "#2f80c0",
    "#4aa3df",
    "#73c2e8",
    "#9bd6f0",
    "#5d7896",
    "#a9bacb",
]

CATEGORICAL_PALETTE = [
    "#264653",
    "#2a9d8f",
    "#e9c46a",
    "#f4a261",
    "#e76f51",
    "#8ab17d",
    "#6d597a",
    "#b56576",
]

MISSING_COLOR = "#c9d6e2"
GRID_COLOR = "#dfe7ef"
TEXT_COLOR = "#183247"
BACKGROUND_COLOR = "#fbfdff"


def save_current(fig, filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def apply_plot_style():
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND_COLOR,
            "axes.facecolor": BACKGROUND_COLOR,
            "axes.edgecolor": "#8ea5bb",
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "figure.titlesize": 15,
            "figure.titleweight": "semibold",
        }
    )


def soften_axes(ax, axis="y"):
    ax.grid(axis=axis, color=GRID_COLOR, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8ea5bb")
    ax.spines["bottom"].set_color("#8ea5bb")


def numeric_columns(df):
    return [col for col in df.columns if col != "ID" and pd.api.types.is_numeric_dtype(df[col])]


def plot_hist_grid(df, columns, filename, title, transform=None, xlabel_prefix=""):
    n_cols = 4
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, max(3.2 * n_rows, 4)))
    axes = axes.flatten()

    for ax, col in zip(axes, columns):
        values = df[col].dropna()
        if transform is not None:
            values = transform(values)
        ax.hist(values, bins=24, color="#1f5f99", edgecolor=BACKGROUND_COLOR, linewidth=0.7)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(f"{xlabel_prefix}{col}", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        soften_axes(ax, axis="y")

    for ax in axes[len(columns) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, y=1.01)
    save_current(fig, filename)


def plot_categorical_counts(df, columns):
    for col in columns:
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        counts = df[col].value_counts(dropna=False).sort_index()
        labels = ["Missing" if pd.isna(idx) else str(int(idx)) if float(idx).is_integer() else str(idx) for idx in counts.index]
        colors = [
            MISSING_COLOR if label == "Missing" else CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
            for i, label in enumerate(labels)
        ]
        ax.bar(labels, counts.values, color=colors)
        ax.set_title(f"COPE {col} distribution", fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=0)
        soften_axes(ax, axis="y")

        save_current(fig, f"cope_categorical_{col}_counts.png")


def plot_missingness(df):
    missing_pct = df.isna().mean().mul(100).sort_values(ascending=False)
    shown = missing_pct[missing_pct > 0]
    if shown.empty:
        shown = missing_pct.head(10)

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.35 * len(shown))))
    ax.barh(shown.index[::-1], shown.values[::-1], color="#2f80c0")
    ax.set_xlabel("Missing (%)")
    ax.set_title("COPE missingness by variable")
    ax.set_xlim(0, max(5, shown.max() * 1.15))
    soften_axes(ax, axis="x")
    save_current(fig, "cope_missingness.png")


def plot_key_boxplots_by_group(df, group_col):
    key_vars = ["totalmins", "pageviews", "posts", "loginwks", "totaldays", "rate_wkpv"]
    key_vars = [col for col in key_vars if col in df.columns]
    group_values = sorted(df[group_col].dropna().unique())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, key_vars):
        grouped = [df.loc[df[group_col] == value, col].dropna() for value in group_values]
        box = ax.boxplot(
            grouped,
            tick_labels=[str(int(value)) for value in group_values],
            showfliers=True,
            patch_artist=True,
            boxprops={"facecolor": "#9bd6f0", "edgecolor": "#12355b", "linewidth": 1.1},
            medianprops={"color": "#12355b", "linewidth": 1.5},
            whiskerprops={"color": "#5d7896", "linewidth": 1.0},
            capprops={"color": "#5d7896", "linewidth": 1.0},
            flierprops={
                "marker": "o",
                "markerfacecolor": "#4aa3df",
                "markeredgecolor": "#12355b",
                "markersize": 3,
                "alpha": 0.65,
            },
        )
        for patch, color in zip(box["boxes"], PALETTE[1:]):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(group_col)
        ax.set_ylabel(col)
        ax.tick_params(axis="both", labelsize=8)
        soften_axes(ax, axis="y")

    for ax in axes[len(key_vars) :]:
        ax.axis("off")

    fig.suptitle(f"Key COPE engagement variables by {group_col}", fontsize=14, y=1.01)
    save_current(fig, f"cope_key_boxplots_by_{group_col}.png")


def main():
    apply_plot_style()
    df = pd.read_csv(DATA_PATH, na_values=["NA", ""])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cat_cols = [col for col in ["cohort", "trt", "registered", "activated"] if col in df.columns]
    num_cols = numeric_columns(df)
    log_cols = [
        col
        for col in [
            "totalmins",
            "logindays",
            "loginwks",
            "pageviews",
            "posts",
            "ptp",
            "ate",
            "totaldays",
            "act_wkpv",
            "dur_wkpv",
            "act_ptp",
            "dur_ptp",
            "act_ate",
            "dur_ate",
        ]
        if col in df.columns
    ]

    plot_hist_grid(
        df,
        num_cols,
        "cope_numeric_histograms.png",
        "COPE numeric variable distributions",
    )
    n_cols = 4
    n_rows = (len(log_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, max(3.2 * n_rows, 4)))
    axes = axes.flatten()
    for ax, col in zip(axes, log_cols):
        values = df[col].dropna().clip(lower=0)
        ax.hist(np.log1p(values), bins=24, color="#4aa3df", edgecolor=BACKGROUND_COLOR, linewidth=0.7)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(f"log1p({col})", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        soften_axes(ax, axis="y")
    for ax in axes[len(log_cols) :]:
        ax.axis("off")
    fig.suptitle("COPE skewed engagement variables, log1p scale", fontsize=14, y=1.01)
    save_current(fig, "cope_log1p_histograms.png")

    plot_categorical_counts(df, cat_cols)
    plot_missingness(df)
    for group_col in ["trt", "cohort"]:
        if group_col in df.columns:
            plot_key_boxplots_by_group(df, group_col)

    summary = df.describe(include="all").transpose()
    summary.index.name = "variable"
    summary.insert(0, "missing_n", df.isna().sum())
    summary.insert(1, "missing_pct", df.isna().mean().mul(100).round(2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "cope_distribution_summary.csv")

    print(f"Saved plots and summary to: {OUT_DIR}")


if __name__ == "__main__":
    main()
