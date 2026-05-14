from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_raw"
OUT_DIR = ROOT / "outputs" / "ppt_dataset_overviews"

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

COPE_PALETTE = [
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


def save_current(fig, dataset, filename):
    out_path = OUT_DIR / dataset
    out_path.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def soften_axes(ax, axis="y"):
    ax.grid(axis=axis, color=GRID_COLOR, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8ea5bb")
    ax.spines["bottom"].set_color("#8ea5bb")


def clean_label(value, prefix=None):
    if pd.isna(value):
        return "Missing"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, str):
        return value
    if float(value).is_integer():
        value = int(value)
    return f"{prefix} {value}" if prefix else str(value)


def sorted_value_counts(series):
    counts = series.value_counts(dropna=False)
    non_missing = [idx for idx in counts.index if not pd.isna(idx)]
    missing = [idx for idx in counts.index if pd.isna(idx)]
    try:
        non_missing = sorted(non_missing)
    except TypeError:
        non_missing = sorted(non_missing, key=str)
    return counts.reindex(non_missing + missing)


def read_csv(name):
    return pd.read_csv(DATA_DIR / f"{name}.csv", na_values=["NA", ""])


def coerce_numeric(df, exclude=None):
    exclude = set(exclude or [])
    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def bar_counts(ax, series, title, xlabel="", prefix=None, palette=None):
    palette = palette or PALETTE
    counts = sorted_value_counts(series)
    labels = [clean_label(idx, prefix=prefix) for idx in counts.index]
    colors = [MISSING_COLOR if label == "Missing" else palette[i % len(palette)] for i, label in enumerate(labels)]
    ax.bar(labels, counts.values, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel or series.name)
    ax.set_ylabel("Count")
    rotation = 25 if any(len(label) > 5 for label in labels) and len(labels) > 3 else 0
    ax.tick_params(axis="x", rotation=rotation)
    if rotation:
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
    soften_axes(ax, axis="y")


def plot_missingness(df, dataset, filename, title, columns=None, top_n=14, bar_color="#2f80c0"):
    data = df[columns] if columns is not None else df
    missing_pct = data.isna().mean().mul(100).sort_values(ascending=True)
    shown = missing_pct.tail(top_n)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = [MISSING_COLOR if value == 0 else bar_color for value in shown.values]
    ax.barh(shown.index, shown.values, color=colors)
    ax.set_xlabel("Missing (%)")
    ax.set_title(title)
    ax.set_xlim(0, max(5, shown.max() * 1.15))
    soften_axes(ax, axis="x")

    for idx, value in enumerate(shown.values):
        if value > 0:
            ax.text(value + max(shown.max() * 0.015, 0.2), idx, f"{value:.1f}%", va="center", fontsize=8)

    save_current(fig, dataset, filename)


def plot_adsl():
    df = read_csv("ADSL")
    df["stdate"] = pd.to_datetime(df["stdate"], errors="coerce")
    df = coerce_numeric(df, exclude=["scrnfl", "stdate", "sttime", "eddate", "edtime"])

    metrics = pd.Series(
        {
            "Total records": len(df),
            "Screened": df["scrnfl"].eq("Y").sum(),
            "Randomised": df["trt"].notna().sum(),
            "Baseline demographics": df["age"].notna().sum(),
            "Start date available": df["stdate"].notna().sum(),
        }
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.barh(metrics.index[::-1], metrics.values[::-1], color=PALETTE[: len(metrics)])
    ax.set_title("ADSL record availability")
    ax.set_xlabel("Number of records")
    ax.set_xlim(0, metrics.max() * 1.16)
    soften_axes(ax, axis="x")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + metrics.max() * 0.018, bar.get_y() + bar.get_height() / 2, f"{int(width)}", va="center")
    save_current(fig, "adsl", "01_adsl_record_availability.png")

    plot_missingness(df, "adsl", "02_adsl_missingness.png", "ADSL missingness overview", top_n=14)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5))
    bar_counts(axes[0], df["trt"], "Treatment arm", "trt", prefix="Arm")
    bar_counts(axes[1], df["cohort"], "Cohort", "cohort", prefix="C")
    fig.suptitle("ADSL trial structure", y=1.02)
    save_current(fig, "adsl", "03_adsl_trial_structure.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5))
    for ax, col, title, color in zip(
        axes,
        ["age", "agecfp"],
        ["Caregiver age", "Care recipient age"],
        ["#1f5f99", "#4aa3df"],
    ):
        values = df[col].dropna()
        ax.hist(values, bins=22, color=color, edgecolor=BACKGROUND_COLOR, linewidth=0.7)
        ax.axvline(values.median(), color="#12355b", linestyle="--", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.text(
            0.96,
            0.92,
            f"n={len(values)}\nmedian={values.median():.1f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ffffff", "edgecolor": "#c9d6e2"},
        )
        soften_axes(ax, axis="y")
    fig.suptitle("ADSL age profile", y=1.02)
    save_current(fig, "adsl", "04_adsl_age_profile.png")


def plot_adqs():
    df = coerce_numeric(read_csv("ADQS"))
    outcome_cols = ["wemwbs", "maks", "eci_neg", "eci_pos", "cw", "cs", "fq", "eq5d", "vas"]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    counts = df["tptnum"].value_counts(dropna=False).sort_index()
    labels = [clean_label(idx, prefix="TPT") for idx in counts.index]
    ax.bar(labels, counts.values, color=PALETTE[: len(counts)])
    ax.set_title("ADQS records by questionnaire timepoint")
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Number of questionnaire records")
    soften_axes(ax, axis="y")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value + counts.max() * 0.02, str(int(value)), ha="center", fontsize=9)
    save_current(fig, "adqs", "01_adqs_records_by_timepoint.png")

    plot_missingness(
        df,
        "adqs",
        "02_adqs_outcome_missingness.png",
        "ADQS outcome missingness",
        columns=outcome_cols + ["ltfu"],
        top_n=12,
    )

    show_cols = ["wemwbs", "maks", "eci_neg", "eci_pos", "cw", "cs", "fq", "eq5d", "vas"]
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 8.2))
    axes = axes.flatten()
    for ax, col in zip(axes, show_cols):
        values = df[col].dropna()
        ax.hist(values, bins=18, color="#1f5f99", edgecolor=BACKGROUND_COLOR, linewidth=0.7)
        ax.set_title(col)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.tick_params(axis="both", labelsize=8)
        soften_axes(ax, axis="y")
    fig.suptitle("ADQS questionnaire score distributions", y=1.01)
    save_current(fig, "adqs", "03_adqs_score_distributions.png")

    selected_outcomes = ["wemwbs", "maks", "eq5d", "vas"]
    mean_scores = df.groupby("tptnum")[selected_outcomes].mean(numeric_only=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    axes = axes.flatten()
    for ax, col, color in zip(axes, selected_outcomes, PALETTE[:4]):
        ax.plot(mean_scores.index, mean_scores[col], marker="o", linewidth=2.1, color=color)
        ax.set_title(col)
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Mean score")
        ax.set_xticks(mean_scores.index)
        soften_axes(ax, axis="y")
    fig.suptitle("ADQS selected mean scores by timepoint", y=1.02)
    save_current(fig, "adqs", "04_adqs_mean_scores_by_timepoint.png")


def plot_cope():
    df = coerce_numeric(read_csv("COPE_Final_Indicators"), exclude=["sdate"])
    engagement_cols = ["totalmins", "loginwks", "pageviews", "posts", "ptp", "ate", "totaldays", "rate_wkpv"]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.2))
    bar_counts(axes[0], df["cohort"], "Cohort", "cohort", prefix="C", palette=COPE_PALETTE)
    bar_counts(axes[1], df["registered"], "Registered", "registered", palette=COPE_PALETTE)
    bar_counts(axes[2], df["activated"], "Activated", "activated", palette=COPE_PALETTE)
    fig.suptitle("COPE participant and access status", y=1.03)
    save_current(fig, "cope", "01_cope_participant_status.png")

    plot_missingness(
        df,
        "cope",
        "02_cope_engagement_missingness.png",
        "COPE engagement missingness",
        columns=engagement_cols,
        top_n=10,
        bar_color=COPE_PALETTE[1],
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    axes = axes.flatten()
    for ax, col, color in zip(axes, ["totalmins", "pageviews", "posts", "loginwks"], COPE_PALETTE[1:5]):
        values = df[col].dropna().clip(lower=0)
        ax.hist(np.log1p(values), bins=22, color=color, edgecolor=BACKGROUND_COLOR, linewidth=0.7)
        ax.set_title(col)
        ax.set_xlabel(f"log1p({col})")
        ax.set_ylabel("Count")
        soften_axes(ax, axis="y")
    fig.suptitle("COPE key engagement distributions", y=1.02)
    save_current(fig, "cope", "03_cope_key_engagement_distributions.png")

    cohort_values = sorted(df["cohort"].dropna().unique())
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    axes = axes.flatten()
    for ax, col in zip(axes, ["totalmins", "pageviews", "posts", "loginwks"]):
        grouped = [df.loc[df["cohort"] == value, col].dropna() for value in cohort_values]
        box = ax.boxplot(
            grouped,
            tick_labels=[str(int(value)) for value in cohort_values],
            patch_artist=True,
            showfliers=True,
            boxprops={"facecolor": "#4aa3df", "edgecolor": "#12355b", "linewidth": 1.0},
            medianprops={"color": "#12355b", "linewidth": 1.4},
            whiskerprops={"color": "#5d7896"},
            capprops={"color": "#5d7896"},
            flierprops={
                "marker": "o",
                "markerfacecolor": "#73c2e8",
                "markeredgecolor": "#12355b",
                "markersize": 3,
                "alpha": 0.65,
            },
        )
        for patch, color in zip(box["boxes"], COPE_PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        ax.set_title(col)
        ax.set_xlabel("Cohort")
        ax.set_ylabel(col)
        soften_axes(ax, axis="y")
    fig.suptitle("COPE engagement by cohort", y=1.02)
    save_current(fig, "cope", "04_cope_engagement_by_cohort.png")


def main():
    apply_plot_style()
    plot_adsl()
    plot_adqs()
    plot_cope()
    print(f"Saved four-chart PPT overviews to: {OUT_DIR}")


if __name__ == "__main__":
    main()
