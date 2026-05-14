from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data_raw" / "ADSL.csv"
OUT_DIR = ROOT / "outputs" / "adsl_overview"

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

MISSING_COLOR = "#c9d6e2"
GRID_COLOR = "#dfe7ef"
TEXT_COLOR = "#183247"
BACKGROUND_COLOR = "#fbfdff"

CATEGORICAL_GROUPS = {
    "Trial structure": ["scrnfl", "trt", "cohort"],
    "Caregiver demographics": ["gender", "ethnic", "employ", "educate", "marry", "living"],
    "Care relationship": ["relate", "gendercfp", "disease", "onset", "multiple", "time"],
}

VARIABLE_LABELS = {
    "scrnfl": "Screened flag",
    "trt": "Treatment arm",
    "cohort": "Cohort",
    "gender": "Caregiver gender",
    "age": "Caregiver age",
    "ethnic": "Ethnicity",
    "employ": "Employment",
    "educate": "Education",
    "marry": "Marital status",
    "relate": "Relationship to care recipient",
    "living": "Living with care recipient",
    "gendercfp": "Care recipient gender",
    "agecfp": "Care recipient age",
    "disease": "Condition group",
    "onset": "Disease onset",
    "multiple": "Multiple conditions",
    "time": "Time since diagnosis",
    "stdate": "Start date",
    "eddate": "End date",
}


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


def load_adsl():
    df = pd.read_csv(DATA_PATH, na_values=["NA", ""])

    date_cols = ["stdate", "eddate"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_cols = [
        col
        for col in df.columns
        if col not in ["scrnfl", "stdate", "sttime", "eddate", "edtime"]
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def plot_sample_flow(df):
    metrics = pd.Series(
        {
            "Total records": len(df),
            "Screened records": df["scrnfl"].eq("Y").sum() if "scrnfl" in df else 0,
            "Randomised records": df["trt"].notna().sum() if "trt" in df else 0,
            "Baseline demographics": df["age"].notna().sum() if "age" in df else 0,
            "Start/end date available": df[["stdate", "eddate"]].notna().all(axis=1).sum()
            if {"stdate", "eddate"}.issubset(df.columns)
            else 0,
        }
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(metrics.index[::-1], metrics.values[::-1], color=PALETTE[: len(metrics)])
    ax.set_xlabel("Number of participants / records")
    ax.set_title("ADSL subject-level overview")
    ax.set_xlim(0, max(metrics.max() * 1.15, 1))
    soften_axes(ax, axis="x")

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + metrics.max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)} ({width / len(df):.1%})",
            va="center",
            fontsize=9,
            color=TEXT_COLOR,
        )

    save_current(fig, "adsl_sample_flow.png")


def plot_missingness(df):
    missing_pct = df.isna().mean().mul(100).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.34 * len(missing_pct))))
    colors = [MISSING_COLOR if value == 0 else "#2f80c0" for value in missing_pct.values]
    ax.barh(missing_pct.index, missing_pct.values, color=colors)
    ax.set_xlabel("Missing (%)")
    ax.set_title("ADSL missingness by variable")
    ax.set_xlim(0, max(5, missing_pct.max() * 1.15))
    soften_axes(ax, axis="x")

    for idx, value in enumerate(missing_pct.values):
        if value > 0:
            ax.text(
                value + missing_pct.max() * 0.015,
                idx,
                f"{value:.1f}%",
                va="center",
                fontsize=8,
                color=TEXT_COLOR,
            )

    save_current(fig, "adsl_missingness.png")


def plot_categorical_grid(df, columns, filename, title):
    columns = [col for col in columns if col in df.columns]
    if not columns:
        return

    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, max(3.4 * n_rows, 4)))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, col in zip(axes, columns):
        counts = sorted_value_counts(df[col])
        prefix = col if col in {"trt", "cohort"} else None
        labels = [clean_label(idx, prefix=prefix) for idx in counts.index]
        colors = [MISSING_COLOR if label == "Missing" else PALETTE[i % len(PALETTE)] for i, label in enumerate(labels)]
        ax.bar(labels, counts.values, color=colors)
        ax.set_title(VARIABLE_LABELS.get(col, col), fontsize=10)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        soften_axes(ax, axis="y")

        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")

    for ax in axes[len(columns) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, y=1.01)
    save_current(fig, filename)


def plot_age_distributions(df):
    age_cols = [col for col in ["age", "agecfp"] if col in df.columns]
    fig, axes = plt.subplots(1, len(age_cols), figsize=(5.6 * len(age_cols), 4.4))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, col, color in zip(axes, age_cols, ["#1f5f99", "#4aa3df"]):
        values = df[col].dropna()
        ax.hist(values, bins=24, color=color, edgecolor=BACKGROUND_COLOR, linewidth=0.7)
        ax.axvline(values.median(), color="#12355b", linestyle="--", linewidth=1.2)
        ax.set_title(VARIABLE_LABELS.get(col, col), fontsize=11)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        soften_axes(ax, axis="y")
        ax.text(
            0.98,
            0.92,
            f"n={len(values)}\nmedian={values.median():.1f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "#ffffff",
                "edgecolor": "#c9d6e2",
            },
        )

    fig.suptitle("ADSL age distributions", fontsize=14, y=1.02)
    save_current(fig, "adsl_age_distributions.png")


def plot_dates(df):
    if "stdate" not in df.columns:
        return

    monthly = df["stdate"].dropna().dt.to_period("M").value_counts().sort_index()
    if monthly.empty:
        return

    labels = [str(period) for period in monthly.index]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(labels, monthly.values, marker="o", color="#1f5f99", linewidth=2.2)
    ax.fill_between(labels, monthly.values, alpha=0.16, color="#4aa3df")
    ax.set_title("ADSL records by start month")
    ax.set_xlabel("Start month")
    ax.set_ylabel("Number of records")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    soften_axes(ax, axis="y")

    save_current(fig, "adsl_start_month_trend.png")


def write_summary(df):
    continuous_cols = [col for col in ["age", "agecfp"] if col in df.columns]
    id_cols = [col for col in ["ID"] if col in df.columns]

    top_values = []
    for col in df.columns:
        if col in continuous_cols or col in id_cols:
            top_values.append("")
        else:
            top_values.append(
                "; ".join(
                    f"{clean_label(idx)}: {count}"
                    for idx, count in sorted_value_counts(df[col]).head(5).items()
                )
            )

    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_missing_n": df.notna().sum(),
            "missing_n": df.isna().sum(),
            "missing_pct": df.isna().mean().mul(100).round(2),
            "unique_n": df.nunique(dropna=True),
            "top_values": top_values,
        }
    )

    if continuous_cols:
        continuous_summary = df[continuous_cols].describe().transpose()
        summary = summary.join(continuous_summary, how="left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.index.name = "variable"
    summary.to_csv(OUT_DIR / "adsl_overview_summary.csv")


def main():
    apply_plot_style()
    df = load_adsl()

    plot_sample_flow(df)
    plot_missingness(df)
    plot_age_distributions(df)
    plot_dates(df)

    for group_name, columns in CATEGORICAL_GROUPS.items():
        filename = f"adsl_{group_name.lower().replace(' ', '_')}.png"
        plot_categorical_grid(df, columns, filename, f"ADSL {group_name.lower()} distributions")

    write_summary(df)
    print(f"Saved ADSL overview plots and summary to: {OUT_DIR}")


if __name__ == "__main__":
    main()
