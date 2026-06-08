from pathlib import Path
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data_raw" / "COPE_Final_Indicators.csv"
DTA_PATH = ROOT / "data_raw" / "COPE_Final_Indicators.dta"

RANDOM_STATE = 42
K_MIN = 2
K_MAX = 8
CHOSEN_K = 5  # Set to None to use silhouette-based selection instead.
CREATE_UMAP = False  # UMAP is off by default because umap-learn is slow to import in this environment.
OUT_DIR = ROOT / "outputs" / (
    f"kmeans_fully_observed_engagement_k{CHOSEN_K}"
    if CHOSEN_K is not None
    else "kmeans_fully_observed_engagement"
)

# Use behavioural engagement measures only. Trial design/access variables are
# kept for interpretation, but are not used to form the clusters.
FEATURE_COLS = [
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
    "sdwkpv",
    "sdptp",
    "sdate",
    "rate_wkpv",
    "rate_ptp",
    "rate_ate",
]

META_COLS = ["ID", "cohort", "trt", "registered", "activated"]
KEY_PROFILE_COLS = ["totalmins", "pageviews", "posts", "loginwks", "totaldays", "rate_wkpv", "rate_ptp", "rate_ate"]

PALETTE = [
    "#12355b",
    "#174a7c",
    "#1f5f99",
    "#2f80c0",
    "#4aa3df",
    "#73c2e8",
    "#9bd6f0",
    "#5d7896",
]
PCA_PALETTE = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]
BOXPLOT_PALETTE = [
    "#264653",
    "#2a9d8f",
    "#e9c46a",
    "#f4a261",
    "#e76f51",
    "#8ab17d",
    "#6d597a",
    "#b56576",
]
GRID_COLOR = "#dfe7ef"
TEXT_COLOR = "#183247"
BACKGROUND_COLOR = "#fbfdff"
DIAGNOSTIC_LINE_COLOR = "#12355b"


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
            "figure.titlesize": 14,
            "figure.titleweight": "semibold",
        }
    )


def save_current(fig, filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def soften_axes(ax, axis="y"):
    ax.grid(axis=axis, color=GRID_COLOR, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8ea5bb")
    ax.spines["bottom"].set_color("#8ea5bb")


def read_engagement_data():
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, na_values=["NA", ""])
    elif DTA_PATH.exists():
        df = pd.read_stata(DTA_PATH)
    else:
        raise FileNotFoundError(
            "Could not find COPE_Final_Indicators.csv or COPE_Final_Indicators.dta "
            f"in {CSV_PATH.parent}"
        )

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def make_fully_observed_features(df):
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected engagement columns: {missing_cols}")

    fully_observed = df.dropna(subset=FEATURE_COLS).copy()
    x_raw = fully_observed[FEATURE_COLS].astype(float)

    if (x_raw < 0).any().any():
        negative_cols = x_raw.columns[(x_raw < 0).any()].tolist()
        raise ValueError(f"log1p transform expects non-negative engagement values: {negative_cols}")

    x_log = np.log1p(x_raw)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_log)
    return fully_observed, x_raw, x_log, x_scaled, scaler


def evaluate_candidate_k(x_scaled):
    max_k = min(K_MAX, len(x_scaled) - 1)
    rows = []
    for k in range(K_MIN, max_k + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
        labels = model.fit_predict(x_scaled)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(x_scaled, labels),
            }
        )
    evaluation = pd.DataFrame(rows)
    best_k = int(evaluation.loc[evaluation["silhouette"].idxmax(), "k"])
    return evaluation, best_k


def fit_kmeans(x_scaled, k):
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
    labels = model.fit_predict(x_scaled) + 1
    return model, labels


def make_cluster_outputs(df, x_scaled, scaler, model, labels):
    clustered = df.copy()
    clustered["cluster"] = labels

    assignment_cols = [col for col in META_COLS if col in clustered.columns] + ["cluster"] + FEATURE_COLS
    assignments = clustered[assignment_cols].sort_values(["cluster", "ID"])

    profile = (
        clustered.groupby("cluster")[FEATURE_COLS]
        .agg(["count", "mean", "median"])
        .sort_index()
    )
    profile.columns = [f"{col}_{stat}" for col, stat in profile.columns]

    meta_profile_cols = [col for col in ["cohort", "trt", "registered", "activated"] if col in clustered.columns]
    meta_profile = clustered.groupby("cluster")[meta_profile_cols].mean(numeric_only=True)
    meta_profile.columns = [f"{col}_mean" for col in meta_profile.columns]

    cluster_sizes = clustered["cluster"].value_counts().sort_index().rename("n")
    cluster_pct = (cluster_sizes / len(clustered) * 100).round(1).rename("pct")
    profile = pd.concat([cluster_sizes, cluster_pct, meta_profile, profile], axis=1)

    centers_log = scaler.inverse_transform(model.cluster_centers_)
    centers_original = np.expm1(centers_log).clip(min=0)
    centers = pd.DataFrame(centers_original, columns=FEATURE_COLS)
    centers.insert(0, "cluster", np.arange(1, len(centers) + 1))

    centers_z = pd.DataFrame(model.cluster_centers_, columns=FEATURE_COLS)
    centers_z.insert(0, "cluster", np.arange(1, len(centers_z) + 1))

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_scores = pca.fit_transform(x_scaled)
    pca_df = pd.DataFrame(
        {
            "ID": clustered["ID"].values,
            "cluster": labels,
            "PC1": pca_scores[:, 0],
            "PC2": pca_scores[:, 1],
        }
    )

    return assignments, profile, centers, centers_z, pca_df, pca


def make_pca_outputs(pca, n_complete_cases):
    component_coefficients = pca.components_.T
    correlation_loadings = component_coefficients * np.sqrt(pca.explained_variance_)

    loadings = pd.DataFrame(
        {
            "variable": FEATURE_COLS,
            "PC1_loading": correlation_loadings[:, 0],
            "PC2_loading": correlation_loadings[:, 1],
            "PC1_abs_loading": np.abs(correlation_loadings[:, 0]),
            "PC2_abs_loading": np.abs(correlation_loadings[:, 1]),
            "PC1_rank": pd.Series(np.abs(correlation_loadings[:, 0])).rank(
                ascending=False, method="min"
            ),
            "PC2_rank": pd.Series(np.abs(correlation_loadings[:, 1])).rank(
                ascending=False, method="min"
            ),
            "PC1_component_coefficient": component_coefficients[:, 0],
            "PC2_component_coefficient": component_coefficients[:, 1],
        }
    )

    variance = pd.DataFrame(
        {
            "component": ["PC1", "PC2"],
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance_percent": pca.explained_variance_ratio_ * 100,
            "cumulative_variance_percent": np.cumsum(pca.explained_variance_ratio_) * 100,
            "n_complete_cases": [n_complete_cases, n_complete_cases],
        }
    )
    return loadings, variance


def make_umap_scores(df, x_scaled, labels):
    try:
        from umap import UMAP
    except ImportError:
        return None

    reducer = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    umap_scores = reducer.fit_transform(x_scaled)
    return pd.DataFrame(
        {
            "ID": df["ID"].values,
            "cluster": labels,
            "UMAP1": umap_scores[:, 0],
            "UMAP2": umap_scores[:, 1],
        }
    )


def plot_k_diagnostics(evaluation):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].plot(evaluation["k"], evaluation["inertia"], marker="o", color=DIAGNOSTIC_LINE_COLOR, linewidth=2)
    axes[0].set_title("Elbow plot")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Within-cluster sum of squares")
    axes[0].set_xticks(evaluation["k"])
    soften_axes(axes[0], axis="y")

    axes[1].plot(evaluation["k"], evaluation["silhouette"], marker="o", color=DIAGNOSTIC_LINE_COLOR, linewidth=2)
    axes[1].set_title("Average silhouette score")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette")
    axes[1].set_xticks(evaluation["k"])
    soften_axes(axes[1], axis="y")

    fig.suptitle("K-means cluster number diagnostics", y=1.02)
    save_current(fig, "kmeans_k_diagnostics.png")


def plot_pca_clusters(pca_df, pca):
    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    for idx, cluster in enumerate(sorted(pca_df["cluster"].unique())):
        subset = pca_df[pca_df["cluster"] == cluster]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=46,
            alpha=0.78,
            color=PCA_PALETTE[idx % len(PCA_PALETTE)],
            edgecolor=BACKGROUND_COLOR,
            linewidth=0.6,
            label=f"Cluster {cluster} (n={len(subset)})",
        )

    explained = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    ax.set_title("K-means clusters projected onto first two PCs")
    ax.legend(frameon=False, loc="best")
    soften_axes(ax, axis="both")
    save_current(fig, "kmeans_pca_clusters.png")


def plot_umap_clusters(umap_df):
    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    for idx, cluster in enumerate(sorted(umap_df["cluster"].unique())):
        subset = umap_df[umap_df["cluster"] == cluster]
        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=46,
            alpha=0.78,
            color=PALETTE[idx % len(PALETTE)],
            edgecolor=BACKGROUND_COLOR,
            linewidth=0.6,
            label=f"Cluster {cluster} (n={len(subset)})",
        )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("K-means clusters projected with UMAP")
    ax.legend(frameon=False, loc="best")
    soften_axes(ax, axis="both")
    save_current(fig, "kmeans_umap_clusters.png")


def plot_cluster_profile_heatmap(centers_z):
    heatmap_data = centers_z.set_index("cluster")
    fig, ax = plt.subplots(figsize=(12.2, 4.8))
    im = ax.imshow(heatmap_data.values, cmap="coolwarm", aspect="auto", vmin=-2.2, vmax=2.2)
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_yticklabels([f"Cluster {int(idx)}" for idx in heatmap_data.index])
    ax.set_title("Cluster centres on standardised log1p engagement variables")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Standardised cluster centre")
    save_current(fig, "kmeans_cluster_centres_heatmap.png")


def plot_key_boxplots(clustered):
    shown_cols = [col for col in KEY_PROFILE_COLS if col in clustered.columns]
    n_cols = 4
    n_rows = (len(shown_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, max(3.1 * n_rows, 4.2)))
    axes = np.atleast_1d(axes).flatten()
    clusters = sorted(clustered["cluster"].unique())

    for ax, col in zip(axes, shown_cols):
        grouped = [np.log1p(clustered.loc[clustered["cluster"] == cluster, col]) for cluster in clusters]
        box = ax.boxplot(
            grouped,
            tick_labels=[str(int(cluster)) for cluster in clusters],
            patch_artist=True,
            showfliers=True,
            boxprops={"facecolor": BOXPLOT_PALETTE[1], "edgecolor": "#264653", "linewidth": 1.0},
            medianprops={"color": "#12355b", "linewidth": 1.4},
            whiskerprops={"color": "#264653"},
            capprops={"color": "#264653"},
            flierprops={
                "marker": "o",
                "markerfacecolor": "#f4a261",
                "markeredgecolor": "#264653",
                "markersize": 3,
                "alpha": 0.65,
            },
        )
        for patch, color in zip(box["boxes"], BOXPLOT_PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        ax.set_title(col)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(f"log1p({col})")
        soften_axes(ax, axis="y")

    for ax in axes[len(shown_cols) :]:
        ax.axis("off")

    fig.suptitle("Key engagement measures by K-means cluster", y=1.01)
    save_current(fig, "kmeans_key_engagement_boxplots.png")


def main():
    apply_plot_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = read_engagement_data()
    fully_observed, x_raw, x_log, x_scaled, scaler = make_fully_observed_features(df)
    evaluation, best_k = evaluate_candidate_k(x_scaled)
    if CHOSEN_K is not None:
        if CHOSEN_K < K_MIN or CHOSEN_K > min(K_MAX, len(x_scaled) - 1):
            raise ValueError(f"CHOSEN_K must be between {K_MIN} and {min(K_MAX, len(x_scaled) - 1)}")
        best_k = CHOSEN_K
    model, labels = fit_kmeans(x_scaled, best_k)

    assignments, profile, centers, centers_z, pca_df, pca = make_cluster_outputs(
        fully_observed,
        x_scaled,
        scaler,
        model,
        labels,
    )
    pca_loadings, pca_variance = make_pca_outputs(pca, len(fully_observed))
    umap_df = make_umap_scores(fully_observed, x_scaled, labels) if CREATE_UMAP else None

    clustered = assignments.copy()
    evaluation.to_csv(OUT_DIR / "kmeans_k_diagnostics.csv", index=False)
    assignments.to_csv(OUT_DIR / "kmeans_cluster_assignments.csv", index=False)
    profile.to_csv(OUT_DIR / "kmeans_cluster_profile.csv")
    centers.to_csv(OUT_DIR / "kmeans_cluster_centres_original_scale.csv", index=False)
    centers_z.to_csv(OUT_DIR / "kmeans_cluster_centres_standardised.csv", index=False)
    pca_df.to_csv(OUT_DIR / "kmeans_pca_scores.csv", index=False)
    pca_loadings.to_csv(OUT_DIR / "pca_loadings.csv", index=False, float_format="%.6f")
    pca_variance.to_csv(OUT_DIR / "pca_explained_variance.csv", index=False, float_format="%.6f")
    if umap_df is not None:
        umap_df.to_csv(OUT_DIR / "kmeans_umap_scores.csv", index=False)

    plot_k_diagnostics(evaluation)
    plot_pca_clusters(pca_df, pca)
    if umap_df is not None:
        plot_umap_clusters(umap_df)
    plot_cluster_profile_heatmap(centers_z)
    plot_key_boxplots(clustered)

    print(f"Read {len(df)} COPE records.")
    print(f"Used {len(fully_observed)} fully observed records for K-means.")
    print(f"Excluded {len(df) - len(fully_observed)} records with missing engagement values.")
    print(f"Selected k={best_k} using maximum average silhouette score.")
    if CREATE_UMAP and umap_df is None:
        print("UMAP plot not created because the 'umap-learn' package is not installed.")
    print(f"Saved K-means outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
