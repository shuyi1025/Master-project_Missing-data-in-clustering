import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


ROOT = Path("/rds/general/user/sp4024/home/final_project")
DATA_DIR = ROOT / "data_raw"
OUT_DIR = ROOT / "outputs" / "simulation_gmm_kmeans"

CSV_PATH = DATA_DIR / "COPE_Final_Indicators.csv"
DTA_PATH = DATA_DIR / "COPE_Final_Indicators.dta"

ID_COL = "ID"
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

K = 5
N_SYNTHETIC = 2000
RANDOM_STATE = 123


def read_cope_data():
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, na_values=["NA", ""])
    elif DTA_PATH.exists():
        df = pd.read_stata(DTA_PATH)
    else:
        raise FileNotFoundError(
            "Could not find COPE_Final_Indicators.csv or COPE_Final_Indicators.dta "
            f"in {DATA_DIR}"
        )

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def make_scaled_complete_cases(df):
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected engagement columns: {missing_cols}")

    complete_cases = df.dropna(subset=FEATURE_COLS).copy()
    x_raw = complete_cases[FEATURE_COLS].astype(float)

    if (x_raw < 0).any().any():
        negative_cols = x_raw.columns[(x_raw < 0).any()].tolist()
        raise ValueError(f"log1p transform expects non-negative values: {negative_cols}")

    x_log = np.log1p(x_raw)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_log)
    x_scaled = pd.DataFrame(x_scaled, columns=FEATURE_COLS, index=complete_cases.index)

    return complete_cases, x_scaled, scaler


def fit_gmm(x_scaled):
    gmm = GaussianMixture(
        n_components=K,
        covariance_type="full",
        n_init=20,
        max_iter=1000,
        reg_covar=1e-6,
        random_state=RANDOM_STATE,
    )
    gmm.fit(x_scaled)

    if not gmm.converged_:
        raise RuntimeError("GMM did not converge.")

    return gmm


def generate_synthetic_data(gmm):
    rng = np.random.default_rng(RANDOM_STATE)

    true_gmm_component = rng.choice(
        np.arange(K),
        size=N_SYNTHETIC,
        p=gmm.weights_,
    )

    x_synthetic = np.zeros((N_SYNTHETIC, len(FEATURE_COLS)))

    for k in range(K):
        row_idx = np.where(true_gmm_component == k)[0]
        if len(row_idx) == 0:
            continue

        x_synthetic[row_idx, :] = rng.multivariate_normal(
            mean=gmm.means_[k],
            cov=gmm.covariances_[k],
            size=len(row_idx),
        )

    x_synthetic = pd.DataFrame(x_synthetic, columns=FEATURE_COLS)
    return x_synthetic, true_gmm_component


def run_full_data_kmeans(x_synthetic):
    kmeans = KMeans(
        n_clusters=K,
        random_state=RANDOM_STATE,
        n_init=50,
    )
    labels = kmeans.fit_predict(x_synthetic) + 1
    return kmeans, labels


def save_outputs(
    complete_cases,
    x_scaled_observed,
    scaler,
    gmm,
    x_synthetic_scaled,
    true_gmm_component,
    kmeans,
    kmeans_labels,
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gmm_parameters = pd.DataFrame(
        {
            "component": np.arange(1, K + 1),
            "mixture_weight": gmm.weights_,
        }
    )
    gmm_parameters.to_csv(OUT_DIR / "gmm_mixture_weights.csv", index=False)

    gmm_means = pd.DataFrame(gmm.means_, columns=FEATURE_COLS)
    gmm_means.insert(0, "component", np.arange(1, K + 1))
    gmm_means.to_csv(OUT_DIR / "gmm_means_standardised.csv", index=False)

    synthetic_scaled = x_synthetic_scaled.copy()
    synthetic_scaled.insert(0, ID_COL, np.arange(1, N_SYNTHETIC + 1))
    synthetic_scaled["true_gmm_component"] = true_gmm_component + 1
    synthetic_scaled["kmeans_full_data_cluster"] = kmeans_labels
    synthetic_scaled.to_csv(OUT_DIR / "synthetic_complete_standardised.csv", index=False)

    synthetic_log = scaler.inverse_transform(x_synthetic_scaled)
    synthetic_original = pd.DataFrame(np.expm1(synthetic_log), columns=FEATURE_COLS).clip(lower=0)
    synthetic_original.insert(0, ID_COL, np.arange(1, N_SYNTHETIC + 1))
    synthetic_original["true_gmm_component"] = true_gmm_component + 1
    synthetic_original["kmeans_full_data_cluster"] = kmeans_labels
    synthetic_original.to_csv(OUT_DIR / "synthetic_complete_original_scale.csv", index=False)

    kmeans_centres = pd.DataFrame(kmeans.cluster_centers_, columns=FEATURE_COLS)
    kmeans_centres.insert(0, "cluster", np.arange(1, K + 1))
    kmeans_centres.to_csv(OUT_DIR / "kmeans_centres_standardised.csv", index=False)

    profile = (
        synthetic_original.groupby("kmeans_full_data_cluster")[FEATURE_COLS]
        .agg(["count", "mean", "median"])
        .sort_index()
    )
    profile.columns = [f"{col}_{stat}" for col, stat in profile.columns]

    cluster_sizes = synthetic_original["kmeans_full_data_cluster"].value_counts().sort_index()
    profile.insert(0, "n", cluster_sizes)
    profile.insert(1, "pct", (cluster_sizes / N_SYNTHETIC * 100).round(1))
    profile.to_csv(OUT_DIR / "kmeans_cluster_profile_original_scale.csv")

    metrics = pd.DataFrame(
        [
            {
                "n_observed_complete_cases": len(complete_cases),
                "n_synthetic": N_SYNTHETIC,
                "k": K,
                "gmm_converged": gmm.converged_,
                "gmm_n_iter": gmm.n_iter_,
                "kmeans_inertia": kmeans.inertia_,  #K-means inertia is the sum of squared distances between each data point and the centroid of the cluster it belongs to.
                "silhouette": silhouette_score(x_synthetic_scaled, kmeans_labels),
                "ari_kmeans_vs_true_gmm_component": adjusted_rand_score(
                    true_gmm_component + 1,
                    kmeans_labels,
                ),
            }
        ]
    )
    metrics.to_csv(OUT_DIR / "benchmark_metrics.csv", index=False)

    x_scaled_observed.to_csv(OUT_DIR / "observed_complete_cases_standardised.csv", index=False)


def main():
    df = read_cope_data()
    complete_cases, x_scaled_observed, scaler = make_scaled_complete_cases(df)

    print(f"Original sample size: {len(df)}")
    print(f"Complete cases used for GMM: {len(complete_cases)}")

    gmm = fit_gmm(x_scaled_observed)
    print(f"GMM converged: {gmm.converged_}")
    print(f"GMM iterations: {gmm.n_iter_}")
    print("GMM mixture weights:")
    print(np.round(gmm.weights_, 3))

    x_synthetic_scaled, true_gmm_component = generate_synthetic_data(gmm)
    print(f"Synthetic data shape: {x_synthetic_scaled.shape}")

    kmeans, kmeans_labels = run_full_data_kmeans(x_synthetic_scaled)
    print("Full-data K-means cluster counts:")
    print(pd.Series(kmeans_labels).value_counts().sort_index())
    print(
        "ARI between K-means labels and true GMM components:",
        round(adjusted_rand_score(true_gmm_component + 1, kmeans_labels), 3),
    )

    save_outputs(
        complete_cases=complete_cases,
        x_scaled_observed=x_scaled_observed,
        scaler=scaler,
        gmm=gmm,
        x_synthetic_scaled=x_synthetic_scaled,
        true_gmm_component=true_gmm_component,
        kmeans=kmeans,
        kmeans_labels=kmeans_labels,
    )
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
