import argparse
import os
import time
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score


ROOT = Path("/rds/general/user/sp4024/home/final_project")
OUTPUT_DIR = ROOT / "outputs" / "simulation204"
SYNTHETIC_SOURCE_PATH = (
    ROOT / "outputs" / "simulation_gmm_kmeans204" / "synthetic_complete_standardised.csv"
)

FEATURE_COLS = [
    "totalmins", "logindays", "loginwks", "pageviews", "posts",
    "ptp", "ate", "totaldays", "act_wkpv", "dur_wkpv",
    "act_ptp", "dur_ptp", "act_ate", "dur_ate", "sdwkpv",
    "sdptp", "sdate", "rate_wkpv", "rate_ptp", "rate_ate",
]

METHOD = "knn"
TRUE_LABEL_COL = "true_gmm_component"
BENCHMARK_LABEL_COL = "kmeans_full_data_cluster"
K = 5
B = 500
MISSINGNESS_MECHANISMS = ["MCAR", "MAR", "MNAR"]
MISSINGNESS_RATES = [0.10, 0.20,0.30,0.40, 0.50]
RANDOM_STATE = 123
KMEANS_N_INIT = 50
KNN_N_NEIGHBORS = 5
MAR_DRIVER_COLS = ["totalmins", "pageviews", "posts"]
MISSINGNESS_LOGIT_SLOPE = -1.25


def seed(*parts):
    value = RANDOM_STATE
    for part in parts:
        if isinstance(part, str):
            part = sum(ord(char) for char in part)
        value = (value * 1_000_003 + int(part)) % (2**32 - 1)
    return value


def zero_based_labels(labels):
    labels = pd.Series(labels).astype(int).to_numpy()
    if labels.min() == 1 and labels.max() == K:
        return labels - 1
    return labels


def load_synthetic_complete_data():
    if not SYNTHETIC_SOURCE_PATH.exists():
        raise FileNotFoundError(
            "Synthetic complete data file not found. Run "
            "scripts/simulation/simulation_gmm_kmeans204.py first. Expected: "
            f"{SYNTHETIC_SOURCE_PATH}"
        )
    df = pd.read_csv(SYNTHETIC_SOURCE_PATH)
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected synthetic feature columns: {missing_cols}")
    return (
        df[FEATURE_COLS].astype(float),
        zero_based_labels(df[TRUE_LABEL_COL]),
        zero_based_labels(df[BENCHMARK_LABEL_COL]),
    )


def run_kmeans(x, random_state):
    model = KMeans(n_clusters=K, random_state=random_state, n_init=KMEANS_N_INIT)
    labels = model.fit_predict(x)
    return model, labels


def make_reference_kmeans_from_labels(x_complete, labels):
    centers = []
    for k in range(K):
        rows = x_complete.loc[labels == k]
        if rows.empty:
            raise ValueError(f"Benchmark cluster {k + 1} has no rows.")
        centers.append(rows.mean(axis=0).to_numpy())
    return SimpleNamespace(cluster_centers_=np.vstack(centers)), labels


def standardize_score(score):
    score = np.asarray(score, dtype=float)
    sd = score.std()
    if sd == 0 or np.isnan(sd):
        return np.zeros_like(score)
    return (score - score.mean()) / sd


def calibrate_intercept(score, target_rate):
    low, high = -20.0, 20.0
    for _ in range(80):
        mid = (low + high) / 2
        probs = 1 / (1 + np.exp(-(mid + MISSINGNESS_LOGIT_SLOPE * score)))
        if probs.mean() < target_rate:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def logistic_missingness_prob(driver, target_rate):
    score = standardize_score(driver)
    intercept = calibrate_intercept(score, target_rate)
    return 1 / (1 + np.exp(-(intercept + MISSINGNESS_LOGIT_SLOPE * score)))


def inject_missingness(x_complete, mechanism, rate, random_state):
    rng = np.random.default_rng(random_state)
    mechanism = mechanism.upper()
    if mechanism == "MCAR":
        mask = rng.random(x_complete.shape) < rate
    elif mechanism == "MAR":
        mask = np.zeros(x_complete.shape, dtype=bool)
        for j, target_col in enumerate(x_complete.columns):
            driver_cols = [
                col for col in MAR_DRIVER_COLS
                if col in x_complete.columns and col != target_col
            ]
            if not driver_cols:
                driver_cols = [col for col in x_complete.columns if col != target_col]
            probs = logistic_missingness_prob(x_complete[driver_cols].mean(axis=1), rate)
            mask[:, j] = rng.random(len(x_complete)) < probs
    elif mechanism == "MNAR":
        mask = np.zeros(x_complete.shape, dtype=bool)
        for j, target_col in enumerate(x_complete.columns):
            probs = logistic_missingness_prob(x_complete[target_col], rate)
            mask[:, j] = rng.random(len(x_complete)) < probs
    else:
        raise ValueError(f"Unknown missingness mechanism: {mechanism}")
    mask = pd.DataFrame(mask, columns=x_complete.columns, index=x_complete.index)
    return x_complete.mask(mask), mask


def as_frame(values, template):
    return pd.DataFrame(values, columns=template.columns, index=template.index)


def impute_knn(x_missing, random_state):
    start = time.perf_counter()
    imputer = KNNImputer(n_neighbors=KNN_N_NEIGHBORS, weights="distance")
    completed = as_frame(imputer.fit_transform(x_missing), x_missing)
    return {
        "completed_datasets": [completed],
        "runtime_seconds": time.perf_counter() - start,
    }


def cluster_proportions(labels):
    counts = np.bincount(labels, minlength=K)
    return counts / counts.sum()


def best_label_permutation(reference_centers, estimated_centers):
    best_perm = None
    best_cost = np.inf
    for perm in permutations(range(K)):
        cost = np.linalg.norm(reference_centers - estimated_centers[list(perm)], axis=1).sum()
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    return list(best_perm)


def compute_metrics(x_completed, estimated_model, estimated_labels, benchmark_model, benchmark_labels):
    perm = best_label_permutation(
        benchmark_model.cluster_centers_,
        estimated_model.cluster_centers_,
    )
    matched_centers = estimated_model.cluster_centers_[perm]
    centroid_distances = np.linalg.norm(
        benchmark_model.cluster_centers_ - matched_centers,
        axis=1,
    )
    estimated_props = cluster_proportions(estimated_labels)[perm]
    benchmark_props = cluster_proportions(benchmark_labels)
    return {
        "ari_vs_full_data_kmeans": adjusted_rand_score(benchmark_labels, estimated_labels),
        "centroid_error_mean": centroid_distances.mean(),
        "centroid_error_max": centroid_distances.max(),
        "cluster_size_error_mean_abs": np.abs(benchmark_props - estimated_props).mean(),
        "silhouette": silhouette_score(x_completed, estimated_labels)
        if len(np.unique(estimated_labels)) > 1
        else np.nan,
    }


def summarise_results(results):
    summary = (
        results.groupby(["mechanism", "target_missing_rate", "method"])
        .agg(
            mean_ari=("ari_vs_full_data_kmeans", "mean"),
            sd_ari=("ari_vs_full_data_kmeans", "std"),
            mean_centroid_error=("centroid_error_mean", "mean"),
            mean_cluster_size_error=("cluster_size_error_mean_abs", "mean"),
            mean_silhouette=("silhouette", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .reset_index()
    )
    summary_path = OUTPUT_DIR / f"simulation_summary_{METHOD}.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path


def run(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mechanisms = args.mechanisms or MISSINGNESS_MECHANISMS
    rates = args.rates or MISSINGNESS_RATES
    b_repetitions = args.b or B
    output_name = args.output or f"simulation_results_{METHOD}.csv"
    if args.smoke:
        rates = args.rates or [0.10]
        b_repetitions = args.b or 1
        output_name = args.output or f"simulation_results_{METHOD}_smoke.csv"

    x_complete, true_gmm_labels, benchmark_labels = load_synthetic_complete_data()
    benchmark_model, benchmark_labels = make_reference_kmeans_from_labels(x_complete, benchmark_labels)
    reference_ari = adjusted_rand_score(true_gmm_labels, benchmark_labels)
    result_path = OUTPUT_DIR / output_name
    print(f"Method: {METHOD}")
    print(f"Loaded complete synthetic records: {len(x_complete)}")
    print(f"Synthetic source: {SYNTHETIC_SOURCE_PATH}")
    print(f"Writing results to: {result_path}")

    rows = []
    for b in range(1, b_repetitions + 1):
        for mechanism in mechanisms:
            for rate in rates:
                x_missing, mask = inject_missingness(
                    x_complete,
                    mechanism,
                    rate,
                    seed("missingness", b, int(rate * 1000), mechanism),
                )
                method_result = impute_knn(
                    x_missing,
                    seed("method", b, int(rate * 1000), mechanism, METHOD),
                )
                metric_rows = []
                for m, x_completed in enumerate(method_result["completed_datasets"], start=1):
                    model, labels = run_kmeans(
                        x_completed,
                        seed("kmeans", b, int(rate * 1000), mechanism, METHOD, m),
                    )
                    metric_rows.append(
                        compute_metrics(x_completed, model, labels, benchmark_model, benchmark_labels)
                    )
                averaged = pd.DataFrame(metric_rows).mean(numeric_only=True).to_dict()
                rows.append(
                    {
                        "replication": b,
                        "n_synthetic": len(x_complete),
                        "synthetic_source": str(SYNTHETIC_SOURCE_PATH),
                        "mechanism": mechanism,
                        "target_missing_rate": rate,
                        "observed_missing_rate": float(mask.to_numpy().mean()),
                        "method": METHOD,
                        "n_imputed_datasets": len(method_result["completed_datasets"]),
                        "runtime_seconds": method_result["runtime_seconds"],
                        "ari_true_gmm_vs_benchmark": reference_ari,
                        **averaged,
                    }
                )
                print(
                    f"b={b} {mechanism} rate={rate:.2f} method={METHOD} "
                    f"ARI={averaged['ari_vs_full_data_kmeans']:.3f}"
                )
    results = pd.DataFrame(rows)
    results.to_csv(result_path, index=False)
    print(f"Saved summary to: {summarise_results(results)}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run the complete KNN simulation script.")
    parser.add_argument("--smoke", action="store_true", help="Run B=1 at 10% missingness.")
    parser.add_argument("--b", type=int, default=None, help=f"Replications. Default: {B}.")
    parser.add_argument("--mechanisms", nargs="+", choices=MISSINGNESS_MECHANISMS, default=None)
    parser.add_argument("--rates", nargs="+", type=float, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    start = time.perf_counter()
    out = run(parse_args())
    print("\nSimulation complete.")
    print(
        out.groupby(["mechanism", "target_missing_rate", "method"])[
            "ari_vs_full_data_kmeans"
        ]
        .mean()
        .round(3)
    )
    print(f"Elapsed seconds: {time.perf_counter() - start:.1f}")
